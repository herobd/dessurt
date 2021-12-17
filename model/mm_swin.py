from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pairing_g_graph_layoutlm import  runLayoutLM
from model.pos_encode import PositionalEncoding, UniformRealEmbedding,PositiveRealEmbedding, ReturnPositionalEncoding
from model.rel_pos_im_transformer import QTransformerLayer
from model.swin_transformer import ConvPatchEmbed, SwinTransformerBlock, PatchMerging
from model.trans_pooling import QPooler
try:
    from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
    from transformers import LayoutLMTokenizer, LayoutLMModel
except:
    pass
from model.special_token_embedder import SpecialTokenEmbedder
from utils.character_tokenizer import CharacterTokenizer
from collections import defaultdict
from timm.models.layers import trunc_normal_
import math, random
from utils.util import calcXYWH
from testtest import PRINT_ATT,ATT_TEXT,attDisplay,NUMS

import timeit



class MmSwin(BaseModel):
    def __init__(self,config):
        super(MmSwin, self).__init__(config)
        self.image_size = config['image_size'] #start at 512?
        dropout = 0 if 'no_dropout' in config and  config['no_dropout'] else 0.1
        lighter_conv_patch_emb = config['lighter_conv_patch_emb'] if 'lighter_conv_patch_emb' in config else False
        init_from_pretrained = config.get('init_from_pretrained',True)
        use_special_question_tokens = config.get('use_special_question_tokens',True)
        self.use_set_length = config.get('use_set_length',True)
        self.max_q_tokens = config.get('max_q_tokens',32)
        self.max_a_tokens = config.get('max_a_tokens',512)

        blocks_per_level = config['blocks_per_level'] #[2,2,6,2] -> in:512,emb:64 then 64,32,16,8
        use_swin = config.get('use_swin',[True]*sum(blocks_per_level  ))
        swin_cross_attention = config.get('swin_cross_attention',[True]*sum(blocks_per_level))
        if blocks_per_level is not None:
            if 'swin_text_downsample_all' in config:
                swin_text_downsample = [[d,d] if type(d) is bool else d for d in config['swin_text_downsample_all']]
                swin_text_downsample_dense=True
                assert (not swin_text_downsample[-1][0] and not swin_text_downsample[-1][1]) and "Shouldn't downsample final. Not used."
                self.downsample_q = sum(d[1] for d in swin_text_downsample)

            else:
                swin_text_downsample = config['swin_text_downsample'] if 'swin_text_downsample' in config else [False]*len(blocks_per_level)
                swin_text_downsample_dense=False
                self.downsample_q = 0

            window_size = config['window_size'] #7
            if type(window_size) is int:
                window_size = [window_size]*len(blocks_per_level)
            d_model = config['decode_dim']
            dim_ff = config['dim_ff']
            nhead = config['decode_num_heads']
            swin_nhead = config['swin_nheads'] #[3,6,12,24] | [2,6,12,12] probably don't need as much later
            im_embed_dim = config['im_embed_dim'] #96 -> 96,192,384,768 | 64->64,128,256,512

        if isinstance(self.image_size,int):
            self.image_size = (self.image_size,self.image_size)

        self.max_pred_len = self.max_a_tokens


        if 'pre_trained' in config:
            pre_trained_patch_emb = config['patch_emb'] if 'patch_emb' in config else None
        else:
            pre_trained_patch_emb = None

        out_token_type = 'char' if config.get('char_output',False) else 'word'
        in_token_type = 'char' if config.get('char_tokens',False) else 'word'

        out_token_type = config.get('out_token_type',out_token_type)
        in_token_type = config.get('in_token_type',in_token_type)


        if in_token_type == 'char':
            self.tokenizer = CharacterTokenizer()
            self.SEP_TOKEN=self.tokenizer.SEP_index
            self.CLS_TOKEN=self.tokenizer.CLS_index
        elif in_token_type == 'word':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.SEP_TOKEN= 102
            self.CLS_TOKEN= 101

        if in_token_type == out_token_type:
            self.decode_tokenizer = self.tokenizer
            self.DECODE_SEP_TOKEN=self.SEP_TOKEN
            self.DECODE_CLS_TOKEN=self.CLS_TOKEN
        elif out_token_type == 'char':
            self.decode_tokenizer = CharacterTokenizer()
            self.DECODE_SEP_TOKEN=self.decode_tokenizer.SEP_index
            self.DECODE_CLS_TOKEN=self.decode_tokenizer.CLS_index


        if init_from_pretrained=='distilbert':
            init_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            init_emb = init_model.embeddings.word_embeddings


        self.text_embedding = nn.Embedding(self.tokenizer.vocab_size, d_model)
        if init_from_pretrained:
            self.text_embedding.weight.data[:,:d_model] = init_emb.weight.data[:,:d_model]

        if use_special_question_tokens:
            self.query_special_token_embedder = SpecialTokenEmbedder(d_model)
            self.query_special_start_token_embedder = SpecialTokenEmbedder(d_model)
        else:
            self.query_special_token_embedder = None
            self.query_special_start_token_embedder = None


        self.q_pos_1d_enc = PositionalEncoding(d_model,dropout=dropout,max_len=1000)
        self.a_pos_1d_enc = PositionalEncoding(d_model,dropout=dropout,max_len=1000,offset_start=1000)



        self.patch_embed =  ConvPatchEmbed(
                img_size=self.image_size, 
                embed_dim=im_embed_dim,
                norm_layer=nn.LayerNorm,
                lighter=lighter_conv_patch_emb,
                in_chans=2) #now includes the mask channel
        if pre_trained_patch_emb is not None:
            checkpoint = torch.load(pre_trained_patch_emb, map_location=lambda storage, location: storage)
            pe_state_dict=self.patch_embed.state_dict()
            for name,value in checkpoint['state_dict']:
                if name.startswith('cnn.'):
                    pe_state_dict[name]=value

            self.patch_embed.load_state_dict(pe_state_dict)

        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution
        self.patch_scale_x = 1/self.patch_embed.patch_size[1]
        self.patch_scale_y = 1/self.patch_embed.patch_size[0]


        self.absolute_2dpos_embed = nn.Parameter(torch.zeros(1, num_patches, im_embed_dim))
        trunc_normal_(self.absolute_2dpos_embed, std=.02)

        mlp_ratio=4.
        qkv_bias=True
        qk_scale=None
        drop_rate=0.
        attn_drop_rate=0.
        drop_path_rate=dropout
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(blocks_per_level))]  # stochastic depth decay rule
        self.swin_layers=nn.ModuleList()
        for level,blocks in enumerate(blocks_per_level):
            d_im = int(im_embed_dim * 2 ** level)
            cur_resolution = (self.patches_resolution[0]//(2**level), self.patches_resolution[1]//(2**level))
            patch_size = (self.image_size[0]/cur_resolution[0],self.image_size[1]/cur_resolution[1])
            for block in range(blocks):
                last = level<len(blocks_per_level)-1 and block == blocks-1
                if (swin_text_downsample_dense and swin_text_downsample[len(self.swin_layers)][1]) or (last and swin_text_downsample[level]):
                    q_pool = QPooler(d_model)
                else:
                    q_pool = None
                do_cross_att = swin_cross_attention[len(self.swin_layers)]
                use_swin_here = use_swin[len(self.swin_layers)]
                self.swin_layers.append( nn.ModuleList( [
                    SwinTransformerBlock(dim=d_im, 
                                input_resolution=cur_resolution,
                                 num_heads=swin_nhead[level], 
                                 window_size=window_size[level],
                                 shift_size=0 if (len(self.swin_layers) % 2 == 0) else window_size[level] // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop_rate, 
                                 attn_drop=attn_drop_rate,
                                 drop_path=dpr[len(self.swin_layers)],
                                 norm_layer=nn.LayerNorm,
                                 sees_docq=do_cross_att) if use_swin_here else None,
                    (nn.Linear(d_model,d_im,bias=False) if d_model!=d_im else nn.Identity()) if do_cross_att else None,
                    QTransformerLayer(d_model,nhead,dim_ff,dropout=dropout),
                    nn.Linear(d_im,d_model,bias=False) if d_model!=d_im else nn.Identity(),
                    PatchMerging(cur_resolution, dim=d_im, norm_layer=nn.LayerNorm) if last else None,
                    None,
                    q_pool
                    ] ) )



        self.final_resolution = cur_resolution
        upsample_net = [nn.ConvTranspose2d(d_im,d_im//2,4,2,1),
                        nn.InstanceNorm2d(d_im//2),
                        nn.Dropout2d(p=0.125,inplace=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(d_im//2,d_im//4,3,1,1),
                        nn.InstanceNorm2d(d_im//4),
                        nn.Dropout2d(p=0.125,inplace=True),
                        nn.ReLU(inplace=True)]
        d_im = d_im//4
        #upsample for rest of Swin blocks
        for i in range(len(blocks_per_level)-1):
            upsample_net+=[ nn.ConvTranspose2d(d_im,d_im//2,4,2,1),
                            nn.InstanceNorm2d(d_im//2),
                            nn.Dropout2d(p=0.125,inplace=True),
                            nn.ReLU(inplace=True)]
            d_im = d_im//2
        #upsample for original CNN encoding
        for i in range(2):
            if d_im>16:
                d_im_out = d_im//2
            else:
                d_im_out = d_im
            upsample_net+=[ nn.ConvTranspose2d(d_im,d_im_out,4,2,1),
                            nn.InstanceNorm2d(d_im_out),
                            nn.Dropout2d(p=0.125,inplace=True),
                            nn.ReLU(inplace=True)]
            d_im = d_im_out
        upsample_net.append(nn.Conv2d(d_im,1,1,1,0))
        upsample_net.append(nn.Sigmoid())
        self.upsample_net= nn.Sequential(*upsample_net)
        
        

        self.answer_decode = nn.Sequential(
                nn.Linear(d_model,self.decode_tokenizer.vocab_size,bias=False),
                nn.LogSoftmax(dim=-1) #except
                )

        self.answer_decode[0].weight = self.text_embedding.weight #Tie weights


        #t#self.opt_history=defaultdict(list)#t#




    #we're building this for fixed images size
    def forward(self,image,ocrRes,questions,answers=None,useCurvedBBs=False,RUN=False,get_tokens=False):


        device = image.device

        im_tokens = self.patch_embed(image)
        im_tokens += self.absolute_2dpos_embed #Swin doesn't use this as it can rely on the biased attention. We need the image tokens to know where they are so they can interact with the document and question tokens
        num_im = im_tokens.size(1)
        #Dropout?
        batch_size = image.size(0)

        #we need to extend batch entries with multiple questions
        repeats = [len(q) for q in questions] #different elements in batch may have different numbers of questions

        #first expand tokens and other things
        repeats_cuda = torch.LongTensor(repeats).to(device)
        im_tokens = torch.repeat_interleave(im_tokens,repeats_cuda,dim=0)
        repeats_cuda = None

        #flatten the questions an answers to a single batch
        questions=[q for bq in questions for q in bq]
        new_batch_size = len(questions)
        if not RUN:
            answers=[a for ba in answers for a in ba]
        else:
            answers=['']*new_batch_size

        #run question+answer through decoder
        if self.query_special_token_embedder is not None:
            _, query_special_tokens = self.query_special_token_embedder(questions)
            questions, query_special_start_tokens = self.query_special_start_token_embedder(questions)

        if self.use_set_length:
            assert self.query_special_token_embedder is not None
            q_t = self.tokenizer(questions,return_tensors="pt",padding='max_length',max_length=self.max_q_tokens,truncation=True)
            q_input_ids = q_t['input_ids'][:,:self.max_q_tokens-1]
            q_attention_mask = q_t['attention_mask'][:,:self.max_q_tokens-1]
            a_t = self.tokenizer(answers,return_tensors="pt",padding='max_length',max_length=self.max_a_tokens,truncation=True)
            a_input_ids = a_t['input_ids'][:,:self.max_a_tokens]
            a_attention_mask = a_t['attention_mask'][:,:self.max_a_tokens]
        else:
            q_t = self.tokenizer(questions,return_tensors="pt",padding=True)
            q_input_ids = q_t['input_ids']
            q_attention_mask = q_t['attention_mask']
            a_t = self.tokenizer(answers,return_tensors="pt",padding=True)
            a_input_ids = a_t['input_ids']
            a_attention_mask = a_t['attention_mask']
        num_q = q_input_ids.size(1)
        num_a = a_input_ids.size(1)-1 #remove last SEP token
        qa_tokens = self.text_embedding(torch.cat((q_input_ids,a_input_ids[:,:-1]),dim=1).to(device))
        q_tokens = qa_tokens[:,:num_q] 
        a_tokens = qa_tokens[:,num_q:] 

        if self.query_special_token_embedder is not None:
            q_tokens = torch.cat((query_special_tokens[:,None,:],q_tokens),dim=1)
            a_tokens[:,0]+=query_special_start_tokens
            num_q+=1
        #DDD
        #a_tokens=torch.autograd.Variable(a_tokens,requires_grad=True)
        #self.start_token=a_tokens
        #self.start_token.retain_grad()


        #the model input ends up being [CLS] Question  [SEP] Answer
        #                             { q tokens     }{ a tokens   }


        q_tokens = self.q_pos_1d_enc(q_tokens)
        q_padding_mask = (1-q_attention_mask).bool()#.to(device) 
        if self.query_special_token_embedder is not None:
            q_padding_mask = torch.cat((torch.BoolTensor(new_batch_size,1).fill_(True),q_padding_mask),dim=1) #for special query token
        if self.downsample_q>0:
            #pad it out
            missing = q_tokens.size(1)% (2**self.downsample_q)
            missing = ((2**self.downsample_q)-missing) % (2**self.downsample_q)
            q_tokens = torch.cat((q_tokens,torch.FloatTensor(new_batch_size,missing,q_tokens.size(2)).fill_(0).to(device)),dim=1)
            q_padding_mask = torch.cat((q_padding_mask,torch.BoolTensor(new_batch_size,missing).fill_(True)),dim=1)
            num_q+=missing
        q_padding_mask = q_padding_mask.to(device)


        a_tokens = self.a_pos_1d_enc(a_tokens)
        a_padding_mask = (1-a_attention_mask[:,1:]).bool().to(device) #remove last SEP
        

        num_all = num_im+num_q+num_a


        all_att_mask = torch.BoolTensor(1,num_all,num_all).fill_(1) #1/0
        all_att_mask[:,-num_a:,-num_a:] = torch.tril(all_att_mask[:,-num_a:,-num_a:])
        all_att_mask[:,:-num_a,-num_a:] = 0 #nothing else attends to a
        all_att_mask = all_att_mask.to(device)


        #Run the Swin and accompanying layers
        level=0
        qa_tokens = torch.cat( (q_tokens,a_tokens),dim=1)
        qa_padding_mask = torch.cat( (q_padding_mask,a_padding_mask), dim=1)
        #convert to 0/-inf as that's what the Swin code expects
        q_padding_mask_inf = torch.FloatTensor(*q_padding_mask.size()).fill_(0).to(device)
        q_padding_mask_inf[q_padding_mask] = float('-inf')

        im_padding_mask = torch.BoolTensor(1,1).fill_(0).expand(new_batch_size,num_im).to(device)
        all_padding_mask = torch.cat( (im_padding_mask,q_padding_mask,a_padding_mask),dim=1)

        if RUN:
            #store results at each layer to reuse
            saved_proj_im_tokens = []
            saved_q_tokens = []
            saved_q_pos = []
            saved_q_padding_mask = []
            saved_a_tokens = []
            ##

        if get_tokens:
            im_tokens = im_tokens.requires_grad_()
            init_im_tokens = im_tokens

        for i,(swin_layer, proj_d2i, layout_layer, proj_i2d, im_downsample, ocr_downsample, q_downsample) in enumerate(self.swin_layers):

            #could be run in parallel
            if PRINT_ATT:
                NUMS.append((num_q,num_a))
            
            if swin_layer is not None:
                if proj_d2i is not None:
                    im_tokens = swin_layer(im_tokens,proj_d2i(q_tokens),
                            docq_padding_mask=q_padding_mask_inf) 
                else:
                    im_tokens = swin_layer(im_tokens)

                proj_im_tokens = proj_i2d(im_tokens)
            #else:
            #   reuse last proj_im_tokens

            if RUN:
                saved_proj_im_tokens.append(proj_im_tokens)
                saved_q_tokens.append(q_tokens)
                saved_q_pos.append(q_pos)
                saved_q_padding_mask.append(q_padding_mask)
                saved_a_tokens.append(a_tokens)
            
            qa_tokens = layout_layer(
                    qa_tokens,
                    torch.cat((proj_im_tokens,qa_tokens),dim=1),
                    all_att_mask[:,-(num_q+num_a):,:],
                    all_padding_mask)
                    

            did_downsample=False
            if im_downsample is not None:
                did_downsample=True
                im_tokens = im_downsample(im_tokens)
                level+=1
                num_im = im_tokens.size(1)
                im_padding_mask = im_padding_mask[:,:num_im]

            q_tokens = qa_tokens[:,:num_q]
            a_tokens = qa_tokens[:,num_q:]
            #num_q_old=num_q
            if q_downsample is not None:
                did_downsample=True
                q_tokens,q_padding_mask = q_downsample(q_tokens,q_padding_mask)
                num_q = q_tokens.size(1)
                qa_tokens = torch.cat( (q_tokens,a_tokens),dim=1)

            if did_downsample:
                num_all = num_im+num_q+num_a
                all_att_mask = all_att_mask[:,-(num_im+num_q+num_a):,-(num_im+num_q+num_a):] #this is uniform except at the end (a), so we can just take the bottom slice of it
                all_padding_mask = torch.cat( (im_padding_mask,q_padding_mask,a_padding_mask), dim=1)


        response = a_tokens

        ##############
        #Visual output
        H,W = self.final_resolution
        #reshape and permute to convert to image
        im_feats = im_tokens.view(new_batch_size,H,W,im_tokens.size(2)).permute(0,3,1,2)
        out_mask = self.upsample_net(im_feats)

                    

        if RUN: #assuming batchsize of 1
            #Forward inference (answer not known)
            assert new_batch_size==1 #just to make stopping easier
            assert num_a==1 #just checking...
            zero = torch.BoolTensor(1,1).fill_(0).to(device) #for creating masks from
            one = torch.BoolTensor(1,1).fill_(1).to(device)

            #response = all_tokens[:,-(num_a):]
            response_decoded = self.answer_decode(response)
            if get_tokens:
                response_decoded_all = response_decoded
            response_greedy_token = response_decoded.argmax(dim=2)

            output_tokens = [response_greedy_token[0,0].item()]
            #print('first token: {}'.format(output_tokens[0]))

            offset = 1

            max_pred_len=self.max_pred_len


            while output_tokens[-1] != self.SEP_TOKEN and offset<max_pred_len:

                ans_emb = self.text_embedding(response_greedy_token)
                ans = self.a_pos_1d_enc(ans_emb,offset=offset)
                num_a += 1



                level=0
                for li,(swin_layer, proj_d2i, layout_layer, proj_i2d, im_downsample, ocr_downsample, q_downsample) in enumerate(self.swin_layers):

                    #could be run in parallel
                    num_im = saved_proj_im_tokens[li].size(1)
                    num_q = saved_q_tokens[li].size(1)

                    proj_im_tokens = saved_proj_im_tokens[li]
                    im_padding_mask = zero.expand(new_batch_size,num_im)#holder_im_padding_mask[:,:num_im]

                    q_tokens = saved_q_tokens[li]
                    q_padding_mask = saved_q_padding_mask[li]

                    a_tokens = saved_a_tokens[li] = torch.cat((saved_a_tokens[li],ans),dim=1)
                    a_padding_mask = zero.expand(new_batch_size,num_a)#holder_a_padding_mask[:,:num_a]

                    all_padding_mask = torch.cat( (im_padding_mask,ocr_padding_mask,q_padding_mask,a_padding_mask), dim=1)
                    all_att_mask = one.expand(new_batch_size,1,num_im+num_ocr+num_q+num_a)


                    ans = layout_layer(
                            a_tokens[:,-1:], #only last token
                            torch.cat((proj_im_tokens,q_tokens,a_tokens),dim=1),
                            all_att_mask,#all_att_mask[:,-(num_q+num_a):,:],
                            all_padding_mask)
                    if im_downsample is not None:
                        level+=1
                #Done Swin (RUN)

                response_decoded = self.answer_decode(ans)
                if get_tokens:
                    response_decoded_all = torch.cat((response_decoded_all,response_decoded),dim=1)
                response_greedy_token = response_decoded.argmax(dim=2)
                assert response_greedy_token.size(1)==1
                

                output_tokens.append(response_greedy_token[0,0].item())
                #print('next token: {}'.format(output_tokens[-1]))
                offset += 1

            
            final_str = self.decode_tokenizer.convert_tokens_to_string(self.decode_tokenizer.convert_ids_to_tokens(output_tokens,skip_special_tokens=True))
            
            if PRINT_ATT:
                attDisplay(image[0],full_ocr_string,'|'+questions[0],'|'+final_str[0]+'^',final_str)

            if get_tokens:
                return response_decoded_all, None, final_str, out_mask, init_im_tokens, init_ocr_tokens
            else:
                return final_str, out_mask #torch.sigmoid(out_mask)
            ############





        response_decoded = self.answer_decode(response)

        #t#time = timeit.default_timer()-ticA#t#
        #t#self.opt_history['transformers'].append(time)#t#
        #t#tic=timeit.default_timer()#t#

        response_greedy_tokens = response_decoded.argmax(dim=2)
        target_decoded = a_input_ids[:,1:]# This has the SEP tokens (and padding), but not CLS (start) token

        #decode the prediction to string
        string_response=[]
        for b in range(len(questions)):
            response_greedy_tokens_b = response_greedy_tokens[b]
            pred_stop = response_greedy_tokens_b==self.DECODE_SEP_TOKEN
            if pred_stop.any():
                stop_index = pred_stop.nonzero(as_tuple=False)[0][0].item()
                response_greedy_tokens_b[stop_index:]=self.DECODE_SEP_TOKEN
            string_response.append(self.decode_tokenizer.convert_tokens_to_string(self.decode_tokenizer.convert_ids_to_tokens(response_greedy_tokens_b,skip_special_tokens=True)))


        #reshape strings into batches
        batch_string_response=[]
        cur_pos=0
        for r in repeats:
            batch_string_response.append(string_response[cur_pos:cur_pos+r])
            cur_pos+=r

        #t#timeA = timeit.default_timer()-ticA#t#
        #t#time = timeit.default_timer()-tic#t#
        #t#self.opt_history['decode'].append(time)#t#
        #t#self.opt_history['full forward'].append(timeA)#t#
        #t#self.print_opt_times()#t#
        #import pdb;pdb.set_trace()
        if PRINT_ATT:
            attDisplay(image[0],'|'+questions[0],'|'+answers[0]+'^',batch_string_response[0])




        return response_decoded, target_decoded.to(device), batch_string_response, out_mask

    #t#def print_opt_times(self):#t#
        #t#for name,times in self.opt_history.items():#t#
            #t#print('time {}({}): {}'.format(name,len(times),np.mean(times)))#t#
            #t#if len(times)>300: #t#
                #t#times.pop(0)   #t#
                #t#if len(times)>600:#t#
                    #t#times.pop(0)#t#
