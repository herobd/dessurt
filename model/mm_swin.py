from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pos_encode import PositionalEncoding, UniformRealEmbedding,PositiveRealEmbedding, ReturnPositionalEncoding, BartLearnedPositionalEmbedding
from model.q_transformer_layer import QTransformerLayer
from model.swin_transformer import ConvPatchEmbed, SwinTransformerBlock, PatchMerging, PatchEmbed
from model.trans_pooling import QPooler
try:
    from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
    from transformers import BartTokenizer, BartModel
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

BEAM_END_THRESH=0.05
TOP_K = 5
CHECK_ONLY_EVERY=15

class MmSwin(BaseModel):
    def __init__(self,config):
        super(MmSwin, self).__init__(config)
        self.image_size = config['image_size'] #input must be this size
        dropout = 0 if config.get('no_dropout',False) else 0.1
        self.conv_patch_emb = config.get('conv_patch_emb',True) #set to False to use original Swin embedding
        lighter_conv_patch_emb = config['lighter_conv_patch_emb'] if 'lighter_conv_patch_emb' in config else False #shallower network
        init_from_pretrained = config.get('init_from_pretrained',True) #the text emebeddings
        use_special_question_tokens = config.get('use_special_question_tokens',True)
        self.use_set_length = config.get('use_set_length',True) #Fixed length input and output
        self.max_q_tokens = config.get('max_q_tokens',20) #fixed length query tokens
        self.max_a_tokens = config.get('max_a_tokens',800) #fixed length response (answer) tokens

        mask_output = config.get('do_mask_output',True) #Turn off to remove CNN mask prediction at the end

        blocks_per_level = config['blocks_per_level'] #Level here relates to the Swin pooling (pooling makes new level)
        use_swin = config.get('use_swin',[True]*sum(blocks_per_level  )) #boolean array (for each layer), if false, the previous layer's visual features are used instead of making new ones
        use_auto = config.get('use_auto',[True]*sum(blocks_per_level  )) #boolean array, if false, the previsou layer's textual features are used instead of making new ones
        swin_cross_attention = config.get('swin_cross_attention',[True]*sum(blocks_per_level)) #boolean array, it false the Swin layers don't cross attend to textural tokens
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

            window_size = config['window_size'] #Swin window size
            if type(window_size) is int:
                window_size = [window_size]*len(blocks_per_level)
            d_model = config['decode_dim'] #width (num feats) of the textual tokens
            dim_ff = config['dim_ff'] #features in fully connected layer of textual transformer layers
            nhead = config['decode_num_heads'] #number of heads in for textual tokens attention
            swin_nhead = config['swin_nheads'] #number of heads for Swin, per level
            im_embed_dim = config['im_embed_dim'] #num features of initial visual tokens

        if isinstance(self.image_size,int):
            self.image_size = (self.image_size,self.image_size)

        self.max_pred_len = self.max_a_tokens


        if 'pre_trained' in config:
            pre_trained_patch_emb = config['patch_emb'] if 'patch_emb' in config else None
        else:
            pre_trained_patch_emb = None


        token_type = config.get('token_type','word')


        if token_type == 'char':
            self.tokenizer = CharacterTokenizer()
            self.SEP_TOKEN=self.tokenizer.SEP_index
            self.CLS_TOKEN=self.tokenizer.CLS_index
        elif token_type == 'word':
            assert init_from_pretrained=='bart'
            self.tokenizer = BartTokenizer.from_pretrained('./cache_huggingface/BART')
            self.SEP_TOKEN= 2
            self.CLS_TOKEN= 0
            #else:
            #    self.tokenizer = DistilBertTokenizer.from_pretrained('./cache_huggingface/distilbert-base-uncased')
            #    self.SEP_TOKEN= 102
            #    self.CLS_TOKEN= 101

        if config.get('form_tokens',False):
            add = ['"answer"',"question","other","header","},{",'"answers":','"content":']
            self.tokenizer.add_tokens(add, special_tokens=True)
        if config.get('NER_tokens',False):
            tokens = ["[NE:{}]".format(cls) for cls in ['N', 'C', 'L', 'T', 'O', 'P', 'G','NORP', 'LAW', 'PER', 'QUANTITY', 'MONEY', 'CARDINAL', 'LOCATION', 'LANGUAGE', 'ORG', 'DATE', 'FAC', 'ORDINAL', 'TIME', 'WORK_OF_ART', 'PERCENT', 'GPE', 'EVENT', 'PRODUCT','ART']]
            self.tokenizer.add_tokens(tokens, special_tokens=True)
        if config.get('rvl_cdip_tokens',False):
            tokens = [
                'letter',
                'form',
                'email',
                'handwritten',
                'advertisement',
                'scientific_report',
                'scientific_publication',
                'specification',
                'file_folder',
                'news_article',
                'budget',
                'invoice',
                'presentation',
                'questionnaire',
                'resume',
                'memo',]
            tokens = ['C:'+cls for cls in tokens]
            self.tokenizer.add_tokens(tokens, special_tokens=True)
        



        if init_from_pretrained=='distilbert':
            init_model = DistilBertModel.from_pretrained('./cache_huggingface/distilbert-base-uncased')
            init_emb = init_model.embeddings.word_embeddings
        elif init_from_pretrained=='bart':
            init_model = BartModel.from_pretrained('./cache_huggingface/BART')
            init_emb = init_model.shared


        self.text_embedding = nn.Embedding(len(self.tokenizer), d_model)
        if init_from_pretrained:
            #copy weights over, allowing for different sized weights
            self.text_embedding.weight.data[:init_emb.weight.size(0),:d_model] = init_emb.weight.data[:,:d_model]

        if use_special_question_tokens:
            #I didn't know how to add tokens to the tokenizer, so I created my own little module to process the text and produce the task token
            self.query_special_token_embedder = SpecialTokenEmbedder(d_model)
            self.query_special_start_token_embedder = SpecialTokenEmbedder(d_model)
        else:
            self.query_special_token_embedder = None
            self.query_special_start_token_embedder = None


        if init_from_pretrained=='bart':
            self.pos_1d_enc = BartLearnedPositionalEmbedding(1026,d_model)
            self.pos_1d_enc.weight.data = init_model.decoder.embed_positions.weight.data
            self.q_pos_1d_enc = self.a_pos_1d_enc = None
            self.pos_enc_adapter = nn.Identity() if d_model == 768 else nn.Linear(768,d_model,bias=False)
        else:
            self.q_pos_1d_enc = PositionalEncoding(d_model,dropout=dropout,max_len=1000)
            self.a_pos_1d_enc = PositionalEncoding(d_model,dropout=dropout,max_len=1000,offset_start=1000)


        if self.conv_patch_emb:
            self.patch_embed =  ConvPatchEmbed(
                    img_size=self.image_size, 
                    embed_dim=im_embed_dim,
                    norm_layer=nn.LayerNorm,
                    lighter=lighter_conv_patch_emb,
                    in_chans=2) #includes the hightlight/mask channel
        else:
            self.patch_embed =  PatchEmbed(
                    img_size=self.image_size, 
                    embed_dim=im_embed_dim,
                    norm_layer=nn.LayerNorm,
                    in_chans=2) #includes the highlight/mask channel
        if pre_trained_patch_emb is not None:
            checkpoint = torch.load(pre_trained_patch_emb, map_location=lambda storage, location: storage)
            pe_state_dict=self.patch_embed.state_dict()
            for name,value in checkpoint['state_dict']:
                if name.startswith('cnn.'):
                    pe_state_dict[name]=value

            self.patch_embed.load_state_dict(pe_state_dict)

        #Swin setup
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
                use_auto_here = use_auto[len(self.swin_layers)]
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
                    QTransformerLayer(d_model,nhead,dim_ff,dropout=dropout) if use_auto_here else None,
                    (nn.Linear(d_im,d_model,bias=False) if d_model!=d_im else nn.Identity()) if use_auto_here else None,
                    PatchMerging(cur_resolution, dim=d_im, norm_layer=nn.LayerNorm) if last else None,
                    None,
                    q_pool
                    ] ) )

        if config.get('use_bart_layer_init',False):
            self.bartLayerInit(init_model) #initialize the textual transformer layers with weigths from BART

        


        self.final_resolution = cur_resolution
        if mask_output:
            #Mask predicting CNN
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
        else:
            self.upsample_net=None
        
        

        self.answer_decode = nn.Linear(d_model,len(self.tokenizer),bias=False)
        self.answer_decode.weight = self.text_embedding.weight #Tie weights
        self.answer_softmax = nn.LogSoftmax(dim=-1)

        
        #experimented with distillation based on DistilBERT, but had to adapt feature sizes
        #Didn't work well
        if 'distillation_dim' in config:
            distillation_dim = config['distillation_dim']
            if distillation_dim!=d_model:
                self.ditillation_adapter = nn.Linear(d_model,distillation_dim,bias=True)
            else:
                self.ditillation_adapter = nn.Identity()
        else:
            self.ditillation_adapter = nn.Identity()




    #we're building this for fixed images size
    def forward(self,image,questions,answers=None,RUN=False,get_tokens=False,distill=False,get_logits=False):
        #image: B C H W
        #questions: List of B lists of strings (allows multiple questions per image, but Dessurt can't actually use this)
        #answers: List of B lists of strings (can be None if RUN is True)
        #RUN: True means true auto_regressive prediction is made. False uses teacher-forcing. Must have batch size of 1
        #get_tokens: Return initial input tokens (was used for saliencey things)
        #distill: Return logits and hidden state 
        #get_logits: Return logits


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
            if not RUN:
                a_t = self.tokenizer(answers,return_tensors="pt",padding='max_length',max_length=self.max_a_tokens+1,truncation=True)
                a_input_ids = a_t['input_ids'][:,:self.max_a_tokens+1]
                a_attention_mask = a_t['attention_mask'][:,:self.max_a_tokens+1]
            else:
                a_t = self.tokenizer(answers,return_tensors="pt",padding=True)
                a_input_ids = a_t['input_ids']
                a_attention_mask = a_t['attention_mask']
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

        a_padding_mask = (1-a_attention_mask[:,:-1]).bool().to(device) #remove last SEP


        

        num_all = num_im+num_q+num_a


        all_att_mask = torch.BoolTensor(1,num_all,num_all).fill_(1) #1/0
        all_att_mask[:,-num_a:,-num_a:] = torch.tril(all_att_mask[:,-num_a:,-num_a:])
        all_att_mask[:,:-num_a,-num_a:] = 0 #nothing else attends to a
        all_att_mask = all_att_mask.to(device)

        if self.q_pos_1d_enc is not None:
            q_tokens = self.q_pos_1d_enc(q_tokens)
            a_tokens = self.a_pos_1d_enc(a_tokens)

        qa_tokens = torch.cat( (q_tokens,a_tokens),dim=1)

        if self.q_pos_1d_enc is None:
            qa_tokens += self.pos_enc_adapter(self.pos_1d_enc(qa_tokens.size()))
            q_tokens = qa_tokens[:,:num_q] 
            a_tokens = qa_tokens[:,num_q:] 


        #Run the Swin and accompanying layers
        level=0
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
            saved_q_padding_mask = []
            saved_a_tokens = []
            ##

        if get_tokens:
            im_tokens = im_tokens.requires_grad_()
            init_im_tokens = im_tokens
            init_a_tokens = a_tokens.requires_grad_()


        for i,(swin_layer, proj_d2i, autoregr_layer, proj_i2d, im_downsample, ocr_downsample, q_downsample) in enumerate(self.swin_layers):

            #could be run in parallel
            if PRINT_ATT:
                NUMS.append((num_q,num_a))
            
            if swin_layer is not None:
                if proj_d2i is not None:
                    im_tokens = swin_layer(im_tokens,proj_d2i(q_tokens),
                            docq_padding_mask=q_padding_mask_inf) 
                else:
                    im_tokens = swin_layer(im_tokens)
                if autoregr_layer is not None:
                    proj_im_tokens = proj_i2d(im_tokens)
                #else:
                #   reuse last proj_im_tokens

            if autoregr_layer is not None:
                if RUN:
                    saved_proj_im_tokens.append(proj_im_tokens)
                    saved_q_tokens.append(q_tokens)
                    saved_q_padding_mask.append(q_padding_mask)
                    saved_a_tokens.append(a_tokens)
                
                qa_tokens = autoregr_layer(
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
        if self.upsample_net is not None:
            H,W = self.final_resolution
            #reshape and permute to convert to image
            im_feats = im_tokens.view(new_batch_size,H,W,im_tokens.size(2)).permute(0,3,1,2)
            out_mask = self.upsample_net(im_feats)
        else:
            out_mask = None

                    
        if RUN: #assuming batchsize of 1
            #Forward inference (answer not known)
            assert new_batch_size==1 #just to make stopping easier
            assert num_a==1 #just checking...
            zero = torch.BoolTensor(1,1).fill_(0).to(device) #for creating masks from
            one = torch.BoolTensor(1,1).fill_(1).to(device)
            max_pred_len=self.max_pred_len

            #response = all_tokens[:,-(num_a):]
            response_decoded = self.answer_decode(response)
            response_decoded = F.softmax(response_decoded,dim=-1)#self.answer_softmax(response_decoded)
            self.preventSpecial(response_decoded)
            if get_tokens:
                response_decoded_all = response_decoded

            if isinstance(RUN,str) and RUN.startswith('beam'):
                beam_search = True
                num_beams = int(RUN[4:])
            else:
                beam_search = False
            if beam_search:
                num_tokens = response_decoded.size(2)
                indexes_b = torch.arange(num_beams).repeat_interleave(num_tokens).to(device)
                indexes_t = torch.arange(num_tokens).repeat(num_beams).to(device)
                indexes_b_cpu = torch.arange(num_beams).repeat_interleave(num_tokens)
                indexes_t_cpu = torch.arange(num_tokens).repeat(num_beams)
                for li in range(len(saved_a_tokens)):
                    saved_a_tokens[li] = saved_a_tokens[li].expand(num_beams,-1,-1)
                    saved_q_tokens[li] = saved_q_tokens[li].expand(num_beams,-1,-1)
                    saved_proj_im_tokens[li] = saved_proj_im_tokens[li].expand(num_beams,-1,-1)

                cur_scores,indices = torch.sort(response_decoded[0,0],descending=True)
                cur_scores = cur_scores[:num_beams]
                indices = indices[:num_beams]
                old_tis = indices
                beam_scores = torch.FloatTensor(num_beams,max_pred_len).zero_().to(device)
                beam_scores[:,0]=cur_scores
                response_discrete_token = old_tis[:,None]

                output_tokens = torch.LongTensor(num_beams,max_pred_len).fill_(1).to(device)
                output_tokens[:,0] = indices

                if indices[0] == self.SEP_TOKEN:
                    best_finish_score = cur_scores[0]
                    best_done_tokens = [self.SEP_TOKEN]
                else:
                    #best_finish_score = -1
                    best_finish_score = torch.FloatTensor([-1])[0].to(device)
                    best_done_tokens=None

            else:
                response_discrete_token = response_decoded.argmax(dim=2)
                output_tokens = [response_discrete_token[0,0].item()]

            offset = 1



            while (beam_search or output_tokens[-1] != self.SEP_TOKEN) and offset<max_pred_len:
                ans = self.text_embedding(response_discrete_token)
                if self.a_pos_1d_enc is None:
                    ans += self.pos_enc_adapter(self.pos_1d_enc(ans.size(),past_key_values_length=num_q+offset))
                else:
                    ans = self.a_pos_1d_enc(ans,offset=offset)
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

                    all_padding_mask = torch.cat( (im_padding_mask,q_padding_mask,a_padding_mask), dim=1)
                    all_att_mask = one.expand(new_batch_size,1,num_im+num_q+num_a)


                    ans = layout_layer(
                            a_tokens[:,-1:], #only last token
                            torch.cat((proj_im_tokens,q_tokens,a_tokens),dim=1),
                            all_att_mask,#all_att_mask[:,-(num_q+num_a):,:],
                            all_padding_mask)
                    if im_downsample is not None:
                        level+=1
                #Done Swin (RUN)

                response_decoded = self.answer_decode(ans)
                response_decoded = F.softmax(response_decoded,dim=-1)#self.answer_softmax(response_decoded)
                self.preventSpecial(response_decoded)
                if get_tokens:
                    response_decoded_all = torch.cat((response_decoded_all,response_decoded),dim=1)

                if beam_search:

                    ###tensorized###
                    response_decoded =response_decoded[:,0]
                        
                    ###W#New beam #######

                    cur_scores = response_decoded+cur_scores[:,None]
                    cur_scores = cur_scores.view(-1)
                    
                    cur_scores,indices = torch.sort(cur_scores,descending=True)
                    cur_scores = cur_scores[:num_beams]
                    indices = indices[:num_beams]

                    ordered_bis = indexes_b[indices]
                    ordered_tis = indexes_t[indices]
                    for li in range(len(saved_a_tokens)):
                        saved_a_tokens[li]=saved_a_tokens[li][ordered_bis]
                    
                    beam_scores=beam_scores[ordered_bis]
                    beam_scores[:,offset]=response_decoded[ordered_bis,ordered_tis]

                    response_discrete_token[:,0] = ordered_tis
                    output_tokens = output_tokens[ordered_bis]
                    output_tokens[:,offset]=ordered_tis



                    if offset%CHECK_ONLY_EVERY==CHECK_ONLY_EVERY-1 and (output_tokens[:TOP_K]==self.SEP_TOKEN).any():
                        for k in range(TOP_K):
                            if (output_tokens[k]==self.SEP_TOKEN).any():
                                output_tokens[0]==output_tokens[k]
                                #best_beam_score = beam_scores[k].sum()/(beam_scores[k]!=0).sum()
                                break
                        break



                else:
                    response_discrete_token = response_decoded.argmax(dim=2)
                    assert response_discrete_token.size(1)==1
                

                    output_tokens.append(response_discrete_token[0,0].item())
                offset += 1
                


            if beam_search:
                output_tokens = output_tokens[0].cpu()
                stop_spots = (output_tokens==self.SEP_TOKEN).nonzero(as_tuple=True)[0]
                if len(stop_spots)>0:
                    output_tokens[stop_spots[0]+1:] = 1 #clear predictions after stop


            
            final_str = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(output_tokens,skip_special_tokens=True))
            
            if PRINT_ATT:
                attDisplay(image[0],full_ocr_string,'|'+questions[0],'|'+final_str[0]+'^',final_str)

            if get_tokens:
                return response_decoded_all, None, final_str, out_mask, init_im_tokens, init_a_tokens
            else:
                return final_str, out_mask #torch.sigmoid(out_mask)
            ############





        response_logits = self.answer_decode(response)
        response_decoded = self.answer_softmax(response_logits)


        response_greedy_tokens = response_decoded.argmax(dim=2)
        target_decoded = a_input_ids[:,1:]# This has the SEP tokens (and padding), but not CLS (start) token


        #decode the prediction to string
        string_response=[]
        for b in range(len(questions)):
            response_greedy_tokens_b = response_greedy_tokens[b]
            pred_stop = response_greedy_tokens_b==self.SEP_TOKEN
            if pred_stop.any():
                stop_index = pred_stop.nonzero(as_tuple=False)[0][0].item()
                response_greedy_tokens_b[stop_index:]=self.SEP_TOKEN
            string_response.append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(response_greedy_tokens_b,skip_special_tokens=True)))


        #reshape strings into batches
        batch_string_response=[]
        cur_pos=0
        for r in repeats:
            batch_string_response.append(string_response[cur_pos:cur_pos+r])
            cur_pos+=r

        if PRINT_ATT:
            attDisplay(image[0],'|'+questions[0],'|'+answers[0]+'^',batch_string_response[0])



        if distill:
            return response_decoded, target_decoded.to(device), batch_string_response, response_logits, self.ditillation_adapter(response), ~a_padding_mask
        elif get_logits:
            return response_decoded, target_decoded.to(device), batch_string_response, out_mask, response_logits
        else:
            return response_decoded, target_decoded.to(device), batch_string_response, out_mask

    def bartLayerInit(self,bart_model):
        #gather decoder layers
        layers=[]
        for swin,_,decoder_layer,_,_,_,_ in self.swin_layers:
            if decoder_layer is not None:
                layers.append(decoder_layer)

        assert len(layers)==10 #hardcoding for this count
        assert len(bart_model.decoder.layers)==6
        these_layers_to_bart = {0:0, 2:1, 4:2, 6:3, 8:4, 9:5}
        for layer_i,bart_i in these_layers_to_bart.items():
            init_layer = bart_model.decoder.layers[bart_i]
            layer = layers[layer_i]
            print('init {} with {}'.format(layer_i,bart_i))
            layer.self_attn.linears[0].weight.data = init_layer.self_attn.q_proj.weight.data
            layer.self_attn.linears[0].bias.data = init_layer.self_attn.q_proj.bias.data
            layer.self_attn.linears[1].weight.data = init_layer.self_attn.k_proj.weight.data
            layer.self_attn.linears[1].bias.data = init_layer.self_attn.k_proj.bias.data
            layer.self_attn.linears[2].weight.data = init_layer.self_attn.v_proj.weight.data
            layer.self_attn.linears[2].bias.data = init_layer.self_attn.v_proj.bias.data
            layer.self_attn.linears[3].weight.data = init_layer.self_attn.out_proj.weight.data
            layer.self_attn.linears[3].bias.data = init_layer.self_attn.out_proj.bias.data

            layer.norm1.weight.data = init_layer.self_attn_layer_norm.weight.data
            layer.norm1.bias.data = init_layer.self_attn_layer_norm.bias.data

            layer.linear1.weight.data = init_layer.fc1.weight.data
            layer.linear1.bias.data = init_layer.fc1.bias.data
            layer.linear2.weight.data = init_layer.fc2.weight.data
            layer.linear2.bias.data = init_layer.fc2.bias.data

            layer.norm2.weight.data = init_layer.final_layer_norm.weight.data
            layer.norm2.bias.data = init_layer.final_layer_norm.bias.data

    def preventSpecial(self, response_decoded):
        response_decoded[:,:,47847]=0 #prevent prediction of '§'
        response_decoded[:,:,4056]=0 #prevent prediction of '¿' (and potentially other special characters
