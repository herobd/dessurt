from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pairing_g_graph_layoutlm import  runLayoutLM
from model.pos_encode import PositionalEncoding, UniformRealEmbedding,PositiveRealEmbedding, ReturnPositionalEncoding
from model.rel_pos_im_transformer import RelPosImTransformerLayer
from model.swin_transformer import ConvPatchEmbed, SwinTransformerBlock, PatchMerging
try:
    from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
    from transformers import LayoutLMTokenizer, LayoutLMModel
except:
    pass
from utils.character_tokenizer import CharacterTokenizer
from collections import defaultdict
from timm.models.layers import trunc_normal_
import math

import timeit

def normalize_bbox2(bbox, dwidth, dheight, twidth, theight, max_dist):
     return [
         max(min(int(max_dist * (bbox[0] / dwidth)),max_dist),0),
         max(min(int(max_dist * (bbox[1] / dheight)),max_dist),0),
         max(min(int(max_dist * (bbox[2] / twidth)),twidth),0),
         max(min(int(max_dist * (bbox[3] / theight)),theight),0),
     ]

class QAImDocGPT(BaseModel):
    def __init__(self,config):
        super(QAImDocGPT, self).__init__(config)
        self.blank_ocr = config['blank_ocr'] if 'blank_ocr' in config else False
        self.image_size = config['image_size'] #start at 512?
        window_size = config['window_size'] #7
        max_dist = math.sqrt(self.image_size[0]**2 + self.image_size[1]**2)
        d_model = config['decode_dim']
        dim_ff = config['dim_ff']
        nhead = config['decode_num_heads']
        blocks_per_level = config['blocks_per_level'] #[2,2,6,2] -> in:512,emb:64 then 64,32,16,8
        swin_nhead = config['swin_nheads'] #[3,6,12,24] | [2,6,12,12] probably don't need as much later
        im_embed_dim = config['im_embed_dim'] #96 -> 96,192,384,768 | 64->64,128,256,512
        dropout = 0 if 'no_dropout' in config and  config['no_dropout'] else 0.1
        lighter_conv_patch_emb = config['lighter_conv_patch_emb'] if 'lighter_conv_patch_emb' in config else False

        if type(window_size) is int:
            window_size = [window_size]*len(blocks_per_level)
        self.max_pred_len = 500


        if 'pre_trained' in config:
            pre_trained_patch_emb = config['patch_emb'] if 'patch_emb' in config else None
        else:
            pre_trained_patch_emb = None

        char_output = config['char_output'] if 'char_output' in config else False
        char_tokens = config['char_tokens'] if 'char_tokens' in config else False
        if char_tokens:
            char_output=False

        if char_tokens:
            self.tokenizer = CharacterTokenizer()
            self.SEP_TOKEN=self.tokenizer.SEP_index
            self.CLS_TOKEN=self.tokenizer.CLS_index
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.SEP_TOKEN= 102
            self.CLS_TOKEN= 101

        if char_output:
            self.decode_tokenizer = CharacterTokenizer()
            self.DECODE_SEP_TOKEN=self.decode_tokenizer.SEP_index
            self.DECODE_CLS_TOKEN=self.decode_tokenizer.CLS_index
        else:
            self.decode_tokenizer = self.tokenizer
            self.DECODE_SEP_TOKEN=self.SEP_TOKEN
            self.DECODE_CLS_TOKEN=self.CLS_TOKEN


        self.answer_embedding = nn.Embedding(self.tokenizer.vocab_size, d_model)
        self.pos_1d_enc = PositionalEncoding(d_model,dropout=dropout,max_len=1000)
        self.word_pos_enc = ReturnPositionalEncoding(d_model,dropout=dropout,max_len=1000)
        self.pos_emb_x = UniformRealEmbedding(d_model,0,self.image_size[1],100)
        self.pos_emb_y = UniformRealEmbedding(d_model,0,self.image_size[0],100)
        self.pos_emb_w = PositiveRealEmbedding(d_model,0,int(0.5*self.image_size[1]),30)
        self.pos_emb_h = PositiveRealEmbedding(d_model,0,int(0.3*self.image_size[0]),30)

        mlp_ratio=4.
        qkv_bias=True
        qk_scale=None
        drop_rate=0.
        attn_drop_rate=0.
        drop_path_rate=dropout
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(blocks_per_level))]  # stochastic depth decay rule

        self.patch_embed =  ConvPatchEmbed(
                img_size=self.image_size, 
                embed_dim=im_embed_dim,
                norm_layer=nn.LayerNorm,
                lighter=lighter_conv_patch_emb)
        if pre_trained_patch_emb is not None:
            checkpoint = torch.load(pre_trained_patch_emb, map_location=lambda storage, location: storage)
            pe_state_dict=self.patch_embed.state_dict()
            for name,value in checkpoint['state_dict']:
                if name.startswith('cnn.'):
                    pe_state_dict[name]=value

            self.patch_embed.load_state_dict(pe_state_dict)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution



        self.absolute_2dpos_embed = nn.Parameter(torch.zeros(1, num_patches, im_embed_dim))
        trunc_normal_(self.absolute_2dpos_embed, std=.02)

        self.layers=nn.ModuleList()
        for level,blocks in enumerate(blocks_per_level):
            d_im = int(im_embed_dim * 2 ** level)
            cur_resolution = (patches_resolution[0]//(2**level), patches_resolution[1]//(2**level))
            patch_size = (self.image_size[0]/cur_resolution[0],self.image_size[1]/cur_resolution[1])
            im_xs = torch.arange(patch_size[1]/2,self.image_size[1],patch_size[1])[None,:].expand(cur_resolution[0],-1)
            im_ys = torch.arange(patch_size[0]/2,self.image_size[0],patch_size[0])[:,None].expand(-1,cur_resolution[1])
            #im_cords = torch.stack((im_xs,im_ys),dim=2).view(patches_resolution[0]*patches_resolution[1],2)
            im_xs = im_xs.contiguous().view(1,cur_resolution[0]*cur_resolution[1])
            im_ys = im_ys.contiguous().view(1,cur_resolution[0]*cur_resolution[1])
            self.register_buffer("im_xs{}".format(level),im_xs,persistent=False)
            self.register_buffer("im_ys{}".format(level),im_ys,persistent=False)
            for block in range(blocks):
                last = level<len(blocks_per_level)-1 and block == blocks-1
                self.layers.append( nn.ModuleList( [
                    SwinTransformerBlock(dim=d_im, 
                                input_resolution=cur_resolution,
                                 num_heads=swin_nhead[level], 
                                 window_size=window_size[level],
                                 shift_size=0 if (len(self.layers) % 2 == 0) else window_size[level] // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop_rate, 
                                 attn_drop=attn_drop_rate,
                                 drop_path=dpr[len(self.layers)],
                                 norm_layer=nn.LayerNorm,
                                 sees_docq=True),
                    nn.Linear(d_model,d_im,bias=False) if d_model!=d_im else nn.Identity(),
                    RelPosImTransformerLayer(d_model,nhead,max_dist,dim_ff,dropout=dropout),
                    nn.Linear(d_im,d_model,bias=False) if d_model!=d_im else nn.Identity(),
                    PatchMerging(cur_resolution, dim=d_im, norm_layer=nn.LayerNorm) if last else None 
                    ] ) )

        self.im_xs=[None]*len(blocks_per_level) #the x,y cords of each patch center for every level/resolution
        self.im_ys=[None]*len(blocks_per_level)
        
                
        self.answer_decode = nn.Sequential(
                nn.Linear(d_model,self.decode_tokenizer.vocab_size),
                nn.LogSoftmax(dim=-1) #except
                )



        #t#self.opt_history=defaultdict(list)#t#


    #we're building this for fixed images size
    def forward(self,image,gtBBs,gtTrans,questions,answers=None,useCurvedBBs=False,RUN=False):
        if self.blank_ocr:
            gtTrans=[[]]*len(questions)
        #torch.autograd.set_detect_anomaly(True)
        #there's got to be a better way...
        for name,buff in self.named_buffers():
            if 'im_xs' in name:
                level=int(name[5:])
                self.im_xs[level]=buff
            elif 'im_ys' in name:
                level=int(name[5:])
                self.im_ys[level]=buff
        #t#ticA=timeit.default_timer()#t#
        device = image.device
        image_size = image.size()

        im_tokens = self.patch_embed(image)
        im_tokens += self.absolute_2dpos_embed #Swin doesn't use this as it can rely on the biased attention. We need the image tokens to know where they are so they can interact with the document and question tokens
        #Dropout?

        all_ocr_bbs=[]
        all_ocr_pos=[]
        max_len=0
        for b,(words,bbs) in enumerate(zip(gtTrans,gtBBs)):
            ocr_bbs = [[0,0,0,0]]
            ocr_pos = [0]
            for i,(word,bb) in enumerate(zip(words,bbs)):
                if useCurvedBBs:
                    x1,y1,x2,y2,r=bb[:5]
                    h = y2-y1
                    w = x2-x1
                    xc = (x1+x2)/2
                    yc = (y1+y2)/2
                else:
                    xc,yc,r,h,w=bb[:5]
                    x1=xc-w
                    x2=xc+w
                    y1=yc-h
                    y2=yc+h
                #bb = normalize_bbox([x1,y1,x2,y2],image_size[3],image_size[2]) #x1y1x2y2
                #bb = normalize_bbox2([xc,yc,w,h],image_size[3],image_size[2],500,300) #x1y1x2y2
                bb = [xc,yc,w,h] #The images should already be normalized
                word_tokens = self.tokenizer.tokenize(word)
                #ocr_ids.extend(word_tokens)
                #total_string+=word+' '
                #word_token_map.append(range(len(ocr_bbs),len(ocr_bbs)+len(word_tokens)))
                ocr_bbs.extend([bb]*len(word_tokens))
                ocr_pos.extend(range(0,len(word_tokens)))
            ocr_bbs.append([0,0,0,0])
            ocr_pos.append(0)
            max_len = max(max_len,len(ocr_bbs))
            all_ocr_bbs.append(ocr_bbs)
            all_ocr_pos.append(ocr_pos)



        total_strings = [' '.join(gtT) for gtT in gtTrans]


        repeats = [len(q) for q in questions] #different elements in batch may have different numbers of questions

        repeat_docs = [[doc]*r for doc,r in zip(total_strings,repeats)]

        im_tokens = torch.repeat_interleave(im_tokens,torch.LongTensor(repeats).to(device),dim=0)


        repeat_docs=[d for bd in repeat_docs for d in bd]
        questions=[q for bq in questions for q in bq]
        if answers is not None:
            answers=[a for ba in answers for a in ba]

        new_batch_size = len(questions)

        #Append question before answer
        if not RUN:
            docqa_str = [doc+':'+q+'[SEP]'+a for doc,q,a in zip(repeat_docs,questions,answers)]
        else:
            docqa_str = [doc+':'+q+'[SEP]' for doc,q in zip(repeat_docs,questions)]
            saved_docqa=[]
            saved_proj_im_tokens=[]
            output_tokens=[]

        #run question+answer through decoder
        docqa_t = self.tokenizer(docqa_str,return_tensors="pt",padding=True)
        docqa_to_emb = docqa_t['input_ids'][:,:-1].detach().to(device) #Remove end (SEP) token, as it doesn't need to predict anythin after that. emb needs to do position
        docqa_emb = self.answer_embedding(docqa_to_emb) 

        docqa_len = docqa_emb.size(1)
        qa_mask = torch.FloatTensor(new_batch_size,docqa_len,1).fill_(1)
        bbs = torch.FloatTensor(new_batch_size,docqa_len,4).fill_(0)#float('NaN'))
        tok_pos = torch.LongTensor(new_batch_size,docqa_len).fill_(0)
        new_i=0
        for b,(ocr_bbs,ocr_pos) in enumerate(zip(all_ocr_bbs,all_ocr_pos)):
            len_docqa = len(ocr_bbs)
            ocr_bbs = torch.FloatTensor(ocr_bbs)[None,...]
            ocr_pos = torch.FloatTensor(ocr_pos)[None,...]
            r =repeats[b]
            bbs[new_i:new_i+r,0:len_docqa,:] = ocr_bbs
            tok_pos[new_i:new_i+r,0:len_docqa] = ocr_pos
            qa_mask[new_i:new_i+r,0:len_docqa] = 0
            new_i+=r
        bbs=bbs.to(device)
        xs=bbs[:,:,0]
        ys=bbs[:,:,1]
        ws=bbs[:,:,2]
        hs=bbs[:,:,3]
        qa_mask = qa_mask.to(device)
        doc_mask = 1-qa_mask

        docqa = self.pos_1d_enc(docqa_emb,qa_mask) #This only applies the 1d position embedding to the q+a part of the docqa
        answer_padding_mask = (1-docqa_t['attention_mask'][:,:-1]).bool().to(device)

        docqa += doc_mask*(self.pos_emb_x(xs) + self.pos_emb_y(ys) + self.pos_emb_w(ws) + self.pos_emb_h(hs) + self.word_pos_enc(tok_pos))

        locs = (docqa_t['input_ids']==self.SEP_TOKEN).nonzero(as_tuple=False)[::2,1] #get first SEP, not second
        att_mask = torch.BoolTensor(new_batch_size,docqa_len,docqa_len).fill_(1) #1/0
        docq_padding_mask = torch.FloatTensor(new_batch_size,docqa_len).fill_(0) #0/-inf
        for b,loc in enumerate(locs):
            att_mask[b,:loc+1,loc+1:]=0 #doc and question tokens cannot attend to answer tokens
            att_mask[b,loc+1:,loc+1:]=torch.tril(att_mask[b,loc+1:,loc+1:]) #causual attention for answer
            docq_padding_mask[b,loc+1:]=float('-inf') #image cannot attend to answers

        att_mask = att_mask.to(device)
        docq_padding_mask = docq_padding_mask.to(device)

        level=0
        for swin_layer, proj_d2i, layout_layer, proj_i2d, downsample in self.layers:

            #could be run in parallel
            im_tokens = swin_layer(im_tokens,proj_d2i(docqa),   #we pass it the answers
                    docq_padding_mask=docq_padding_mask) #but we'll mask them out
            proj_im_tokens = proj_i2d(im_tokens)
            if RUN:
                saved_docqa.append(docqa)
                saved_proj_im_tokens.append(proj_im_tokens)
            docqa = layout_layer(
                    docqa,xs,ys,
                    proj_im_tokens,
                    self.im_xs[level].expand(new_batch_size,-1),
                    self.im_ys[level].expand(new_batch_size,-1),
                    docqa_mask=att_mask,
                    docqa_padding_mask=answer_padding_mask,
                    pos_mask = doc_mask)


            

            if downsample is not None:
                im_tokens = downsample(im_tokens)
                #im_xs = (im_xs[:,0::2]+im_xs[:,1::2])/2
                #im_ys = (im_ys[:,0::2]+im_ys[:,1::2])/2
                level+=1

        if RUN: #assuming batchsize of 1
            assert docqa.size(0)==1 #just to make stopping easier
            response_decoded = self.answer_decode(docqa)
            response_greedy_token = response_decoded.argmax(dim=2)
            response_greedy_token = response_greedy_token[:,-1:] #only care about last

            output_tokens.append(response_greedy_token[0,0].item())

            offset = docqa.size(1)

            max_pred_len=self.max_pred_len
            holder_xs = torch.cat((xs,torch.FloatTensor(new_batch_size,max_pred_len).fill_(0).to(device)),dim=1)
            holder_ys = torch.cat((ys,torch.FloatTensor(new_batch_size,max_pred_len).fill_(0).to(device)),dim=1)
            holder_answer_padding_mask = torch.FloatTensor(new_batch_size,max_pred_len).fill_(0).to(device) #assumes here batch size of 1
            holder_doc_mask = torch.cat((doc_mask,torch.FloatTensor(new_batch_size,max_pred_len,1).fill_(0).to(device)),dim=1)

            holder_att_mask = torch.BoolTensor(new_batch_size,max_pred_len,max_pred_len).fill_(1) #1/0
            #assume batch size of 1
            holder_att_mask[0,:docqa_len,docqa_len:]=0 #doc and question tokens cannot attend to answer tokens
            holder_att_mask[0,docqa_len:,docqa_len:]=torch.tril(holder_att_mask[0,docqa_len:,docqa_len:]) #causual attention for answer
            holder_att_mask = holder_att_mask.to(device)

            while output_tokens[-1] != self.SEP_TOKEN and saved_docqa[0].size(1)<max_pred_len:

                ans_emb = self.answer_embedding(response_greedy_token)
                ans = self.pos_1d_enc(ans_emb,offset=offset)
                level=0
                for li,(swin_layer, proj_d2i, layout_layer, proj_i2d, downsample) in enumerate(self.layers):
                    saved_docqa[li] = torch.cat((saved_docqa[li],ans),dim=1)
                    xs = holder_xs[:,:saved_docqa[li].size(1)]
                    ys = holder_ys[:,:saved_docqa[li].size(1)]
                    att_mask = holder_att_mask[:,:saved_docqa[li].size(1),:saved_docqa[li].size(1)]
                    answer_padding_mask = holder_answer_padding_mask[:,:saved_docqa[li].size(1)]
                    doc_mask = holder_doc_mask[:,:saved_docqa[li].size(1)]
                    ans = layout_layer(
                            saved_docqa[li],xs,ys,
                            saved_proj_im_tokens[li],
                            self.im_xs[level].expand(new_batch_size,-1),
                            self.im_ys[level].expand(new_batch_size,-1),
                            docqa_mask=att_mask,
                            docqa_padding_mask=answer_padding_mask,
                            pos_mask = doc_mask,
                            auto_regressive=ans) #we tell it to only use the last answer token as the keys
                    if downsample is not None:
                        level+=1

                response_decoded = self.answer_decode(ans)
                response_greedy_token = response_decoded.argmax(dim=2)
                assert response_greedy_token.size(1)==1
                

                output_tokens.append(response_greedy_token[0,0].item())
                offset += 1


            
            final_str = self.decode_tokenizer.convert_tokens_to_string(self.decode_tokenizer.convert_ids_to_tokens(output_tokens,skip_special_tokens=True))
            return final_str
        ############



        response = docqa
        response_decoded = self.answer_decode(response.reshape(-1,response.size(2)))
        response_decoded = response_decoded.view(new_batch_size,docqa_len,-1)


        #t#time = timeit.default_timer()-ticA#t#
        #t#self.opt_history['transformers'].append(time)#t#
        #t#tic=timeit.default_timer()#t#

        response_greedy_tokens = response_decoded.argmax(dim=2)
        locs = (docqa_t['input_ids']==self.SEP_TOKEN).nonzero(as_tuple=False)[::2,1] #get first SEP, not second
        mask_response = torch.ones(response_decoded.size())
        for b,loc in enumerate(locs):
            #for the question (before SEP)
            docqa_t['input_ids'][b,:loc]=0 #zero GT where question is (we don't want to train that)
            #response_decoded[b,:loc]*=0 #zero pred
            mask_response[b,:loc]*=0
        response_decoded = response_decoded*mask_response.to(device)
        target_decoded = docqa_t['input_ids'][:,1:]# This has the SEP tokens (and padding), but not CLS (start) token

        #decode the prediction to string
        string_response=[]
        for b in range(len(questions)):
            response_greedy_tokens_b = response_greedy_tokens[b]
            response_greedy_tokens_b = response_greedy_tokens_b[locs[b]:] #only take response
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
        return response_decoded, target_decoded.to(device), batch_string_response

    #t#def print_opt_times(self):#t#
        #t#for name,times in self.opt_history.items():#t#
            #t#print('time {}({}): {}'.format(name,len(times),np.mean(times)))#t#
            #t#if len(times)>300: #t#
                #t#times.pop(0)   #t#
                #t#if len(times)>600:#t#
                    #t#times.pop(0)#t#
