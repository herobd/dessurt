from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pairing_g_graph_layoutlm import  runLayoutLM
from model.pos_encode import PositionalEncoding, UniformRealEmbedding,PositiveRealEmbedding, ReturnPositionalEncoding
from model.rel_pos_im_transformer import RelPosImTransformerLayer, RelPosTransformerLayer
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
from utils.util import calcXYWH

import timeit

def normalize_bbox2(bbox, dwidth, dheight, twidth, theight, max_dist):
     return [
         max(min(int(max_dist * (bbox[0] / dwidth)),max_dist),0),
         max(min(int(max_dist * (bbox[1] / dheight)),max_dist),0),
         max(min(int(max_dist * (bbox[2] / twidth)),twidth),0),
         max(min(int(max_dist * (bbox[3] / theight)),theight),0),
     ]

class QAImDocGPT2(BaseModel):
    def __init__(self,config):
        super(QAImDocGPT2, self).__init__(config)
        self.blank_ocr = config['blank_ocr'] if 'blank_ocr' in config else False
        self.image_size = config['image_size'] #start at 512?
        blocks_per_level = config['swin_blocks_per_level'] #[2,2,6,2] -> in:512,emb:64 then 64,32,16,8
        dropout = 0 if 'no_dropout' in config and  config['no_dropout'] else 0.1
        lighter_conv_patch_emb = config['lighter_conv_patch_emb'] if 'lighter_conv_patch_emb' in config else False

        full_layers = config['full_layers']
        fd_model = config['full_dim']
        fdim_ff = config['fdim_ff']
        fnhead = config['full_num_heads']
        if blocks_per_level is not None:
            window_size = config['window_size'] #7
            if type(window_size) is int:
                window_size = [window_size]*len(blocks_per_level)
            d_model = config['decode_dim']
            dim_ff = config['dim_ff']
            nhead = config['decode_num_heads']
            swin_nhead = config['swin_nheads'] #[3,6,12,24] | [2,6,12,12] probably don't need as much later
            im_embed_dim = config['im_embed_dim'] #96 -> 96,192,384,768 | 64->64,128,256,512
        else:
            d_model = fd_model
            im_embed_dim = fd_model

        if type(self.image_size) is int:
            self.image_size = (self.image_size,self.image_size)
        max_dist = math.sqrt(self.image_size[0]**2 + self.image_size[1]**2)
        self.max_pred_len = 500


        if 'pre_trained' in config:
            pre_trained_patch_emb = config['patch_emb'] if 'patch_emb' in config else None
        else:
            pre_trained_patch_emb = None

        char_output = config['char_output']
        char_tokens = config['char_tokens']
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


        self.text_embedding = nn.Embedding(self.tokenizer.vocab_size, d_model)
        self.ocr_emb = nn.Linear(97,d_model,bias=False)
        self.q_pos_1d_enc = PositionalEncoding(d_model,dropout=dropout,max_len=1000)
        self.a_pos_1d_enc = PositionalEncoding(d_model,dropout=dropout,max_len=1000,offset_start=1000)
        self.ocr_1dpos_enc = ReturnPositionalEncoding(d_model,dropout=dropout,max_len=1000)
        self.ocr_pos_emb_x = UniformRealEmbedding(d_model,0,self.image_size[1],100)
        self.ocr_pos_emb_y = UniformRealEmbedding(d_model,0,self.image_size[0],100)
        self.ocr_pos_emb_w = PositiveRealEmbedding(d_model,0,int(0.5*self.image_size[1]),30)
        self.ocr_pos_emb_h = PositiveRealEmbedding(d_model,0,int(0.3*self.image_size[0]),30)


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

        if blocks_per_level is not None:
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
                                     sees_docq=True),
                        nn.Linear(d_model,d_im,bias=False) if d_model!=d_im else nn.Identity(),
                        RelPosImTransformerLayer(d_model,nhead,max_dist,dim_ff,dropout=dropout),
                        nn.Linear(d_im,d_model,bias=False) if d_model!=d_im else nn.Identity(),
                        PatchMerging(cur_resolution, dim=d_im, norm_layer=nn.LayerNorm) if last else None,
                        None,#ocr downsample
                        None #question downsample
                        ] ) )

            self.im_xs=[None]*len(blocks_per_level) #the x,y cords of each patch center for every level/resolution
            self.im_ys=[None]*len(blocks_per_level)
        else:
            self.swin_layers=None
            d_im = im_embed_dim
            cur_resolution = patches_resolution
            patch_size = (self.image_size[0]/cur_resolution[0],self.image_size[1]/cur_resolution[1])
            im_xs = torch.arange(patch_size[1]/2,self.image_size[1],patch_size[1])[None,:].expand(cur_resolution[0],-1)
            im_ys = torch.arange(patch_size[0]/2,self.image_size[0],patch_size[0])[:,None].expand(-1,cur_resolution[1])
            im_xs = im_xs.contiguous().view(1,cur_resolution[0]*cur_resolution[1])
            im_ys = im_ys.contiguous().view(1,cur_resolution[0]*cur_resolution[1])
            self.register_buffer("im_xs{}".format(0),im_xs,persistent=False)
            self.register_buffer("im_ys{}".format(0),im_ys,persistent=False)
            self.im_xs=[None]
            self.im_ys=[None]
        
        
        if d_im != fd_model:
            self.im_transition = nn.Linear(d_im,fd_model)
        else:
            self.im_transition = nn.Identity()
        if d_model != fd_model:
            self.ocrqa_transition = nn.Linear(d_model,fd_model)
        else:
            self.ocrqa_transition = nn.Identity()

        self.full_layers = nn.ModuleList()
        for im_pool_p, ocr_pool_p, q_pool_p in full_layers:
            if im_pool_p=='n':
                im_pool = None
            else:
                raise NotImplementedError('unknown image pooling method: {}'.format(im_pool_p))
            if ocr_pool_p=='n':
                ocr_pool = None
            elif ocr_pool_p=='p':
                ocr_pool = nn.Conv2d(fd_model,fd_model,kernel_size=4,stride=2,padding=1)
            else:
                raise NotImplementedError('unknown ocr pooling method: {}'.format(ocr_pool_p))
            if q_pool_p=='n':
                q_pool = None
            elif q_pool_p=='p':
                q_pool = nn.Conv2d(fd_model,fd_model,kernel_size=4,stride=2,padding=1)
            else:
                raise NotImplementedError('unknown question pooling method: {}'.format(q_pool_p))

            layer = RelPosTransformerLayer(fd_model,fnhead,max_dist,fdim_ff,dropout=dropout)

            self.full_layers.append(nn.ModuleList([im_pool,ocr_pool,q_pool,layer]))

        self.answer_decode = nn.Sequential(
                nn.Linear(fd_model,self.decode_tokenizer.vocab_size),
                nn.LogSoftmax(dim=-1) #except
                )


        #t#self.opt_history=defaultdict(list)#t#


    #we're building this for fixed images size
    def forward(self,image,ocrRes,questions,answers=None,useCurvedBBs=False,RUN=False):
        if self.blank_ocr:
            ocrRes=[[]]*len(questions)
        torch.autograd.set_detect_anomaly(True)
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

        im_tokens = self.patch_embed(image)
        im_tokens += self.absolute_2dpos_embed #Swin doesn't use this as it can rely on the biased attention. We need the image tokens to know where they are so they can interact with the document and question tokens
        num_im = im_tokens.size(1)
        #Dropout?

        all_ocr_res=[]
        all_ocr_bbs=[]
        all_ocr_1dpos=[]
        max_len=0
        for b,res_im in enumerate(ocrRes):
            ocr_res = []#[torch.zeros_like(preds[0])]
            ocr_bbs = []#[[0,0,0,0]]
            ocr_1dpos = []#[0]
            for i,(bb,(string,char_prob),score) in enumerate(res_im):
                #spread x,y location along span
                tlX,tlY = bb[0]
                trX,trY = bb[1]
                brX,brY = bb[2]
                blX,blY = bb[3]
                lX,lY,rX,rY,width,height = calcXYWH(tlX,tlY,trX,trY,brX,brY,blX,blY)
                cX = (lX+rX)/2
                cY = (lY+rY)/2
                #we'll use the entire height for each part
                start_point = torch.FloatTensor([lX,lY])
                end_point = torch.FloatTensor([rX,rY])
                step = 1/(char_prob.size(0)-1)
                forward = torch.arange(1,-(step-0.00001),-step)[:,None] #add xy dim
                backward = torch.arange(0,1+(step-0.00001),step)[:,None]
                assert forward.size(0)==char_prob.size(0) and backward.size(0)==char_prob.size(0)
                all_xy = forward*start_point + backward*end_point



                #rather than taking the whol sequence, we'll just take the probabilities at character predictions (nonblank)
                char_pred = char_prob.argmax(dim=1)
                char_loc = char_pred!=0
                new_char_prob = char_prob[char_loc] #this will be on the GPU
                new_xy = all_xy[char_loc]
                wh = torch.FloatTensor([width,height])[None,:].expand(new_xy.size(0),-1)

                bbs = torch.cat( (new_xy,wh), dim=1)

                #We'll append one additional entry at the begining. Has 0 for all probs and uses the center xy
                first_bb = torch.FloatTensor([[cX,cY,width,height]])
                bbs = torch.cat( (first_bb,bbs), dim=0)
                new_char_prob = torch.cat( (torch.zeros_like(new_char_prob[0:1]),new_char_prob), dim=0)
                ocr_res.append(new_char_prob)
                ocr_bbs.append(bbs)
                ocr_1dpos.extend(range(0,new_char_prob.size(0)))
            
            #ocr_bbs.append([0,0,0,0])
            #ocr_pos.append(0)
            max_len = max(max_len,len(ocr_1dpos))
            all_ocr_res.append(torch.cat(ocr_res,dim=0))
            all_ocr_bbs.append(torch.cat(ocr_bbs,dim=0))
            all_ocr_1dpos.append(ocr_1dpos)
            assert len(ocr_1dpos) ==  all_ocr_bbs[-1].size(0)

        #padding
        ocr_padding_mask = torch.BoolTensor(len(all_ocr_res),max_len).fill_(0) #0 / 1 on when a padded value
        for i in range(len(all_ocr_res)):
            if all_ocr_res[i].size(0)<max_len:
                diff = max_len - all_ocr_res[i].size(0)
                all_ocr_res[i] = F.pad(all_ocr_res[i],(0,0,0,diff))
                all_ocr_bbs[i] = F.pad(all_ocr_bbs[i],(0,0,0,diff))
                all_ocr_1dpos[i] += [0]*diff
                ocr_padding_mask[i,-diff:]=1

        ocr_tokens = self.ocr_emb(torch.stack(all_ocr_res,dim=0))
        ocr_bbs = torch.stack(all_ocr_bbs,dim=0).to(device)
        ocr_1dpos = torch.LongTensor(all_ocr_1dpos).to(device)
        ocr_padding_mask = ocr_padding_mask.to(device)
        all_ocr_res=None
        all_ocr_bbs=None
        all_ocr_1dpos=None
        num_ocr = ocr_tokens.size(1)

        #we need to extend batch entries with multiple questions
        repeats = [len(q) for q in questions] #different elements in batch may have different numbers of questions

        #first expand tokens and other things
        repeats_cuda = torch.LongTensor(repeats).to(device)
        im_tokens = torch.repeat_interleave(im_tokens,repeats_cuda,dim=0)
        ocr_tokens = torch.repeat_interleave(ocr_tokens,repeats_cuda,dim=0)
        ocr_bbs = torch.repeat_interleave(ocr_bbs,repeats_cuda,dim=0)
        ocr_1dpos = torch.repeat_interleave(ocr_1dpos,repeats_cuda,dim=0)
        ocr_padding_mask = torch.repeat_interleave(ocr_padding_mask,repeats_cuda,dim=0)
        repeats_cuda = None

        #flatten the questions an answers to a single batch
        questions=[q for bq in questions for q in bq]
        new_batch_size = len(questions)
        if not RUN:
            answers=[a for ba in answers for a in ba]
        else:
            answers=['']*new_batch_size
            saved_docqa=[]
            saved_proj_im_tokens=[]
            output_tokens=[]

        #run question+answer through decoder
        q_t = self.tokenizer(questions,return_tensors="pt",padding=True)
        num_q = q_t['input_ids'].size(1)-1 #remove last SEP token
        a_t = self.tokenizer(answers,return_tensors="pt",padding=True)
        num_a = a_t['input_ids'].size(1)-1 #remove last SEP token
        qa_tokens = self.text_embedding(torch.cat((q_t['input_ids'],a_t['input_ids'][:,1:-1]),dim=1).to(device)) #strip CLS and SEP off of answers, will recieve questions SEP
        q_tokens = qa_tokens[:,:num_q] #no SEP
        if not RUN:
            a_tokens = qa_tokens[:,num_q:] #gets SEP at beginning

        #the model input ends up being [CLS] Question  [SEP] Answer
        #                             { q tokens     }{ a tokens   }

        xs=ocr_bbs[:,:,0]
        ys=ocr_bbs[:,:,1]
        ws=ocr_bbs[:,:,2]
        hs=ocr_bbs[:,:,3]
        ocr_pos = ocr_bbs[:,:,0:2] #just x,y

        q_tokens = self.q_pos_1d_enc(q_tokens)
        q_padding_mask = (1-q_t['attention_mask'][:,:-1]).bool().to(device) #remove last SEP
        if not RUN:
            a_tokens = self.a_pos_1d_enc(a_tokens)
            a_padding_mask = (1-a_t['attention_mask'][:,:-1]).bool().to(device) #remove last SEP

        ocr_tokens += self.ocr_pos_emb_x(xs) + self.ocr_pos_emb_y(ys) + self.ocr_pos_emb_w(ws) + self.ocr_pos_emb_h(hs) + self.ocr_1dpos_enc(ocr_1dpos)

        num_all = num_im+num_ocr+num_q+num_a
        all_att_mask = torch.BoolTensor(1,num_all,num_all).fill_(1) #1/0
        all_att_mask[:,-num_a:,-num_a] = torch.tril(all_att_mask[:,-num_a:,-num_a])
        all_att_mask = all_att_mask.to(device)

        #make position (2d) masks. Controls whether relative position attention bias is applied
        ocr_pos_mask = ocr_padding_mask[:,:,None]
        q_pos_mask = torch.FloatTensor(1,num_q,1).fill_(0).to(device)
        a_pos_mask = torch.FloatTensor(1,num_a,1).fill_(0).to(device)

        q_pos = torch.FloatTensor(1,num_q,2).fill_(0).to(device)
        a_pos = torch.FloatTensor(1,num_a,2).fill_(0).to(device)

        q_pos_ex = q_pos.expand(new_batch_size,-1,-1)
        a_pos_ex = a_pos.expand(new_batch_size,-1,-1)
        q_pos_mask_ex = q_pos_mask.expand(new_batch_size,-1,-1)
        a_pos_mask_ex = a_pos_mask.expand(new_batch_size,-1,-1)


        level=0
        if self.swin_layers is not None:
            ocrqa_tokens = torch.cat( (ocr_tokens,q_tokens,a_tokens),dim=1)
            ocrqa_padding_mask = torch.cat( (ocr_padding_mask,q_padding_mask,a_padding_mask), dim=1)
            ocrq_padding_mask = torch.cat( (ocr_padding_mask,q_padding_mask,torch.zeros_like(a_padding_mask)), dim=1)
            #convert to 0/-inf as that's what the Swin code expects
            ocrq_padding_mask_inf = torch.where(ocrq_padding_mask,torch.zeros_like(ocrq_padding_mask),torch.empty_like(ocrq_padding_mask).fill_(float('-inf')))
            ocrqa_pos_mask = torch.cat( (ocr_pos_mask,q_pos_mask_ex,a_pos_mask_ex), dim=1)
            ocrqa_pos = torch.cat( (ocr_pos,q_pos_ex,a_pos_ex), dim=1)
            #ocrqa_att_mask = torch.BoolTensor(1,num_ocr+num_q+num_a,num_ocr+num_q+num_a).fill_(1) #1/0
            #ocrqa_att_mask[:,-num_a:,-num_a] = torch.tril(ocrqa_att_mask[:,-num_a:,-num_a])
            ocrqa_att_mask = all_att_mask[:,num_im:,num_im:]
            for swin_layer, proj_d2i, layout_layer, proj_i2d, im_downsample, ocr_downsample, q_downsample in self.swin_layers:

                #could be run in parallel

                im_tokens = swin_layer(im_tokens,proj_d2i(ocrqa_tokens),   #we pass it the answers
                        docq_padding_mask=ocrq_padding_mask_inf) #but we'll mask them out
                proj_im_tokens = proj_i2d(im_tokens)
                if RUN:
                    saved_ocrqa.append(ocrqa_tokens)
                    saved_proj_im_tokens.append(proj_im_tokens)
                ocrqa_tokens = layout_layer(
                        ocrqa_tokens,
                        ocrqa_pos[:,:,0], #x
                        ocrqa_pos[:,:,1], #y
                        proj_im_tokens,
                        self.im_xs[level].expand(new_batch_size,-1),
                        self.im_ys[level].expand(new_batch_size,-1),
                        docqa_mask=ocrqa_att_mask.expand(new_batch_size,-1,-1),
                        docqa_padding_mask=ocrqa_padding_mask,
                        pos_mask = ocrqa_pos_mask)


                

                if im_downsample is not None:
                    im_tokens = im_downsample(im_tokens)
                    level+=1
                    num_im = im_tokens.size(1)
                if ocr_downsample is not None:
                    ocr_tokens = ocrqa_tokens[:,:num_ocr]
                    ocr_tokens,ocr_pos,ocr_padding_mask = ocr_downsample(ocr_tokens,ocr_pos,ocr_padding_mask)
                    num_ocr = ocr_tokens.size(1)
                    ocr_pos_mask = ocr_padding_mask[:,:,None]#torch.FloatTensor(new_batch_size,ocr_tokens.size(1),1).fill_(1).to(device)
                if q_downsample is not None:
                    q_tokens = ocrqa_tokens[:,num_ocr:num_ocr+num_q]
                    q_tokens,q_padding_mask = q_downsample(q_tokens,q_padding_mask)
                    num_q = q_tokens.size(1)
                    q_pos = q_pos[:,:num_q]
                    q_pos_mask = q_pos_mask[:,:num_q]#torch.FloatTensor(new_batch_size,num_q,1).fill_(0).to(device)
                    q_pos_ex = q_pos.expand(new_batch_size,-1,-1)
                    q_pos_mask_ex = q_pos_mask.expand(new_batch_size,-1,-1)

                if im_downsample is not None or ocr_downsample is not None or q_downsample is not None:
                    ocrqa_att_mask = ocrqa_att_mask[:,-(num_ocr+num_q+num_a):,-(num_ocr+num_q+num_a):]
                if ocr_downsample is not None or q_downsample is not None:
                    ocrqa_tokens = torch.cat( (ocr_tokens,q_tokens,a_tokens),dim=1)
                    ocrq_padding_mask = torch.cat( (ocr_padding_mask,q_padding_mask,torch.zeros_like(a_padding_mask)), dim=1)
                    ocrq_padding_mask_inf = torch.where(ocrq_padding_mask,torch.zeros_like(ocrq_padding_mask),torch.empty_like(ocrq_padding_mask).fill_('-inf'))
                    ocrq_pos_mask = torch.cat( (ocr_pos_mask,q_pos_mask,blank_a_pos_mask), dim=1)
                    ocrq_pos = torch.cat( (ocr_pos,q_pos,blank_a_pos), dim=1)
            
            ocrqa_tokens = self.ocrqa_transition(ocrqa_tokens)
            ocr_tokens = ocrqa_tokens[:,:num_ocr]
            q_tokens = ocrqa_tokens[:,num_ocr:num_ocr+num_q]
            a_tokens = ocrqa_tokens[:,num_ocr+num_q:num_ocr+num_q+num_a]
        #Swin DONE

        im_pos = torch.stack( (self.im_xs[level],self.im_ys[level]),dim=2).expand(new_batch_size,-1,-1)
        im_pos_mask = torch.FloatTensor(1,num_im,1).fill_(1).expand(new_batch_size,-1,-1).to(device)
        im_padding_mask = torch.BoolTensor(1,num_im).fill_(0).expand(new_batch_size,-1).to(device)
        im_tokens = self.im_transition(im_tokens)

        num_all = num_im+num_ocr+num_q+num_a
        all_att_mask = all_att_mask[:,-num_all:,-num_all:] 

        all_tokens = torch.cat( (im_tokens,ocr_tokens,q_tokens,a_tokens),dim=1)
        all_pos = torch.cat( (im_pos,ocr_pos,q_pos_ex,a_pos_ex),dim=1)
        all_pos_mask = torch.cat( (im_pos_mask,ocr_pos_mask,q_pos_mask_ex,a_pos_mask_ex),dim=1)
        all_padding_mask = torch.cat( (im_padding_mask,ocr_padding_mask,q_padding_mask,a_padding_mask),dim=1)
        for im_downsample, ocr_downsample, q_downsample, layer in self.full_layers:
            if im_downsample is not None:
                im_tokens = all_tokens[:,:num_im]
                im_tokens,im_pos = im_downsample(im_tokens,im_pos)
                num_im = im_tokens.size(1)
                im_pos_mask = im_pos_mask[:,:num_im]#torch.FloatTensor(new_batch_size,num_im,1).fill_(1).to(device)
                im_padding_mask = im_padding_mask[:,:num_im]#torch.BoolTensor(new_batch_size,num_im).fill_(1).to(device)
            if ocr_downsample is not None:
                ocr_tokens = all_tokens[:,num_im:num_im+num_ocr]
                ocr_tokens,ocr_pos,ocr_padding_mask = ocr_downsample(ocr_tokens,ocr_pos,ocr_padding_mask)
                num_ocr = ocr_tokens.size(1)
                ocr_pos_mask = ocr_padding_mask[:,:,None]#torch.FloatTensor(new_batch_size,ocr_tokens.size(1),1).fill_(1).to(device)
            if q_downsample is not None:
                q_tokens = all_tokens[:,num_im+num_ocr:num_im+num_ocr+num_q]
                q_tokens,q_padding_mask = q_downsample(q_tokens,q_padding_mask)
                num_q = q_tokens.size(1)
                q_pos_mask = q_pos_mask[:,:num_q]#torch.FloatTensor(new_batch_size,num_q,1).fill_(0).to(device)
                q_pos = q_pos[:,:num_q]
                q_pos_ex = q_pos.expand(new_batch_size,-1,-1)
                q_pos_mask_ex = q_pos_mask.expand(new_batch_size,-1,-1)

            if im_downsample is not None or ocr_downsample is not None or q_downsample is not None:
                num_all = num_im+num_ocr+num_q+num+a
                all_tokens = torch.cat( (im_tokens,ocr_tokens,q_tokens,a_tokens),dim=1)
                all_pos = torch.cat( (im_pos,ocr_pos,q_pos_ex,a_pos_ex),dim=1)
                all_pos_mask = torch.cat( (im_pos_mask,ocr_pos_mask,q_pos_mask_ex,a_pos_mask_ex),dim=1)
                all_padding_mask = torch.cat( (im_padding_mask,ocr_padding_mask,q_padding_mask,a_padding_mask),dim=1)
                #all_att_mask = torch.BoolTensor(new_batch_size,all_tokens.size(1),all_tokens.size(1)).fill_(1).to(device)
                #all_att_mask[:,-num_a:,-num_a] = a_att_mask
                all_att_mask = all_att_mask[:,-num_all:,-num_all:] #becuase the a part doesnt change (only part non-1) we can just trim down


            #all_tokens = torch.cat( (im_tokens,ocr_tokens,q_tokens,a_tokens),dim=1)
            all_tokens = layer(
                    all_tokens,
                    all_pos,
                    all_pos_mask, #which tokens have pos (x,y)
                    all_att_mask, #mask future answer tokens
                    all_padding_mask #padding on q and a tokens
                    )

            #im_tokens = all_tokens[:,:num_im]
            #ocr_tokens = all_tokens[:,num_im:num_im+num_ocr]
            #q_tokens = all_tokens[:,num_im+num_ocr:num_im+num_ocr+num_q]
            #a_tokens = all_tokens[:,num_im+num_ocr+num_q:num_im+num_ocr+num_q+num_a]

                

        if RUN: #assuming batchsize of 1
            raise NotImplementedError('need to do RUN still')
            assert docqa.size(0)==1 #just to make stopping easier
            response_decoded = self.answer_decode(docqa)
            response_greedy_token = response_decoded.argmax(dim=2)
            response_greedy_token = response_greedy_token[:,-1:] #only care about last

            output_tokens.append(response_greedy_token[0,0].item())

            offset = docqa.size(1)

            max_pred_len=self.max_pred_len
            holder_xs = torch.cat((xs,torch.FloatTensor(new_batch_size,max_pred_len).fill_(0).to(device)),dim=1)
            holder_ys = torch.cat((ys,torch.FloatTensor(new_batch_size,max_pred_len).fill_(0).to(device)),dim=1)
            holder_answer_padding_mask = torch.BoolTensor(new_batch_size,max_pred_len).fill_(0).to(device) #assumes here batch size of 1
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



        response = all_tokens[:,-(num_a):]


        response_decoded = self.answer_decode(response)

        #t#time = timeit.default_timer()-ticA#t#
        #t#self.opt_history['transformers'].append(time)#t#
        #t#tic=timeit.default_timer()#t#

        response_greedy_tokens = response_decoded.argmax(dim=2)
        target_decoded = a_t['input_ids'][:,1:]# This has the SEP tokens (and padding), but not CLS (start) token

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
        return response_decoded, target_decoded.to(device), batch_string_response

    #t#def print_opt_times(self):#t#
        #t#for name,times in self.opt_history.items():#t#
            #t#print('time {}({}): {}'.format(name,len(times),np.mean(times)))#t#
            #t#if len(times)>300: #t#
                #t#times.pop(0)   #t#
                #t#if len(times)>600:#t#
                    #t#times.pop(0)#t#
