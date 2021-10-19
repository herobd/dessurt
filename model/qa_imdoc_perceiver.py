from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pairing_g_graph_layoutlm import  runLayoutLM
from model.pos_encode import PositionalEncoding, UniformRealEmbedding,PositiveRealEmbedding, ReturnPositionalEncoding
from model.rel_pos_im_transformer import RelPosQTransformerLayer, RelPosTransformerLayer
from model.swin_transformer import ConvPatchEmbed, SwinTransformerBlock, PatchMerging
from model.trans_pooling import OCRPooler, QPooler
try:
    from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
    from transformers import LayoutLMTokenizer, LayoutLMModel
except:
    pass
from utils.character_tokenizer import CharacterTokenizer
from collections import defaultdict
from timm.models.layers import trunc_normal_
import math, random
from utils.util import calcXYWH

import timeit

def normalize_bbox2(bbox, dwidth, dheight, twidth, theight, max_dist):
     return [
         max(min(int(max_dist * (bbox[0] / dwidth)),max_dist),0),
         max(min(int(max_dist * (bbox[1] / dheight)),max_dist),0),
         max(min(int(max_dist * (bbox[2] / twidth)),twidth),0),
         max(min(int(max_dist * (bbox[3] / theight)),theight),0),
     ]

#https://discuss.pytorch.org/t/differentiable-affine-transforms-with-grid-sample/79305
def affineTransform(x,out_size,scalew=1, scaleh=1, rot=0):
    #trans_scale = torch.FloatTensor([
    #    [scalew, 0, -x.shape[1]//2],
    #    [0, scaleh, -x.shape[0]//2],
    #    [0,0,1]
    #])
    scale = torch.FloatTensor([
        [scalew, 0, 0],
        [0, scaleh, 0],
        [0,0,1]
    ])
    rot = torch.FloatTensor([
        [math.cos(rot), -math.sin(rot), 0],
        [math.sin(rot), math.cos(rot), 0],
        [0,0,1]
    ])
    #retrans = torch.FloatTensor([
    #    [1,0, out_size[1]//2],
    #    [0,1, out_size[0]//2],
    #    [0,0,1]
    #])
    #theta = torch.matmul(retrans,torch.matmul(rot,trans_scale))[None,0:2]
    #theta = rot[None,0:2]
    theta = torch.matmul(scale,rot)[None,0:2]
    grid = F.affine_grid(theta, out_size, align_corners=False)
    return F.grid_sample(x, grid.to(x.device), align_corners=False,mode='nearest')

class QAImDocPerceiver(BaseModel):
    def __init__(self,config):
        super(QAImDocPerceiver, self).__init__(config)
        self.blank_ocr = config['blank_ocr'] if 'blank_ocr' in config else False
        self.ocr_in_image = config['grid_ocr'] if 'grid_ocr' in config else False
        self.image_size = config['image_size'] #start at 512?
        dropout = 0 if 'no_dropout' in config and  config['no_dropout'] else 0.1
        lighter_conv_patch_emb = config['lighter_conv_patch_emb'] if 'lighter_conv_patch_emb' in config else False

        blocks_per_level = config['swin_blocks_per_level'] #[2,2,6,2] -> in:512,emb:64 then 64,32,16,8
        full_layers = config['full_layers']
        if full_layers is not None:
            fd_model = config['full_dim']
            fnhead = config['full_num_heads']
            fdim_ff = config['fdim_ff']
        else:
            fd_model=None
        if blocks_per_level is not None:
            if 'swin_text_downsample_all' in config:
                swin_text_downsample = [[d,d] if type(d) is bool else d for d in config['swin_text_downsample_all']]
                swin_text_downsample_dense=True
                assert (not swin_text_downsample[-1][0] and not swin_text_downsample[-1][1]) and "Shouldn't downsample final. Not used."
                self.downsample_ocr = sum(d[0] for d in swin_text_downsample)
                self.downsample_q = sum(d[1] for d in swin_text_downsample)

            else:
                swin_text_downsample = config['swin_text_downsample'] if 'swin_text_downsample' in config else [False]*len(blocks_per_level)
                swin_text_downsample_dense=False
                self.downsample_ocr = sum(swin_text_downsample)

            window_size = config['window_size'] #7
            if type(window_size) is int:
                window_size = [window_size]*len(blocks_per_level)
            d_model = config['decode_dim']
            if fd_model is None:
                fd_model = d_model
            dim_ff = config['dim_ff']
            nhead = config['decode_num_heads']
            swin_nhead = config['swin_nheads'] #[3,6,12,24] | [2,6,12,12] probably don't need as much later
            im_embed_dim = config['im_embed_dim'] #96 -> 96,192,384,768 | 64->64,128,256,512
        else:
            d_model = fd_model
            im_embed_dim = fd_model

        if isinstance(self.image_size,int):
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
        self.ocr_out_dim = 97
        self.one_hot_conf = 0.9
        self.zero_hot_conf = (1-self.one_hot_conf)/(self.ocr_out_dim-1)
        if not self.ocr_in_image:
            self.ocr_emb = nn.Sequential(
                    nn.Conv1d(self.ocr_out_dim,d_model,3,padding=1), #this will mix neighboring instances....
                    nn.ReLU(True),
                    nn.Conv1d(d_model,d_model,3,padding=1),
                    )
                    #nn.Linear(97,d_model,bias=False)
            self.ocr_1dpos_enc = ReturnPositionalEncoding(d_model,dropout=dropout,max_len=1000)
            self.ocr_seqid_enc = ReturnPositionalEncoding(d_model,dropout=dropout,max_len=1000,offset_start=2000)
            self.ocr_pos_emb_x = UniformRealEmbedding(d_model,0,self.image_size[1],100)
            self.ocr_pos_emb_y = UniformRealEmbedding(d_model,0,self.image_size[0],100)
            self.ocr_pos_emb_w = PositiveRealEmbedding(d_model,0,int(0.5*self.image_size[1]),30)
            self.ocr_pos_emb_h = PositiveRealEmbedding(d_model,0,int(0.3*self.image_size[0]),30)
        else:
            self.embed_ocr_grid = nn.Linear(self.ocr_out_dim,im_embed_dim)

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

        #dim=32?
        #logits dim=100
        self.perciever = PerceiverIO(
                depth = 26,
                num_latents = 256,
                latent_dim=1280
                dim = 768, #input dim
                queries_dim = 768, #output dim
                cross_heads = 8,
                latent_heads = 8,
                **kwargs #decoder_ff=True?
            )


        self.final_resolution = self.patches_resolution
        upsample_net = [nn.ConvTranspose2d(d_im,d_im//2,4,2,1),
                        nn.InstanceNorm2d(d_im//2),
                        nn.Dropout2d(p=0.125,inplace=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(d_im//2,d_im//4,3,1,1),
                        nn.InstanceNorm2d(d_im//4),
                        nn.Dropout2d(p=0.125,inplace=True),
                        nn.ReLU(inplace=True)
                        nn.ConvTranspose2d(d_im//4,d_im//8,4,2,1),
                        nn.InstanceNorm2d(d_im//8),
                        nn.Dropout2d(p=0.125,inplace=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(d_im//8,1,1,1,0),
                        nn.Sigmoid()]
        self.upsample_net= nn.Sequential(*upsample_net)
        
        

        self.answer_decode = nn.Sequential(
                nn.Linear(fd_model,self.decode_tokenizer.vocab_size),
                nn.LogSoftmax(dim=-1) #except
                )


        #t#self.opt_history=defaultdict(list)#t#




    #we're building this for fixed images size
    def forward(self,image,ocr_results,questions,answers=None,RUN=False):


        #t#ticA=timeit.default_timer()#t#
        device = image.device

        im_tokens = self.patch_embed(image)
        im_tokens += self.absolute_2dpos_embed #Swin doesn't use this as it can rely on the biased attention. We need the image tokens to know where they are so they can interact with the document and question tokens
        query_im_tokens = im_tokens
        num_im = im_tokens.size(1)
        #Dropout?

        if self.ocr_seperate_tokens:
            ocr_tokens,ocr_bbs,ocr_1dpos,ocr_seqid,ocr_padding_mask = self.prepareOCRTokens(ocr_results)
        else:
            ocr_padding_mask = torch.BoolTensor(batch_size,0)
            ocr_tokens =None
            ocr_bbs = torch.FloatTensor(batch_size,4)
            ocr_1dpos = torch.LongTensor(batch_size,0)
            ocr_seqid = torch.LongTensor(batch_size,0)

        if self.ocr_in_image:
            im_tokens += self.appendOCRToVisual(ocrRes)

        ocr_bbs = ocr_bbs.to(device)
        ocr_1dpos = ocr_1dpos.to(device)
        ocr_seqid = ocr_seqid.to(device)
        ocr_padding_mask = ocr_padding_mask.to(device)
        num_ocr = ocr_padding_mask.size(1)

        repeats = [len(q) for q in questions] #different elements in batch may have different numbers of questions
        assert all(r==1 for r in repeats)
        #for now, no multi questions. In the future, each question (and answer) will have special encoding and all go together (appended along len dim, not batch dim)

        questions=[q for bq in questions for q in bq]
        new_batch_size = len(questions)
        if not RUN:
            answers=[a for ba in answers for a in ba]
        else:
            answers=['']*new_batch_size

        #run question+answer through decoder
        q_t = self.tokenizer(questions,return_tensors="pt",padding=True)
        num_q = q_t['input_ids'].size(1)
        a_t = self.tokenizer(answers,return_tensors="pt",padding=True)
        num_a = a_t['input_ids'].size(1)-1 #remove last SEP token
        qa_tokens = self.text_embedding(torch.cat((q_t['input_ids'],a_t['input_ids'][:,:-1]),dim=1).to(device))
        q_tokens = qa_tokens[:,:num_q] 
        a_tokens = qa_tokens[:,num_q:] 

        #the model input ends up being [CLS] Question  [SEP] Answer
        #                             { q tokens     }{ a tokens   }

        #xs=ocr_bbs[:,:,0]
        #ys=ocr_bbs[:,:,1]
        #ws=ocr_bbs[:,:,2]
        #hs=ocr_bbs[:,:,3]
        #ocr_pos = ocr_bbs[:,:,0:2] #just x,y

        q_tokens = self.q_pos_1d_enc(q_tokens)
        q_padding_mask = (1-q_t['attention_mask']).bool()#.to(device) 
        q_padding_mask = q_padding_mask.to(device)


        a_tokens = self.a_pos_1d_enc(a_tokens)
        a_padding_mask = (1-a_t['attention_mask'][:,1:]).bool().to(device) #remove last SEP
        
        if num_ocr>0:
            ocr_tokens += self.ocr_pos_emb_x(xs) + self.ocr_pos_emb_y(ys) + self.ocr_pos_emb_w(ws) + self.ocr_pos_emb_h(hs) + self.ocr_1dpos_enc(ocr_1dpos) + self.ocr_seqid_enc(ocr_seqid)
        else:
            ocr_tokens = torch.FloatTensor(new_batch_size,0,q_tokens.size(2)).to(device)

        num_all = num_im+num_ocr+num_q+num_a

        #make position (2d) masks. Controls whether relative position attention bias is applied
        #ocr_pos_mask = (~ocr_padding_mask[:,:,None]).float()


        #Put full input together. im, ocr,question

        input_tokens = torch.cat( (im_tokens,ocr_tokens,q_tokens), dim=1)
        input_padding_mask = torch.cat( (im_padding_mask,ocr_padding_mask,q_padding_mask), dim=1)
        
        #Put query together
        query_tokens = torch.cat((query_im_tokens,self.query_a_tokens),dim=1)

        #Run through Perceiver
        output = self.perciever(input_tokens,input_padding_mask,query_tokens)

        num_out_im_tokens = self.patches_resolution[0]*self.patches_resolution[1]

        im_tokens = output[:,:num_out_im_tokens]
        a_tokens = output[:,num_out_im_tokens:]


        ##############
        #Visual output
        H,W = self.patches_resolution
        #reshape and permute to convert to image
        im_feats = im_tokens.view(batch_size,H,W,im_tokens.size(2)).permute(0,3,1,2)
        out_mask = self.upsample_net(im_feats)
        
        #############
        #Text output
        response_decoded = self.answer_decode(a_tokens)

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


        return response_decoded, target_decoded.to(device), batch_string_response, out_mask

    #t#def print_opt_times(self):#t#
        #t#for name,times in self.opt_history.items():#t#
            #t#print('time {}({}): {}'.format(name,len(times),np.mean(times)))#t#
            #t#if len(times)>300: #t#
                #t#times.pop(0)   #t#
                #t#if len(times)>600:#t#
                    #t#times.pop(0)#t#

    def prepareForOCRTokens(self,ocr_results):
        all_ocr_res=[]
        all_ocr_bbs=[]
        all_ocr_1dpos=[]
        all_ocr_seqid=[]
        max_len=0
        batch_size = len(ocr_results)
            if ocrRes is None:
                ocrRes = [[]]*batch_size
            for b,res_im in enumerate(ocr_results):
                ocr_res = []#[torch.zeros_like(preds[0])]
                ocr_bbs = []#[[0,0,0,0]]
                ocr_1dpos = []#[0]
                ocr_seqid = []#[0]
                for i,(bb,(string,char_prob),score) in enumerate(res_im):
                    if len(string)==0 or (self.blank_ocr and self.train and random.random()<self.blank_ocr):
                        continue
                    #spread x,y location along span
                    tlX,tlY = bb[0]
                    trX,trY = bb[1]
                    brX,brY = bb[2]
                    blX,blY = bb[3]
                    lX,lY,rX,rY,width,height,rot = calcXYWH(tlX,tlY,trX,trY,brX,brY,blX,blY)
                    #cX = (lX+rX)/2
                    cY = (lY+rY)/2
                    #we'll use the entire height for each part
                    start_point = torch.FloatTensor([lX,cY])
                    end_point = torch.FloatTensor([rX,cY])
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

                    #We'll append additional entries at the begining and end. We'll first be sure there is one ad the front back, and then add however many needed to the back so that pooling always is self contained
                    start_bb = bbs[0:1]
                    end_bb = bbs[-1:]
                    empty_prob = -1*torch.ones_like(new_char_prob[0:1])
                    if self.downsample_ocr>0:
                        missing = (2+new_char_prob.size(0))% (2**self.downsample_ocr)
                        missing = ((2**self.downsample_ocr)-missing) % (2**self.downsample_ocr)
                        add_front = 1
                        add_back = 1+missing
                        bbs = torch.cat( ([start_bb]*add_front) + [bbs] + ([end_bb]*add_back), dim=0)
                        new_char_prob = torch.cat( ([empty_prob]*add_front) + [new_char_prob] + ([empty_prob]*add_back), dim=0)

                    ocr_res.append(new_char_prob)
                    ocr_bbs.append(bbs)
                    assert new_char_prob.size(0) == bbs.size(0)
                    ocr_1dpos.extend(range(0,new_char_prob.size(0)))
                    ocr_seqid.extend([len(ocr_bbs)]*new_char_prob.size(0))

                
                #ocr_bbs.append([0,0,0,0])
                #ocr_pos.append(0)
                max_len = max(max_len,len(ocr_1dpos))
                if len(ocr_res)>0:
                    all_ocr_res.append(torch.cat(ocr_res,dim=0))
                    all_ocr_bbs.append(torch.cat(ocr_bbs,dim=0))
                    all_ocr_1dpos.append(ocr_1dpos)
                    all_ocr_seqid.append(ocr_seqid)
                else:
                    all_ocr_res.append(torch.FloatTensor(0,97).to(device))
                    all_ocr_bbs.append(torch.FloatTensor(0,4))#.to(device))
                    all_ocr_1dpos.append([])
                    all_ocr_seqid.append([])

        #padding
        ocr_padding_mask = torch.BoolTensor(len(all_ocr_res),max_len).fill_(0) #0 / 1 on when a padded value
        for i in range(len(all_ocr_res)):
            if all_ocr_res[i].size(0)<max_len:
                diff = max_len - all_ocr_res[i].size(0)
                all_ocr_res[i] = F.pad(all_ocr_res[i],(0,0,0,diff))
                all_ocr_bbs[i] = F.pad(all_ocr_bbs[i],(0,0,0,diff))
                all_ocr_1dpos[i] += [0]*diff
                all_ocr_seqid[i] += [0]*diff
                ocr_padding_mask[i,-diff:]=1
        if max_len!=0:
            ocr_tokens = self.ocr_emb(torch.stack(all_ocr_res,dim=0).permute(0,2,1)).permute(0,2,1)
        else:
            ocr_tokens =None
        ocr_bbs = torch.stack(all_ocr_bbs,dim=0)
        ocr_1dpos = torch.LongTensor(all_ocr_1dpos)
        ocr_seqid = torch.LongTensor(all_ocr_seqid)

        return ocr_tokens,ocr_bbs,ocr_1dpos,ocr_seqid,ocr_padding_mask

    def appendOCRToVisual(self,ocr_results):
        batch_size = len(ocr_results)
        ocr_grid = torch.FloatTensor(batch_size,self.ocr_out_dim,*self.patches_resolution).fill_(-1).to(device)
        for b,res_im in enumerate(ocr_results):
            ocr_res = []#[torch.zeros_like(preds[0])]
            ocr_bbs = []#[[0,0,0,0]]
            ocr_1dpos = []#[0]
            ocr_seqid = []#[0]
            if res_im is None:
                res_im = []
            for i,(bb,(string,char_prob),score) in enumerate(res_im):
                #We will draw the character probabilities into the image using the 
                #print('XX inserting grid {}/{}'.format(i,len(res_im)))
                # bounding box
                tlX,tlY = bb[0]
                trX,trY = bb[1]
                brX,brY = bb[2]
                blX,blY = bb[3]
                tlX *= self.patch_scale_x
                tlY *= self.patch_scale_y
                trX *= self.patch_scale_x
                trY *= self.patch_scale_y
                brX *= self.patch_scale_x
                brY *= self.patch_scale_y
                blX *= self.patch_scale_x
                blY *= self.patch_scale_y
                lX,lY,rX,rY,width,height,rot = calcXYWH(tlX,tlY,trX,trY,brX,brY,blX,blY)
                
                left_x = min(tlX,trX,blX,brX)
                top_y = min(tlY,trY,blY,brY)
                h=max(tlY,trY,blY,brY) - min(tlY,trY,blY,brY)
                w=max(tlX,trX,blX,brX) - min(tlX,trX,blX,brX)
                patch_size = (1,self.ocr_out_dim,max(1,round(h)),max(1,round(w)))
                if isinstance(char_prob,list):
                    tensor = torch.FloatTensor(len(char_prob),self.ocr_out_dim).fill_(self.zero_hot_conf)
                    tensor[range(len(char_prob)),char_prob]=self.one_hot_conf
                    char_prob=tensor.to(device)

                im_patch = affineTransform(
                        char_prob.permute(1,0)[None,:,None],#.to(device),#make sequance an image,
                        patch_size, #canvas to draw in
                        w/width,
                        h/height, #expand height of char prob to fill vert space
                        rot)

                #check boundaries in case patch is off image
                if round(top_y)<0:
                    y_start = -round(top_y)
                    top_y=0
                else:
                    y_start=0
                if round(top_y)+im_patch.size(2)>=ocr_grid.size(2):
                    y_end=ocr_grid.size(2)-(round(top_y)+im_patch.size(2)+1)
                else:
                    y_end=im_patch.size(2)
                if round(left_x)<0:
                    x_start = -round(left_x)
                    left_x=0
                else:
                    x_start=0
                if round(left_x)+im_patch.size(3)>=ocr_grid.size(3):
                    x_end=ocr_grid.size(3)-(round(left_x)+im_patch.size(3)+1)
                else:
                    x_end=im_patch.size(3)
                #correct if boundaries are bad
                im_patch = im_patch[:,:,y_start:y_end,x_start:x_end]

                mask = im_patch[0].sum(dim=0)!=0
                ocr_grid[b,:,
                        round(top_y):round(top_y)+mask.size(0),
                        round(left_x):round(left_x)+mask.size(1)][:,mask] = im_patch[0][:,mask]

                #x_step = (rX-lX)/char_prob.size(0)
                #y_step = (rY-lY)/char_prob.size(0)
                #to_x = [round(lX+i*x_step) for i in range(char_prob.size(0))]
                #to_y = [round(lY+i*y_step) for i in range(char_prob.size(0))]
                #ocr_grid[b,:,to_y,to_x] = char_prob
        
        ocr_grid = ocr_grid.permute(0,2,3,1).view(batch_size,-1,self.ocr_out_dim) #flatten
        return self.embed_ocr_grid(ocr_grid)
