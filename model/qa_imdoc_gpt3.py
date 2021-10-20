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
from testtest import PRINT_ATT,ATT_TEXT,attDisplay,NUMS

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

class QAImDocGPT3(BaseModel):
    def __init__(self,config):
        super(QAImDocGPT3, self).__init__(config)
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
            im_xs = torch.arange(patch_size[1]/2,self.image_size[1],patch_size[1])[None,:].expand(cur_resolution[0],-1)
            im_ys = torch.arange(patch_size[0]/2,self.image_size[0],patch_size[0])[:,None].expand(-1,cur_resolution[1])
            #im_cords = torch.stack((im_xs,im_ys),dim=2).view(patches_resolution[0]*patches_resolution[1],2)
            im_xs = im_xs.contiguous().view(1,cur_resolution[0]*cur_resolution[1])
            im_ys = im_ys.contiguous().view(1,cur_resolution[0]*cur_resolution[1])
            self.register_buffer("im_xs{}".format(level),im_xs,persistent=False)
            self.register_buffer("im_ys{}".format(level),im_ys,persistent=False)
            for block in range(blocks):
                last = level<len(blocks_per_level)-1 and block == blocks-1
                if (swin_text_downsample_dense and swin_text_downsample[len(self.swin_layers)][0]) or (last and swin_text_downsample[level]):
                    ocr_pool = OCRPooler(d_model)
                else:
                    ocr_pool = None
                if (swin_text_downsample_dense and swin_text_downsample[len(self.swin_layers)][1]) or (last and swin_text_downsample[level]):
                    q_pool = QPooler(d_model)
                else:
                    q_pool = None
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
                    RelPosQTransformerLayer(d_model,nhead,max_dist,dim_ff,dropout=dropout),
                    nn.Linear(d_im,d_model,bias=False) if d_model!=d_im else nn.Identity(),
                    PatchMerging(cur_resolution, dim=d_im, norm_layer=nn.LayerNorm) if last else None,
                    ocr_pool,
                    q_pool
                    ] ) )

        self.im_xs=[None]*len(blocks_per_level) #the x,y cords of each patch center for every level/resolution
        self.im_ys=[None]*len(blocks_per_level)

        #def hookf(name):
        #    print('registered '+name)
        #    def hook(module,gradin,gradout):
        #        print(name+' grad in')
        #        for gin in gradin:
        #            print(gin.size())
        #        print(name+' grad out')
        #        for gout in gradout:
        #            print(gout.size())

        #for i,(swin,lin1,rel,lin2,pm,ocrp,qp) in enumerate(self.swin_layers):
        #    if ocrp is not None:
        #        ocrp.register_backward_hook(hookf('ocrpool [{}]'.format(i)))
        #    if qp is not None:
        #        qp.register_backward_hook(hookf('qpool[{}]'.format(i)))

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
        
        
        if d_im != fd_model:
            self.im_transition = nn.Linear(d_im,fd_model)
        else:
            self.im_transition = nn.Identity()
        if d_model != fd_model:
            self.ocrqa_transition = nn.Linear(d_model,fd_model)
        else:
            self.ocrqa_transition = nn.Identity()
        
        if full_layers is not None:
            self.full_layers = nn.ModuleList()
            for im_pool_p, ocr_pool_p, q_pool_p in full_layers:
                if im_pool_p=='n':
                    im_pool = None
                elif im_pool_p=='p':
                    im_pool = AttentionPrunning( cur_resolution )
                else:
                    raise NotImplementedError('unknown image pooling method: {}'.format(im_pool_p))

                if ocr_pool_p=='n':
                    ocr_pool = None
                elif ocr_pool_p=='p':
                    ocr_pool = OCRPooler(fd_model)#nn.Conv2d(fd_model,fd_model,kernel_size=4,stride=2,padding=1)
                else:
                    raise NotImplementedError('unknown ocr pooling method: {}'.format(ocr_pool_p))

                if q_pool_p=='n':
                    q_pool = None
                elif q_pool_p=='p':
                    q_pool = QPooler(fd_model)
                else:
                    raise NotImplementedError('unknown question pooling method: {}'.format(q_pool_p))

                layer = RelPosTransformerLayer(fd_model,fnhead,max_dist,fdim_ff,dropout=dropout)

                self.full_layers.append(nn.ModuleList([im_pool,ocr_pool,q_pool,layer]))
        else:
            self.full_layers = None

        self.answer_decode = nn.Sequential(
                nn.Linear(fd_model,self.decode_tokenizer.vocab_size),
                nn.LogSoftmax(dim=-1) #except
                )


        #t#self.opt_history=defaultdict(list)#t#




    #we're building this for fixed images size
    def forward(self,image,ocrRes,questions,answers=None,useCurvedBBs=False,RUN=False):
        ##DDD
        #image = torch.autograd.Variable(image,requires_grad=True)
        #self.image=image
        #self.image.retain_grad()
        ##DDD


        #if self.blank_ocr:
        #   ocrRes=[[]]*len(questions)
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

        im_tokens = self.patch_embed(image)
        im_tokens += self.absolute_2dpos_embed #Swin doesn't use this as it can rely on the biased attention. We need the image tokens to know where they are so they can interact with the document and question tokens
        num_im = im_tokens.size(1)
        #Dropout?

        all_ocr_res=[]
        all_ocr_bbs=[]
        all_ocr_1dpos=[]
        all_ocr_seqid=[]
        max_len=0
        batch_size = image.size(0)
        if self.ocr_in_image:
            ocr_grid = torch.FloatTensor(batch_size,self.ocr_out_dim,*self.patches_resolution).fill_(-1).to(device)
            for b,res_im in enumerate(ocrRes):
                ocr_res = []#[torch.zeros_like(preds[0])]
                ocr_bbs = []#[[0,0,0,0]]
                ocr_1dpos = []#[0]
                ocr_seqid = []#[0]
                if PRINT_ATT:
                    full_ocr_string='$'
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
                all_ocr_res.append(torch.FloatTensor(0,97).to(device))
                all_ocr_bbs.append(torch.FloatTensor(0,4))#.to(device))
                all_ocr_1dpos.append([])
                all_ocr_seqid.append([])
            
            ocr_grid = ocr_grid.permute(0,2,3,1).view(batch_size,-1,self.ocr_out_dim) #flatten
            im_tokens += self.embed_ocr_grid(ocr_grid)
        else:
            if ocrRes is None:
                ocrRes = [[]]*batch_size
            if PRINT_ATT:
                assert image.size(0)==1
            for b,res_im in enumerate(ocrRes):
                ocr_res = []#[torch.zeros_like(preds[0])]
                ocr_bbs = []#[[0,0,0,0]]
                ocr_1dpos = []#[0]
                ocr_seqid = []#[0]
                if PRINT_ATT:
                    full_ocr_string='$'
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

                    if PRINT_ATT:
                        full_ocr_string+='|'+string
                
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
                if PRINT_ATT:
                    full_ocr_string+='^'

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
        ocr_bbs = torch.stack(all_ocr_bbs,dim=0).to(device)
        ocr_1dpos = torch.LongTensor(all_ocr_1dpos).to(device)
        ocr_seqid = torch.LongTensor(all_ocr_seqid).to(device)
        ocr_padding_mask = ocr_padding_mask.to(device)
        all_ocr_res=None
        all_ocr_bbs=None
        all_ocr_1dpos=None
        all_ocr_seqid=None
        num_ocr = max_len#ocr_tokens.size(1)

        #we need to extend batch entries with multiple questions
        repeats = [len(q) for q in questions] #different elements in batch may have different numbers of questions

        #first expand tokens and other things
        repeats_cuda = torch.LongTensor(repeats).to(device)
        im_tokens = torch.repeat_interleave(im_tokens,repeats_cuda,dim=0)
        if ocr_tokens is not None:
            ocr_tokens = torch.repeat_interleave(ocr_tokens,repeats_cuda,dim=0)
        ocr_bbs = torch.repeat_interleave(ocr_bbs,repeats_cuda,dim=0)
        ocr_1dpos = torch.repeat_interleave(ocr_1dpos,repeats_cuda,dim=0)
        ocr_seqid = torch.repeat_interleave(ocr_seqid,repeats_cuda,dim=0)
        ocr_padding_mask = torch.repeat_interleave(ocr_padding_mask,repeats_cuda,dim=0)
        repeats_cuda = None

        #flatten the questions an answers to a single batch
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

        #DDD
        #a_tokens=torch.autograd.Variable(a_tokens,requires_grad=True)
        #self.start_token=a_tokens
        #self.start_token.retain_grad()


        #the model input ends up being [CLS] Question  [SEP] Answer
        #                             { q tokens     }{ a tokens   }

        xs=ocr_bbs[:,:,0]
        ys=ocr_bbs[:,:,1]
        ws=ocr_bbs[:,:,2]
        hs=ocr_bbs[:,:,3]
        ocr_pos = ocr_bbs[:,:,0:2] #just x,y

        q_tokens = self.q_pos_1d_enc(q_tokens)
        q_padding_mask = (1-q_t['attention_mask']).bool()#.to(device) 
        if self.downsample_q>0:
            #pad it out
            missing = q_tokens.size(1)% (2**self.downsample_q)
            missing = ((2**self.downsample_q)-missing) % (2**self.downsample_q)
            q_tokens = torch.cat((q_tokens,torch.FloatTensor(new_batch_size,missing,q_tokens.size(2)).fill_(0).to(device)),dim=1)
            q_padding_mask = torch.cat((q_padding_mask,torch.BoolTensor(new_batch_size,missing).fill_(True)),dim=1)
            num_q+=missing
        q_padding_mask = q_padding_mask.to(device)


        a_tokens = self.a_pos_1d_enc(a_tokens)
        a_padding_mask = (1-a_t['attention_mask'][:,1:]).bool().to(device) #remove last SEP
        
        if num_ocr>0:
            ocr_tokens += self.ocr_pos_emb_x(xs) + self.ocr_pos_emb_y(ys) + self.ocr_pos_emb_w(ws) + self.ocr_pos_emb_h(hs) + self.ocr_1dpos_enc(ocr_1dpos) + self.ocr_seqid_enc(ocr_seqid)
        else:
            #ocr_tokens = self.ocr_pos_emb_x(xs) + self.ocr_pos_emb_y(ys) + self.ocr_pos_emb_w(ws) + self.ocr_pos_emb_h(hs) + self.ocr_1dpos_enc(ocr_1dpos) + self.ocr_seqid_enc(ocr_seqid)
            ocr_tokens = torch.FloatTensor(new_batch_size,0,q_tokens.size(2)).to(device)

        num_all = num_im+num_ocr+num_q+num_a

        #make position (2d) masks. Controls whether relative position attention bias is applied
        ocr_pos_mask = (~ocr_padding_mask[:,:,None]).float()

        q_pos = torch.FloatTensor(1,num_q,2).fill_(0).to(device)

        q_pos = q_pos.expand(new_batch_size,-1,-1)
        q_pos_mask = torch.FloatTensor(1,1,1).fill_(0).expand(new_batch_size,num_q,-1).to(device)
        
        a_pos = torch.FloatTensor(1,num_a,2).fill_(0).to(device)
        a_pos_mask = torch.FloatTensor(1,num_a,1).fill_(0).to(device)
        a_pos = a_pos.expand(new_batch_size,-1,-1)
        a_pos_mask = a_pos_mask.expand(new_batch_size,-1,-1)

        all_att_mask = torch.BoolTensor(1,num_all,num_all).fill_(1) #1/0
        all_att_mask[:,-num_a:,-num_a:] = torch.tril(all_att_mask[:,-num_a:,-num_a:])
        all_att_mask[:,:-num_a,-num_a:] = 0 #nothing else attends to a
        all_att_mask = all_att_mask.to(device)


        #Run the Swin and accompanying layers
        level=0
        qa_tokens = torch.cat( (q_tokens,a_tokens),dim=1)
        qa_padding_mask = torch.cat( (q_padding_mask,a_padding_mask), dim=1)
        ocrq_padding_mask = torch.cat( (ocr_padding_mask,q_padding_mask), dim=1)
        #convert to 0/-inf as that's what the Swin code expects
        ocrq_padding_mask_inf = torch.FloatTensor(*ocrq_padding_mask.size()).fill_(0).to(device)
        ocrq_padding_mask_inf[ocrq_padding_mask] = float('-inf')
        qa_pos_mask = torch.cat( (q_pos_mask,a_pos_mask), dim=1)
        qa_pos = torch.cat( (q_pos,a_pos), dim=1)

        im_pos_mask = torch.FloatTensor(1,1,1).fill_(1).expand(new_batch_size,num_im,-1).to(device  )
        all_pos_mask = torch.cat((im_pos_mask,ocr_pos_mask,q_pos_mask,a_pos_mask),dim=1)

        im_padding_mask = torch.BoolTensor(1,1).fill_(0).expand(new_batch_size,num_im).to(device)
        all_padding_mask = torch.cat( (im_padding_mask,ocr_padding_mask,q_padding_mask,a_padding_mask),dim=1)

        if RUN:
            #store results at each layer to reuse
            saved_proj_im_tokens = []
            saved_ocr_tokens = []
            saved_ocr_pos = []
            saved_ocr_padding_mask = []
            saved_q_tokens = []
            saved_q_pos = []
            saved_q_padding_mask = []
            saved_a_tokens = []
            ##

        for i,(swin_layer, proj_d2i, layout_layer, proj_i2d, im_downsample, ocr_downsample, q_downsample) in enumerate(self.swin_layers):

            #could be run in parallel
            if PRINT_ATT:
                NUMS.append((num_ocr,num_q,num_a))

            ocrq_tokens = torch.cat( (ocr_tokens,q_tokens),dim=1)
            im_tokens = swin_layer(im_tokens,proj_d2i(ocrq_tokens),
                    docq_padding_mask=ocrq_padding_mask_inf) 
            proj_im_tokens = proj_i2d(im_tokens)

            if RUN:
                saved_proj_im_tokens.append(proj_im_tokens)
                saved_ocr_tokens.append(ocr_tokens)
                saved_ocr_pos.append(ocr_pos)
                saved_ocr_padding_mask.append(ocr_padding_mask)
                saved_q_tokens.append(q_tokens)
                saved_q_pos.append(q_pos)
                saved_q_padding_mask.append(q_padding_mask)
                saved_a_tokens.append(a_tokens)
            
            qa_tokens = layout_layer(
                    qa_tokens,
                    qa_pos[:,:,0], #x
                    qa_pos[:,:,1], #y
                    qa_pos_mask,
                    #all_tokens,
                    torch.cat((proj_im_tokens,ocr_tokens,qa_tokens),dim=1),
                    torch.cat((self.im_xs[level].expand(new_batch_size,-1),ocr_pos[:,:,0],qa_pos[:,:,0]),dim=1),
                    torch.cat((self.im_xs[level].expand(new_batch_size,-1),ocr_pos[:,:,1],qa_pos[:,:,1]),dim=1),
                    all_pos_mask,
                    all_att_mask[:,-(num_q+num_a):,:],
                    all_padding_mask)
                    

            did_downsample=False
            if im_downsample is not None:
                did_downsample=True
                im_tokens = im_downsample(im_tokens)
                level+=1
                num_im = im_tokens.size(1)
                im_padding_mask = im_padding_mask[:,:num_im]
                im_pos_mask = im_pos_mask[:,:num_im]

            q_tokens = qa_tokens[:,:num_q]
            a_tokens = qa_tokens[:,num_q:]
            #num_ocr_old=num_ocr
            if ocr_downsample is not None:
                did_downsample=True
                ocr_tokens,ocr_pos,ocr_padding_mask = ocr_downsample(ocr_tokens,ocr_pos,ocr_padding_mask)
                num_ocr = ocr_tokens.size(1)
                ocr_pos_mask = (~ocr_padding_mask[:,:,None]).float()
            #num_q_old=num_q
            if q_downsample is not None:
                did_downsample=True
                q_tokens,q_padding_mask = q_downsample(q_tokens,q_padding_mask)
                num_q = q_tokens.size(1)
                q_pos = q_pos[:,:num_q]
                q_pos_mask = q_pos_mask[:,:num_q]#torch.FloatTensor(new_batch_size,num_q,1).fill_(0).to(device)
                qa_tokens = torch.cat( (q_tokens,a_tokens),dim=1)
                qa_pos = torch.cat( (q_pos,a_pos), dim=1)
                qa_pos_mask = torch.cat( (q_pos_mask,a_pos_mask), dim=1)

            if did_downsample:
                num_all = num_im+num_ocr+num_q+num_a
                all_att_mask = all_att_mask[:,-(num_im+num_ocr+num_q+num_a):,-(num_im+num_ocr+num_q+num_a):] #this is uniform except at the end (a), so we can just take the bottom slice of it
                all_pos_mask = torch.cat((im_pos_mask,ocr_pos_mask,q_pos_mask,a_pos_mask),dim=1)
                all_padding_mask = torch.cat( (im_padding_mask,ocr_padding_mask,q_padding_mask,a_padding_mask), dim=1)
            if ocr_downsample is not None or q_downsample is not None:
                ocrq_padding_mask = torch.cat( (ocr_padding_mask,q_padding_mask), dim=1)
                ocrq_padding_mask_inf = torch.FloatTensor(*ocrq_padding_mask.size()).fill_(0).to(device)
                ocrq_padding_mask_inf[ocrq_padding_mask] = float('-inf')


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
                    #num_ocr = saved_ocr_tokens[li].size(1)
                    num_q = saved_q_tokens[li].size(1)

                    proj_im_tokens = saved_proj_im_tokens[li]
                    im_padding_mask = zero.expand(new_batch_size,num_im)#holder_im_padding_mask[:,:num_im]
                    im_pos_mask = one.expand(new_batch_size,num_im,1)#holder_im_pos[:,:num_im]

                    ocr_tokens = saved_ocr_tokens[li]
                    ocr_pos = saved_ocr_pos[li]
                    ocr_padding_mask = saved_ocr_padding_mask[li]
                    ocr_pos_mask = (~ocr_padding_mask[:,:,None]).float()

                    q_tokens = saved_q_tokens[li]
                    q_pos = saved_q_pos[li]
                    q_padding_mask = saved_q_padding_mask[li]
                    q_pos_mask = zero.expand(new_batch_size,num_q)[:,:,None] #holder_q_pos_mask[:,:num_q]

                    a_tokens = saved_a_tokens[li] = torch.cat((saved_a_tokens[li],ans),dim=1)
                    a_pos = zero.expand(new_batch_size,num_a,2).float()#holder_a_pos[:,:num_a]
                    a_padding_mask = zero.expand(new_batch_size,num_a)#holder_a_padding_mask[:,:num_a]
                    a_pos_mask = a_padding_mask[:,:,None]#holder_a_pos_mask[:,:num_a]

                    all_pos_mask = torch.cat((im_pos_mask,ocr_pos_mask,q_pos_mask,a_pos_mask),dim=1)
                    all_padding_mask = torch.cat( (im_padding_mask,ocr_padding_mask,q_padding_mask,a_padding_mask), dim=1)
                    all_att_mask = one.expand(new_batch_size,1,num_im+num_ocr+num_q+num_a)


                    ans = layout_layer(
                            a_tokens[:,-1:], #only last token
                            a_pos[:,-1:,0], #x
                            a_pos[:,-1:,1], #y
                            a_pos_mask[:,-1:],
                            #all_tokens,
                            torch.cat((proj_im_tokens,ocr_tokens,q_tokens,a_tokens),dim=1),
                            torch.cat((self.im_xs[level].expand(new_batch_size,-1),ocr_pos[:,:,0],q_pos[:,:,0],a_pos[:,:,0]),dim=1),
                            torch.cat((self.im_xs[level].expand(new_batch_size,-1),ocr_pos[:,:,1],q_pos[:,:,1],a_pos[:,:,1]),dim=1),
                            all_pos_mask,
                            all_att_mask,#all_att_mask[:,-(num_q+num_a):,:],
                            all_padding_mask)
                    if im_downsample is not None:
                        level+=1
                #Done Swin (RUN)

                response_decoded = self.answer_decode(ans)
                response_greedy_token = response_decoded.argmax(dim=2)
                assert response_greedy_token.size(1)==1
                

                output_tokens.append(response_greedy_token[0,0].item())
                #print('next token: {}'.format(output_tokens[-1]))
                offset += 1

            
            final_str = self.decode_tokenizer.convert_tokens_to_string(self.decode_tokenizer.convert_ids_to_tokens(output_tokens,skip_special_tokens=True))
            
            if PRINT_ATT:
                attDisplay(image[0],full_ocr_string,'|'+questions[0],'|'+final_str[0]+'^',final_str)
            return final_str, out_mask #torch.sigmoid(out_mask)
            ############





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
        if PRINT_ATT:
            attDisplay(image[0],full_ocr_string,'|'+questions[0],'|'+answers[0]+'^',batch_string_response[0])




        return response_decoded, target_decoded.to(device), batch_string_response, out_mask

    #t#def print_opt_times(self):#t#
        #t#for name,times in self.opt_history.items():#t#
            #t#print('time {}({}): {}'.format(name,len(times),np.mean(times)))#t#
            #t#if len(times)>300: #t#
                #t#times.pop(0)   #t#
                #t#if len(times)>600:#t#
                    #t#times.pop(0)#t#
