from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pairing_g_graph_layoutlm import  runLayoutLM
from model.pos_encode import PositionalEncoding, UniformRealEmbedding,PositiveRealEmbedding, ReturnPositionalEncoding
from model.swin_transformer import ConvPatchEmbed, SwinTransformerBlock, PatchMerging
from model.perceiver_io import BARTAttentionLayer
#try:
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from transformers import BartTokenizer, BartModel
from transformers import LayoutLMTokenizer, LayoutLMModel
#except:
#    pass
from model.special_token_embedder import SpecialTokenEmbedder
from model.cnn_hwr import ResConvPatchEmbed
from model.part_frozen_embedding import PartFrozenEmbedding
from utils.character_tokenizer import CharacterTokenizer
from utils.bytepair_tokenizer import BytePairTokenizer
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

class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)

class Donut(BaseModel):
    def __init__(self,config):
        super(Donut, self).__init__(config)
        self.image_size = config['image_size'] #start at 512?
        self.max_q_tokens = config.get('max_q_tokens',32)
        self.max_a_tokens = config.get('max_a_tokens',512)
        dropout = 0 if 'no_dropout' in config and  config['no_dropout'] else 0.0625
        lighter_conv_patch_emb = config['lighter_conv_patch_emb'] if 'lighter_conv_patch_emb' in config else False

        blocks_per_level = config['swin_blocks_per_level'] #[2,2,6,2] -> in:512,emb:64 then 64,32,16,8
        bart_layers = config['bart_layers']
        d_model = config['decoder_dim']

        init_from_pretrained = config.get('init_from_pretrained')
        learned_pos_emb = config.get('learned_pos_emb',False)

        window_size = config['window_size'] #7
        if type(window_size) is int:
            window_size = [window_size]*len(blocks_per_level)
        swin_nhead = config['swin_nheads'] #[3,6,12,24] | [2,6,12,12] probably don't need as much later
        im_embed_dim = config['im_embed_dim'] #96 -> 96,192,384,768 | 64->64,128,256,512

        if isinstance(self.image_size,int):
            self.image_size = (self.image_size,self.image_size)
        max_dist = math.sqrt(self.image_size[0]**2 + self.image_size[1]**2)
        self.max_pred_len = 500


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
            if init_from_pretrained=='bart':
                self.tokenizer = BartTokenizer.from_pretrained('./cache_huggingface/BART')
            else:
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.SEP_TOKEN= 102
            self.CLS_TOKEN= 101
        elif in_token_type == 'bp':
            self.tokenizer = BytePairTokenizer()
            self.SEP_TOKEN=self.tokenizer.SEP_index
            self.CLS_TOKEN=self.tokenizer.CLS_index

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
        elif init_from_pretrained=='bart':
            init_model = BartModel.from_pretrained('./cache_huggingface/BART')
            init_emb = init_model.shared
        else:
            init_model = None
            init_emb = None

        if in_token_type == 'bp': #we'll use the pre-trained embedding
            self.text_embedding = PartFrozenEmbedding(self.tokenizer.vocab_size,self.tokenizer.pretrained_dim(),d_model-self.tokenizer.pretrained_dim(),torch.FloatTensor(self.tokenizer.get_pretrained()))
        else:
            self.text_embedding = nn.Embedding(self.tokenizer.vocab_size, d_model)
            if in_token_type == 'word': #we'll use the pre-trained embedding to initialize
                
                self.text_embedding.weight.data[:,:d_model] = init_emb.weight.data[:,:d_model]

        if learned_pos_emb:
            self.pos_1d_enc = BartLearnedPositionalEmbedding(1026,d_model)
            self.pos_1d_enc.weight.data = init_model.decoder.embed_positions.weight.data
            self.q_pos_1d_enc = self.a_pos_1d_enc = None
        else:
            self.q_pos_1d_enc = PositionalEncoding(d_model,dropout=dropout,max_len=1000)
            self.a_pos_1d_enc = PositionalEncoding(d_model,dropout=dropout,max_len=1000,offset_start=1000)

        self.query_special_token_embedder = SpecialTokenEmbedder(d_model)


        if config.get('use_res_cnn_embed'):
            self.patch_embed = ResConvPatchEmbed(
                    img_size=self.image_size,
                    embed_dim=im_embed_dim,
                    in_chans=2)
        else:
            self.patch_embed =  ConvPatchEmbed(
                    img_size=self.image_size, 
                    embed_dim=im_embed_dim,
                    norm_layer=nn.LayerNorm,
                    lighter=lighter_conv_patch_emb,
                    in_chans=2) #now includes the mask channel
        if pre_trained_patch_emb is not None:
            checkpoint = torch.load(pre_trained_patch_emb, map_location=lambda storage, loc: storage)
            pe_state_dict=self.patch_embed.state_dict()
            for name,load_value in checkpoint['state_dict'].items():
                if name.startswith('cnn.'):
                    init_value = pe_state_dict[name]
                    init_size = init_value.size()
                    load_size = load_value.size()
					
                    dims=-1
                    for dim in range(len(load_size)):
                        if init_size[dim]!=load_size[dim]:
                            dims=dim
                    if dims>-1:
                        #brain surgery
                        if dims==0:
                            init_value[:load_size[0]] = load_value[:init_size[0]]
                        elif dims==1:
                            init_value[:load_size[0],:load_size[1]] = load_value[:init_size[0],:init_size[1]]
                        elif dims==2:
                            init_value[:load_size[0],:load_size[1],:load_size[2]] = load_value[:init_size[0],:init_size[1],:init_size[2]]
                        elif dims==3:
                            init_value[:load_size[0],:load_size[1],:load_size[2],:load_size[3]] = load_value[:init_size[0],:init_size[1],:init_size[2],:init_size[3]]
                        else:
                            raise NotImplementedError('no Brain Surgery above 4 dims')
                        pe_state_dict[name]=init_value
                    else:
                        pe_state_dict[name]=load_value

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
                                 sees_docq=False),
                    PatchMerging(cur_resolution, dim=d_im, norm_layer=nn.LayerNorm) if last else None
                    ] ) )


        self.decoder_transformer = nn.ModuleList()
        for x in range(bart_layers):
            self.decoder_transformer.append( BARTAttentionLayer(
                main_dim = d_model,
                encoder_dim = d_im,
                main_heads = config.get('self_nheads',8),
                main_dim_head = config.get('self_head_dim',32)) )

        if init_from_pretrained=='bart':
            #copy weights from pretrained model (huggingface)
            #We DON'T copy the cross attention (encoder attn) weights as I'm assuming the modality shift means they wouldn't be helpful
            assert len(self.decoder_transformer) == len(init_model.decoder.layers)
            for i,layer in enumerate(self.decoder_transformer):
                init_layer = init_model.decoder.layers[i]
                layer.self_att.fn.to_q.weight.data = init_layer.self_attn.q_proj.weight.data
                layer.self_att.fn.to_q.bias.data = init_layer.self_attn.q_proj.bias.data
                layer.self_att.fn.to_k.weight.data = init_layer.self_attn.k_proj.weight.data
                layer.self_att.fn.to_k.bias.data = init_layer.self_attn.k_proj.bias.data
                layer.self_att.fn.to_v.weight.data = init_layer.self_attn.v_proj.weight.data
                layer.self_att.fn.to_v.bias.data = init_layer.self_attn.v_proj.bias.data
                layer.self_att.fn.to_out.weight.data = init_layer.self_attn.out_proj.weight.data
                layer.self_att.fn.to_out.bias.data = init_layer.self_attn.out_proj.bias.data

                layer.self_att.norm.weight.data = init_layer.self_attn_layer_norm.weight.data
                layer.self_att.norm.bias.data = init_layer.self_attn_layer_norm.bias.data

                layer.ff.fn.net[0].weight.data = init_layer.fc1.weight.data
                layer.ff.fn.net[0].bias.data = init_layer.fc1.bias.data
                layer.ff.fn.net[2].weight.data = init_layer.fc2.weight.data
                layer.ff.fn.net[2].bias.data = init_layer.fc2.bias.data

                layer.ff.norm.weight.data = init_layer.final_layer_norm.weight.data
                layer.ff.norm.bias.data = init_layer.final_layer_norm.bias.data


        
        

        self.answer_decode = nn.Sequential(
                nn.Linear(d_model,self.decode_tokenizer.vocab_size,bias=False),
                nn.LogSoftmax(dim=-1) #except
                )

        self.answer_decode[0].weight = self.text_embedding.weight #Tie weights


        #t#self.opt_history=defaultdict(list)#t#




    #we're building this for fixed images size
    def forward(self,image,ocrRes,questions,answers=None,RUN=False,get_tokens=False):
        device = image.device

        im_tokens = self.patch_embed(image)
        im_tokens += self.absolute_2dpos_embed #Swin doesn't use this as it can rely on the biased attention. We need the image tokens to know where they are so they can interact with the document and question tokens
        num_im = im_tokens.size(1)

        if get_tokens:
            init_im_tokens = im_tokens
            init_ocr_tokens = None
        #Dropout?


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

        questions, query_special_tokens = self.query_special_token_embedder(questions)

        #run question+answer through decoder
        q_t = self.tokenizer(questions,return_tensors="pt",padding='max_length',max_length=self.max_q_tokens,truncation=True)
        q_tokens = q_t['input_ids'][:,:self.max_q_tokens-1]
        num_q = q_tokens.size(1)
        a_t = self.tokenizer(answers,return_tensors="pt",padding='max_length',max_length=self.max_a_tokens,truncation=True)
        a_tokens = a_t['input_ids'][:,:self.max_a_tokens]
        num_a = a_tokens.size(1)-1 #remove last SEP token
        qa_tokens = self.text_embedding(torch.cat((q_tokens,a_tokens[:,:-1]),dim=1).to(device))
        q_tokens = qa_tokens[:,:num_q] 
        a_tokens = qa_tokens[:,num_q:] 

        q_tokens = torch.cat((query_special_tokens[:,None,:],q_tokens),dim=1)
        num_q+=1

        if self.q_pos_1d_enc is not None:
            q_tokens = self.q_pos_1d_enc(q_tokens)
            #q_padding_mask = (1-q_t['attention_mask']).bool()#.to(device) 
            #q_padding_mask = q_padding_mask.to(device)


            a_tokens = self.a_pos_1d_enc(a_tokens)
            #a_padding_mask = (1-a_t['attention_mask'][:,1:]).bool().to(device) #remove last SEP

        qa_tokens = torch.cat((q_tokens,a_tokens),dim=1)

        if self.q_pos_1d_enc is None:
            qa_tokens += self.pos_1d_enc(qa_tokens.size())
        
        qa_att_mask = torch.BoolTensor(1,num_q+num_a,num_q+num_a).fill_(1) #1/0
        qa_att_mask[:,-num_a:,-num_a:] = torch.tril(qa_att_mask[:,-num_a:,-num_a:])
        qa_att_mask[:,:-num_a,-num_a:] = 0 #nothing else attends to a

        #compute padding for each question
        qa_att_mask = qa_att_mask.expand(new_batch_size,-1,-1).clone()
        q_attention = q_t['attention_mask'][:,:self.max_q_tokens]
        qa_att_mask[:,:,:num_q] *= q_attention[:,None,:]==1
        #we don't need padding for answer since it doesn't matter what happens with those later tokens

        qa_att_mask = qa_att_mask.to(device)


        #qa_tokens = torch.cat( (q_tokens,a_tokens),dim=1)
        #qa_padding_mask = torch.cat( (q_padding_mask,a_padding_mask), dim=1)




        for i,(swin_layer,im_downsample) in enumerate(self.swin_layers):


            im_tokens = swin_layer(im_tokens)
            if im_downsample is not None:
                im_tokens = im_downsample(im_tokens)

        for layer in self.decoder_transformer:
            qa_tokens = layer(qa_tokens,im_tokens,qa_att_mask)
        
        a_tokens = qa_tokens[:,-num_a:]

        response = a_tokens

        out_mask = None

                    

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
