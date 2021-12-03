#From Phil Wang's Perviever code (MIT license)
#https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_io.py
from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None, context_dim2 = None, context_dim3 = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None
        self.norm_context2 = nn.LayerNorm(context_dim2) if exists(context_dim2) else None
        self.norm_context3 = nn.LayerNorm(context_dim3) if exists(context_dim3) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)
        if exists(self.norm_context2):
            context2 = kwargs['context2']
            normed_context2 = self.norm_context2(context2)
            kwargs.update(context2 = normed_context2)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, v_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        if v_dim is None:
            v_dim = context_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = True) #Official implementation uses bias (defaults to True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias = True)
        self.to_v = nn.Linear(context_dim, v_dim, bias = True)
        self.to_out = nn.Linear(v_dim, v_dim, bias=True)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            if len(mask.size())==2:
                #mask = rearrange(mask, 'b ... -> b (...)')
                mask = repeat(mask, 'b j -> (b h) () j', h = h)
            else:
                mask = repeat(mask, 'b i j -> (b h) i j', h =h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class Attention2Context(nn.Module):
    def __init__(self, query_dim, context_dim1, context_dim2, heads = 8, dim_head = 64, v_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        #context_dim = default(context_dim, query_dim)
        if v_dim is None:
            v_dim = query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = True) #Official implementation uses bias (defaults to True)
        self.to_k1 = nn.Linear(context_dim1, inner_dim, bias = True)
        self.to_v1 = nn.Linear(context_dim1, v_dim, bias = True)
        self.to_k2 = nn.Linear(context_dim2, inner_dim, bias = True)
        self.to_v2 = nn.Linear(context_dim2, v_dim, bias = True)
        self.to_out = nn.Linear(v_dim, v_dim, bias=True)

    def forward(self, x, context, context2, mask1 = None, mask2 = None):
        h = self.heads

        q = self.to_q(x)
        #context = default(context, x)
        k1 = self.to_k1(context)
        v1 = self.to_v1(context)
        k2 = self.to_k2(context2)
        v2 = self.to_v2(context2)

        k = torch.cat((k1,k2),dim=1)
        v = torch.cat((v1,v2),dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask1 is not None or mask2 is not None:
            device = context.device
            if mask1 is None:
                batch_size = mask2.size(0)
                if len(mask2.size())==2:
                    mask1 = torch.BoolTensor(batch_size,context.size(1)).fill_(1).to(device)
                else:
                    mask1 = torch.BoolTensor(batch_size,x.size(1),context.size(1)).fill_(1).to(device)
            if mask2 is None:
                batch_size = mask1.size(0)
                if len(mask1.size())==2:
                    mask2 = torch.BoolTensor(batch_size,context2.size(1)).fill_(1).to(device)
                else:
                    mask2 = torch.BoolTensor(batch_size,x.size(1),context2.size(1)).fill_(1).to(device)
            mask = torch.cat((mask1,mask2),dim=2)
            #mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            if len(mask.size())==2:
                mask = repeat(mask, 'b j -> (b h) () j', h = h)
            else:
                mask = repeat(mask, 'b i j -> (b h) i j', h =h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class Attention3Context(nn.Module):
    def __init__(self, query_dim, context_dim1, context_dim2, context_dim3, heads = 8, dim_head = 64, v_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        #context_dim = default(context_dim, query_dim)
        if v_dim is None:
            v_dim = query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = True) #Official implementation uses bias (defaults to True)
        self.to_k1 = nn.Linear(context_dim1, inner_dim, bias = True)
        self.to_v1 = nn.Linear(context_dim1, v_dim, bias = True)
        self.to_k2 = nn.Linear(context_dim2, inner_dim, bias = True)
        self.to_v2 = nn.Linear(context_dim2, v_dim, bias = True)
        self.to_k3 = nn.Linear(context_dim3, inner_dim, bias = True)
        self.to_v3 = nn.Linear(context_dim3, v_dim, bias = True)
        self.to_out = nn.Linear(v_dim, v_dim, bias=True)

    def forward(self, x, context, context2, context3, mask1 = None, mask2 = None, mask3 = None):
        h = self.heads

        q = self.to_q(x)
        #context = default(context, x)
        k1 = self.to_k1(context)
        v1 = self.to_v1(context)
        k2 = self.to_k2(context2)
        v2 = self.to_v2(context2)
        k3 = self.to_k3(context3)
        v3 = self.to_v3(context3)

        k = torch.cat((k1,k2,k3),dim=1)
        v = torch.cat((v1,v2,v3),dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask1 is not None or mask2 is not None or mask3 is not None:
            batch_size = x.size(0)
            device = context.device
            if mask1 is None:
                if len(mask2.size())==2 if mask2 is not None else len(mask3.size())==2:
                    mask1 = torch.BoolTensor(batch_size,context.size(1)).fill_(1).to(device)
                else:
                    mask1 = torch.BoolTensor(batch_size,x.size(1),context.size(1)).fill_(1).to(device)
            if mask2 is None:
                if len(mask1.size())==2:
                    mask2 = torch.BoolTensor(batch_size,context2.size(1)).fill_(1).to(device)
                else:
                    mask2 = torch.BoolTensor(batch_size,x.size(1),context2.size(1)).fill_(1).to(device)
            if mask3 is None:
                if len(mask1.size())==2:
                    mask3 = torch.BoolTensor(batch_size,context3.size(1)).fill_(1).to(device)
                else:
                    mask3 = torch.BoolTensor(batch_size,x.size(1),context3.size(1)).fill_(1).to(device)
            elif len(mask3.size())==2 and len(mask2.size())==3:
                mask3 = mask3[:,None,:].expand(-1,x.size(1),-1)

            mask = torch.cat((mask1,mask2,mask3),dim=2)
            max_neg_value = -torch.finfo(sim.dtype).max
            if len(mask.size())==2:
                mask = repeat(mask, 'b j -> (b h) () j', h = h)
            else:
                mask = repeat(mask, 'b i j -> (b h) i j', h =h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class PerceiverI(nn.Module):
    def __init__(
        self,
        *,
        block_specification,
        dim,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False
    ):
        super().__init__()
        if num_latents>0:
            self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        #self.cross_attend_blocks = nn.ModuleList([
        #    PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head, v_dim=latent_dim), context_dim = dim),
        #    PreNorm(latent_dim, FeedForward(latent_dim))
        #])

        #get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        #get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        #get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        #self.layers = nn.ModuleList([])
        #cache_args = {'_cache': weight_tie_layers}

        #for i in range(depth):
        #    self.layers.append(nn.ModuleList([
        #        get_latent_attn(**cache_args),
        #        get_latent_ff(**cache_args)
        #    ]))

        self.cross_blocks = nn.ModuleList([])
        self.inner_block_count = []
        for blk_spec in block_specification:
            if len(blk_spec) == 2:
                num_self_att_per_block, num_blocks = blk_spec
                rev_cross=False
            else:
                num_self_att_per_block, num_blocks, rev_cross = blk_spec

            cross_att = PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head, v_dim=latent_dim), context_dim = dim)
            cross_ff = PreNorm(latent_dim, FeedForward(latent_dim))

            if rev_cross:
                rev_cross_att = PreNorm(dim, Attention(dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head, v_dim=dim), context_dim = latent_dim)
                rev_cross_ff = PreNorm(dim, FeedForward(dim))
            else:
                rev_cross_att = rev_cross_ff = None

            self_att = nn.ModuleList([])
            self_ff = nn.ModuleList([])
            for i in range(num_self_att_per_block):
                self_att.append(PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head)))
                self_ff.append(PreNorm(latent_dim, FeedForward(latent_dim)))
            self.cross_blocks.append(nn.ModuleList([
                cross_att,
                cross_ff,
                rev_cross_att,
                rev_cross_ff,
                self_att,
                self_ff]))
            self.inner_block_count.append(num_blocks)

            

    def forward(
        self,
        data,
        mask = None,
        latents = None
    ):
        b, *_, device = *data.shape, data.device
        if latents is None:
            x = repeat(self.latents, 'n d -> b n d', b = b)
        else:
            x = latents


        for (cross_att, cross_ff, rev_cross_att, rev_cross_ff, self_att, self_ff),num_blocks in zip(self.cross_blocks,self.inner_block_count):
            x = cross_att(x, context = data, mask = mask) + x
            x = cross_ff(x) + x


            for i in range(num_blocks):
                for att,ff in zip(self_att, self_ff):
                    x = att(x) + x
                    x = ff(x) + x

            if rev_cross_att is not None:
                data = rev_cross_att(data, context = x) + data #no masking, as it doesn't matter if we write to padded data locations
                data = rev_cross_ff(data) + data

        return x,data

class DecoderO(nn.Module):
    def __init__(
        self,
        *,
        queries_dim,
        logits_dim = None,
        latent_dim = 512,
        cross_heads = 1,
        cross_dim_head = 64,
        decoder_ff = True
    ):
        super().__init__()
        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head, v_dim = queries_dim), context_dim = latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(
        self,
        x,
        queries
    ):
        b = x.shape[0]

        # make sure queries contains batch dimension

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b = b)

        # cross attend from decoder queries to latents
        
        latents = self.decoder_cross_attn(queries, context = x)

        # optional decoder feedforward

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # final linear out

        return self.to_logits(latents)

class DoubleDecoderO(nn.Module):
    def __init__(
        self,
        *,
        queries_dim,
        logits_dim = None,
        latent_dim = 512,
        input_dim =256,
        cross_heads = 1,
        cross_dim_head = 64,
        decoder_ff = True
    ):
        super().__init__()
        self.decoder_cross_attn = PreNorm(queries_dim, Attention2Context(queries_dim, latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, v_dim = queries_dim), context_dim = latent_dim, context_dim2=input_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(
        self,
        latent_tokens,
        input_tokens,
        queries
    ):
        b = latent_tokens.shape[0]

        # make sure queries contains batch dimension

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b = b)

        # cross attend from decoder queries to latents
        
        output = self.decoder_cross_attn(queries, context = latent_tokens, context2 = input_tokens)

        # optional decoder feedforward

        if exists(self.decoder_ff):
            output = output + self.decoder_ff(output)

        # final linear out

        return self.to_logits(output)

# Perceiver LM example

class PerceiverLM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.perceiver_io = PerceiverIO(
            dim = dim,
            queries_dim = dim,
            logits_dim = num_tokens,
            **kwargs
        )

    def forward(
        self,
        x,
        mask = None
    ):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        logits = self.perceiver_io(x, mask = mask, queries = x)
        return logits


class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        main_dim,
        cross_dim,
        cross_heads = 1,
        main_dim_head = 64,
        cross_dim_head = 64,
    ):
        super().__init__()

        self.cross_att = PreNorm(main_dim, Attention(main_dim, cross_dim, heads = cross_heads, dim_head = cross_dim_head, v_dim=main_dim), context_dim = cross_dim)
        self.cross_ff = PreNorm(main_dim, FeedForward(main_dim))

            

    def forward(
        self,
        data,
        cross_data,
        mask = None
    ):
        b, *_, device = *data.shape, data.device
        x = data

        x = self.cross_att(x, context = cross_data, mask = mask) + x
        x = self.cross_ff(x) + x

        return x

class AutoRegressiveAttention(nn.Module):
    def __init__(
        self,
        main_dim,
        input_len,
        main_heads=1,
        main_dim_head = 64,
    ):
        super().__init__()

        self.main_att = PreNorm(main_dim, Attention(main_dim, main_dim, heads = main_heads, dim_head = main_dim_head, v_dim=main_dim))
        self.main_ff = PreNorm(main_dim, FeedForward(main_dim))

        self.register_buffer("mask", torch.tril(torch.BoolTensor(1,input_len,input_len).fill_(1)))
        

    def forward(
        self,
        data
    ):
        b, *_, device = *data.shape, data.device
        x = data

        x = self.main_att(x, mask = self.mask[:,:x.size(1),:x.size(1)].expand(x.size(0),-1,-1)) + x
        x = self.main_ff(x) + x

        return x

class LatentAutoRegressiveAttention(nn.Module):
    def __init__(
        self,
        main_dim,
        output_len,
        latent_dim,
        main_heads=1,
        main_dim_head = 64,
        cross_heads =1,
        cross_dim_head = 64
    ):
        super().__init__()

        self.main_att = PreNorm(main_dim, Attention2Context(main_dim, latent_dim, main_dim, heads = cross_heads, dim_head = cross_dim_head, v_dim = main_dim), context_dim = latent_dim, context_dim2=main_dim)
        self.main_ff = PreNorm(main_dim, FeedForward(main_dim))

        self.register_buffer("mask", torch.tril(torch.BoolTensor(1,output_len,output_len).fill_(1)))
        #self.register_buffer("latent_mask", torch.BoolTensor(1,latent_len,latent_len).fill_(1))
        

    def forward(
        self,
        data,
        latent,
        in_tokens=None,
        in_mask=None,
        last_token=None
    ):
        b, *_, device = *data.shape, data.device
        if last_token is None:
            x = data
        else:
            x = last_token #for runnning auto_regressive prediction
            #only predicts one next token

        x = self.main_att(x, context=latent, context2=data, mask2 = self.mask[:,:x.size(1),:data.size(1)].expand(x.size(0),-1,-1)) + x
        x = self.main_ff(x) + x

        return x

class AllAutoRegressiveAttention(nn.Module):
    def __init__(
        self,
        main_dim, 
        output_len,
        latent_dim,
        input_dim,
        main_heads=1,
        main_dim_head = 64,
        cross_heads =1,
        cross_dim_head = 64
    ):
        super().__init__()

        self.main_att = PreNorm(main_dim, Attention3Context(main_dim, latent_dim, main_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, v_dim = main_dim), context_dim = latent_dim, context_dim2=main_dim, context_dim3=input_dim)
        self.main_ff = PreNorm(main_dim, FeedForward(main_dim))

        self.register_buffer("autoR_mask", torch.tril(torch.BoolTensor(1,output_len,output_len).fill_(1)))
        #self.register_buffer("latent_mask", torch.BoolTensor(1,latent_len,latent_len).fill_(1))
        

    def forward(
        self,
        data,
        latent,
        in_tokens,
        in_mask,
        last_token=None
    ):
        b, *_, device = *data.shape, data.device
        if last_token is None:
            x = data
        else:
            x = last_token #for runnning auto_regressive prediction
            #only predicts one next token

        x = self.main_att(x, context=latent, context2=data, context3=in_tokens, mask2 = self.autoR_mask[:,:x.size(1),:data.size(1)].expand(x.size(0),-1,-1), mask3 = in_mask) + x
        x = self.main_ff(x) + x

        return x
