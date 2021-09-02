DEBUG=0
import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, LayerNorm
from torch import Tensor
import torch.nn.functional as F
from .attention import PosBiasedMultiHeadedAttention
from torch.nn.modules.transformer import _get_activation_fn

class RelPosQTransformerLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, max_dist, dim_feedforward=2048, dropout=0.1, activation="relu",fixed=True):
        super(RelPosQTransformerLayer, self).__init__()
        self.self_attn = PosBiasedMultiHeadedAttention(nhead,d_model,max_dist,dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.fixed=fixed


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(RelativePosQTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, 
            query_tokens: Tensor, 
            query_x: Tensor, 
            query_y: Tensor, 
            query_pos_mask: Tensor,
            all_tokens: Tensor, 
            all_tokens_x: Tensor, 
            all_tokens_y: Tensor, 
            all_pos_mask: Tensor,
            full_mask = None, 
            all_padding_mask= None,
            ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            query_tokens: B,Q,D
            query_x,query_y: B,Q
            query_pos_mask: B,Q (bool)
            all_tokens: B,A,D
            all_tokens_x,all_tokens_y: B,A
            all_pos_mask: B,A (bool)
            full_mask: B,Q,A (?) This will be a subpart of all_mask[:,num_q+num_a,num_all] This is attention
            all_padding_mask: B,A

        """
        
        full_pos_mask = all_pos_mask[:,None,:].expand(-1,query_x.size(1),-1,1) * query_pos_mask[:,:,None].expand(-1,-1,all_x.size(1),1)
        docqa2 = self.self_attn(query_tokens, all_tokens, all_tokens, query_x,query_y,all_x,all_y, 
                mask=full_mask,
                key_padding_mask=all_padding_mask,
                pos_mask=full_pos_mask)

        docqa = docqa + self.dropout1(docqa2)
        docqa = self.norm1(docqa)
        docqa2 = self.linear2(self.dropout(self.activation(self.linear1(docqa))))
        docqa = docqa + self.dropout2(docqa2)
        docqa = self.norm2(docqa)
        return docqa


class RelPosImTransformerLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, max_dist, dim_feedforward=2048, dropout=0.1, activation="relu",fixed=True):
        super(RelPosImTransformerLayer, self).__init__()
        self.self_attn = PosBiasedMultiHeadedAttention(nhead,d_model,max_dist,dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.fixed=fixed


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(RelativePositionTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, 
            docqa: Tensor, 
            docqa_x: Tensor, 
            docqa_y: Tensor, 
            im_tokens: Tensor, 
            im_tokens_x: Tensor, 
            im_tokens_y: Tensor, 
            docqa_mask = None, 
            docqa_padding_mask= None,
            pos_mask=None,
            auto_regressive=None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            docqa: the set to the encoder layer (required). (batch,length,dim)
            docqa_x: the x position of each element of set (batch,length)
            docqa_y: the y position of each element of set (batch,length)
            docqa_mask: the mask for the docqa sequence (optional).
            docqa_key_padding_mask: the mask for the docqa keys per batch (optional). False/True (batch,length)
            pos_mask: mask for which places actually have a position 1/0 (batch,lengthDoc,lenfthDoc+Im,1)

        """
        assert (docqa.size(0)==docqa_x.size(0)) and (docqa.size(1)==docqa_x.size(1)) 
        assert docqa_x.size()==docqa_y.size()
        batch_size = docqa.size(0)
        if pos_mask is not None:
            l = pos_mask.size(1)
            assert l == docqa.size(1)
            new_pos_mask = pos_mask[:,None,:].expand(-1,l,-1,1) * pos_mask[:,:,None].expand(-1,-1,l,1)
        else:
            new_pos_mask = None
        full = torch.cat((im_tokens,docqa),dim=1)
        full_x = torch.cat((im_tokens_x,docqa_x),dim=1)
        full_y = torch.cat((im_tokens_y,docqa_y),dim=1)
        im_padding_mask = torch.BoolTensor(batch_size,im_tokens.size(1)).fill_(0).to(docqa_padding_mask.device)
        full_padding_mask = torch.cat((im_padding_mask,docqa_padding_mask),dim=1)
        if self.fixed:
            full_pos_mask= torch.cat(((~im_padding_mask[:,:,None]).float(),pos_mask),dim=1)
        else:
            full_pos_mask= torch.cat((im_padding_mask[:,:,None],pos_mask),dim=1)
        full_pos_mask= pos_mask[:,:,None].expand(-1,-1,full_pos_mask.size(1),1) * full_pos_mask[:,None,:].expand(-1,pos_mask.size(1),-1,1)
        imdoc_mask = torch.BoolTensor(batch_size,docqa.size(1),im_tokens.size(1)).fill_(1).to(docqa_mask.device)
        full_mask = torch.cat((imdoc_mask,docqa_mask),dim=-1)
        
        if auto_regressive is None:
            docqa2 = self.self_attn(docqa, full, full, docqa_x,docqa_y,full_x,full_y, mask=full_mask,
                                  key_padding_mask=full_padding_mask,pos_mask=full_pos_mask)
        else:
            docqa = auto_regressive
            docqa2 = self.self_attn(auto_regressive, full, full, docqa_x[:,-1:],docqa_y[:,-1:],full_x,full_y, 
                    mask=full_mask[:,-1:], key_padding_mask=full_padding_mask,pos_mask=full_pos_mask[:,-1:])

        docqa = docqa + self.dropout1(docqa2)
        docqa = self.norm1(docqa)
        docqa2 = self.linear2(self.dropout(self.activation(self.linear1(docqa))))
        docqa = docqa + self.dropout2(docqa2)
        docqa = self.norm2(docqa)
        return docqa




class RelPosTransformerLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, max_dist, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(RelPosTransformerLayer, self).__init__()
        self.self_attn = PosBiasedMultiHeadedAttention(nhead,d_model,max_dist,dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(RelativePositionTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, 
            tokens: Tensor, 
            pos: Tensor, 
            pos_mask: Tensor, # 1/0
            att_mask: Tensor, 
            padding_mask: Tensor, #False/True
            auto_regressive=None) -> Tensor:
        r"""Pass the input through the encoder layer.
        
        All masks are to be broadcast from the batch dim
        Args:
            tokens: the set to the encoder layer (required). (batch,length,dim)
            pos: the x,y position of each element of set (batch,length,2)
            pos_mask: mask for which places actually have a position 1/0 (batch,length,1)
            att_mask: the attention mask for sequence (optional). (batch,length,length)
            padding_mask: the mask for the keys per batch (optional, True if padded value). False/True(batch,length)

        """
        batch_size = tokens.size(0)
        l = pos_mask.size(1)
        assert l == tokens.size(1)
        new_pos_mask = pos_mask[:,None,:].expand(-1,l,-1,1) * pos_mask[:,:,None].expand(-1,-1,l,1)

        if auto_regressive is None:
            tokens2 = self.self_attn(tokens, tokens, tokens, 
                    pos[:,:,0],pos[:,:,1],
                    pos[:,:,0],pos[:,:,1], 
                    mask=att_mask,
                    key_padding_mask=padding_mask,
                    pos_mask=new_pos_mask)
        else:
            tokens2 = self.self_attn(auto_regressive, tokens, tokens, 
                    pos[:,-1:,0],pos[:,-1:,1],
                    pos[:,:,0],pos[:,:,1], 
                    mask=att_mask[:,-1:], 
                    key_padding_mask=padding_mask,
                    pos_mask=new_pos_mask[:,-1:,-1:])
            tokens = auto_regressive

        tokens = tokens + self.dropout1(tokens2)
        tokens = self.norm1(tokens)
        tokens2 = self.linear2(self.dropout(self.activation(self.linear1(tokens))))
        tokens = tokens + self.dropout2(tokens2)
        tokens = self.norm2(tokens)
        return tokens
