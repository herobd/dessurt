import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, LayerNorm
from torch import Tensor
import torch.nn.functional as F
from .attention import PosBiasedMultiHeadedAttention
from torch.nn.modules.transformer import _get_activation_fn, _get_clones

#def _get_activation_fn(activation):
#    if activation == "relu":
#        return F.relu
#    elif activation == "gelu":
#        return F.gelu
#
#    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
class RelPosImTransformerEncoderLayer(nn.Module):
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
        super(RelPosImTransformerEncoderLayer, self).__init__()
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
            docqa: Tensor, 
            docqa_x: Tensor, 
            docqa_y: Tensor, 
            im_tokens: Tensor, 
            im_tokens_x: Tensor, 
            im_tokens_y: Tensor, 
            docqa_mask = None, 
            docqa_key_padding_mask= None,
            pos_mask=None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            docqa: the set to the encoder layer (required). (batch,length,dim)
            docqa_x: the x position of each element of set (batch,length)
            docqa_y: the y position of each element of set (batch,length)
            docqa_mask: the mask for the docqa sequence (optional).
            docqa_key_padding_mask: the mask for the docqa keys per batch (optional). 1/0 (batch,length)
            pos_mask: mask for which places actually have a position 1/0 (batch,lengthDoc,lenfthDoc+Im,1)

        """
        assert (docqa.size(0)==docqa_x.size(0)) and (docqa.size(1)==docqa_x.size(1)) 
        assert docqa_x.size()==docqa_y.size()
        if pos_mask is not None:
            l = pos_mask.size(1)
            assert l == docqa.size(1)
            new_pos_mask = pos_mask[:,None,:].expand(-1,l,-1,1) * pos_mask[:,:,None].expand(-1,-1,l,1)
        else:
            new_pos_mask = None

        full = torch.cat((im_tokens,docqa),dim=1)
        full_x = torch.cat((im_tokens_x,docqa_x),dim=1)
        full_y = torch.cat((im_tokens_y,docqa_y),dim=1)
        im_padding_mask = torch.FloatTensor(batch_size,im_tokens.size(1)).fill_(1).to(im_padding_mask,docqa_padding_mask.device)
        full_padding_mask = torch.cat((im_padding_mask,docqa_padding_mask),dim=1)
        full_pos_mask= torch.cat((im_padding_mask,pos_mask),dim=1)
        imdoc_mask = torch.FloatTensor(batch_size,docqa.size(1),im_tokens.size(1)).fill_(1).to(mask.device)
        full_mask = torch.cat((imdoc_mask,mask),dim=-1)

        docqa2 = self.self_attn(docqa, full, full, docqa_x,docqa_y,full_x,full_y, mask=full_mask,
                              key_padding_mask=full_padding_mask,pos_mask=full_pos_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class RelativePositionTransformerEncoderLayer(nn.Module):
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
        super(RelativePositionTransformerEncoderLayer, self).__init__()
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

    def forward(self, src: Tensor, src_x: Tensor, src_y: Tensor, src_mask = None, src_key_padding_mask= None,pos_mask=None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the set to the encoder layer (required). (batch,length,dim)
            src_x: the x position of each element of set (batch,length)
            src_y: the y position of each element of set (batch,length)
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional). (batch,length)

        Shape:
            see the docs in Transformer class.
        """
        assert (src.size(0)==src_x.size(0)) and (src.size(1)==src_x.size(1)) 
        assert src_x.size()==src_y.size()
        if pos_mask is not None:
            l = pos_mask.size(1)
            assert l == src.size(1)
            new_pos_mask = pos_mask[:,None,:].expand(-1,l,-1,1) * pos_mask[:,:,None].expand(-1,-1,l,1)
        else:
            new_pos_mask = None

        src2 = self.self_attn(src, src, src, src_x,src_y,src_x,src_y, mask=src_mask,
                              key_padding_mask=src_key_padding_mask,pos_mask=new_pos_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class PositionBiasedTransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(PositionBiasedTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, src_x: Tensor, src_y: Tensor, mask = None, src_key_padding_mask = None, pos_mask=None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output,src_x,src_y, src_mask=mask, src_key_padding_mask=src_key_padding_mask,pos_mask=pos_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
