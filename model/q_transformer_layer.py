
import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, LayerNorm
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_activation_fn

class QTransformerLayer(nn.Module):
    r"""Attention for query tokens.
    This implemented as attention between two sets of tokens (the all_tokens includes the query tokens)

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

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",fixed=True):
        super(QTransformerLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(nhead,d_model,dropout=dropout)
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
        super(QTransformerLayer, self).__setstate__(state)

    def forward(self, 
            query_tokens: Tensor, 
            all_tokens: Tensor, 
            full_mask = None, 
            all_padding_mask= None,
            ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            query_tokens: B,Q,D
            all_tokens: B,A,D
            full_mask: B,Q,A (?) This will be a subpart of all_mask[:,num_q+num_a,num_all] This is attention
            all_padding_mask: B,A

        """
        
        response = self.self_attn(query_tokens, all_tokens, all_tokens,
                mask=full_mask,
                key_padding_mask=all_padding_mask,
                )

        query_tokens = query_tokens + self.dropout1(response)
        query_tokens = self.norm1(query_tokens)
        response = self.linear2(self.dropout(self.activation(self.linear1(query_tokens))))
        query_tokens = query_tokens + self.dropout2(response)
        query_tokens = self.norm2(query_tokens)
        return query_tokens
