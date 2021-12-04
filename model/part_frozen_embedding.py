import torch
import torch.nn as nn


class PartFrozenEmbedding(nn.Module):
    "Split nn.Embedding where one half has frozen weights"
    def __init__(self,vocab_size,frozen_dim,learn_dim,frozen_pretrained):
        super(PartFrozenEmbedding, self).__init__()
        self.frozen = nn.Embedding(vocab_size,frozen_dim)
        self.frozen.requires_grad = False #freeze
        self.frozen.weight.data = frozen_pretrained

        self.learn = nn.Embedding(vocab_size,learn_dim)

    def forward(self,x):
        return torch.cat((self.frozen(x),self.learn(x)),dim=-1)

