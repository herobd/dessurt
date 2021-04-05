from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pairing_g_graph_layoutlm import PairingGGraphLayoutLM



class DecoderOnPairing(BaseModel):
    def __init__(self):
        self.pairing_model = ?



    def forward(image,gtBBs,gtTrans,question,answer=None):
        graph_feats = self.pairing_model(image,gtBBs=gtBBs,useGTBBs=True,gtTrans=gtTrans,return_feats=True)
        layoutlm_feats = runLayoutLM(image.size(),gtBBs,gtTrans,image.device,self.layoutlm_tokenizer,self.layoutlm)

        inputs = self.question_tokenizer(question, return_tensors="pt", padding=True)
        inputs = {k:i.to(device) for k,i in inputs.items()}
        question_feats = self.question_languagemodel(**inputs).last_hidden_state

        graph_feats = self.change_graph(graph_feats)
        layoutlm_feats =self.change_layoutlm


