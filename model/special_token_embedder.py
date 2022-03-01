import torch
from torch import nn

#This module finds the query token, removes from the string and gets the appropriate Embedding vector
#It returns the modified strings and the tensor of the embeddings
class SpecialTokenEmbedder(nn.Module):
    def __init__(self,emb_dim):
        super(SpecialTokenEmbedder, self).__init__()
    
        tokens = [
                #para qa
                'kb~','k0~','su~','s0~','up~','u0~','dn~','d0~','^^~','^0~','vv~','v0~','0;~','0w~','w0>',';0>','rm>','mk>','mm~','re~','r0~','b0~','bk~','00~','0p~',
                #form qa
                'al~','z0~','z0>','zs~','zs>','zm~','zm>',
                'g0~','g0>','gs~','gs>','gm~','gm>',
                'c$~','cs~',
                'l0~','l0>',
                'l~','l>',
                'v0~','v0>',
                'v~','v>',
                'h0~','h0>',
                'hd~','hd>',
                'u1~','u1>',
                'uh~','uh>',
                'q0~','q0>',
                'qu~','qu>',
                'fi~','t~','ri~','ci~','$r~','$c~','ar~','ac~','rh~','ch~','rh>','ch>','zs~','gs~',
                'f0~','pr~','p0~','f1~','p1~','t0~','r*~','c*~','#r~','#c~','%r~','%c~','%r>','%c>','ar>','ac>','r@~','c@~','r&~','c&~','r&>','c&>','0t~','t#>',
                #added (para)
                'infillread~','infillread0~','proper_su~','proper_s0~',
                #distillation (masked language model)
                'mlm>',
                #added (para, for IAM)
                'read_block>', 'read_block0>',
                #question answering
                'natural_q~',
                #NER
                'ne>','ne~','ner_line>','ner_text~','ner_full>',
                #Forms
                'json>','link-both~','link-box~','link-text~',
                'linkdown-both~','linkdown-box~','linkdown-text~',
                'linkup-both~','linkup-box~','linkup-text~',
                'json~',
                'list_row_headers~','list_column_headers~',
                'full_row~','full_col~',
                'full_row0~','full_col0~',
                #RVL CDIP
                'classify>',
                #census (parital)
                'all-name~','all-given name~','all-age~','record~',
                #new NER
                'ner_full_c1>','ner_line_c1>',
                #SROIE
                'sroie>',
                ]
        self.get_index = {s:i for i,s in enumerate(tokens)}
        self.emb = nn.Embedding(len(tokens),emb_dim)

    def forward(self,questions):
        device = self.emb.weight.device
        return_strings=[]
        gathered_indexes=[]
        for q in questions:
            index=None
            try:
                tilda = q.index('~')
            except ValueError:
                tilda = 999999999999
            try:
                bracket = q.index('>')
            except ValueError:
                bracket = 999999999999
            pos = min(tilda,bracket)
            query = q[:pos+1]

            gathered_indexes.append(self.get_index[query])

            return_strings.append(q[pos+1:])

        gathered_indexes = torch.LongTensor(gathered_indexes).to(device)
        return return_strings, self.emb(gathered_indexes)



