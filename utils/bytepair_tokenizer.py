import torch
try:
    from bpemb import BPEmb
except:
    pass

class BytePairTokenizer():
    def __init__(self):
        self.emb = BPEmb(lang="en",vs=100000)
        self.SEP_index=2 #eos
        self.CLS_index=1 #bos
        self.vocab_size=100000

    
    #These two functions are in convert_ids_to_tokens, so convert_tokens_to_string doesn't do anything
    def convert_ids_to_tokens(self,id_tensor,skip_special_tokens):
        assert skip_special_tokens
        return self.emb.decode_ids(id_tensor.tolist())


    def convert_tokens_to_string(self,tokens):
        return tokens

    def tokenize(self,string):
        raise NotImplementedError('didnt think this was needed')

    def __call__(self,list_strings,return_tensors="pt",padding=True):
        assert return_tensors=="pt" and padding
        tokens = self.emb.encode_ids_with_bos_eos(list_strings)
        length = max([len(b) for b in tokens])
        batch_size = len(tokens)
        input_ids = torch.LongTensor(batch_size,length).fill_(0)
        attention_mask = torch.LongTensor(batch_size,length).fill_(0)
        for b,toks in enumerate(tokens):
            input_ids[b,0:len(toks)]=torch.LongTensor(toks)
            attention_mask[b,:len(toks)]=1

        return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
                }

    def get_pretrained(self):
        return self.emb.vectors
    def pretrained_dim(self):
        return self.emb.vectors.shape[1]

