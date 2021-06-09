import torch

class CharacterTokenizer():
    def __init__(self):
        self.char_to_idx = {"[NONE]":0, "\t": 1, " ": 2, "!": 3, "\"": 4, "#": 5, "$": 6, "%": 7, "&": 8, "'": 9, "(": 10, ")": 11, "*": 12, "+": 13, ",": 14, "-": 15, ".": 16, "/": 17,  "0": 18, "1": 19, "2": 20, "3": 21, "4": 22, "5": 23, "6": 24, "7": 25, "8": 26, "9": 27, ":": 28, ";": 29, "<": 30, "=": 31, ">": 32, "?": 33, "@": 34, "A": 35, "B": 36, "C": 37, "D": 38, "E": 39, "F": 40, "G": 41, "H": 42, "I": 43, "J": 44, "K": 45, "L": 46, "M": 47, "N": 48, "O": 49, "P": 50, "Q": 51, "R": 52, "S": 53, "T": 54, "U": 55, "V": 56, "W": 57, "X": 58, "Y": 59, "Z": 60, "[": 61, "\\": 62, "]": 63, "_": 64, "`": 65, "a": 66, "b": 67, "c": 68, "d": 69, "e": 70, "f": 71, "g": 72, "h": 73, "i": 74, "j": 75, "k": 76, "l": 77, "m": 78, "n": 79, "o": 80, "p": 81, "q": 82, "r": 83, "s": 84, "t": 85, "u": 86, "v": 87, "w": 88, "x": 89, "y": 90, "z": 91, "~": 92, "\u00a0": 93, "\u00a3": 94, "\u00a7": 95, "\u00ad": 96, "\u00b0": 97, "\u00b1": 98, "\u00bd": 99, "\u00c9": 100, "\u00d7": 101, "\u00e0": 102, "\u00e1": 103, "\u00e3": 104, "\u00e4": 105, "\u00e5": 106, "\u00e7": 107, "\u00e8": 108, "\u00e9": 109, "\u00ea": 110, "\u00eb": 111, "\u00ed": 112, "\u00f3": 113, "\u00fc": 114, "\u0394": 115, "\u03b1": 116, "\u03b2": 117, "\u03b3": 118, "\u03bc": 119, "\u2002": 120, "\u2013": 121, "\u2014": 122, "\u2018": 123, "\u2019": 124, "\u201c": 125, "\u201d": 126, "\u2022": 127, "\u2026": 128, "\u2033": 129, "\ufb01": 130, "\u0000":131, "\u0001": 132}#"[SEP]": 131, "[CLS]": 132}
        self.idx_to_char = ["[NONE]","\t", " ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "~", "\u00a0", "\u00a3", "\u00a7", "\u00ad", "\u00b0", "\u00b1", "\u00bd", "\u00c9", "\u00d7", "\u00e0", "\u00e1", "\u00e3", "\u00e4", "\u00e5", "\u00e7", "\u00e8", "\u00e9", "\u00ea", "\u00eb", "\u00ed", "\u00f3", "\u00fc", "\u0394", "\u03b1", "\u03b2", "\u03b3", "\u03bc", "\u2002", "\u2013", "\u2014", "\u2018", "\u2019", "\u201c", "\u201d", "\u2022", "\u2026", "\u2033", "\ufb01",'\u0000','\u0001']#"[SEP]","[CLS]"]
        self.SEP_index=131
        self.CLS_index=132
        self.vocab_size=133


    def convert_ids_to_tokens(self,id_tensor,skip_special_tokens):
        assert skip_special_tokens
        return [self.idx_to_char[idt.item()] for idt in id_tensor if idt<131]

    def convert_tokens_to_string(self,tokens):
        return ''.join(tokens)

    def tokenize(self,string):
        string = string.replace('[SEP]','\u0000').replace('[CLS]','\u0001')
        return [c for c in string if c in self.char_to_idx]

    def __call__(self,list_strings,return_tensors="pt",padding=True):
        assert return_tensors=="pt" and padding
        tokens = [ [self.char_to_idx[c] for c in string.replace('[SEP]','\u0000').replace('[CLS]','\u0001') if c in self.char_to_idx] for string in list_strings]
        length = max([len(b) for b in tokens])+2
        batch_size = len(tokens)
        input_ids = torch.LongTensor(batch_size,length).fill_(0)
        attention_mask = torch.LongTensor(batch_size,length).fill_(0)
        for b,toks in enumerate(tokens):
            input_ids[b,0]=self.CLS_index
            input_ids[b,1:len(toks)+1]=torch.LongTensor(toks)
            input_ids[b,len(toks)+1]=self.SEP_index
            attention_mask[b,:len(toks)+2]=1

        return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
                }


