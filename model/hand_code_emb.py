
import torch
import torch.nn as nn
import numpy as np
import re

###
#These features were crafted for the FUNSD (tobacco) dataset5
###

def read_list(filepaths):
    if type(filepaths) is str:
        filepaths=[filepaths]
    lines=[]
    for filepath is filepaths:
        with open(filepath) as f:
            lines += f.readlines()
    return set([l.lower().strip() for l in lines])

class HandCodeEmb(nn.Module):
    def __init__(self,out_size):
        super(HandCodeEmb, self).__init__()

        self.name_list = read_list('data/text_lists/names_only_f300.txt')
        self.place_list = read_list(['data/text_lists/states_and_cities.txt','data/text_lists/countries.txt'])
        self.color_list = read_list('data/text_lists/colors.txt')
        self.business_list = read_list('data/text_lists/businesss_terms.txt')
        self.advertising_list = read_list('data/text_lists/advertisings_terms.txt')

        self.feature_checks = [
            hasNumeral,
            hasMoney,
            hasDayNumber,
            hasYearNumber,
            hasMonth,
            hasDate,
            hasDateNoYear,
            hasTime,
            hasWrittenNumber,
            hasName,
            hasPlace,
            hasBusiness,
            hasAd,
            hasColor,
            countCaps,
            countNumerals,
            countName,
            isX,
            noLetters,
            questionName,
            questionDate,
            questionPlace,
            questionNumber,
            questionMoney,
            questionTime
            ]

        self.num_feats = len(self.feature_checks)

        self.emb_layers = nn.Sequential(
                nn.Linear(self.num_feats,out_size),
                nn.LeakyReLU(0.2,True),
                nn.Linear(out_size,out_size),
                nn.ReLU(True)
                )


    
    def forward(self,transcriptions):
        features = torch.FloatTensor(len(transcriptions),self.num_feats).zero_()
        for i, trans in enumerate(transcriptions):
            trans = trans.strip()
            for j, check in enumerate(self.feature_checks):
                features[i,j] = self.check(trans)

        features = features.to(the device)
        return self.emb_layers(features)




    def hasNumeral(self,s):
        return 1 if re.search(r'\d',s) is not None else 0
    def hasMoney(self,s):
        return 1 if re.search(r'\$\d+(.\d+)?',s) is not None else 0

    def hasDayNumber(self,s):
        return 1 if re.search(r'(^|[^\w])[012]?\d(([^\w]|$)',s) is not None else 0

    def hasYearNumber(self,s):
        return 1 if re.search(r'(^|[^\w])((1[789])|20)\d\d([^\w]|$)',s) is not None else 0

    def hasMonth(self,s):
        return 1 if re.search(r'(^|[^\w])(jan|january|feb|febuary|mar|march|apr|april|may|jun|june|aug|august|jul|july|sep|september|sept|oct|october|nov|november|dec|december)([^\w]|$)',s,flags=re.IGNORECASE)) is not None else 0

    def hasDate(self,s):
        return 1 if re.search(r'(^|[^\w])1?\d[\/-][0123]?\d[\/-]([12]\d)?\d\d([^\w]|$)',s) is not None else 0
    def hasDateNoYear(self,s):
        return 1 if re.search(r'(^|[^\w])[01]?\d[\/-][0123]?\d([^\w]|$)',s) is not None else 0
    def hasTime(self,s):
        return 1 if re.search(r'(^|[^\w])[012]?\d\s?:\s?[0-5]?\d([^\w]|a|p|$)',s) is not None else 0

    def hasWrittenNumber(self,s):
        return 1 if re.search(r'(^|[^\w])(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\w+teen|twenty-?\w*|thirty-?\w*|fourty-?\w*|fifty-?\w*|sixty-?\w*|seventy-?\w*|eighty-?\w*|ninety-?\w*|hundred|thousand|million)([^\w]|$)',s,flags=re.IGNORECASE)) is not None else 0

    def hasName(self,s):
        return 1 if s.lower() in self.names_list else 0
    def hasPlace(self,s):
        return 1 if s.lower() in self.place_list else 0
    def hasBusiness(self,s):
        return 1 if s.lower() in self.business_list else 0
    def hasAd(self,s):
        return 1 if s.lower() in self.ad_list else 0
    def hasColor(self,s):
        return 1 if s.lower() in self.color_list else 0
    def countCaps(self,s):
        caps = len([x for x in s if (x>='A' and x<='Z')])
        lower = len([x for x in s if (x>='a' and x<='z')])
        return caps/(lower+caps)
    def countNumerals(self,s):
        n =  len([x for x in s if (x>='0' and x<='9')])
        return n/len(s)
    def countName(self,s):
        words = s.split(' ')
        c = len([w for w in words if win self.names_list])
        return c/len(words)
    def isX(self,s):
        return s=='x' or s=='X'
    def noLetters(self,s):
        return 1 if re.search(r'[a-zA-Z]',s) is None else 0

    def questionName(self,s):
        return 1 if re.search(r'(^|[^\w])(requested by|requestors?|editors?|reporters?|reported by|received by|leader|supervisors?|managers?|initialed by|submitted by|signed|recipients?|prepared by|approved by|authors?|names?|signatures?|clients?|present|written by|investigators?)([^\w]|$)',s,flags=re.IGNORECASE) is not None or re.match(r'(to|from):?',s,flags=re.IGNORECASE) is not None else 0
    def questionDate(self,s):
        return 1 if re.search(r'(^|[^\w])(date|until|deadline|year|day|month)([^\w]|$)',s,flags=re.IGNORECASE) is not None else 0
    def questionPlace(self,s):
        return 1 if re.search(r'(^|[^\w])(city|place|country|address|location|markets?|zones?)([^\w]|$)',s,flags=re.IGNORECASE) is not None else 0
    def questionNumber(self,s):
        return 1 if re.search(r'score|number|length|weight|rate|width|density|volume|circumference|#|quantity|no\.|count|code|phone',s,flags=re.IGNORECASE) is not None else 0
    def questionMoney(self,s):
        return 1 if re.search(r'(^|[^\w])(costs?|paid|budgets?|balances?|\$|values?|prices?)([^\w]|$)',s,flags=re.IGNORECASE) is not None else 0
    def questionTime(self,s):
        return 1 if re.search(r'(^|[^\w])(time|hours?)([^\w]|$)',s,flags=re.IGNORECASE) is not None else 0






