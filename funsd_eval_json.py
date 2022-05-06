import os
import json
import logging
import argparse
import torch
from model import *
from logger import Logger
from trainer import *
from data_loader import getDataLoader
import math
from collections import defaultdict
import pickle
#import requests
import warnings
from utils import img_f
import editdistance
import random
import re
from transformers import BartTokenizer
try:
    import easyocr
except:
    pass

end_token = '‡'
np_token = '№'
blank_token = 'ø'

CUT_BACK=False

def findNonEscaped(s,target):
    chopped=0
    while True:
        loc = s.find(target)
        if loc<=0 or s[loc-1]!='\\':
            return loc+chopped
        else:
            s=s[loc+1:]
            chopped+=loc+1
def rfindNonEscaped(s,target):
    while True:
        loc = s.rfind(target)
        if loc<=0 or s[loc-1]!='\\':
            return loc
        else:
            s=s[:loc]



def norm_ed(s1,s2):
    return editdistance.eval(s1.lower(),s2.lower())/max(len(s1),len(s2),1)

def derepeat(s):
    #hardcoded error the model makes a lot (for some reason)
    s = s.replace('{"*": "question"},','')
    s = s.replace(', {"*": "question"}','')
    #very rough
    while True:
        #m = re.search(r'(.......+)\1\1\1\1\1\1\1+',s) #8 chars, 7 repeat
        m = re.search(r'(.......+)\1\1\1\1\1+',s) #8 chars, 5 repeat
        if m is None:
            break

        start,end = m.span()
        #end-=len(m[1]) #keep one
        s = s[:start]+s[end:]
    return s
def findUnmatched(s):
    b_stack=[]
    c_stack=[]
    in_quote=False
    escaping=False
    for i,c in enumerate(s):
        if  c=='\\' and not escaping:
            escaping=True
        else:
            if not in_quote:
                if c=='[':
                    b_stack.append(i)
                elif c==']':
                    b_stack.pop()
                elif c=='{':
                    c_stack.append(i)
                elif c=='}':
                    c_stack.pop()
                elif c=='"' and not escaping:
                    in_quote=True
                    #print('start quote {}'.format(s[i-5:i+5]))
                    #print('                 ^')
            elif c=='"' and not escaping:
                in_quote=False
                #print('end quote {}'.format(s[i-5:i+5]))
                #print('               ^')
            if escaping:
                escaping=False

    return b_stack[-1] if len(b_stack) > 0 else -1, c_stack[-1] if len(c_stack) > 0 else -1

def getFormData(model,img,tokenizer,quiet=False,beam_search=False):
    question='json>'
    answer,out_mask = model(img,[[question]],RUN=True if not beam_search else f'beam{beam_search}')
    if not quiet:
        print('PRED:: '+answer)
    num_calls=1
    total_char_pred=len(answer)
    answer = derepeat(answer)
    total_answer = answer
    cut_tokens=[]
    for i in range(5): #shouldn't need to be more than 4 calls for test set, but often more are done to dig out of repeating ruts
        if end_token in total_answer:
            break
        num_calls+=1
        
        #how much of a lead? Need it to fit tokenwise in the 20 limit
        if total_answer.startswith('[question]ø'):
            total_answer='[{"'
            potentialoverlap=total_answer
            prompt=total_answer
            immune=True
        else:
            immune=False

            if CUT_BACK:
                tokens = tokenizer.encode(total_answer)
                if len(tokens)>600:
                    cut = tokens[-100:]
                    if cut not in cut_tokens:
                        cut_tokens.append(cut)
                        tokens = tokens[:-100]
                        total_answer = tokenizer.decode(tokens,skip_special_tokens=True)
            else:
                tokens = tokenizer.encode(answer)

            tokens_potentialoverlap = tokens[-5:]
            potentialoverlap = tokenizer.decode(tokens[-7:],skip_special_tokens=True)
            tokens = tokens[-25:-4] #allow for overlap
            prompt = tokenizer.decode(tokens,skip_special_tokens=True)


        question = 'json~'+prompt
        answer,out_mask = model(img,[[question]],RUN=True if not beam_search else f'beam{beam_search}')
        total_char_pred += len(answer)
        if not quiet:
            print('CONT:: '+answer)
        len_before = len(answer)
        answer = answer.replace('ø[question]','')
        answer = answer.replace('[question]ø','')
        answer = derepeat(answer)
        len_after = len(answer)

        if len_before>0 and len_after/len_before<0.25 and not immune:
            break #bad repeating going on
        
        #find overlapping region
        #import pdb;pdb.set_trace()
        OVERLAP_THRESH=0.3
        best_ed=OVERLAP_THRESH
        perfect_match=False
        for ci in range(len(potentialoverlap)):
            po_old = potentialoverlap[ci:]
            po_new = answer[:len(po_old)]
            if po_old==po_new:
                answer = answer[len(po_old):]
                perfect_match=True
                break
            else:
                ed = norm_ed(po_old,po_new)
                if ed<best_ed:
                    best_ed = ed
                    best_answer=answer[len(po_old):]
        if not perfect_match and best_ed<OVERLAP_THRESH:
            answer=best_answer
        total_answer+=answer
    
    final_char_pred = len(total_answer)
    pred_data = fixLoadJSON(total_answer)
    return pred_data,  final_char_pred/total_char_pred

def fixLoadJSON(pred):
    pred_data = None

    if pred.startswith('[question]ø'):
        return []
    #becuase I used backslash as newline, there are often mistakes predicting where it does't do the double backslash. Try and fix this:
    pred = re.sub('([^\\\\])\\\\([a-zA-Z 0-9])',r'\1\\\\\2',pred)

    #speed things up, fix no comma error
    pred = re.sub('}{|} {','}, {',pred)

    start_len = len(pred)
    end_token_loc = pred.find(end_token)
    if end_token_loc != -1:
        pred = pred[:end_token_loc]
    counter=50
    last_char=-1
    last_len=len(pred)

    pred_steps=[pred]
    pred_edits=['init']
    pred_chars=[-1]
    try: 
        while pred_data is None:
            if len(pred)>start_len+1020 or counter==0:
                #assert False
                #import pdb;pdb.set_trace()
                pred_edits.append('TRUNCATE')
                pred = pred[:char]
                
            pred = pred.replace(',,',',')
            pred = pred.replace('{{','{')
            try:
                pred_data = json.loads(pred)
            except json.decoder.JSONDecodeError as e:
                sections = '{}'.format(e)
                #print(sections)
                sections=sections.replace("':'","';'")
                sections = sections.split(':')
                #if len(sections)==3:
                #    err,typ,loc =sections
                #else:
                typ,loc = sections

                assert 'line 1' in loc
                loc_char = loc.find('char ')
                loc_char_end = loc.rfind(')')
                char = int(loc[loc_char+5:loc_char_end])
                
                if last_char>=char and len(pred)>=last_len:
                    counter -=1
                last_char = char
                last_len = len(pred)
                
                if "Expecting ',' delimiter" in typ:

                    if char==len(pred) or (char==len(pred)-1 and pred[char]==']'):
                        #closing ] or }?
                        #bracket = pred.rfind('[')
                        #curley = pred.rfind('{')
                        bracket,curley = findUnmatched(pred)
                        assert bracket!=-1 or curley!=-1
                        if bracket>curley:
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'add ]')
                            pred+=']'
                        else:
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'add }')
                            pred+='}'
                    elif counter<20:
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'blindly add ","')
                        pred = pred[:char]+','+pred[char:]
                    elif pred[char]==':':
                        #it didn't close a list
                        if pred[:char-1].rfind('[')>pred[:char-1].rfind('{'):
                            assert pred[char-1]=='"'
                            open_quote = pred[:char-1].rfind('"')
                            assert open_quote!=-1
                            comma = pred[:open_quote].rfind(',')
                            bracket = pred[:open_quote].rfind('[')
                            #assert comma != -1
                            if comma>bracket:
                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'insert list close and new object start')
                                pred = pred[:comma]+']},{'+pred[comma+1:]
                            else:
                                #unless this is a table, we want seperate objects
                                curley = pred[:bracket].rfind('{')
                                sub = pred[curley+1:bracket]
                                if 'headers' in sub or 'cells' in sub:
                                    #table
                                    pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'end list (no new objec)')
                                    pred = pred[:bracket+1]+'],'+pred[bracket+1:]
                                else:
                                    #check if this is fill-in-prose
                                    open_quote = findNonEscaped(pred[char:],'"')
                                    open_quote += char
                                    close_quote = findNonEscaped(pred[open_quote+1:],'"')
                                    close_quote += open_quote+1
                                    class_maybe = pred[open_quote+1:close_quote]
                                    if class_maybe=='answer':
                                        #is fill-in-prose
                                        close_quote_b = rfindNonEscaped(pred[:bracket],'"')
                                        open_quote_b = rfindNonEscaped(pred[:close_quote_b],'"')
                                        #assert pred[open_quote_b+1:close_quote_b]=='answers'
                                        close_bracket = pred[char:].find(']')
                                        close_curley = pred[char:].find('}')
                                        if close_bracket<close_curley:
                                            close_bracket += char
                                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char],pred[char+1:char+10])+' shortening to fill-in-prose, removed {} AND ]'.format(pred[open_quote_b:bracket+1]))
                                            pred = pred[:open_quote_b]+pred[bracket+1:close_bracket]+pred[close_bracket+1:]
                                        else:
                                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char],pred[char+1:char+10])+' shortening to fill-in-prose, removed {}'.format(pred[open_quote_b:bracket+1]))
                                            pred = pred[:open_quote_b]+pred[bracket+1:]
                                    else:
                                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'(2nd) insert list close and new object start')
                                        pred = pred[:bracket+1]+']},{'+pred[bracket+1:]
                        else:
                            #double colon/value
                            close_curly = pred[char:].find('{')
                            if close_curly != -1:
                                close_curly+=char
                            #remove it 
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'removing double colon/value')
                            pred = pred[:char-1]+pred[close_curly:]
                    elif pred[char]==']' and pred[char-1]=='"':
                        #assert pred[:char-1].rfind('[')<pred[:char-1].rfind('{')
                        if char==len(pred)-1 and pred[char]==']' and pred[char-1]=='"':
                            #missing }?
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'closing object (end of doc)')
                            pred = pred[:char]+'}'+pred[char:]
                        elif char<len(pred)-1 and pred[char+1]!='}':
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'closing object')
                            pred = pred[:char]+'}'+pred[char:]
                        else:
                            #this may be just a unopened list closing, in which case we'll remove it
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'remove list closing (not opened)')
                            pred = pred[:char]+pred[char+1:]
                    elif pred[char]==']' and pred[char-1]==']':
                        #double list closure
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'remove list closure (double)')
                        pred = pred[:char]+pred[char+1:]
                    elif pred[char-1]=='"':
                        #prev_quote = pred[:char-1].rfind('"')
                        prev_quote = rfindNonEscaped(pred[:char-1],'"')
                        prev_comma = pred[:char-1].rfind(',')
                        prev_colon = pred[:char-1].rfind(':')
                        if prev_quote>prev_comma and prev_quote>prev_colon and prev_colon>prev_comma:

                            #we have an unterminated list?
                            #next_quote = pred[char:].find('"')
                            next_quote = findNonEscaped(pred[char:],'"')
                            if next_quote!=-1:
                                next_quote += char
                            else:
                                next_quote = 999999999999999
                            next_curly = pred[char:].find('}')
                            if next_curly!=-1:
                                next_curly += char
                            else:
                                next_curly = 999999999999999
                            next_bracket = pred[char:].find(']')
                            if next_bracket!=-1:
                                next_bracket += char
                            else:
                                next_bracket = 999999999999999
                            next_end = min(next_curly,next_bracket)
                            if next_quote>char and next_quote<next_end:
                                #This is an incorrectly started value string
                                #we'll just remove it
                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'remove value string')
                                pred = pred[:char]+pred[next_end:]
                            elif pred[char]=='}':
                                    pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'close list (at end of object)')
                                    pred = pred[:char]+']'+pred[char:]
                            elif pred[char]=='"' and (pred[char+1]=='}' or pred[char+1]==']'):
                                #extra close quote, remove
                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'remove double quote')
                                pred = pred[:char]+pred[char+1:]
                            elif next_quote==999999999999999:
                                #just cut it off
                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred  [char:char+1],pred[char+1:char+10])+'remove ending characters')
                                pred = pred[:char]
                            else: 
                                assert False        
                        else:
                            prev_quote = rfindNonEscaped(pred[:char-1],'"')
                            prev_curley = pred[:prev_quote].rfind('{')
                            prev_bracket = pred[:prev_quote].rfind('[')
                            #next_quote = pred[char:].find('"')
                            next_quote = findNonEscaped(pred[char:],'"')
                            next_quote += char
                            #next_colon = pred[char:].find(':')
                            if pred[next_quote+1]==':' and prev_bracket>prev_curley:# and prev_colon<prev_quote:
                                #We're in a list, so close it
                                #and start the quote we sould be in
                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'end list, start object+quote')
                                pred = pred[:char]+']}, {"'+pred[char:]
                                #else
                                #maybe it shouldn't have closed
                                #import pdb;pdb.set_trace()
                                #pred = pred[:char-1]+pred[char:]
                            elif prev_bracket>prev_curley:
                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'added a comma yeo')
                                pred=pred[:char]+',"'+pred[char:]
                            else:
                                close_curly = pred[char:].find('}')
                                close_curly+=char
                                if close_curly>next_quote:
                                    pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'added a colon')
                                    pred=pred[:char]+':"'+pred[char:]
                                elif (pred[char-2]==',' or pred[char-3]==',') and (pred[char-3]=='"' or pred[char-4]=='"'):
                                    #missed closequote?
                                    comma = pred[:char].rfind(',')
                                    pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'added a quote, dfd')
                                    pred=pred[:comma]+'"'+pred[comma:]
                                elif pred[char]!='[' and pred[char]!=']' and pred[char]!='{' and pred[char]!='}' and pred[char]!=':' and pred[char]!=',':
                                    pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'remove the quote (merge) blindly')
                                    pred=pred[:char-1]+pred[char:]

                                else:
                                    pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'added a comma blindly (this could be bad)')
                                    pred=pred[:char]+', '+pred[char:]

                                    #assert False

                    elif pred[char]=='{' and (pred[char-1]=='}' or pred[char-2]=='}'):
                        #forgot a comma
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'add a comma between objs')
                        pred = pred[:char]+','+pred[char:]
                    elif pred[char]==',':
                        bracket_close = pred[char:].find(']')
                        if bracket_close!=-1:
                            bracket_close+=char
                        else:
                            bracket_close=9999999
                        curley_close = pred[char:].find('}')
                        if curley_close!=-1:
                            curley_close+=char
                        else:
                            curley_close=9999999
                        end = min(bracket_close,curley_close)
                        quote_locations=[]
                        in_quote=False
                        escaped=False
                        for c in range(char,end):
                            if not escaped and pred[c]=='"':
                                quote_locations.append(c)
                                in_quote = not in_quote
                            elif not in_quote and pred[c]==',':
                                break
                            elif not escaped and pred[c]=='\\':
                                escaped=True
                            else:
                                escaped=False
                        if len(quote_locations)==3:
                            #remove bad middle quote
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'remove extra comma')
                            pred=pred[:quote_locations[1]]+pred[quote_locations[1]+1:]
                        else:
                            assert False
                    #elif pred[char]!='[' and pred[char]!='{' and pred[char]!=']' and pred[char!='}' and pred[char]!='"':
                    #        next_quote = findNonEscaped(pred[char:],'"')
                    #        if next_quote!=-1:
                    #            next_quote += char
                    #        else:
                    #            next_quote=999999999
                    #        
                    #        next_close_bracket=pred[char:].find(']')
                    #        if next_close_bracket==-1:
                    #            next_close_bracket=999999999999
                    #        next_close_curley=pred[char:].find('}')
                    #        if next_close_curley==-1:
                    #            next_close_curley=999999999999
                    #        next_close = min(next_close_curley,next_close_bracket)
                    #        next_close+=char

                    #        if next_quote<next_close:

                    elif pred[char]!=':' and pred[char]!='"':
                        next_quote = findNonEscaped(pred[char:],'"')
                        if next_quote!=-1:
                            next_quote+=char
                        else:
                            next_quote=99999999
                        bracket_close = pred[char:].find(']')
                        if bracket_close!=-1:
                            bracket_close+=char
                        else:
                            bracket_close=9999999
                        curley_close = pred[char:].find('}')
                        if curley_close!=-1:
                            curley_close+=char
                        else:
                            curley_close=9999999
                        end = min(bracket_close,curley_close)
                        quote_locations=[]
                        in_quote=True
                        escaped=False
                        for c in range(char,end):
                            if not escaped and pred[c]=='"':
                                quote_locations.append(c)
                                in_quote = not in_quote
                            elif not in_quote and pred[c]==',':
                                break
                            elif not escaped and pred[c]=='\\':
                                escaped=True
                            else:
                                escaped=False
                        if len(quote_locations)%2==1:
                            #uneven quotes, add one
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'add a comma and quote')
                            pred = pred[:char]+',"'+pred[char:]
                        else:
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'well, it asked for a comma2')
                            pred = pred[:char]+','+pred[char:]
                            #assert False
                    else:
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'well, it asked for a comma')
                        pred = pred[:char]+','+pred[char:]
                        #assert False
                elif 'Unterminated string starting at' in typ:
                    pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'close quote')
                    pred+='"'

                elif 'Expecting value' in typ:
                    if char==len(pred) and pred[char-1]==':':
                        #We'll just remove this incomplete prediction
                        bracket = pred.rfind('{')
                        assert bracket > pred.rfind('}')
                        comma = pred[:bracket].rfind(',')
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'remove incomplete predition')
                        pred = pred[:comma]
                    elif char==len(pred) and pred[char-1]!='"':
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'blank string')
                        pred+='""'
                    elif char==len(pred)-1 and pred[char]!='"':
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'(2nd) blank string')
                        pred+='""'
                    elif counter<20:
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'blindly add " (value)')
                        pred=pred[:char]+'"'+pred[char:]
                    elif pred[char]=='}' and pred[:char].rfind('{')<=pred[:char].rfind('}'):
                        #random extra close curelybrace
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'remove extra close curley')
                        pred = pred[:char]+pred[char+1:]
                    elif pred[char-1]=='"' and pred[char:].find('"')+1==pred[char:].find(':'):
                        #forgot to seperate something
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'add comma + quote')
                        pred = pred[:char]+', "'+pred[char:]
                    elif pred[char:].startswith('answers') or pred[char:].startswith('"answers') or pred[char:].startswith(' "answers'):
                        #we need to add this to the previous entity
                        if pred[char:].startswith('answers'):
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'(2nd) add comma + quote')
                            prepend=', "'
                        elif pred[char:].startswith(' "answers'):
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'add comma')
                            prepend=','
                        else:
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'2nd add comma')
                            prepend=', '
                        #prev_quote = pred[:char].rfind('"')
                        prev_quote = rfindNonEscaped(pred[:char],'"')
                        prev_curly = pred[:char].rfind('}')
                        prev_comma = pred[:char].rfind(',')
                        if prev_curly > prev_quote and prev_curly+1==prev_comma:
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'cut something? {}'.format(pred[prev_curly:prev_comma+1]))
                            pred = pred[:prev_curly]+prepend+pred[prev_comma+1:]
                        else:
                            assert False
                    elif ',' == pred[char-1]:
                        next_quote = findNonEscaped(pred[char:],'"')
                        if next_quote!=-1:
                            bracket_close = pred[char:].find(']')
                            if bracket_close!=-1:
                                bracket_close+=char
                            else:
                                bracket_close=9999999
                            curley_close = pred[char:].find('}')
                            if curley_close!=-1:
                                curley_close+=char
                            else:
                                curley_close=9999999
                            end = min(bracket_close,curley_close,len(pred))
                            quote_locations=[]
                            in_quote=False
                            escaped=False
                            for c in range(char,end):
                                if not escaped and pred[c]=='"':
                                    quote_locations.append(c)
                                    in_quote = not in_quote
                                elif not in_quote and pred[c]==',':
                                    break
                                elif not escaped and pred[c]=='\\':
                                    escaped=True
                                else:
                                    escaped=False
                            next_quote+=char
                            if ','==pred[next_quote+1] or ']'==pred[next_quote+1] or '}'==pred[next_quote+1]:
                                #forgot open quote. Add it
                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'adding open quote')
                                pred=pred[:char]+'"'+pred[char:]
                            if char>len(pred)-6:
                                #just cut it
                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'removing bad end')
                                pred=pred[:char]
                            elif pred[char-2:char]=='},' or pred[char-3:char]=='}, ':
                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'adding open obj')
                                pred=pred[:char]+'{'+pred[char:]
                            elif pred[char-2:char]==': ' or pred[char-2:char]==', ' or pred[char-1]==':' or pred[char-1]==',':
                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'adding open string')
                                pred=pred[:char]+'"'+pred[char:]
                                
                            else:
                                assert False
                        else:
                            assert False
                    elif pred[char:].startswith('question"') or pred[char:].startswith('answer"') or pred[char:].startswith('other"') and (pred[char-1]==':' or pred[char-2]==':' ): 
                        #missed open quote
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'adding open quote2')
                        pred=pred[:char]+'"'+pred[char:] 

                    elif pred[char-1]==':' or (pred[char-1]==' ' and pred[char-2]==':'):
                        bracket_close = pred[char:].find(']')
                        if bracket_close!=-1:
                            bracket_close+=char
                        else:
                            bracket_close=9999999
                        curley_close = pred[char:].find('}')
                        if curley_close!=-1:
                            curley_close+=char
                        else:
                            curley_close=9999999
                        end = min(bracket_close,curley_close,len(pred))
                        quote_locations=[]
                        in_quote=True
                        escaped=False
                        signature=''
                        for c in range(char,end):
                            if not escaped and pred[c]=='"':
                                quote_locations.append(c)
                                in_quote = not in_quote
                                signature+='"'
                            elif not in_quote and pred[c]==',':
                                break
                            elif not in_quote and pred[c]==':':
                                signature+=':'
                            elif not escaped and pred[c]=='\\':
                                escaped=True
                            else:
                                escaped=False
                        if len(quote_locations)==1 and signature[0]=='"':
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'adding open quote (obj)')
                            pred=pred[:char]+'"'+pred[char:]
                        elif signature[:3]=='":"':
                            #merge this with prev string
                            close_quote = pred[:char].rfind('"')
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'merging/appending to strings')
                            pred=pred[:close_quote]+pred[char:]
                        else:
                            colon = pred[:char].rfind(':')
                            end_prev_quote = rfindNonEscaped(pred[:colon],'"')
                            open_prev_quote = rfindNonEscaped(pred[:end_prev_quote],'"')
                            prev_thing = pred[open_prev_quote+1:end_prev_quote]
                            if prev_thing in ['answers','column headers',' row headers','content']:
                                assert False
                            else:
                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'adding open quote (Should be a string)')
                                pred=pred[:char]+'"'+pred[char:]
                                
                                

                    elif pred[char]!=']' and pred[char]!='[' and pred[char]!='{' and pred[char]!='}' and pred[char]!=':' and pred[char]!=',':
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'adding open quote(just for anything really)')
                        pred=pred[:char]+'"'+pred[char:] 
                    
                    elif pred[char]==',':
                        #remove it
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'removing comma (wants value)')
                        pred=pred[:char]+pred[char+1:] 
                    else:
                        assert False

                elif "Expecting ';' delimiter" in typ:

                    if char==len(pred):
                        #what things have colon? class, answers, content
                        if pred.endswith('"content"') or pred.endswith('"answers"') or pred.endswith('"cells"') or pred.endswith('"row headers"') or pred.endswith('"column headers"'):
                            comma= pred.rfind(',')
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'close curly')
                            pred = pred[:comma]+'}'
                        else:
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'add class pred')
                            pred+=': "other"}'
                    elif counter<20:
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'add ":" blindly')
                        pred = pred[:char]+':'+pred[char:]
                    else:
                        fixed = False
                        #first check if this is a bad ", maybe unescaped
                        #import pdb;pdb.set_trace()

                        if pred[char]=='"' and (pred[char-1]=='"' or pred[char-2]=='"'):
                            #extra quotes in there
                            #find the ned quote and remove until then
                            next_quote = findNonEscaped(pred[char+1:],'"')
                            assert next_quote!=-1
                            next_quote += char+1

                            next_close_quote = findNonEscaped(pred[next_quote+1:],'"')
                            assert next_close_quote!=-1
                            next_close_quote+=next_quote+1
                            p = pred[next_quote+1:next_close_quote]
                            if p in ('answer','question','header','other'):

                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'replace extra quote with ":": {}'.format(pred[char:next_quote]))
                                pred = pred[:char]+':'+pred[next_quote:]
                            else:
                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'replace extra quote with ",": {}'.format(pred[char:next_quote]))
                                pred = pred[:char]+','+pred[next_quote:]

                            fixed=True

                            
                        elif pred[char-1]=='"':
                            #quote = pred[char:].find('"')
                            quote = findNonEscaped(pred[char:],'"')
                            colon = pred[char:].find(':')
                            quote += char
                            colon += char
                            if colon-1==quote and pred[colon+1]==' ':
                                #skip colon and space
                                if pred[colon+2:].startswith('"question"') or pred[colon+2:].startswith('"other"') or pred[colon+2:].startswith('"header"') or pred[colon+2:].startswith('"answer"'):
                                    #yes, escape the "
                                    pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'escape a quote')
                                    pred = pred[:char-1]+'\\'+pred[char-1:]
                                    fixed=True

                        
                        if not fixed:
                            if pred[char-1]=='"':
                                #find opening quote
                                open_quote = rfindNonEscaped(pred[:char-1],'"')
                                bracket = pred[:open_quote].rfind('{')
                                colon = open_quote
                            else:
                                bracket = pred[:char-1].rfind('{')
                                colon = char-1
                            
                            #There can be colons in strings, but it's unlikely a string starts with colon, so just be sure there's an (unescaped) quote before it
                            colons = [m.end()-1 for m in re.finditer(r'[^\\]":',pred[:colon])]
                            if len(colons)==0:
                                colon=-1
                            else:
                                colon = colons[-1]

                            if bracket>colon:
                                #this is missing the class prediction
                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'add just class')
                                pred = pred[:char]+': "other"'+pred[char:]
                            else:
                                #extra data?
                                #open_quote= pred[colon:].find('"')
                                open_quote = findNonEscaped(pred[colon:],'"')
                                assert open_quote!=-1
                                open_quote += colon
                                #close_quote= pred[open_quote+1:].find('"')
                                close_quote = findNonEscaped(pred[open_quote+1:],'"')

                                assert close_quote!=-1
                                close_quote += open_quote+1

                                assert close_quote<char

                                pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'remove thing: {}'.format(pred[close_quote+1:char]))
                                pred =pred[:close_quote+1]+pred[char:] #REMOVE
                            
                elif 'Expecting property name enclosed in double quotes' in typ:

                    if char==len(pred) or char==len(pred)-1:
                        if pred[-1]=='"':
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'peel back')
                            pred = pred[:-1]
                            bracket = pred.rfind('{')
                            if bracket>pred.rfind('"'):
                                pred = pred[:bracket]
                        else:
                            pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'append curly')
                            if pred[-1]==',':
                                pred=pred[:-1]
                            pred+='}'
                    elif counter<20:
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'blindly add "')
                        pred=pred[:char]+'"'+pred[char:]
                    elif pred[char]=='{':
                        #forgot to close object
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'close obj }')
                        pred=pred[:char]+'}'+pred[char:]
                    elif pred[char]=='}':
                        #extra comma
                        comma = pred[char:].rfind(',')
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'remove extra comma in object')
                        pred=pred[:comma]+pred[char:]
                    
                    else:
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'blindly add " (end)')
                        pred=pred[:char]+'"'+pred[char:]
                elif 'Expecting value' in typ:

                    if pred[-1]==',' and (char==len(pred) or char==len(pred)-1):
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'remove end comma')
                        pred=pred[:-1]
                    else:
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'blindly add " (value)')
                        pred=pred[:char]+'"'+pred[char:]
                elif 'Extra data' in typ :
                    if len(pred)==char:
                        assert pred[-1]==','
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'re,pve emd comma')
                        pred = pred[:-1]
                    elif pred[char-1]==']':
                        #closed bracket too early?
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'insert comma')
                        pred = pred[:char-1]+','+pred[char:]
                    elif (pred[char-1]==']' or pred[char-2]==']') and pred[char]=='{' and char<len(pred)-2:
                        close_bracket = pred[:char].rfind(']')
                        pred = pred[:close_bracket]+', '+pred[char:]
                    else:
                        assert False
                elif 'Invalid' in typ and 'escape' in typ:
                    if  pred[char-1:char+1] == '\\u':
                        #doesn't have number of char. Just remove
                        pred_edits.append('{}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10])+'fix escape')
                        pred = pred[:char-1]+pred[char+1:]
                    else:
                        assert False
                else:
                    assert False
                
                #print('corrected pred: '+pred)
                pred_steps.append(pred)
                pred_chars.append(char)
    except Exception as e:
        print('ERROR correcting JSON')
        for charx,p,did in zip(pred_chars,pred_steps,pred_edits):
            print('======== char {} =='.format(charx))
            print(did)
            print(p)
        print('currect context: {}<{}>{} '.format(pred[char-10:char],pred[char:char+1],pred[char+1:char+10]))
        raise e
    return pred_data

class Entity():
    def __init__(self,text,cls,identity):
        #print('Created entitiy: {}'.format(text))
        self.text=text
        self.text_lines = text.split('\\')
        self.cls=cls
        self.id=identity

    def __repr__(self):
        return '({} :: {})'.format(self.text,self.cls)
def parseDict(header,entities,links):
    if header=='':
        return []
    to_link=[]
    is_table=False
    row_headers = None
    col_headers = None
    cells = None
    my_text = None
    return_ids=[]
    double_entity=[]
    for text,value in header.items():
        if text=='content':
            if isinstance(value,list):
                for thing in reversed(value):
                    to_link+=parseDict(thing,entities,links)
            else:
                assert isinstance(value,dict)
                to_link+=parseDict(value,entities,links)
        elif text=='answers':
            if not isinstance(value,list):
                value=[value]
            for a in reversed(value):
                if isinstance(a,str):
                    a_id=len(entities)
                    entities.append(Entity(a,'answer',a_id))
                    to_link.append(a_id)
                else:
                    assert isinstance(a,dict)
                    to_link+=parseDict(a,entities,links)
        elif text=='row headers':
            assert isinstance(value,list)
            row_headers = value
            is_table = True
        elif text=='column headers':
            assert isinstance(value,list)
            col_headers = value
            is_table = True
        else:
            if isinstance(value,str):
                if my_text is not None:
                    #merged entity?
                    double_entity.append((my_text,my_class,to_link))
                    #my_id=len(entities)
                    #entities.append(Entity(my_text,my_class,my_id))
                    #for other_id in to_link:
                    #    links.append((my_id,other_id))
                    #return_ids.append(my_id)
                    to_link = []
                my_text = text
                my_class = value
            elif isinstance(value,list) and text=='cells':
                is_table=True
                cells = value
            elif isinstance(value,list) and my_text is None:
                #potentially bad qa?
                my_text = text
                my_class = 'question'
                for a in reversed(value):
                    assert isinstance(a,str)
                    a_id=len(entities)
                    entities.append(Entity(a,'answer',a_id))
                    to_link.append(a_id)
    if not is_table:
        if my_text is not None:
            my_id=len(entities)
            entities.append(Entity(my_text,my_class,my_id))
            for other_id in to_link:
                links.append((my_id,other_id))
            return_ids.append(my_id)
        else:
            return_ids+=to_link

    else:
        #a table
        if cells is not None:
            cell_ids = defaultdict(dict)
            for r,row in reversed(list(enumerate(cells))):
                for c,cell in reversed(list(enumerate(row))):
                    if cell is not None:
                        c_id = len(entities)
                        entities.append(Entity(cell,'answer',c_id))
                        cell_ids[r][c]=c_id
                        #if row_headers is not None and len(row_ids)>r:
                        #    links.append((row_ids[r],c_id))
                        #if col_headers is not None and len(col_ids)>c:
                        #    links.append((col_ids[c],c_id))
        if row_headers is not None:
            #row_ids = list(range(len(entities),len(entities)+len(row_headers)))
            row_ids=[]
            for rh in reversed(row_headers):
                if rh is not None:
                    if '<<' in rh and '>>' in rh:
                        #subheader
                        raise NotImplementedError("subheader is not implemented")
                    row_ids.append(len(entities))
                    entities.append(Entity(rh,'question',len(entities)))
        else:
            row_ids = []
        if col_headers is not None:
            #col_ids = list(range(len(entities),len(entities)+len(col_headers)))
            col_ids = []
            for ch in reversed(col_headers):
                if ch is not None:
                    if '<<' in ch and '>>' in ch:
                        #subheader
                        raise NotImplementedError("subheader is not implemented")
                    col_ids.append(len(entities))
                    entities.append(Entity(ch,'question',len(entities)))
        else:
            col_ids = []
        if cells is not None:
            for r,row in reversed(list(enumerate(cells))):
                for c,cell in reversed(list(enumerate(row))):
                    if cell is not None:
                        c_id = cell_ids[r][c]
                        if row_headers is not None and len(row_ids)>r:
                            links.append((row_ids[r],c_id))
                        if col_headers is not None and len(col_ids)>c:
                            links.append((col_ids[c],c_id))
    

        return_ids+=row_ids+col_ids

    for my_text,my_class,to_link in double_entity:
        my_id=len(entities)
        entities.append(Entity(my_text,my_class,my_id))
        for other_id in to_link:
            links.append((my_id,other_id))
        return_ids.append(my_id)

    return return_ids




def main(resume,config,addToConfig,gpu=False,do_pad=False,test=False,draw=False,max_qa_len=None,quiet=False,BROS=False,ENTITY_MATCH_THRESH=0.6,LINK_MATCH_THRESH=0.6,DEBUG=False,beam_search=False,write=False):
    TRUER=True #False makes this do pair-first alignment, which is kind of cheating
    np.random.seed(1234)
    torch.manual_seed(1234)
    if DEBUG:
        print("DEBUG")
        print("EBUG")
        print("EBUG")
    
    
    #too_long_gen_thresh=10

    if resume is not None:
        checkpoint = torch.load(resume, map_location=lambda storage, location: storage)
        print('loaded {} iteration {}'.format(checkpoint['config']['name'],checkpoint['iteration']))
        if config is None:
            config = checkpoint['config']
        else:
            config = json.load(open(config))
        for key in config.keys():
            if 'pretrained' in key:
                config[key]=None
    else:
        checkpoint = None
        config = json.load(open(config))
    config['optimizer_type']="none"
    config['trainer']['use_learning_schedule']=False
    config['trainer']['swa']=False
    if not gpu:
        config['cuda']=False
    else:
        config['cuda']=True
        config['gpu']=gpu


    do_ocr=config['trainer']['do_ocr'] if 'do_ocr' in config['trainer'] else False
    if do_ocr and do_ocr!='no':
        ocr_reader = easyocr.Reader(['en'],gpu=config['cuda'])
    if addToConfig is not None:
        for add in addToConfig:
            addTo=config
            printM='added config['
            for i in range(len(add)-2):
                try:
                    indName = int(add[i])
                except ValueError:
                    indName = add[i]
                addTo = addTo[indName]
                printM+=add[i]+']['
            value = add[-1]
            if value=="":
                value=None
            elif value[0]=='[' and value[-1]==']':
                value = value[1:-1].split('-')
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value == 'None':
                            value=None
            addTo[add[-2]] = value
            printM+=add[-2]+']={}'.format(value)
            print(printM)
            #if (add[-2]=='useDetections' or add[-2]=='useDetect') and 'gt' not in value:
            #    addDATASET=True
    if max_qa_len is None:
        #max_qa_len=config['data_loader']['max_qa_len'] if 'max_qa_len' in config['data_loader'] else config['data_loader']['max_qa_len_out']
        max_qa_len_in = config['data_loader'].get('max_qa_len_in',640)
        max_qa_len_out = config['data_loader'].get('max_qa_len_out',2560,)

        
    if checkpoint is not None:
        if 'swa_state_dict' in checkpoint and checkpoint['iteration']>config['trainer']['swa_start']:
            state_dict = checkpoint['swa_state_dict']
            #SWA  leaves the state dict with 'module' in front of each name and adds extra params
            new_state_dict = {key[7:]:value for key,value in state_dict.items() if key.startswith('module.')}
            print('Loading SWA model')
        else:
            state_dict = checkpoint['state_dict']
            #DataParaellel leaves the state dict with 'module' in front of each name
            new_state_dict = {
                    (key[7:] if key.startswith('module.') else key):value for key,value in state_dict.items()
                    }

        model = eval(config['arch'])(config['model'])
        if 'query_special_start_token_embedder.emb.weight' in new_state_dict:
            loading_special = new_state_dict['query_special_start_token_embedder.emb.weight']
            model_special = model.state_dict()['query_special_start_token_embedder.emb.weight']

            if loading_special.size(0) != model_special.size(0):
                model_special[:loading_special.size(0)] = loading_special[:model_special.size(0)]
                new_state_dict['query_special_start_token_embedder.emb.weight'] = model_special
        if 'query_special_token_embedder.emb.weight' in new_state_dict:
            loading_special = new_state_dict['query_special_token_embedder.emb.weight']
            model_special = model.state_dict()['query_special_token_embedder.emb.weight']

            if loading_special.size(0) != model_special.size(0):
                model_special[:loading_special.size(0)] = loading_special[:model_special.size(0)]
                new_state_dict['query_special_token_embedder.emb.weight'] = model_special
        model.load_state_dict(new_state_dict)
    else:
        model = eval(config['arch'])(config['model'])

    model.eval()
    if gpu:
        model = model.cuda()

    if do_pad is not None:
        do_pad = do_pad.split(',')
        if len(do_pad)==1:
            do_pad+=do_pad
        do_pad = [int(p) for p in do_pad]
    else:
        do_pad = config['model']['image_size']
        if type(do_pad) is int:
            do_pad = (do_pad,do_pad)



    ##DAT##
    config['data_loader']['shuffle']=False
    config['validation']['shuffle']=False
    config['data_loader']['eval']=True
    config['validation']['eval']=True

    # change to graph pair dataset
    config['data_loader']['data_set_name']='FUNSDGraphPair'
    config['data_loader']['data_dir']='../data/FUNSD'
    config['data_loader']['crop_params']=None
    config['data_loader']['batch_size']=1
    config['data_loader']['split_to_lines']=True
    config['data_loader']['color']=False
    config['data_loader']['rescale_range']=[1,1]
    if DEBUG:
        config['data_loader']['num_workers']=0

    config['validation']['data_set_name']='FUNSDGraphPair'
    config['validation']['data_dir']='../data/FUNSD'
    config['validation']['crop_params']=None
    config['validation']['batch_size']=1
    config['validation']['split_to_lines']=True
    config['validation']['color']=False
    config['validation']['rescale_range']=[1,1]

    if not test:
        data_loader, valid_data_loader = getDataLoader(config,'train')
    else:
        valid_data_loader, data_loader = getDataLoader(config,'test')
        data_loader = valid_data_loader
    valid_iter = iter(valid_data_loader)

    num_classes = len(valid_data_loader.dataset.classMap)

    total_entity_true_pos =0
    total_entity_pred =0
    total_entity_gt =0
    total_rel_true_pos =0
    total_rel_pred =0
    total_rel_gt =0
    total_entity_true_pos2 =0
    total_entity_pred2 =0
    total_entity_gt2 =0
    total_rel_true_pos2 =0
    total_rel_pred2 =0
    total_rel_gt2 =0

    tokenizer = BartTokenizer.from_pretrained('./cache_huggingface/BART')

    to_write = {}

    going_DEBUG=False
    with torch.no_grad():
        for instance in valid_iter:
            groups = instance['gt_groups']
            classes_lines = instance['bb_gt'][0,:,-num_classes:]
            loc_lines = instance['bb_gt'][0,:,0:2] #x,y
            bb_lines = instance['bb_gt'][0,:,[5,10,7,12]].long()
            pairs = instance['gt_groups_adj']
            transcription_lines = instance['transcription']
            #transcription_lines = [s if cased else s for s in transcription_lines]
            img = instance['img'][0]
            if not quiet:
                print()
                print(instance['imgName'])

            if DEBUG and (not going_DEBUG and instance['imgName']!='92327794'):
                continue
            going_DEBUG=True

            gt_line_to_group = instance['targetIndexToGroup']

            transcription_groups = []
            transcription_firstline = []
            pos_groups = []
            for group in groups:
                transcription_groups.append('\\'.join([transcription_lines[t] for t in group]))
                transcription_firstline.append(transcription_lines[group[0]])
                pos_groups.append(loc_lines[group[0]])


            classes = [classes_lines[group[0]].argmax() for group in groups]
            gt_classes = [data_loader.dataset.index_class_map[c] for c in classes]

            if draw:
                draw_img = (255*(1-img.permute(1,2,0).expand(-1,-1,3).numpy())).astype(np.uint8)
                #import pdb;pdb.set_trace()
            
            if do_pad and (img.shape[1]<do_pad[0] or img.shape[2]<do_pad[1]):
                diff_x = do_pad[1]-img.shape[2]
                diff_y = do_pad[0]-img.shape[1]
                p_img = torch.FloatTensor(img.size(0),do_pad[0],do_pad[1]).fill_(-1)#np.zeros(do_pad,dtype=img.dtype)
                pad_y = diff_y//2
                pad_x = diff_x//2
                if diff_x>=0 and diff_y>=0:
                    p_img[:,diff_y//2:do_pad[0]-(diff_y//2 + diff_y%2),diff_x//2:do_pad[1]-(diff_x//2 + diff_x%2)] = img
                elif diff_x<0 and diff_y>=0:
                    p_img[:,diff_y//2:do_pad[0]-(diff_y//2 + diff_y%2),:] = img[:,:,(-diff_x)//2:-((-diff_x)//2 + (-diff_x)%2)]
                elif diff_x>=0 and diff_y<0:
                    p_img[:,:,diff_x//2:do_pad[1]-(diff_x//2 + diff_x%2)] = img[:,(-diff_y)//2:-((-diff_y)//2 + (-diff_y)%2),:]
                else:
                    p_img = img[:,(-diff_y)//2:-((-diff_y)//2 + (-diff_y)%2),(-diff_x)//2:-((-diff_x)//2 + (-diff_x)%2)]
                img=p_img

                loc_lines[:,0]+=pad_x
                loc_lines[:,1]+=pad_y
                #bb_lines[:,0]+=pad_x
                #bb_lines[:,1]+=pad_y
                #bb_lines[:,2]+=pad_x
                #bb_lines[:,3]+=pad_y

            img = img[None,...] #re add batch 
            img = torch.cat((img,torch.zeros_like(img)),dim=1) #add blank mask channel

            if gpu:
                img = img.cuda()
            
            #First find tables, as those are done seperately (they should do multiline things alread)
            #TODO multi-step pred for long forms

            #print GT
            #print('==GT form==')
            #for ga,gb in pairs:
            #    print('{} [{}] <=> {} [{}]'.format(transcription_groups[ga],gt_classes[ga],transcription_groups[gb],gt_classes[gb]))
            #print()
            

            pred_data, good_char_pred_ratio = getFormData(model,img,tokenizer,quiet,beam_search)

            
            if not quiet:
                print('==Corrected==')
                print(json.dumps(pred_data,indent=2))

            if write:
                to_write[instance['imgName']]=pred_data
            
            #get entities and links
            pred_entities=[]
            pred_links=[]
            for thing in pred_data[::-1]: #build pred_entities in reverse
                if isinstance(thing,dict):
                    parseDict(thing,pred_entities,pred_links)
                #elif thing=='':
                #    pass
                #else:
                #    print('non-dict at document level: {}'.format(thing))
                #    assert False
                #    #import pdb;pdb.set_trace()


            #we're going to do a check for repeats of the last entity. This frequently happens
            if len(pred_entities)>0:
                last_entity = pred_entities[-1]
                remove = None
                entities_with_link = None
                for i in range(len(pred_entities)-2,0,-1):
                    if pred_entities[i].text==last_entity.text and pred_entities[i].cls==last_entity.cls:
                        if entities_with_link is None:
                            entities_with_link = set()
                            for a,b in pred_links:
                                entities_with_link.add(a)
                                entities_with_link.add(b)

                        if i not in entities_with_link:
                            remove=i+1
                        else:
                            break
                    else:
                        break
                if remove is not None:
                    if not quiet:
                        print('removing duplicate end entities: {}'.format(pred_entities[remove:]))
                        pred_entities = pred_entities[:remove]
                
            #align entities to GT ones
            #pred_to_gt={}
            #for g_i,gt in enumerate(transcription_groups):
            #    closest_dist=9999999
            #    closest_e_i=-1
            #    for e_i,entity in pred_entities:
            #        dist
            #should find pairs in GT with matching text and handle these seperately/after

            if TRUER:
                DIST_SCORE=600
                #order entities by y
                gt_entities = []
                for i,(text,text_firstline,(x,y),cls) in enumerate(zip(transcription_groups,transcription_firstline,pos_groups,gt_classes)):
                    gt_entities.append((i,x,y,text_firstline if BROS else text,cls))
                #gt_entities.sort(key=lambda a:a[2])

                pos_in_gt=0
                last_x=0
                last_y=0
                last_text=None
                gt_to_pred = defaultdict(list)
                all_scores = defaultdict(dict)
                for p_i,entity in reversed(list(enumerate(pred_entities))):
                    #if 'NAME' in entity.text:
                    #    import pdb; pdb.set_trace()
                    if BROS:
                        p_text = entity.text_lines[0]
                    else: 
                        p_text = entity.text
                    has_link=False
                    for a,b in pred_links:
                        if p_i==a or p_i==b:
                            has_link=True
                            break
                    #if 'cc' in p_text:
                    #    import pdb; pdb.set_trace()
                    if last_text == p_text:
                        #the model maybe double predicted? In anycase, there will be a 0 distance if we use the last_x/y, so we'll rewind one step back
                        last_x = last2_x
                        last_y = last2_y
                    best_score=9999999
                    for g_i,x,y,g_text,cls in gt_entities:

                        text_dist = norm_ed(p_text,g_text)
                        if text_dist<LINK_MATCH_THRESH:# and cls==entity.cls: cheating a little if we use class

                            #dist = abs(last_y-y) + 0.1*abs(last_x-x)#math.sqrt((last_y-y)**2)+((last_x-x)**2))
                            #asymetric, penalize up alignment more than down
                            dist = 0.15*abs(last_x-x)
                            if last_y<y:
                                dist+= y-last_y
                            else:
                                dist+= 1.2*(last_y-y)
                            score = text_dist + dist/DIST_SCORE

                            #if '532' in p_text:
                            #    import pdb;pdb.set_trace()

                            #adjust score if there are any links
                            #This obiviously doesn't effect these comprisons, but will help it ownership fights
                            if has_link:
                                score -= 0.03


                            all_scores[p_i][g_i]=score

                            if score<best_score:
                                align_g_i = g_i
                                align_x = x
                                align_y = y
                                best_score = score
                    if best_score<9999999:
                        gt_to_pred[align_g_i].append((p_i,best_score))
                        last2_x = last_x
                        last2_y = last_y
                        last_x=align_x
                        last_y=align_y
                        last_text = p_text

                #Now, we potentially aligned multiple pred entities to gt entities
                #We need to resolve these by finding the best alternate match for the worse match
                new_gt_to_pred={}#[None]*len(groups)
                pred_to_gt={}
                new_gt_to_pred_scores={}
                to_realign=[]
                for g_i,p_is in gt_to_pred.items():
                    if len(p_is)>1:
                        #import pdb;pdb.set_trace()
                        #we need to check if any links to the p_is have already been aligned.
                        #if so, we'll want to keep that consistant
                        new_pis=[]
                        for p_i,score in p_is:
                            aligned_link = False
                            for a,b in pred_links:
                                other_i = None
                                if a==p_i:
                                    other_i = b
                                elif b==p_i:
                                    other_i = a
                                if other_i in pred_to_gt:
                                    #The link has been aligned to a gt
                                    aligned_link=True
                                    break
                            if aligned_link:
                                new_pis.append((p_i,score-0.05))
                            else:
                                new_pis.append((p_i,score))
                        p_is = new_pis
                        p_is.sort(key=lambda a:a[1])

                    new_gt_to_pred[g_i]=p_is[0][0] #best score gets it
                    new_gt_to_pred_scores[g_i]=p_is[0][1]
                    pred_to_gt[p_is[0][0]]=g_i
                    for p_i,_ in p_is[1:]:
                        to_realign.append(p_i)
                debug_count=0
                while len(to_realign)>0:
                    if debug_count>50:
                        print('infinite loop')
                        assert False
                        #import pdb;pdb.set_trace()
                    debug_count+=1

                    doing = to_realign
                    to_realign = []
                    for p_i in doing:
                        scores = [(g_i,score) for g_i,score in all_scores[p_i].items()]
                        best_score=9999999
                        for g_i,score in all_scores[p_i].items():
                            if score<best_score:
                                can_match = g_i not in new_gt_to_pred or score<new_gt_to_pred_scores[g_i]
                                if can_match:
                                    align_g_i=g_i
                                    best_score=score
                        if best_score<9999999:
                            if align_g_i in new_gt_to_pred:
                                to_realign.append(new_gt_to_pred[align_g_i])
                            new_gt_to_pred[align_g_i]=p_i
                            new_gt_to_pred_scores[align_g_i]=best_score
                            pred_to_gt[p_i]=align_g_i
                        #else:
                            #unmatched

                
                entities_truepos = 0
                for g_i,p_i in new_gt_to_pred.items():
                    if gt_classes[g_i]==pred_entities[p_i].cls:
                        if norm_ed(pred_entities[p_i].text,transcription_groups[g_i])<ENTITY_MATCH_THRESH:
                            entities_truepos+=1
                        #print('A hit G:{} <> P:{}'.format(transcription_groups[g_i],pred_entities[p_i].text))

                rel_truepos = 0
                good_pred_pairs = set()
                for g_i1,g_i2 in pairs:
                    #if (g_i1==18 and g_i2==55) or (g_i2==18 and g_i1==55):
                    #    import pdb;pdb.set_trace()
                    if g_i1 in new_gt_to_pred and g_i2 in new_gt_to_pred:
                        p_i1 = new_gt_to_pred[g_i1]
                        p_i2 = new_gt_to_pred[g_i2]
                        if gt_classes[g_i1]==pred_entities[p_i1].cls and gt_classes[g_i2]==pred_entities[p_i2].cls:
                            if (p_i1,p_i2) in pred_links or (p_i2,p_i1) in pred_links:
                                rel_truepos+=1
                                if draw:
                                    if (p_i1,p_i2) in pred_links:
                                        good_pred_pairs.add((p_i1,p_i2))
                                    else:
                                        good_pred_pairs.add((p_i2,p_i1))
                    
                        
                if draw:
                    #pred_to_gt = {p_i:g_i for g_i,p_i in new_gt_to_pred.items()}
                    bad_pred_pairs=set(pred_links)-good_pred_pairs

                ############
            else:
                #Cheating
                gt_pair_hit=[False]*len(pairs)
                rel_truepos=0
                pred_to_gt=defaultdict(list)
                good_pred_pairs = []
                bad_pred_pairs = []
                for p_a,p_b in pred_links:
                    e_a = pred_entities[p_a]
                    e_b = pred_entities[p_b]

                    a_aligned = pred_to_gt.get(p_a,-1)
                    b_aligned = pred_to_gt.get(p_b,-1)

                    best_score = 99999
                    best_gt_pair = -1
                    for pairs_i,(g_a,g_b) in enumerate(pairs):
                        #can't match to a gt pair twice
                        if gt_pair_hit[pairs_i]:
                            continue

                        if a_aligned==-1 and b_aligned==-1:

                            if BROS:
                                dist_aa = norm_ed(transcription_firstline[g_a],e_a.text_lines[0]) if e_a.cls==gt_classes[g_a] else 99
                                dist_bb = norm_ed(transcription_firstline[g_b],e_b.text_lines[0]) if e_b.cls==gt_classes[g_b] else 99
                                dist_ab = norm_ed(transcription_firstline[g_a],e_b.text_lines[0]) if e_b.cls==gt_classes[g_a] else 99
                                dist_ba = norm_ed(transcription_firstline[g_b],e_a.text_lines[0]) if e_a.cls==gt_classes[g_b] else 99
                            else:
                                dist_aa = norm_ed(transcription_groups[g_a],e_a.text) if e_a.cls==gt_classes[g_a] else 99
                                dist_bb = norm_ed(transcription_groups[g_b],e_b.text) if e_b.cls==gt_classes[g_b] else 99
                                dist_ab = norm_ed(transcription_groups[g_a],e_b.text) if e_b.cls==gt_classes[g_a] else 99
                                dist_ba = norm_ed(transcription_groups[g_b],e_a.text) if e_a.cls==gt_classes[g_b] else 99
                            
                            if dist_aa+dist_bb < dist_ab+dist_ba and dist_aa<LINK_MATCH_THRESH and dist_bb<LINK_MATCH_THRESH:
                                score = dist_aa+dist_bb
                                if score<best_score:
                                    best_score = score
                                    best_gt_pair = pairs_i
                                    matching = (g_a,g_b)
                            elif dist_ab<LINK_MATCH_THRESH and dist_ba<LINK_MATCH_THRESH:
                                score = dist_ab+dist_ba
                                if score<best_score:
                                    best_score = score
                                    best_gt_pair = pairs_i
                                    matching = (g_b,g_a)
                        elif a_aligned!=-1 and b_aligned!=-1:
                            if g_a == a_aligned and g_b == b_aligned:
                                matching = (g_a,g_b)
                                best_gt_pair = pairs_i
                                break #can't get better than this if restricting alignment
                            elif g_a == b_aligned and g_b == a_aligned:
                                matching = (g_b,g_a)
                                best_gt_pair = pairs_i
                                break #can't get better than this if restricting alignment
                        else:
                            #only one is aligned
                            if a_aligned!=-1:
                                p_loose = p_b
                                e_loose = e_b
                                if g_a == a_aligned:
                                    g_have = g_a
                                    g_other = g_b
                                elif g_b == a_aligned:
                                    g_have = g_b
                                    g_other = g_a
                                else:
                                    continue #not match for aligned
                            else:
                                p_loose = p_a
                                e_loose = e_a
                                if g_a == b_aligned:
                                    g_have = g_a
                                    g_other = g_b
                                elif g_b == b_aligned:
                                    g_have = g_b
                                    g_other = g_a
                                else:
                                    continue

                            if BROS:
                                score = norm_ed(transcription_firstline[g_other],e_loose.text_lines[0]) if e_loose.cls==gt_classes[g_other] else 99
                            else:
                                score = norm_ed(transcription_groups[g_other],e_loose.text) if e_loose.cls==gt_classes[g_other] else 99
                            if score<best_score and score<LINK_MATCH_THRESH:
                                matching = (g_have,g_other) if a_aligned!=-1 else (g_other,g_have)
                                best_gt_pair = pairs_i


                    if best_gt_pair!=-1:
                        gt_pair_hit[best_gt_pair]=True
                        pred_to_gt[p_a] = matching[0]
                        pred_to_gt[p_b] = matching[1]
                        rel_truepos+=1
                        good_pred_pairs.append((p_a,p_b))
                    else:
                        bad_pred_pairs.append((p_a,p_b))

                    #    rel_FP+=1
                assert rel_truepos==sum(gt_pair_hit)
                rel_recall = sum(gt_pair_hit)/len(pairs) if len(pairs)>0 else 1
                rel_prec = rel_truepos/len(pred_links) if len(pred_links)>0 else 1

                #Now look at the entities. We have some aligned already, do the rest
                gt_entities_hit = [[] for i in range(len(transcription_groups))]
                to_align = []
                for p_i in range(len(pred_entities)):
                    if p_i in pred_to_gt:
                        gt_entities_hit[pred_to_gt[p_i]].append(p_i)
                    else:

                        to_align.append(p_i)

                #resolve ambiguiotity
                for p_is in gt_entities_hit:
                    if len(p_is)>1:
                        #can only align one
                        cls= pred_entities[p_is[0]].cls
                        for e_i in p_is[1:]:
                            assert pred_entities[e_i].cls == cls

                for p_i in to_align:
                    e_i = pred_entities[p_i]
                    best_score = 999999999
                    match = None
                    for g_i,p_is in enumerate(gt_entities_hit):
                        if len(p_is)==0:
                            if BROS:
                                score = norm_ed(transcription_firstline[g_i],e_i.text_lines[0]) if e_i.cls==gt_classes[g_i] else 99
                            else:
                                score = norm_ed(transcription_groups[g_i],e_i.text) if e_i.cls==gt_classes[g_i] else 99
                            if score<LINK_MATCH_THRESH and score<best_score:
                                best_score = score
                                match = g_i

                    if match is None:
                        #false positive? Split entity?
                        if not quiet:
                            print('No match found for pred entitiy: {}'.format(e_i.text))
                        #import pdb;pdb.set_trace()
                        pass
                    else:
                        gt_entities_hit[match].append(p_i)
                        pred_to_gt[p_i]=match

                #check completion of entities (pred have all the lines)
                entities_truepos=0
                for g_i,p_i in enumerate(gt_entities_hit):
                    if len(p_i)>0:
                        p_i=p_i[0]
                        p_lines = pred_entities[p_i].text_lines
                        g_lines = transcription_groups[g_i].split('\\')

                        if len(p_lines)==len(g_lines):
                            entities_truepos+=1
                        elif not quiet:
                            print('Incomplete entity')
                            print('    GT:{}'.format(g_lines))
                            print('  pred:{}'.format(p_lines))
            #######End cheating       

                    
            entity_recall = entities_truepos/len(transcription_groups) if len(transcription_groups) else 1
            entity_prec = entities_truepos/len(pred_entities) if len(pred_entities)>0 else 1
            rel_recall = rel_truepos/len(pairs) if len(pairs)>0 else 1
            rel_prec = rel_truepos/len(pred_links) if len(pred_links)>0 else 1

            total_entity_true_pos += entities_truepos
            total_entity_pred += len(pred_entities)
            total_entity_gt += len(groups)
            assert entities_truepos<=len(pred_entities)
            assert entities_truepos<=len(groups)
            if not quiet:
                print('Entity precision: {}'.format(entity_prec))
                print('Entity recall:    {}'.format(entity_recall))
                print('Entity Fm:        {}'.format(2*entity_recall*entity_prec/(entity_recall+entity_prec) if entity_recall+entity_prec>0 else 0))
                print('Rel precision: {}'.format(rel_prec))
                print('Rel recall:    {}'.format(rel_recall))
                print('Rel Fm:        {}'.format(2*rel_recall*rel_prec/(rel_recall+rel_prec) if rel_recall+rel_prec>0 else 0))
            else:
                print('{} (calls:{}, goodChars:{}) EntityFm: {},  RelFm: {}'.format(instance['imgName'],
                    -1,#num_calls,
                    good_char_pred_ratio,
                    2*entity_recall*entity_prec/(entity_recall+entity_prec) if entity_recall+entity_prec>0 else 0,2*rel_recall*rel_prec/(rel_recall+rel_prec) if rel_recall+rel_prec>0 else 0))

            total_rel_true_pos += rel_truepos
            total_rel_pred += len(pred_links)
            total_rel_gt += len(pairs)



            if draw:
                mid_points=[]
                for p_i,entity in enumerate(pred_entities):
                    if p_i in pred_to_gt:
                        g_i = pred_to_gt[p_i]
                        cls = entity.cls
                        if cls=='header':
                            color=(0,0,255) #header
                        elif cls=='question':
                            color=(0,255,255) #question
                        elif cls=='answer':
                            color=(255,255,0) #answer
                        elif cls=='other':
                            color=(255,0,255) #other 

                        x1,y1,x2,y2 = bb_lines[groups[g_i][0]]
                        for l_i in groups[g_i][1:]:
                            x1_,y1_,x2_,y2_ = bb_lines[l_i]
                            x1=min(x1,x1_)
                            y1=min(y1,y1_)
                            x2=max(x2,x2_)
                            y2=max(y2,y2_)
                        mid_points.append(((x1+x2)//2,(y1+y2)//2))
                        if cls == gt_classes[g_i]:
                            img_f.rectangle(draw_img,(x1,y1),(x2,y2),color,2)
                        else:
                            x,y = mid_points[-1]
                            draw_img[y-3:y+3,x-3:x+3]=color

                    else:
                        if not quiet:
                            print('unmatched entity: {}'.format(entity.text))
                        best=9999999999
                        for g_i,g_text in enumerate(transcription_groups):
                            s = norm_ed(g_text,entity.text)
                            if s<best:
                                best=s
                                best_g = g_i

                        x1,y1,x2,y2 = bb_lines[groups[best_g][0]]
                        mid_points.append((x1,(y1+y2)//2))

                for p_a,p_b in bad_pred_pairs:
                    x1,y1 = mid_points[p_a]
                    x2,y2 = mid_points[p_b]
                    img_f.line(draw_img,(x1,y1+1),(x2,y2+1),(255,0,0),2)
                for p_a,p_b in good_pred_pairs:
                    x1,y1 = mid_points[p_a]
                    x2,y2 = mid_points[p_b]
                    img_f.line(draw_img,(x1,y1),(x2,y2),(0,255,0),2)

                img_f.imshow('f',draw_img)
                img_f.show()
        print('======================')


        total_entity_prec = total_entity_true_pos/total_entity_pred
        total_entity_recall = total_entity_true_pos/total_entity_gt
        total_entity_F = 2*total_entity_prec*total_entity_recall/(total_entity_recall+total_entity_prec) if total_entity_recall+total_entity_prec>0 else 0

        print('Total entity recall, prec, Fm:\t{:.3}\t{:.3}\t{:.3}'.format(total_entity_recall,total_entity_prec,total_entity_F))


        total_rel_prec = total_rel_true_pos/total_rel_pred
        total_rel_recall = total_rel_true_pos/total_rel_gt
        total_rel_F = 2*total_rel_prec*total_rel_recall/(total_rel_recall+total_rel_prec) if total_rel_recall+total_rel_prec>0 else 0
        print('Total rel recall, prec, Fm:\t{:.3}\t{:.3}\t{:.3}'.format(total_rel_recall,total_rel_prec,total_rel_F))

        if write:
            with open(write, 'w') as f:
                json.dump(to_write,f)


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='run QA model on image(s)')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-p', '--pad', default=None, type=str,
                        help='pad image to this size (square)')
    parser.add_argument('-f', '--config', default=None, type=str,
                        help='config override')
    parser.add_argument('-a', '--addtoconfig', default=None, type=str,
                        help='Arbitrary key-value pairs to add to config of the form "k1=v1,k2=v2,...kn=vn".  You can nest keys with k1=k2=k3=v')
    parser.add_argument('-T', '--test', default=False, action='store_const', const=True,
                        help='run test set (default: False)')
    parser.add_argument('-B', '--BROS', default=False, action='store_const', const=True,
                        help='evaluate matching using only first line of entities (default: False)')
    parser.add_argument('-q', '--quiet', default=False, action='store_const', const=True,
                        help='prevent pred prints (default: False)')
    parser.add_argument('-m', '--max-qa-len', default=None, type=int,
                        help='max len for questions')
    parser.add_argument('-d', '--draw', default=False, action='store_const', const=True,
                        help='display image with pred annotated (default: False)')
    parser.add_argument('-D', '--DEBUG', default=False, action='store_const', const=True,
                        help='d')
    parser.add_argument('-E', '--ENTITY_MATCH_THRESH', default=0.6, type=float,
                        help='Edit distance required to have pred entity match a GT one for entity detection')
    parser.add_argument('-L', '--LINK_MATCH_THRESH', default=0.6, type=float,
                        help='Edit distance required to have pred entity match a GT one for linking')
    parser.add_argument('-b', '--beam_search', default=False, type=int,
            help='number of beams (default: not beam search)')
    parser.add_argument('-w', '--write', default=False, type=str,
                        help='path to write all jsons to')

    args = parser.parse_args()

    addtoconfig=[]
    if args.addtoconfig is not None:
        split = args.addtoconfig.split(',')
        for kv in split:
            split2=kv.split('=')
            addtoconfig.append(split2)

    config = None
    if args.checkpoint is None and args.config is None:
        print('Must provide checkpoint (with -c)')
        exit()
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            main(args.checkpoint,args.config,addtoconfig,True,do_pad=args.pad,test=args.test,max_qa_len=args.max_qa_len, draw=args.draw, quiet=args.quiet,BROS=args.BROS,ENTITY_MATCH_THRESH=args.ENTITY_MATCH_THRESH,LINK_MATCH_THRESH=args.LINK_MATCH_THRESH,DEBUG=args.DEBUG,beam_search=args.beam_search,write=args.write)
    else:
        main(args.checkpoint,args.config, addtoconfig,do_pad=args.pad,test=args.test,max_qa_len=args.max_qa_len, draw=args.draw,quiet=args.quiet,BROS=args.BROS,ENTITY_MATCH_THRESH=args.ENTITY_MATCH_THRESH,LINK_MATCH_THRESH=args.LINK_MATCH_THRESH,DEBUG=args.DEBUG,beam_search=args.beam_search,write=args.write)
