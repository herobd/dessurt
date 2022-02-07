import torch
from transformers import GPT2Tokenizer,GPT2LMHeadModel
import sys
import os
import re
import random
from datasets import load_dataset, load_from_disk
from utils.util import ensure_dir
import warnings
import json
#warnings.filterwarnings("ignore")
import timeit

def addPair(q,a,pairs,seen_pairs,seen_qs,seen_as):
    pair = q+':'+a
    if pair in seen_pairs:
        return False #done, repeating
    for sq in range(len(seen_qs)-1):
        if seen_qs[sq]==seen_qs[-1] and seen_qs[sq+1]==q:
            return False #repeating fields
    all_same_a=len(seen_as)>=3
    for sa in seen_as[-3:]:
        if sa!=a:
            all_same_a=False
            break
    if all_same_a:
        return False
    pairs.append((q,a))
    seen_pairs.add(pair)
    seen_qs.append(q)
    seen_as.append(a)
    return True

def parseOutput(output):
    #remove endoftext
    endindex = output.find('<|endoftext|>')
    if endindex>=0:
        output = output[:endindex]

    output = re.sub('\n+','\n',output) #collapse double newlines
    lines = output.split('\n') #get lines

    lines = lines[1:] #discard first prompt line
    pairs=[]
    seen_pairs=set()
    seen_qs=[]
    seen_as=[]
    i=0
    while i < len(lines):
        qa = lines[i].split(':')
        if len(qa)==2:
            q=qa[0].strip()
            a=qa[1].strip()
            if len(a)==0:
                #potentially multiline answer?
                if i<len(lines)-1:
                    next_line = lines[i+1]
                    if next_line.count(':')>1:
                        i+=1
                    else:
                        q = qa[0].strip()
                        a = next_line.strip()
                        if not addPair(q,a,pairs,seen_pairs,seen_qs,seen_as):
                            break
                        i+=2
                else:
                    i+=1
            else:
                if not addPair(q,a,pairs,seen_pairs,seen_qs,seen_as):
                    break
                i+=1
        else:
            i+=1

    return pairs

GOAL=100000 
gpuN=0
if gpuN is not None:
    device = torch.device('cuda:' + str(gpuN))
else:
    device = torch.device('cpu')
#prompt_text='This form is to be filled out regarding Cleopatra'#,'Information regarding Cleopatra']
#prompt_text = sys.argv[1]
length = 256
temperature = 0.85 #lower is more greedy in sampling
k=0
p=0.9
repetition_penalty=1.0 #primarily for CTRL
num_return_sequences = 3
stop_token = None

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

if os.path.exists('DIR'):
    with open('DIR') as f:
        cache_path = f.readline().strip()
else:
    cache_path = '../data/wiki_cache' #/Data6/davis/data_cache
    ensure_dir(cache_path)
if not os.path.exists(os.path.join(cache_path,'dataset_info.json')):
    wikipedia = load_dataset('wikipedia', '20200501.en', cache_dir=cache_path)['train']
    wikipedia.save_to_disk(cache_path)
else:
    wikipedia = load_from_disk(cache_path)

#print('start')
documents=[]
all_pairs = set()
all_qs = set()

num_pair_with_number = 0
num_q_with_number = 0
most_with_number = 0.002

#init
heads = [
        'This form is to be filled out.',
        'This form has been filled out.',
        'Form #',
        'This form is to be filled out regarding {}.',
        'This form contains information about {}.',
        ]
for q in ['Location','Name','Details']:
    all_qs.add(q)
n=6
months=['January','February','March','April','May','June','July','August','September','October','November','December','Jan','Feb','Mar','Apr','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for i in range(60):
    if i%n==0:
        all_pairs.add(('Date','{}/{}/{}'.format(random.randrange(1,13),random.randrange(1,32),random.randrange(1600,2100))))
    elif i%n==1:
        all_pairs.add(('Date','{}/{}/{:02}'.format(random.randrange(1,13),random.randrange(1,32),random.randrange(0,100))))
    elif i%n==2:
        all_pairs.add(('Date','{:02}/{:02}/{}'.format(random.randrange(1,13),random.randrange(1,32),random.randrange(1600,2100))))
    elif i%n==3:
        all_pairs.add(('Date','{:02}/{:02}/{:02}'.format(random.randrange(1,13),random.randrange(1,32),random.randrange(0,100))))
    elif i%n==4:
        all_pairs.add(('Date','{} {} {}'.format(random.randrange(1,32),random.choice(months),random.randrange(1600,2100))))
    elif i%n==5:
        all_pairs.add(('Date','{} {}, {}'.format(random.choice(months),random.randrange(1,32),random.randrange(1600,2100))))

start = timeit.default_timer()
while len(documents)<GOAL:
    prompt_head = random.choice(heads)
    if '{}' in prompt_head:
        topic = wikipedia[random.randrange(wikipedia.num_rows)]['title']
        prompt_head = prompt_head.format(topic)
    elif '#' in prompt_head:
        fake = '{}'.format(random.randrange(10000))
        if random.random()<0.5:
            letter=random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
            if random.random()<0.5:
                fake = letter+fake
            else:
                fake += letter
        prompt_head = prompt_head.replace('#',fake)
        topic = None
    else:
        topic = None

    first_pair = random.choice(tuple(all_pairs))
    second_q = random.choice(tuple(all_qs))

    prompt_text = prompt_head+'\n'+first_pair[0]+': '+first_pair[1]+'\n'+second_q+':'

    #with warnings.catch_warnings():
    #    warnings.filterwarnings("ignore")
    inputs = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = inputs.to(device)

    output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=num_return_sequences,
        )
    #inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    #outputs = model(**inputs, labels=inputs["input_ids"])
    #logits = outputs.logits

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        #print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(stop_token) if stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
        #print(total_sequence)
        pairs = parseOutput(total_sequence)
        #print('=== PAIRS ===')
        #for q,a in pairs:
        #    print(q+':'+(' '*max(1,20-len(q)))+' '+a)
        #all_pairs.update(pairs)
        for q,a in pairs:
            if re.search(r'\d',q):
                if num_pair_with_number/len(all_pairs)<most_with_number:
                    l=len(all_pairs)
                    all_pairs.add((q,a))
                    if len(all_pairs)>l:
                        num_pair_with_number+=1

                if num_q_with_number/len(all_qs)<most_with_number:
                    l=len(all_qs)
                    all_qs.add(q)
                    if len(all_qs)>l:
                        num_q_with_number+=1
            else:
                all_pairs.add((q,a))
                all_qs.add(q)
        if topic is not None:
            #be sure we're on topic
            pairs = pairs[1:]
        if len(pairs)>0:
            documents.append((topic,pairs))


    print('{}/{}'.format(len(documents),GOAL))
    print('num pair {}/{}'.format(num_pair_with_number,len(all_pairs)))
    print('num q {}/{}'.format(num_q_with_number,len(all_qs)))

#check for possible lists
for topic,qas in documents:
    for i in range(len(qas)):
        q,a = qas[i]
        s = a.split(',')
        if len(s)>3:
            min_l=len(s[0])
            max_l=len(s[0])
            for part in s[1:]:
                min_l = min(min_l,len(part))
                max_l = max(max_l,len(part))
            if max_l-min_l<7 and max_l<50:
                #assume this is a list
                lst = [part.strip() for part in s]
                if all(lst[0]==l for l in lst[1:]):
                    #bad (repeated) list
                    qas[i]=(q,lst[0])
                else:
                    qas[i]=(q,lst)
                    #print('Found list!')
                    #print(qas[i])

print('end')
print(timeit.default_timer()-start)

with open('synthetic_forms.json','w') as out:
    json.dump(documents,out,indent=2)
