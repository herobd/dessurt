from utils.filelock import FileLock, FileLockException
import threading
from synthetic_text_gen import SyntheticWord
from datasets import load_dataset
import os, random, re
import unicodedata
import numpy as np
from utils import img_f
import argparse

def threadFunction(self,nums):
    #check all locks can be opened
    bad_locks=[]
    for i in nums:
        try:
            with self.locks[i]:
                pass
        except FileLockException:
            bad_locks.append(i)

    #delete bad locks and re-generate these first
    for i in bad_locks:
        os.system('rm {}'.format(os.path.join(self.dir_paths[i],'*'))) #clear it out INCLUDING THE LOCK
        self.generate(i)
    #start iterating inifinitely
    while True:
        for i in nums:
            self.generate(i)
        print('{} finished gen pass'.format(nums))

class GenDaemon:
    def __init__(self,font_dir,out_dir,num_held):
        self.gen = SyntheticWord(font_dir)
        self.used_thresh=2
        self.multi_para = 0.1
        self.num_held = num_held
        self.text_data = load_dataset('wikipedia', '20200501.en', cache_dir='/Data6/davis/data_cache')['train']
        
        self.prune_headers = ["See also", "Gallery", "External media", "History", "Notes"]
        self.wiki_end_keywords = ['References','Sources','External links']
        self.wiki_end_keywords = ['\n'+k+'\n' for k in self.wiki_end_keywords] + ['\n'+k+' \n' for k in self.wiki_end_keywords] + ['\nCategory:']

        self.locks = []
        self.dir_paths = []
        for i in range(num_held):
            self.dir_paths.append(os.path.join(out_dir,'{}'.format(i)))
            count_path = os.path.join(self.dir_paths[i],'count')
            self.locks.append(FileLock(count_path,timeout=3))

    def generate(self,i):

        words = self.getTextSample()
        #text = re.sub(r'\s+',' ',text) #replace all whitespace with space
        #words = text.split(' ')

        font,font_name,fontN,fontN_name = self.gen.getFont()
        try:
            #self.locks[i].acquire()
            with self.locks[i]:
                with open(os.path.join(self.dir_paths[i],'count')) as f:
                    used = int(f.read())
                if used >=self.used_thresh: #no need to rewrite an unused section
                        
                    with open(os.path.join(self.dir_paths[i],'words.txt'),'w') as f:
                        for wi,word in enumerate(words):
                            if re.match(r'\d',word):
                                _font = fontN
                            else:
                                _font = font
                            if word[-1]=='\n':
                                f.write(word+'¶\n')
                                word = word[:-1]
                            else:
                                f.write(word+'\n')
                            img = self.gen.getRenderedText(_font,word)

                            img = (img*255).astype(np.uint8)

                            img_path = os.path.join(self.dir_paths[i],'{}.png'.format(wi))

                            img_f.imwrite(img_path,img)

                    with open(os.path.join(self.dir_paths[i],'count'),'w') as f:
                        f.write('0') #number of times used

            #self.locks[i].release()
        except FileLockException:
            print('Daemon could not unlock {}'.format(i))
            pass 
    
    def getTextSample(self):
        instance_i = random.randrange(self.text_data.num_rows)
        text = self.text_data[instance_i]['text']
        text = unicodedata.normalize(text)#.decode('utf')


        #We first want to cut off the end of the wikipedia article, which has the references and stuff 
        for keyword in self.wiki_end_keywords:
            cut_i = text.find(keyword)
            if cut_i>-1:
                break
        if cut_i>-1:
            text = text[:cut_i]

        #break by paragraph (double newline)
        paras = text.split('\n\n')

        paras = [para for para in paras if para.strip() not in self.prune_headers]

        #select para or paras
        if self.multi_para<random.random():
            num_paras = random.randrange(2,4)
            if num_paras<len(paras):
                start_p = random.randrange(len(paras)-num_paras)
                paras = paras[start_p:start_p+num_paras]
            para = '\n '.join(paras)
        else:
            para = random.choice(paras)
            i=0
            while para.count(' ')<=1 and i <10: #avoid 1-2 word para
                para = random.choice(paras)
                i+=1
        
        #normalize whitespace
        para=re.sub(r'  ',r' ',para)
        para=re.sub(r' ?\n ?',r'\n ',para)

        #text = re.sub('r\s+',' ',text) #normalize whitespace
        para = para.strip()
        words = para.split(' ') #this includes newlines!

        #if len(words)<=num_words:
        #    return words
        #else:
        #    start_w = random.randrange(len(words)-num_words)
        #    return words[start_w:start_w+num_words]
        return words


    
    def run(self,N):
        #Spawn N threads with their own subset of instances to update
        threads=[]
        start_n=0
        for n in range(1,N):
            end_n = round((n/N)*self.num_held)
            nums = range(start_n,end_n)
            threads.append(threading.Thread(target=threadFunction, args=(self,nums), daemon=True))
            threads[-1].start()
            start_n=end_n

        #Have this tread running too
        nums = range(start_n,self.num_held)
        threadFunction(self,nums)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run GenDaemon to continuously generate word images from wikipedia (en) text')
    parser.add_argument('-f', '--font_dir', type=str, help='directory containing fonts')
    parser.add_argument('-o', '--out_dir', type=str, help='directory for storing generated images')
    parser.add_argument('-n', '--num_held', type=int, help='number of stored paragraphs (subdirs)')
    parser.add_argument('-N', '--num_threads', type=int, help='number of stored paragraphs (subdirs)')

    args = parser.parse_args()


    daemon = GenDaemon(args.font_dir,args.out_dir,args.num_held)
    daemon.run(args.num_threads)
