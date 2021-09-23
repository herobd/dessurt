from utils.filelock import FileLock, FileLockException
from utils.util import ensure_dir
import threading
from synthetic_text_gen import SyntheticWord
from data_sets.wiki_text import getWikiArticle
import os, random, re, time
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
        did =0
        for i in nums:
            did += self.generateSave(i)
        if did/len(nums) < 0.2: #if we refreshed less than 20% of the data
            time.sleep(60*30) #wait 30 minutes before contining
        #print('{} finished gen pass'.format(nums))

class GenDaemon:
    def __init__(self,font_dir,out_dir=None,num_held=0):
        self.gen = SyntheticWord(font_dir)
        self.used_thresh=-1
        self.multi_para = 0.1
        self.num_held = num_held

        self.locks = []
        self.dir_paths = []
        for i in range(num_held):
            self.dir_paths.append(os.path.join(out_dir,'{}'.format(i)))
            ensure_dir(self.dir_paths[i])
            count_path = os.path.join(self.dir_paths[i],'count')
            self.locks.append(FileLock(count_path,timeout=3))

    def generateSave(self,i):
        did=False
        try:
            #self.locks[i].acquire()
            with self.locks[i]:
                if os.path.exists(os.path.join(self.dir_paths[i],'count')):
                    with open(os.path.join(self.dir_paths[i],'count')) as f:
                        used = f.read()
                        if used[0]=='i':
                            used=999999
                        else:
                            used = int(used)
                else:
                    used=999999
                if used >=self.used_thresh: #no need to rewrite an unused section
                    did=True
                    with open(os.path.join(self.dir_paths[i],'count'),'w') as f:
                        f.write('incomplete') 
                    
                    generated = self.generate()
                    with open(os.path.join(self.dir_paths[i],'words.txt'),'w') as f:
                        for wi,(word,img) in enumerate(generated):
                            img_path = os.path.join(self.dir_paths[i],'{}.png'.format(wi))

                            img_f.imwrite(img_path,img)
                            f.write(word+'\n')

                    with open(os.path.join(self.dir_paths[i],'count'),'w') as f:
                        f.write('0') #number of times used

            #self.locks[i].release()
        except FileLockException:
            print('Daemon could not unlock {}'.format(i))
            pass 

        return did

    def generate(self):
        words = self.getTextSample()
        #text = re.sub(r'\s+',' ',text) #replace all whitespace with space
        #words = text.split(' ')

        font,font_name,fontN,fontN_name = self.gen.getFont()
        out_words = []   
        for word in words:
            if re.match(r'\d',word):
                _font = fontN
            else:
                _font = font
            if word[-1]=='\n':
                newline=True
                word = word[:-1]
            else:
                newline=False
            img = self.gen.getRenderedText(_font,word)
            if img is None:
                #retry generation
                if _font!=fontN:
                    img = self.gen.getRenderedText(fontN,word)

                if img is None:
                    _,_,font2,font2_name = self.gen.getFont()
                    img = self.gen.getRenderedText(font2,word)

                if img is None:
                    print('no image for word: {}'.format(word))
                    continue
            
            img = (img*255).astype(np.uint8)
            if newline:
                #f.write(word+'¶\n')
                out_words.append((word+'¶',img))
            else:
                #f.write(word+'\n'
                out_words.append((word,img))
        
        if len(words)>0:
            return out_words
        else:
            return self.generate()

    
    def getTextSample(self):
        paras = getWikiArticle()

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
        para=re.sub(r' +',r' ',para)
        para=re.sub(r' ?\n ?',r'\n ',para)

        #text = re.sub('r\s+',' ',text) #normalize whitespace
        para = para.strip()
        #print('++++++\n'+para+'\n++++++++++')
        words = para.split(' ') #this includes newlines!

        #if len(words)<=num_words:
        #    return words
        #else:
        #    start_w = random.randrange(len(words)-num_words)
        #    return words[start_w:start_w+num_words]
        #Don't know why empty words happens, but remove them
        words = [w for w in words if len(w)>0]
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
