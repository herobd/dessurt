from utils.filelock import FileLock, FileLockException
from utils.util import ensure_dir
import threading
from synthetic_text_gen import SyntheticWord
from data_sets import getWikiArticle, getWikiDataset
import os, random, re, time
import unicodedata
import numpy as np
from utils import img_f
import argparse




#This class is used to generate the text (using synthetic_text_gen)
#It originally was going to be a daemon, constantly running in the background making new word images for the datasets to read in, but I realized the generation was fast enough that having the datasets multithreaded would probably be good enough.
#I only used in as an object to be called. The daemon functionality is not fully implemented (and none of my datasets have the interface to use the daemon data)

#Default arguments for non-daemon functionality
class GenDaemon:
    def __init__(self,font_dir,out_dir=None,num_held=0,simple=False,no_wiki=False,clear_fonts=False):
        self.gen = SyntheticWord(font_dir,clear=clear_fonts)
        if not no_wiki:
            self.wiki_dataset = getWikiDataset()
        self.used_thresh=-1
        self.multi_para = 0.1
        self.num_held = num_held
        self.simple = simple #for debugging purposese

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

    #With default arguments, samples a wikipedia article and generates the word images
    #Can also provide the text, which will be split along spaces and those word images generated.
    #It returns a list of (text,image) pairs
    #Can also return font for reuse (generate multiple things with same font)
    def generate(self,text=None,font=None,ret_font=False,used=None):
        if text is None:
            words = self.getTextSample()
        else:
            words = text.split(' ')

        if font is None:
            font,font_name,fontN,fontN_name = self.gen.getFont()
            if used is not None:
                while font_name in used:
                    font,font_name,fontN,fontN_name = self.gen.getFont()
                used.add(font_name)
                    
            #if font_name!=fontN_name:
            #    print('@got font pair {} / {}'.format(font_name,fontN_name))
        else:
            font,fontN=font
            font_name=fontN_name=None
        out_words = []   
        for word in words:
            if len(word)==0:
                continue
            if word[-1]=='\n':
                newline=True
                word = word[:-1]
            else:
                newline=False
            img,word_new = self.genImageForWord(word,font,fontN,font_name,fontN_name)
            if img is None:
                #try again, with different fonts
                for retry in range(3):
                    t_font,t_font_name,t_fontN,t_fontN_name = self.gen.getFont()
                    img,word_new = self.genImageForWord(word,t_font,t_fontN)
                    if img is not None:
                        break

                if img is None:
                    continue

            if newline:
                #f.write(word+'¶\n')
                out_words.append((word_new+'¶',img))
            else:
                #f.write(word+'\n'
                out_words.append((word_new,img))
        
        if len(words)>0:
            if ret_font:
                return out_words,(font,fontN)
            return out_words
        else:
            assert text is None
            return self.generate()


    #used by generate()
    #Sometimes generation fails due to weird font/character combination, this will try to regenerate a word image.
    def genImageForWord(self,word,font,fontN,font_name=None,fontN_name=None):
        if re.search(r'\d',word):
            _font = fontN
            _font_name = fontN_name
        else:
            _font = font
            _font_name = font_name
        img,word_new = self.gen.getRenderedText(_font,word)
        if img is None:
            #retry generation
            if _font!=fontN:
                _font_name = fontN_name
                img,word_new = self.gen.getRenderedText(fontN,word)

            if img is None:
                _,_,font2,font2_name = self.gen.getFont()
                _font_name = font2_name
                img,word_new = self.gen.getRenderedText(font2,word)

            if img is None:
                return None,None
        
        img = (img*255).astype(np.uint8)
        return img,word_new

    def getBrackets(self,font,paren):
        font,fontN=font
        o,c = self.gen.getBrackets(font,paren)
        return 255*o,255*c

    #get a wikipedia article split into words
    def getTextSample(self):
        if self.simple:
            words = []
            word_count = random.randint(3,10)
            for w in range(word_count):
                word=''
                for i in range(random.randrange(1,6)):
                    word += random.choice('abcde')
                words.append(word)
            paras = [' '.join(words)]
        else:
            paras = getWikiArticle(dataset=self.wiki_dataset)

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

        para = para.strip()
        words = para.split(' ') #this includes newlines!

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

#Thread job when running as daemon
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

#Run as daemon
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run GenDaemon to continuously generate word images from wikipedia (en) text')
    parser.add_argument('-f', '--font_dir', type=str, help='directory containing fonts')
    parser.add_argument('-o', '--out_dir', type=str, help='directory for storing generated images')
    parser.add_argument('-n', '--num_held', type=int, help='number of stored paragraphs (subdirs)')
    parser.add_argument('-N', '--num_threads', type=int, help='number of stored paragraphs (subdirs)')

    args = parser.parse_args()


    daemon = GenDaemon(args.font_dir,args.out_dir,args.num_held)
    daemon.run(args.num_threads)


