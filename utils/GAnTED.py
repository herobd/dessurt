from zss import simple_distance, Node
import itertools
import math
import random
from multiprocessing import Pool

#GAnTED metric: Greedy-Aligned normalized Tree Edit Distance

def customEditDistance(pred,gt):
    #This permits the wildcard character '¿' and wildcard glob '§' that are pressent in the NAF transcription GT

    #code modified from https://stackoverflow.comdd/a/19928962/1018830
    len_1=len(pred)

    len_2=len(gt)

    x =[[0]*(len_2+1) for _ in range(len_1+1)]#the matrix whose last element ->edit distance

    for i in range(0,len_1+1): #initialization of base case values

        x[i][0]=i
    for j in range(0,len_2+1):

        x[0][j]=j
    for i in range (1,len_1+1):

        for j in range(1,len_2+1):

            if pred[i-1]==gt[j-1] or gt[j-1]=='¿':
                x[i][j] = min(x[i-1][j-1], x[i][j-1]+1, x[i-1][j]+1)

            else:
                x[i][j]= min(x[i][j-1],x[i-1][j],x[i-1][j-1])+(1 if gt[j-1]!='§' else 0)

    return x[i][j]

#Thread function
#Compute nTED when child is moved to move_to
def scoreOfLocation(data):
    without,move_to,node_i,node_ni,child,pred,gt,match_thresh = data
    level = [pred]
    i=0
    while len(level)>0 and i<node_i:
        next_level=[]
        for ni,node in enumerate(level):
            next_level+=node.children
        level = next_level
        i+=1
    node = level[node_ni]
    node.children = without[:move_to]+[child]+without[move_to:]
    score = simple_distance(pred,gt,label_dist=lambda a,b:matchNEditDistance(a,b,match_thresh))
    #assert init!=score
    return score, move_to


def GAnTED(pred,gt,match_thresh=1,num_processes=1):
    getScore = lambda :simple_distance(pred,gt,label_dist=lambda a,b:matchNEditDistance(a,b,match_thresh))
    best_score = getScore()

    if num_processes>1:
        pool = Pool(processes=num_processes)
        chunk = 3
    else:
        pool = None

    level = [pred]
    i=0
    while len(level)>0:
        #print('level {} has {}'.format(i,len(level)))


        next_level=[]
        for ni,node in enumerate(level):
            next_level+=node.children
            
            original_children=node.children

            did=[]
            child_i=0
            while child_i<len(original_children):
                #print('child {}/{}'.format(child_i,len(original_children)))
                #permutation_range = range(len(original_children)
                #TEST tight

                child = original_children[child_i]
                if child not in did:
                    permutation_range = range(max(0,child_i-10),min(len(original_children),child_i+10))
                    without = original_children[:child_i]+original_children[child_i+1:]
                    best=None
                    if pool is None:
                        for move_to in permutation_range:
                            if move_to==child_i:
                                continue
                            #print(f'{move_to} / {len(original_children)}')
                            node.children = without[:move_to]+[child]+without[move_to:]
                            score=getScore()
                            if score<best_score:
                                best = move_to
                                best_score = score
                    else:
                        jobs=[(without,move_to,i,ni,child,pred,gt,match_thresh) for move_to in permutation_range if move_to!=child_i]
                        results = pool.imap_unordered(scoreOfLocation, jobs, chunksize=chunk)
                        for score,move_to in results:
                            if score<best_score:
                                best = move_to
                                best_score = score


                    if best is None:
                        child_i+=1
                    else:
                        original_children = without[:best]+[child]+without[best:]
                        if best<=child_i:
                            child_i+=1
                    did.append(child)
                else:
                    child_i+=1

            node.children=original_children
                    

        level = next_level
        i+=1

    return best_score/simple_distance(gt, Node(''),  label_dist=nEditDistance)
        


def nEditDistance(a,b):
    if a is None:
        a=''
    if b is None:
        b=''
    if len(a)==0 and len(b)==0:
        return 0
    return customEditDistance(a,b)/(0.5*(len(a)+len(b)))

#This has a threshold like ANLS
def matchNEditDistance(a,b,thresh=0.5):
    if a is None:
        a=''
    if b is None:
        b=''
    if len(a)==0 and len(b)==0:
        return 0
    ned = customEditDistance(a,b)/(0.5*(len(a)+len(b)))
    if ned<thresh:
        return ned
    else:
        return 1

def nTED(pred,gt):
    empty_tree = Node("") #not sure if there's a better way to define this
    return simple_distance(gt,pred,label_dist=nEditDistance)/simple_distance(gt, empty_tree,label_dist=nEditDistance)


def printTree(node,depth=0):
    print((' '*depth)+node.label)
    for c in node.children:
        printTree(c,depth+1)

#For testing with random orders
def shuffleTree(node):
    random.shuffle(node.children)
    for c in node.children:
        shuffleTree(c)

#Make representing abigious tables easier in tree
class TableNode(Node):
    def __init__(self,row_headers,col_headers,cells,title=None):
        super().__init__(title if title is not None else "")
        self.row_headers=row_headers
        self.col_headers=col_headers
        self.cells=cells
        self.title=title

        self.refresh()
        self.children = self.row_major.children

    def refresh(self):
        row_major = Node("") 
        col_major = Node("") 

        #assumes no double nesting of row/col headers
        r=0
        for header in self.row_headers:
            if isinstance(header,str):
                row = Node(header)
                if self.cells is not None:
                    for cell in self.cells[r]:
                        row.addkid(Node(cell))
                row_major.addkid(row)

                row = Node(header)
                col_major.addkid(row)
                r+=1
            elif header is not None:
                super_header,sub_headers = header
                row = Node(super_header)
                minor_row = Node(super_header)
                for sub_header in sub_headers:
                    sub_row = Node(sub_header)
                    if self.cells is not None:
                        for cell in self.cells[r]:
                            row.addkid(Node(cell))
                    row.addkid(sub_row)

                    sub_row = Node(sub_header)
                    minor_row.addkid(sub_row)
                    r+=1
                row_major.addkid(row)
                col_major.addkid(minor_row)
        
        c=0
        for header in self.col_headers:
            if isinstance(header,str):
                col = Node(header)
                if self.cells is not None:
                    for row in self.cells:
                        col.addkid(Node(row[c]))
                col_major.addkid(col)

                col = Node(header)
                row_major.addkid(col)
                c+=1
            elif header is not None:
                super_header,sub_headers = header
                col = Node(super_header)
                minor_col = Node(super_header)
                for sub_header in sub_headers:
                    sub_col = Node(sub_header)
                    if self.cells is not None:
                        for row in self.cells:
                            col.addkid(Node(row[c]))
                    col.addkid(sub_col)

                    sub_col = Node(sub_header)
                    minor_col.addkid(sub_col)
                    c+=1
                row_major.addkid(minor_col)
                col_major.addkid(col)


        self.row_major = row_major
        self.col_major = col_major



    def set_row_major(self,row_major):
        self.refresh() #as things get scrambled when GAnTED is called
        if row_major:
            self.children = self.row_major.children
        else:
            self.children = self.col_major.children

    def __sub__(self, other):
        if isinstance(other,TableNode):
            return min(simple_distance(self.row_major,other.row_major),
                       simple_distance(self.col_major,other.col_major),)
        else:
            return min(simple_distance(self.row_major,other),
                       simple_distance(self.col_major,other),)

class FormNode(Node):
    def __init__(self,label):
        super().__init__(label if label is not None else "")
        assert isinstance(self.label,str)

    def addkid(self,child):
        assert isinstance(child,Node)
        super().addkid(child)

    def __sub__(self, other):
        if isinstance(other,TableNode):
            return min(simple_distance(self.row_major,other.row_major),
                       simple_distance(self.col_major,other.col_major),)
        else:
            return simple_distance(self,other)
