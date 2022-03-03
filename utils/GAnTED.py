from zss import simple_distance, Node

#GAnTED metric: Greedy-Aligned normalized Tree Edit Distance

def greedyAlign(A,B):
    scores=[]
    for a_pos,a_child in enumerate(A.children):
        for b_pos,b_child in enumerate(B.children):
            print('Greedy {}/{}, {}/{}'.format(a_pos,len(A.children),b_pos,len(B.children)))
            score = nTED(a_child,b_child)
            scores.append((a_pos,b_pos,score))
    
    scores.sort(key=lambda a:a[-1])

    new_a_children=[None]*max(len(A.children),len(B.children))
    
    unused_a = set(range(len(A.children)))
    used_b =set()
    for a_pos,b_pos,score in scores:
        if a_pos in unused_a and b_pos not in used_b:
            new_a_children[b_pos]=A.children[a_pos]
            unused_a.remove(a_pos)
            used_b.add(b_pos)

    #new_a_children has blank placeholders for the B spots that didn't have a match (B had mode children)
    for a_child,b_child in zip(new_a_children,B.children):
        if a_child is not None:
            greedyAlign(a_child,b_child) #recursice. This will get expensive


    new_a_children = [child for child in new_a_children if child is not None]

    unused = [A.children[ai] for ai in unused_a]
    new_a_children += unused

    A.children = new_a_children


def nTED(pred,gt):
    empty_tree = Node("") #not sure if there's a better way to define this
    return simple_distance(gt,pred)/simple_distance(gt, empty_tree)

def GAnTED(pred,gt):
    print(str(pred))
    print('================')
    print(str(gt))
    greedyAlign(pred,gt)
    return nTED(pred,gt)

 



class TableNode(Node):
    def __init__(self,row_headers,col_headers,cells,title=None):
        super().__init__(title if not None else "")
        self.row_headers=row_headers
        self.col_headers=col_headers
        self.cells=cells
        self.title=title

        row_major = Node("") #should have class?
        col_major = Node("") #should have class?

        #assumes no double nesting of row/col headers
        r=0
        for header in self.row_headers:
            if isinstance(header,str):
                row = Node(header)
                for cell in self.cells[r]:
                    row.addkid(Node(cell))
                row_major.addkid(row)

                row = Node(header)
                col_major.addkid(row)
                r+=1
            else:
                super_header,sub_headers = header
                row = Node(super_header)
                minor_row = Node(super_header)
                for sub_header in sub_headers:
                    sub_row = Node(sub_header)
                    for cell in self.cells[r]:
                        row.addkid(Node(cell))
                    row.addkid(sub_row)

                    sub_row = Node(header)
                    minor_row.addkid(sub_row)
                    r+=1
                row_major.addkid(row)
                col_major.addkid(minor_row)
        
        c=0
        for header in self.col_headers:
            if isinstance(header,str):
                col = Node(header)
                #for cell in self.cells[r]:
                for row in self.cells:
                    col.addkid(Node(row[c]))
                col_major.addkid(col)

                col = Node(header)
                row_major.addkid(col)
                c+=1
            else:
                super_header,sub_headers = header
                col = Node(super_header)
                minor_col = Node(super_header)
                for sub_header in sub_headers:
                    sub_col = Node(sub_header)
                    #for cell in self.cells[r]:
                    for row in self.cells:
                        col.addkid(Node(row[c]))
                    col.addkid(sub_col)

                    sub_col = Node(header)
                    minor_col.addkid(sub_col)
                    c+=1
                row_major.addkid(minor_col)
                col_major.addkid(col)


        self.row_major = row_major
        self.col_major = col_major

        self.children = self.row_major.children


    def set_row_major(self,row_major):
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

    def __sub__(self, other):
        if isinstance(other,TableNode):
            return min(simple_distance(self.row_major,other.row_major),
                       simple_distance(self.col_major,other.col_major),)
        else:
            return simple_distance(self,other)
