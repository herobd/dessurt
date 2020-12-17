import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import sys,json,os
from utils import img_f


image_dir = sys.argv[1]
image_name = sys.argv[2]
with open(os.path.join(image_dir,'{}_saliency_info.json'.format(image_name))) as f:
    info = json.load(f)
num_giters = info['num_giters']
edge_indexes = info['edge_indexes']
node_infos = info['node_info']
root = tk.Tk() #initailize window

root.geometry("{}x{}".format(754,1000))

giter='all'
imgs={}
H=0
W=0
cur_node1=None
cur_node2=None
cur_giter=None
for ei,(node1,node2) in enumerate(edge_indexes[-1]):
    for giter in range(num_giters):
        imgs['{}_{}_{}'.format(min(node1,node2),max(node1,node2),giter)] = img_f.imread(os.path.join(image_dir,'{}_saliency__{}_graph_g{}.png'.format(image_name,ei,giter)))
    imgs['{}_{}_all'.format(min(node1,node2),max(node1,node2),giter)] = img_f.imread(os.path.join(image_dir,'{}_saliency__{}_graph_all.png'.format(image_name,ei)))
    imgs['{}_{}_pix'.format(min(node1,node2),max(node1,node2),giter)] = img_f.imread(os.path.join(image_dir,'{}_saliency__{}_pixels.png'.format(image_name,ei)))
    if ei==0:
        selected=[node1,node2]
        only='{}_{}_all'.format(min(node1,node2),max(node1,node2),giter)

for key in imgs:
    H = imgs[key].shape[0]
    W = imgs[key].shape[1]
    H=max(H,H)
    W=max(W,W)
    #canvas = tk.Canvas(root,  width=W, height=H)
    img =  ImageTk.PhotoImage(image=Image.fromarray(imgs[key]))
    #canvas.create_image(0,0,anchor='nw',image=img)
    #print('canvas created {} {} {}'.format(key,H,W))

    #imgs[key]=canvas
    imgs[key]=(img,H,W)
cur_img = None


def changeImage(node1,node2,giter=None):
    global cur_node1,cur_node2,cur_giter,imgs,prev_node1,prev_node2,prev_giter,root
    if giter is None:
        giter='all'
    key = '{}_{}_{}'.format(min(node1,node2),max(node1,node2),giter)
    if key in imgs:
        prev_node1 = cur_node1
        prev_node2 = cur_node2
        prev_giter = cur_giter
        img,H,W = imgs['{}_{}_{}'.format(min(node1,node2),max(node1,node2),giter)]
        if cur_img is not None:
            cur_img.place_forget()
        canvas = tk.Canvas(root,  width=W, height=H)
        canvas.create_image(0,0,anchor='nw',image=img)
        canvas.place(x=0,y=0)

        cur_node1=node1
        cur_node2=node2
        cur_giter=giter
        return True
    else:
        return False

def undoImage():
    global cur_node1,cur_node2,cur_giter,imgs,prev_node1,prev_node2,prev_giter
    changeImage(prev_node1,prev_node2,prev_giter)

def previewImage(new_node):
    global cur_node1,cur_node2,cur_giter,imgs,prev_node1,prev_node2,prev_giter,selected
    return changeImage(selected[1],new_node)
def setImage(new_node):
    global cur_node1,cur_node2,cur_giter,imgs,prev_node1,prev_node2,prev_giter,selected
    selected.pop()
    selected.append(new_node)

changeImage(*selected)
#root.mainloop()


class HoverButton(tk.Button):
    def __init__(self,master, node_id, **kw):
        #tk.Button.__init__(self,master=master,**kw)
        tk.Frame.__init__(self,master=master,**kw)
        self.node_id=node_id
        #self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.click)
        self.exists=False


    def on_enter(self, e):
        self.exists=previewImage(self.node_id)

    def on_leave(self, e):
        if self.exists:
            undoImage()
    
    def click(self,e):
        setImage(self.node_id)




#root.geometry("{}x{}".format(controller.W,controller.H))
buttons=[]
for node_id, node_info in enumerate(node_infos[-1]):
    x1,x2,y1,y2 = node_info
    h = y2-y1+1
    w = x2-x1+1
    abutton = HoverButton(root,node_id,width=w,height=h)#,bg='blue')
    abutton.place(x=x1,y=y1)
    buttons.append(abutton)



root.mainloop()
