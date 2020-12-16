import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import sys,json,os
from utils import img_f

#Nothing selected, show node gradients (both) when hover
#Node selected, show node gradients, on hover show edge gradients


class Controller():
    def __init__(self,root,image_dir,image_name,edge_indexes,num_giters):
        self.giter='all'
        self.imgs={}
        self.H=0
        self.W=0
        self.cur_node1=None
        self.cur_node2=None
        self.cur_giter=None
        for ei,(node1,node2) in enumerate(edge_indexes[-1]):
            for giter in range(num_giters):
                self.imgs['{}_{}_{}'.format(min(node1,node2),max(node1,node2),giter)] = img_f.imread(os.path.join(image_dir,'{}_saliency__{}_graph_g{}.png'.format(image_name,ei,giter)))
            self.imgs['{}_{}_all'.format(min(node1,node2),max(node1,node2),giter)] = img_f.imread(os.path.join(image_dir,'{}_saliency__{}_graph_all.png'.format(image_name,ei)))
            self.imgs['{}_{}_pix'.format(min(node1,node2),max(node1,node2),giter)] = img_f.imread(os.path.join(image_dir,'{}_saliency__{}_pixels.png'.format(image_name,ei)))
            if ei==0:
                self.selected=[node1,node2]
                only='{}_{}_all'.format(min(node1,node2),max(node1,node2),giter)
        
        for key in self.imgs:
            if key==only:
                H = self.imgs[key].shape[0]
                W = self.imgs[key].shape[1]
                self.H=max(H,self.H)
                self.W=max(W,self.W)
                canvas = tk.Canvas(root,  width=W, height=H)
                #canvas.place(x=0,y=0)
                img =  ImageTk.PhotoImage(image=Image.fromarray(self.imgs[key]))
                canvas.create_image(0,0,anchor='nw',image=img)
                print('canvas created {} {} {}'.format(key,H,W))

                self.imgs[key]=canvas
        self.cur_img = None
        self.changeImage(*self.selected)
        #root.mainloop()


    def changeImage(self,node1,node2,giter=None):
        self.prev_node1 = self.cur_node1
        self.prev_node2 = self.cur_node2
        self.prev_giter = self.cur_giter
        if giter is None:
            giter=self.giter
        img = self.imgs['{}_{}_{}'.format(min(node1,node2),max(node1,node2),giter)]
        if self.cur_img is not None:
            self.cur_img.place_forget()
        img.place(x=0,y=0)

        self.cur_node1=node1
        self.cur_node2=node2
        self.cur_giter=giter

    def undoImage(self):
        self.changeImage(self.prev_node1,self.prev_node2,self.prev_giter)

    def previewImage(self,new_node):
        self.changeImage(self.selected[1],new_node)
    def setImage(self,new_node):
        self.selected.pop()
        self.selected.append(new_node)
        self.changeImage(*self.selected)


class HoverButton(tk.Button):
    def __init__(self,master, controller, node_id, **kw):
        #tk.Button.__init__(self,master=master,**kw)
        tk.Frame.__init__(self,master=master,**kw)
        self.controller=controller
        self.node_id=node_id
        #self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.click)


    def on_enter(self, e):
        self.controller.previewImage(self.node_id)

    def on_leave(self, e):
        self.controller.undoImage()
    
    def click(self,e):
        self.controller.setImage(self.node_id)


image_dir = sys.argv[1]
image_name = sys.argv[2]
with open(os.path.join(image_dir,'{}_saliency_info.json'.format(image_name))) as f:
    info = json.load(f)
num_giters = info['num_giters']
edge_indexes = info['edge_indexes']
node_infos = info['node_info']
root = tk.Tk() #initailize window

root.geometry("{}x{}".format(754,1000))


controller = Controller(root,image_dir,image_name,edge_indexes,num_giters)
#root.geometry("{}x{}".format(controller.W,controller.H))
buttons=[]
for node_id, node_info in enumerate(node_infos[-1]):
    x1,x2,y1,y2 = node_info
    h = y2-y1+1
    w = x2-x1+1
    abutton = HoverButton(root,controller,node_id,width=w,height=h)#,bg='blue')
    abutton.place(x=x1,y=y1)
    buttons.append(abutton)



root.mainloop()
