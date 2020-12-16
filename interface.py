import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

#Nothing selected, show node gradients (both) when hover
#Node selected, show node gradients, on hover show edge gradients


class Controller():
    def __init__(self,root,image_dir,image_name,edge_indexes,num_giters):
        self.giter='all'

        for ei,(n1,n2) in enumerate(edge_indexes):
            for giter in range(num_giters):
                self.imgs['{}_{}_{}'.format(min(node1,node2),max(node1,node2),giter)] = img_f.imread(os.path.join(image_dir,'{}_saliency__{}_graph_g{}.png'.format(image_name,ei,giter)))
            self.imgs['{}_{}_all'.format(min(node1,node2),max(node1,node2),giter)] = img_f.imread(os.path.join(image_dir,'{}_saliency__{}_graph_all.png'.format(image_name,ei)))
            self.imgs['{}_{}_pix'.format(min(node1,node2),max(node1,node2),giter)] = img_f.imread(os.path.join(image_dir,'{}_saliency__{}_pixels.png'.format(image_name,ei)))
        
        for key in self.imgs:
            H = self.imgs[key].shape[0]
            W = self.imgs[key].shape[1]
            self.H=max(H,self.H)
            self.W=max(W,self.W)
            canvas = tk.Canvas(root,  width=W, height=H)
            #canvas.place(x=0,y=0)
            img =  ImageTk.PhotoImage(image=Image.fromarray(self.imgs[key]))
            canvas.create_image(0,0,anchor='nw',image=img)

            self.imgs[key]=canvas



    def changeImage(self,node1,node2,giter=None):
        self.prev_node1 = self.cur_node1
        self.prev_node2 = self.cur_node2
        self.prev_giter = self.cur_giter
        if giter is None:
            giter=self.giter
        img = self.imgs['{}_{}_{}'.format(min(node1,node2),max(node1,node2),giter)]
        self.cur_img.place_forget()
        self.img.place(x=0,y=0)

        self.cur_node1=node1
        self.cur_node2=node2
        self.cur_giter=giter

    def undoImage(self):
        self.changeImages(self.prev_node1,self.prev_node2,self.prev_giter)


class HoverButton(tk.Button):
    def __init__(self,master, controller, selected, node_id, **kw):
        #tk.Button.__init__(self,master=master,**kw)
        tk.Frame.__init__(self,master=master,**kw)
        self.controller=controller
        self.selected=selected
        self.node_id=node_id
        #self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.click)


    def on_enter(self, e):
        self.controller.changeImage(self.selected[1],self.node_id)

    def on_leave(self, e):
        self.controller.undoImage()
    
    def click(self,e):
        self.selected.pop()
        self.selected.append(self.node_id)
        self.controller.changeImage(self.selected[0],self.node_id)


image_dir = sys.argv[1]
image_name = sys.argv[2]
with open(os.path.joing(image_dir,'{}_info.json'.format(image_name))) as f:
    info = json.load(f)
num_giters = info['num_giters']
edge_indexes = info['edge_indexes']
node_infos = info['node_info']
root = tk.Tk() #initailize window



controller = Controller(root,image_dir,image_name,edge_indexes,num_giters)
root.geometry("{}x{}".format(controller.W,controller.H))
selected=[0,1]
buttons=[]
for node_id, node_info in enumerate(node_infos[-1]):
    x1,x2,y1,y2 = node_info
    h = y2-y1+1
    w = x2-x1+1
    abutton = HoverButton(master=root,controller,selected,node_id,width=w,height=h)#,bg='blue')
    abutton.place(x=x1,y=y1)
    buttons.append(abutton)

controller.changeImage(0,1) #startout with a selection


print('start')
root.mainloop()
print('end')
