import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

#Nothing selected, show node gradients (both) when hover
#Node selected, show node gradients, on hover show edge gradients

H=W=400


#alabel = tk.Label(master=root,text=text)
##alabel.place(x=3,y=3)

class Controller():
    def __init__(self,image_dir,image_name,edge_indexes,num_giters):
        self.giter='all'

        for ei,(n1,n2) in enumerate(edge_indexes):
            for giter in range(num_giters):
                self.imgs['{}_{}_{}'.format(min(node1,node2),max(node1,node2),giter)] = img_f.imread(os.path.join(image_dir,'{}_saliency__{}_graph_g{}.png'.format(image_name,ei,giter))
            self.imgs['{}_{}_all'.format(min(node1,node2),max(node1,node2),giter)] = img_f.imread(os.path.join(image_dir,'{}_saliency__{}_graph_all.png'.format(image_name,ei))
            self.imgs['{}_{}_pix'.format(min(node1,node2),max(node1,node2),giter)] = img_f.imread(os.path.join(image_dir,'{}_saliency__{}_pixels.png'.format(image_name,ei))


    def change_image(self,node1,node2,giter=None):
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

    def undo_image(self):
        self.change_images(self.prev_node1,self.prev_node2,self.prev_giter)


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
        self.controller.change_image(self.selected[1],self.node_id)

    def on_leave(self, e):
        self.controller.undo_image()
    
    def click(self,e):
        self.selected.pop()
        self.selected.append(self.node_id)
        self.controller.change_image(self.selected[0],self.node_id)

root = tk.Tk() #initailize window

root.geometry("{}x{}".format(W,H))

text='Hello world'
#alabel.pack()


canvas = tk.Canvas(root,  width=W, height=H)
canvas.place(x=0,y=0)

array = np.zeros([H,W,3],dtype=np.uint8)
array[10:20,:,1]=255
array[:,50:140,0]=255

img =  ImageTk.PhotoImage(image=Image.fromarray(array))
canvas.create_image(0,0,anchor='nw',image=img)

bcanvas = tk.Canvas(root,  width=W, height=H)

barray = np.zeros([H,W,3],dtype=np.uint8)
barray[10:20,:,2]=255
barray[:,50:140,1]=255

bimg =  ImageTk.PhotoImage(image=Image.fromarray(barray))
bcanvas.create_image(0,0,anchor='nw',image=bimg)
bcanvas.x=0
bcanvas.y=0

#lines
bcanvas.create_line(23,100,23,100

blabel = tk.Label(master=bcanvas,text=text)
blabel.x=50
blabel.y=10

buttons=[]
for node_id, node_info in enumerate(node_infos[-1]):
    abutton = HoverButton(master=root,controller,selected,node_id,width=25,height=25)#,bg='blue')
    abutton.place(x=200,y=250)
    buttons.append(abutton)

canvas2 = tk.Canvas(root,  width=25, height=25)
canvas2.pack()#place(x=0,y=0)
array2 = np.zeros([25,25,3],dtype=np.uint8)
array2[:,:,2]=255

img2 =  ImageTk.PhotoImage(image=Image.fromarray(array2))
canvas2.create_image(200,250,anchor='nw',image=img2)

print('start')
root.mainloop()
print('end')
