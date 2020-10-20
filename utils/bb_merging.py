import torch
import numpy as np
import math
from utils.util import pointDistance
from utils import img_f

def xyrwh_TextLine(bb):
    assert(False and "untested")
    x,y,r,w,h = bb[1:6]
    theta = math.atan2(h,w)
    theta_final = theta-r
    x1 = x -L*math.cos(theta_final)
    y1 = x +L*math.sin(theta_final)
    x2 = x +L*math.cos(theta_final)
    y2 = x -L*math.sin(theta_final)
    return TextLine(torch.FloatTensor([bb[0],x1,y1,x2,y2,r,*bb[6:]]))

def check_point_angles(points):
    if len(points)==4:
        d01 = pointDistance(points[0],points[1])
        d12 = pointDistance(points[1],points[2])
        d23 = pointDistance(points[2],points[3])
        d03 = pointDistance(points[3],points[0])
        d13 = pointDistance(points[3],points[1])#diagonal
        d02 = pointDistance(points[0],points[2])#diagonal

        a0 = math.acos((d01**2 + d03**2 - d13**2)/(2*d01*d03))
        a1 = math.acos((d01**2 + d12**2 - d02**2)/(2*d01*d12))
        a2 = math.acos((d12**2 + d23**2 - d13**2)/(2*d12*d23))
        a3 = math.acos((d23**2 + d03**2 - d02**2)/(2*d23*d03))
        assert( (a0>0 and a1>0 and a2>0 and a3>0) or 
                (a0<0 and a1<0 and a2<0 and a3<0) )

class TextLine:
    #two constructors. One takes vector, other two TextLines and merges them
    def __init__(self,pred_bb_info=None,other=None,clone=None):
        if clone is not None:
            self.all_conf = list(clone.all_conf) if clone.all_conf is not None else None
            self.all_cls = list(clone.all_cls) if clone.all_cls is not None else None
            self.all_primitive_rects = list(clone.all_primitive_rects)
            self.all_angles = list(clone.all_angles)

            self.cls=clone.cls
            self.conf = clone.conf
            self.median_angle = clone.median_angle
            self.height = clone.height
            self.width = clone.width
            self.std_r = clone.std_r
            self.r_left = clone.r_left
            self.r_right = clone.r_right

            self.poly_points = clone.poly_points.copy() if clone.poly_points is not None else None
            self.center_point = clone.center_point
            self.point_pairs = list(clone.point_pairs) if clone.point_pairs is not None else None
        elif other is None:

            pred_bb_info = pred_bb_info.cpu().detach()
            self.all_conf = [pred_bb_info[0].item()]
            self.all_cls = [pred_bb_info[6:].numpy()]
            #assert(self.all_cls[0].shape[0]==4)


            if pred_bb_info[2]>pred_bb_info[4]: #I think this only occured after changing things on a trained model...
                tmp = pred_bb_info[2]
                pred_bb_info[2]=pred_bb_info[4]
                pred_bb_info[4]=tmp
            if pred_bb_info[1]>pred_bb_info[3]: #I think this only occured after changing things on a trained model...
                tmp = pred_bb_info[1]
                pred_bb_info[1]=pred_bb_info[3]
                pred_bb_info[3]=tmp
            if abs(pred_bb_info[2]-pred_bb_info[4])<1: #detector sometimes predicts flat BBs
                pred_bb_info[2]-=1
                pred_bb_info[4]+=1
            if abs(pred_bb_info[1]-pred_bb_info[3])<1:
                pred_bb_info[1]-=1
                pred_bb_info[3]+=1

            self.all_primitive_rects = [ np.array([[pred_bb_info[1].item(),pred_bb_info[2].item()],[pred_bb_info[3].item(),pred_bb_info[2].item()],[pred_bb_info[3].item(),pred_bb_info[4].item()],[pred_bb_info[1].item(),pred_bb_info[4].item()]]) ] #tl, tr, bt, bl

            assert(self.all_primitive_rects[0][1,1]!=self.all_primitive_rects[0][2,1])
            
            self.all_angles = [pred_bb_info[5].item()]

            #to skip a call to compute(), we'll solve this as it's one rectangle
            self.cls=self.all_cls[0]
            self.conf=np.array(self.all_conf[0])
            self.median_angle =  self.all_angles[0]
            if self.median_angle > math.pi:
                self.median_angle -= 2*math.pi
            elif self.median_angle < -math.pi:
                self.median_angle += 2*math.pi

            if self.median_angle>=-math.pi/4 and self.median_angle<=math.pi/4:
                #horz=True
                #readright=True
                top_points = [self.all_primitive_rects[0][0],self.all_primitive_rects[0][1]]
                bot_points = [self.all_primitive_rects[0][2],self.all_primitive_rects[0][3]]
                self.height = self.all_primitive_rects[0][3][1]-self.all_primitive_rects[0][0][1]
                self.width = self.all_primitive_rects[0][1][0]-self.all_primitive_rects[0][0][0]
            elif self.median_angle>=-math.pi*3/4 and self.median_angle<=-math.pi/4:
                #horz=False
                #readup=False
                top_points = [self.all_primitive_rects[0][1],self.all_primitive_rects[0][2]]
                bot_points = [self.all_primitive_rects[0][0],self.all_primitive_rects[0][3]]
                self.width = self.all_primitive_rects[0][3][1]-self.all_primitive_rects[0][0][1]
                self.height = self.all_primitive_rects[0][1][0]-self.all_primitive_rects[0][0][0]
            elif self.median_angle>=math.pi/4 and self.median_angle<=math.pi*3/4:
                #horz=False
                #readup=True
                top_points = [self.all_primitive_rects[0][3],self.all_primitive_rects[0][0]]
                bot_points = [self.all_primitive_rects[0][2],self.all_primitive_rects[0][1]]
                self.width = self.all_primitive_rects[0][3][1]-self.all_primitive_rects[0][0][1]
                self.height = self.all_primitive_rects[0][1][0]-self.all_primitive_rects[0][0][0]
            else:
                #horz=True
                #readright=False
                top_points = [self.all_primitive_rects[0][2],self.all_primitive_rects[0][3]]
                bot_points = [self.all_primitive_rects[0][1],self.all_primitive_rects[0][0]]
                self.height = self.all_primitive_rects[0][3][1]-self.all_primitive_rects[0][0][1]
                self.width = self.all_primitive_rects[0][1][0]-self.all_primitive_rects[0][0][0]
            
            self.poly_points = np.array( top_points+bot_points )

            check_point_angles(self.poly_points)

            self.center_point = self.poly_points.mean(axis=0)
            self.point_pairs=list(zip(top_points,bot_points))
                #assert(type(self.cls) is np.ndarray)
            self.std_r = 0
            self.r_left = self.median_angle
            self.r_right = self.median_angle
            
        else:
            if pred_bb_info.all_conf is not None and other.all_conf is not None:
                self.all_conf =  pred_bb_info.all_conf+other.all_conf
            else:
                self.all_conf = [pred_bb_info.getConf(),other.getConf()]
            if pred_bb_info.all_cls is not None and other.all_cls is not None:
                self.all_cls =  pred_bb_info.all_cls+other.all_cls
            else:
                self.all_cls = [pred_bb_info.getCls(),other.getCls()]
            self.all_primitive_rects =  pred_bb_info.all_primitive_rects+other.all_primitive_rects
            self.all_angles =  pred_bb_info.all_angles+other.all_angles
            self.median_angle = None
            self.poly_points=None
            self.point_pairs=None
            self.cls=None
            self.conf=None

    def merge(self,other):
        self.all_primitive_rects+=other.all_primitive_rects
        self.all_angles+=other.all_angles
        if self.all_conf is not None and other.all_conf is not None:
            self.all_conf+=other.all_conf
        else:
            self.all_conf = [self.getConf(),other.getConf()]
        if self.all_cls is not None and other.all_cls is not None:
            self.all_cls+=other.all_cls
        else:
            self.all_cls = [self.getCls(),other.getCls()]
        self.median_angle = None
        self.poly_points=None
        self.point_pairs=None
        self.cls=None
        self.conf=None

    def compute(self):

        self.median_angle = np.median(self.all_angles)
        if self.median_angle > math.pi:
            self.median_angle -= 2*math.pi
        elif self.median_angle < -math.pi:
            self.median_angle += 2*math.pi

        if self.median_angle>=-math.pi/4 and self.median_angle<=math.pi/4:
            horz=True
            readright=True
        elif self.median_angle>=-math.pi*3/4 and self.median_angle<=-math.pi/4:
            horz=False
            readup=False
        elif self.median_angle>=math.pi/4 and self.median_angle<=math.pi*3/4:
            horz=False
            readup=True
        else:
            horz=True
            readright=False

        if horz:
            primitive_rects = self.all_primitive_rects
        else:
            primitive_rects = [ [[r[0][1],r[0][0]],[r[3][1],r[3][0]],[r[2][1],r[2][0]],[r[1][1],r[1][0]]] for r in self.all_primitive_rects]
        #print ('primitive {}'.format(primitive_rects))

        #get outer points
        #select windows of points
        #for each window
        #  regress top and bottom lines
        #  average angles
        #  fit new lines to points
        #  take middle points, in terms of a perpendicual line connecting the two lines
        #Ends? Take the end points (according to perp line) of the estimated lines

        primitive_rects_top = list(primitive_rects)
        primitive_rects_top.sort(key=lambda rect:rect[0][0])
        #remove duplicates (or near duplicates)
        toremove = [i for i in range(len(primitive_rects_top)-1) if abs(primitive_rects_top[i][0][0]-primitive_rects_top[i+1][0][0])<0.001 and abs(primitive_rects_top[i][0][1]-primitive_rects_top[i+1][0][1])<0.001 and abs(primitive_rects_top[i][1][0]-primitive_rects_top[i+1][1][0])<0.001 and abs(primitive_rects_top[i][1][1]-primitive_rects_top[i+1][1][1])<0.001]
        toremove.reverse()
        for r in toremove:
            del primitive_rects_top[r]
        top_points=[]#[primitive_rects_top[0][0]]
        last_point_top=primitive_rects_top[0][0]
        #At a high level, this extracts the top border of the polygon created by unioning all rects
        for rect in primitive_rects_top:
            #print('top_points: {}'.format(top_points))
            #print(' adding {} {}'.format(rect[0], rect[1]))
            if last_point_top[0]>rect[1][0]: #the prior rectangle extendes beyond this one
                #find the first point beyon rect (may not be last)
                first_i=-2 #"first" and "second" are relative to traversing top_points in reverse
                while top_points[first_i][0]>rect[1][0]:
                    first_i-=1
                first_i+=1 #move to beyond
                #while first_i>-len(top_points): there is only one strech because we process boxes in order (only steps up, going backward)
                if top_points[first_i][1]>rect[0][1]:
                    second_i = first_i-1
                    while top_points[second_i][1]>rect[0][1] and top_points[second_i][0]>rect[0][0]:
                        second_i-=1
                        #print('second {}: {}'.format(second_i,top_points[second_i]))
                    inter_first=[rect[1][0],top_points[first_i][1]]
                    second_i = len(top_points)+second_i
                    if top_points[second_i][1]<=rect[0][1]:
                        inter_second=[top_points[second_i][0],rect[0][1]]
                        top_points = top_points[:second_i+1]+[inter_second,rect[1],inter_first]+top_points[first_i:]
                    else:
                        inter_second=[rect[0][0],top_points[second_i][1]]
                        top_points = top_points[:second_i+1]+[inter_second,rect[0],rect[1],inter_first]+top_points[first_i:]
                #if rect[0][1]<last_point_top[1]:
                #    #we need to step back all toppoints until we reach where rect[0][0] is
                #    #anyplace rect is above, we need to redo the points
                #    ri=-1
                #    #while top_points[ri][0]<rect[0]
                #    #oldwrong
                #    top_points.append( [rect[0][0],last_point_top[1]] )
                #    top_points.append( rect[0] )
                #    top_points.append( rect[1] )
                #    top_points.append( [rect[1][0],last_point_top[1]] )
                ##otherwise it doesn't matter
                assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
            else:
                if rect[0][0]>=last_point_top[0]: #this rectangle starts after the last
                    top_points.append(rect[0]) #just add it
                    assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
                else:
                    second_i = -1
                    while top_points[second_i][1]>rect[0][1] and top_points[second_i][0]>rect[0][0]:
                        second_i-=1
                    if top_points[second_i][1]<=rect[0][1]:
                        inter_second=[top_points[second_i][0],rect[0][1]]
                        second_i = len(top_points)+second_i
                        top_points = top_points[:second_i+1]+[inter_second]
                    else:
                        inter_second=[rect[0][0],top_points[second_i][1]]
                        second_i = len(top_points)+second_i
                        top_points = top_points[:second_i+1]+[inter_second,rect[0]]
                #elif rect[0][1]<last_point_top[1]: #if the rect/line im adding is above
                #    if top_points[-2][0]>rect[0][0]: #if prior line was clipped, we need to work with the intersection point top_points[-2]
                #        if top_points[-3][1]>rect[0][1]:
                #            intersect_point = [rect[0][0],top_points[-2][1]]
                #            top_points[-3]=intersect_point
                #            top_points[-2]=rect[0]
                #            top_points.pop()
                #            assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
                #        else:
                #            intersect_point=[top_points[-3][0],rect[0][1]]
                #            top_points[-2]=intersect_point
                #            top_points.pop()
                #            assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
                #    else:
                #        intersect_point = [rect[0][0],last_point_top[1]]
                #        top_points[-1]=intersect_point
                #        top_points.append(rect[0])
                #        assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
                #elif rect[0][1]>last_point_top[1]: #if the rect/line im adding is below
                #    if top_points[-2][0]>rect[0][0]:
                #        if top_points[-3][1]>rect[0][1]:
                #            last_two_points = top_points[-2:]
                #            intersect_point_3 = [rect[0][0],top_points[-3][1]]
                #            top_points[-3]=intersect_point_3
                #            top_points[-2]=rect[0]
                #            intersect_point_2 = [top_points[-2][0],rect[0][1]]
                #            top_points[-1]=intersect_point_2
                #            top_points+=last_two_points
                #            assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
                #        intersection_point_1 = [top_points[-1][0],rect[1][1]]
                #        top_points.append(intersection_point_1)
                #        assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))

                #    else:
                #        intersect_point = [last_point_top[0],rect[0][1]]
                #        top_points.append(intersect_point)
                #        assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
                #else:
                #    #it's straight across, we'll just average the two outside points to provide a good midpoint
                #    #import pdb;pdb.set_trace()
                #    top_points[-1]=[(top_points[-2][0]+rect[1][0])/2,(top_points[-2][1]+rect[1][1])/2]
                #    assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
                assert(rect[1][0]>=top_points[-1][0])
                top_points.append(rect[1])
                assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
                if math.sqrt((top_points[-1][0]-top_points[-2][0])**2 + (top_points[-1][1]-top_points[-2][1])**2)<0.0001:
                    top_points.pop()
            last_point_top = top_points[-1]

        primitive_rects_bot = list(primitive_rects)
        primitive_rects_bot.sort(key=lambda rect:rect[3][0])
        #remove duplicates (or near duplicates)
        toremove = [i for i in range(len(primitive_rects_bot)-1) if abs(primitive_rects_bot[i][3][0]-primitive_rects_bot[i+1][3][0])<0.001 and abs(primitive_rects_bot[i][3][1]-primitive_rects_bot[i+1][3][1])<0.001 and abs(primitive_rects_bot[i][2][0]-primitive_rects_bot[i+1][2][0])<0.001 and abs(primitive_rects_bot[i][2][1]-primitive_rects_bot[i+1][2][1])<0.001]
        toremove.reverse()
        for r in toremove:
            del primitive_rects_bot[r]
        bot_points=[]#[primitive_rects_bot[0][3]]
        last_point_bot=primitive_rects_bot[0][3]
        for rect in primitive_rects_bot:
            #print('bot_points: {}'.format(bot_points))
            #print(' adding {} {}'.format(rect[3], rect[2]))
            #if abs(rect[3][0]-35)<1 and abs(rect[3][1]-386)<1:
            #    import pdb;pdb.set_trace()
            if last_point_bot[0]>rect[2][0]: #the prior rectangle extendes beyond this one
                #find the first point beyon rect (may not be last)
                first_i=-2 #"first" and "second" are relative to traversing bot_points in reverse
                while bot_points[first_i][0]>rect[2][0]:
                    first_i-=1
                first_i+=1 #move to beyond
                #while first_i>-len(bot_points): there is only one strech because we process boxes in order (only steps up, going backward)
                if bot_points[first_i][1]<rect[3][1]:
                    second_i = first_i-1
                    while bot_points[second_i][1]<rect[3][1] and bot_points[second_i][0]>rect[3][0]:
                        second_i-=1
                        #print('second {}: {}'.format(second_i,bot_points[second_i]))
                    inter_first=[rect[2][0],bot_points[first_i][1]]
                    second_i = len(bot_points)+second_i
                    if bot_points[second_i][1]>=rect[3][1]:
                        inter_second=[bot_points[second_i][0],rect[3][1]]
                        bot_points = bot_points[:second_i+1]+[inter_second,rect[2],inter_first]+bot_points[first_i:]
                    else:
                        inter_second=[rect[3][0],bot_points[second_i][1]]
                        bot_points = bot_points[:second_i+1]+[inter_second,rect[3],rect[2],inter_first]+bot_points[first_i:]
            else:
                if rect[3][0]>=last_point_bot[0]:
                    bot_points.append(rect[3])
                else:
                    #walk backwards past all previous points this new line subsumes
                    second_i = -1
                    while bot_points[second_i][1]<rect[3][1] and bot_points[second_i][0]>rect[3][0]:
                        second_i-=1
                    if bot_points[second_i][1]>=rect[3][1]:
                        inter_second=[bot_points[second_i][0],rect[3][1]]
                        second_i = len(bot_points)+second_i
                        bot_points = bot_points[:second_i+1]+[inter_second]
                    else:
                        inter_second=[rect[3][0],bot_points[second_i][1]]
                        second_i = len(bot_points)+second_i
                        bot_points = bot_points[:second_i+1]+[inter_second,rect[3]]
                #elif rect[3][1]>last_point_bot[1]:
                #    if bot_points[-2][0]>rect[3][0]: #if prior line was clipped, we need to work with the intersection point bot_points[-2]
                #        if bot_points[-3][1]<rect[3][1]:
                #            intersect_point = [rect[3][0],bot_points[-2][1]]
                #            bot_points[-3]=intersect_point
                #            bot_points[-2]=rect[3]
                #            bot_points.pop()
                #        else:
                #            intersect_point=[bot_points[-3][0],rect[3][1]]
                #            bot_points[-2]=intersect_point
                #            bot_points.pop()
                #    else:
                #        intersect_point = [rect[3][0],last_point_bot[1]]
                #        bot_points[-1]=intersect_point
                #        bot_points.append(rect[3])
                #elif rect[3][1]<last_point_bot[1]:
                #    if bot_points[-2][0]>rect[3][0]:
                #        if bot_points[-3][1]<rect[3][1]:
                #            last_two_points = bot_points[-2:]
                #            intersect_point_3 = [rect[3][0],bot_points[-3][1]]
                #            bot_points[-3]=intersect_point_3

                #            bot_points[-2]=rect[3]

                #            intersect_point_2 = [bot_points[-2][0],rect[3][1]]
                #            bot_points[-1]=intersect_point_2
                #            bot_points+=last_two_points
                #        intersection_point_1 = [bot_points[-1][0],rect[2][1]]
                #        bot_points.append(intersection_point_1)

                #    else:
                #        intersect_point = [last_point_bot[0],rect[3][1]]
                #        bot_points.append(intersect_point)
                ##else:
                ##    #it's straight across, we'll just average the two overlapped points to provide a good midpoint
                ##    bot_points[-1]=[(bot_points[-2][0]+rect[2][0])/2,(bot_points[-2][1]+rect[2][1])/2]

                assert(rect[2][0]>=bot_points[-1][0])
                bot_points.append(rect[2])
            last_point_bot = bot_points[-1]
            #print('\n')
            #print(bot_points)


        #all_top_points = [rect[0] for rect in primitive_rects]+[rect[1] for rect in primitive_rects]
        #all_top_points.sort(key=lambda p:p[0])
        #top_points=[all_top_points[0]]
        #for point in all_top_points[1:]:
        #    if point[1]<top_points[-1][1]:
        #        top_points.append([point[0],top_points[-1][1]])
        #        top_points.append(point)
        #    elif point[1]>top_points[-1][1]:

        #all_bot_points = [rect[3] for rect in primitive_rects]+[rect[2] for rect in primitive_rects]
        #all_bot_points.sort(key=lambda p:p[0])

            #print(bot_points)

        #remove points that are actually the same
        #for i in range(len(top_points)-1):
        #    if top_points[i][0]==top_points[i+1][0] and top_points[i][1]==top_points[i+1][1]:
        #        toremove.append(i)

        #top_points.reverse()
        #bot_points.reverse()


        #print('top ps {}'.format(top_points))
        #print('bot ps {}'.format(bot_points))

        #smooth
        #for i in range(1,len(top_points)-1):
        #    new_top_points.append((top_points[i-1]+top_points[i]+top_points
        top_total_distance=0
        for i in range(len(top_points)-1):
            top_total_distance += math.sqrt((top_points[i][0]-top_points[i+1][0])**2 + (top_points[i][1]-top_points[i+1][1])**2)
        bot_total_distance=0
        for i in range(len(bot_points)-1):
            bot_total_distance += math.sqrt((bot_points[i][0]-bot_points[i+1][0])**2 + (bot_points[i][1]-bot_points[i+1][1])**2)
        
        #DEBUG#
        self._top_points=list(top_points)
        self._bot_points=list(bot_points)
        if not horz:
            self._top_points = [(y,x) for x,y in self._top_points]
            self._bot_points = [(y,x) for x,y in self._bot_points]
        #DEBUG#

    
        #step size is average "height"
        top_points_np=np.array(top_points)
        bot_points_np=np.array(bot_points)
        step_size = np.linalg.norm(top_points_np.mean(axis=0)-bot_points_np.mean(axis=0))
        #if step_size==0: #Detection error with flat box
        #    for p in top_points:
        #        p[1]-=2
        #    for p in bot_points:
        #        p[1]+=2
        #    step_size=4
        top_points_np= bot_points_np= None

        #step_size_top= (2*step_size/top_total_distance)/(1/top_total_distance + 1/bot_total_distance)
        #step_size_bot = (2*step_size/bot_total_distance)/(1/top_total_distance + 1/bot_total_distance)
        step_size_top = 2*step_size/((bot_total_distance/top_total_distance)+1)
        step_size_bot = 2*step_size/((top_total_distance/bot_total_distance)+1)

        assert(abs(step_size_bot+step_size_top-2*step_size)<0.001)

        #we'll check for large gaps (long lines) and split them. This should help the later chunking
        #toadd=[ for i in range(len(top_points)-1) if ]
        toadd=[]
        max_length = step_size_top/2
        for i in range(len(top_points)-1):
            dist = math.sqrt((top_points[i][0]-top_points[i+1][0])**2 + (top_points[i][1]-top_points[i+1][1])**2)
            lengths = math.floor(dist/max_length)
            if lengths>0:
                subdist = dist/lengths
                for t in np.arange(1/lengths,0.999999999,1/lengths):
                    x = t*top_points[i+1][0] + (1-t)*top_points[i][0]
                    y = t*top_points[i+1][1] + (1-t)*top_points[i][1]
                    toadd.append((i+1,[x,y]))
        toadd.reverse()
        #print('adding {} points to top'.format(len(toadd)))
        #print('top_points before: {}'.format(top_points))
        for i,p in toadd:
            top_points.insert(i,p)
        #print('top_points after: {}'.format(top_points))
        assert(all([top_points[i][0]<=top_points[i+1][0]+0.000001 for i in range(len(top_points)-1)]))

        toadd=[]
        max_length = step_size_bot/2
        for i in range(len(bot_points)-1):
            dist = math.sqrt((bot_points[i][0]-bot_points[i+1][0])**2 + (bot_points[i][1]-bot_points[i+1][1])**2)
            lengths = math.floor(dist/max_length)
            if lengths>0:
                subdist = dist/lengths
                for t in np.arange(1/lengths,0.999999999,1/lengths):
                    x = t*bot_points[i+1][0] + (1-t)*bot_points[i][0]
                    y = t*bot_points[i+1][1] + (1-t)*bot_points[i][1]
                    toadd.append((i+1,[x,y]))
        toadd.reverse()
        #print('adding {} points to bot'.format(len(toadd)))
        #print('bot_points before: {}'.format(bot_points))
        for i,p in toadd:
            bot_points.insert(i,p)
        #print('bot_points after: {}'.format(bot_points))

        TEST_top_total_distance=0
        for i in range(len(top_points)-1):
            TEST_top_total_distance += math.sqrt((top_points[i][0]-top_points[i+1][0])**2 + (top_points[i][1]-top_points[i+1][1])**2)
        TEST_bot_total_distance=0
        for i in range(len(bot_points)-1):
            TEST_bot_total_distance += math.sqrt((bot_points[i][0]-bot_points[i+1][0])**2 + (bot_points[i][1]-bot_points[i+1][1])**2)
        assert(abs(top_total_distance-TEST_top_total_distance)<0.000001)
        assert(abs(bot_total_distance-TEST_bot_total_distance)<0.000001)

        top_points=np.array(top_points)
        bot_points=np.array(bot_points)

        final_points=[]
        
        t_i=1
        b_i=1
        accum_top_distance=0
        accum_bot_distance=0
        top_distance = 0
        bot_distance = 0
        while t_i<len(top_points) or b_i<len(bot_points):
            #print('top {:.2f}/{:.2f}, bot {:.2f}/{:.2f}'.format(accum_top_distance,top_total_distance,accum_bot_distance,bot_total_distance))
            #print('\ttop {:.2f}/{:.2f}, bot {:.2f}/{:.2f}'.format(accum_top_distance/step_size_top,top_total_distance/step_size_top,accum_bot_distance/step_size_bot,bot_total_distance/step_size_bot))
            if top_distance<step_size_top:
                top_points_in_step=[top_points[t_i-1]]
            if bot_distance<step_size_bot:
                bot_points_in_step=[bot_points[b_i-1]]
            added=0
            while (top_distance<step_size_top or t_i>=len(top_points)-2) and t_i<len(top_points):
                last_top_distance=np.linalg.norm(top_points_in_step[-1]-top_points[t_i])
                top_distance+=last_top_distance
                accum_top_distance+=last_top_distance
                top_points_in_step.append(top_points[t_i])
                #print('t_i:{}, added {},\tdistance:{}'.format(t_i,top_points[t_i],top_distance))
                t_i+=1
                added+=1
            if added>2 and top_distance-step_size_top>-(top_distance-last_top_distance-step_size_top) and t_i<len(top_points)-2:
                top_distance-=last_top_distance
                accum_top_distance-=last_top_distance
                t_i-=1
                r=top_points_in_step.pop()
                #print('t_i:{}, removed {},\tdistance:{}'.format(t_i,r,top_distance))
            #if added>0:
            #    accum_top_distance+=top_distance
            top_distance = top_distance-step_size #reset the top_distance with the residual to keep the two sides in sync

            added=0
            while (bot_distance<step_size_bot or b_i>=len(bot_points)-2) and b_i<len(bot_points):
                last_bot_distance=np.linalg.norm(bot_points_in_step[-1]-bot_points[b_i])
                bot_distance+=last_bot_distance
                accum_bot_distance+=last_bot_distance
                bot_points_in_step.append(bot_points[b_i])
                #print('b_i:{}, added {}, distance:{}'.format(b_i,bot_points[b_i],bot_distance))
                b_i+=1
                added+=1
            if added>2 and bot_distance-step_size_bot>-(bot_distance-last_bot_distance-step_size_bot) and b_i<len(bot_points)-2:
                bot_distance-=last_bot_distance
                accum_bot_distance-=last_bot_distance
                b_i-=1
                r=bot_points_in_step.pop()
                #print('b_i:{}, removed {}, distance:{}'.format(b_i,r,bot_distance))
            #if added>0:
            #    accum_bot_distance+=bot_distance
            bot_distance = bot_distance-step_size #reset the bot_distance with the residual to keep the two sides in sync
            #print('diff top: {}, bot:{}'.format(top_distance,bot_distance))
            #print('F top {:.2f}/{:.2f}, bot {:.2f}/{:.2f}'.format(accum_top_distance,top_total_distance,accum_bot_distance,bot_total_distance))


            if len(top_points_in_step)==1:
                assert(t_i==len(top_points))
                top_points_in_step = [top_points[-2],top_points_in_step[0]]
            if len(bot_points_in_step)==1:
                assert(b_i==len(bot_points))
                bot_points_in_step = [bot_points[-2],bot_points_in_step[0]]
            assert(len(top_points_in_step)>1 and len(bot_points_in_step)>1)
            top_points_in_step = np.array(top_points_in_step)
            s_top,c_top = np.polyfit(top_points_in_step[:,0],top_points_in_step[:,1], 1)

            bot_points_in_step = np.array(bot_points_in_step)
            s_bot,c_bot = np.polyfit(bot_points_in_step[:,0],bot_points_in_step[:,1], 1)

            s = (s_top+s_bot)/2
            top_new_c = (top_points_in_step[:,1]-s*top_points_in_step[:,0]).mean()
            bot_new_c = (bot_points_in_step[:,1]-s*bot_points_in_step[:,0]).mean()


            #now we need a midpoint for both
            #we'll first average the first and last points
            #this will be projected onto the two lines

            mean_point = (top_points_in_step[0]+top_points_in_step[-1]+bot_points_in_step[0]+bot_points_in_step[-1])/4
            mag_top = (mean_point[0]+(mean_point[1]-top_new_c)*s)/(1+s**2)
            top_point = [mag_top,mag_top*s+top_new_c]
            mag_bot = (mean_point[0]+(mean_point[1]-bot_new_c)*s)/(1+s**2)
            bot_point = [mag_bot,mag_bot*s+bot_new_c]

            if len(final_points)==0: #We haven't added a point, so we'll add an end point using the first points
                mean_first_point = (top_points_in_step[0]+bot_points_in_step[0])/2
                mag_top = (mean_first_point[0]+(mean_first_point[1]-top_new_c)*s)/(1+s**2)
                top_first_point = [mag_top,mag_top*s+top_new_c]
                mag_bot = (mean_first_point[0]+(mean_first_point[1]-bot_new_c)*s)/(1+s**2)
                bot_first_point = [mag_bot,mag_bot*s+bot_new_c]
                final_points.append([top_first_point,bot_first_point])
            #assert(top_point[0]!=bot_point[0] or top_point[1]!=bot_point[1])
            #assert(top_point[1]<bot_point[1])
            final_points.append([top_point,bot_point])

        #now we'll add the other end point
        mean_end_point = (top_points_in_step[-1]+bot_points_in_step[-1])/2
        mag_top = (mean_end_point[0]+(mean_end_point[1]-top_new_c)*s)/(1+s**2)
        top_end_point = [mag_top,mag_top*s+top_new_c]
        mag_bot = (mean_end_point[0]+(mean_end_point[1]-bot_new_c)*s)/(1+s**2)
        bot_end_point = [mag_bot,mag_bot*s+bot_new_c]
        final_points.append([top_end_point,bot_end_point])

        #Hack, because weird stuff is getting merged at the begining of training.
        for i in range(len(final_points)-1):
            #check if the lines cross at all
            if (
                    doIntersect(final_points[i][0],final_points[i+1][0],final_points[i][1],final_points[i+1][1]) or 
                    (i>0 and doIntersect(final_points[i-1][0],final_points[i][0],final_points[i][1],final_points[i+1][1])) or 
                    (i>0 and doIntersect(final_points[i][0],final_points[i+1][0],final_points[i-1][1],final_points[i][1]))
                    ):
                tmp=final_points[i+1][0]
                final_points[i+1][0]=final_points[i+1][1]
                final_points[i+1][1]=tmp

        #print('final pair points: {}'.format(final_points))
        if horz:
            if readright:
                self.point_pairs = final_points
            else:
                final_points.reverse()
                self.point_pairs = [[b,t] for t,b in final_points]
        else:
            if readup:
                self.point_pairs = [[[b[1],b[0]],[t[1],t[0]]] for t,b in final_points]
            else:
                final_points.reverse()
                self.point_pairs = [[[t[1],t[0]],[b[1],b[0]]] for t,b in final_points]

        top_points,bot_points = zip(*self.point_pairs)
        top_points=list(top_points)
        bot_points=list(bot_points)
        bot_points.reverse()
        self.poly_points = np.array( top_points+bot_points )
        check_point_angles(self.poly_points)

        self.center_point = self.poly_points.mean(axis=0)
            #assert(type(self.cls) is np.ndarray)
        self.std_r = np.std(self.all_angles)
        self.r_left = math.atan2(top_points[0][1]-bot_points[0][1],bot_points[0][0]-top_points[0][0])
        self.r_right = math.atan2(top_points[-1][1]-bot_points[-1][1],bot_points[-1][0]-top_points[-1][0])
        
        horz_sum=0
        vert_sum=0
        for i,(top,bot) in enumerate(self.point_pairs):
            if i>=0:
                horz_sum += math.sqrt((top[0]-top_points[i-1][0])**2 +(top[1]-top_points[i-1][1])**2)
                horz_sum += math.sqrt((bot[0]-bot_points[i-1][0])**2 +(bot[1]-bot_points[i-1][1])**2)
            vert_sum += math.sqrt((top[0]-bot[0])**2 + (top[0]- bot[0])**2)
        self.height = vert_sum/len(self.point_pairs)
        self.width = horz_sum/2



    def getReadPosition(self):
        if self.median_angle is None:
            self.compute()
        angle = self.median_angle
        slope = -math.tan(angle)
        xc,yc = self.center_point

        if math.isinf(slope):
            if angle==math.pi/2:
                return xc
            elif angle==-math.pi/2:
                return -xc
            else:
                assert(False)
        elif slope == 0:
            if angle==0:
                return yc
            elif angle==-math.pi:
                return -yc
            else:
                assert(False)
        else:
            b=yc-slope*xc

            #we define a parameteric line which defines the read direction (perpediculat to this text lines slope) and find the location parametrically. Smaller values of t are earlier in read order
            #x = +/- sqrt(1/(1+1/slope**2)) * t
            #y = +/- sqrt(1/(1+slope**2)) * t
            #The +/- must be determined using the actual slope

            if angle<math.pi/2 and angle>=0:
                sign_x=1
                sign_y=1
            elif angle>math.pi/2: # and angle<=math.pi:
                sign_x=1
                sign_y=-1
            elif angle>-math.pi/2 and angle<0:
                sign_x=-1
                sign_y=1
            elif angle<-math.pi/2:
                sign_x=-1
                sign_y=-1
            else:
                assert(False)



            t = b/(sign_y*math.sqrt(1/(1+slope**2))-slope*sign_x*math.sqrt(1/(1+(1/slope**2))))
            return t.item()
        
    def polyYs(self):
        if self.poly_points is None:
            self.compute()
        #return [p[1] for p in self.poly_points]
        return self.poly_points[:,1]
    def polyXs(self):
        if self.poly_points is None:
            self.compute()
        #return [p[0] for p in self.poly_points]
        return self.poly_points[:,0]
    def polyPoints(self):
        if self.poly_points is None:
            self.compute()
        return self.poly_points
    def pairPoints(self):
        if self.point_pairs is None:
            self.compute()
        return self.point_pairs


    def getGrid2(self,height):
        if self.point_pairs is None:
            self.compute()

        chunks=[]
        for i in range(len(self.point_pairs)-1):
            tl,bl = self.point_pairs[i]
            tr,br = self.point_pairs[i+1]

            hAvg = ( math.sqrt((tl[0]-bl[0])**2 + (tl[1]-bl[1])**2) + math.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2) )/2
            wAvg = ( math.sqrt((tl[0]-tr[0])**2 + (tl[1]-tr[1])**2) + math.sqrt((bl[0]-br[0])**2 + (bl[1]-br[1])**2) )/2
            scale = height/hAvg
            width = round(wAvg*scale)

            T = img_f.getAffineTransform(np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]]),np.array([tl,tr,br,bl]))
            #T = torch.from_numpy(T[0:2]) #affine so we don;t actually need third
            T = torch.from_numpy(T)

            ys = torch.arange(height).view(height,1,1).expand(height,width,1)
            xs = torch.arange(width).view(1,width,1).expand(height,width,1)
            orig_points = torch.cat([xs,ys,torch.ones_like(xs)],dim=0)

            new_points = T.float().mm(orig_points.float().view(3,height*width))
            new_points = new_points.view(3,height,width).permute(1,2,0)[...,:2]
            chunks.append(new_points)

        return torch.cat(chunks,dim=1)

    def boundingRect(self):
        if self.poly_points is None:
            self.compute()
        minx,miny = self.poly_points.min(axis=0)
        maxx,maxy = self.poly_points.max(axis=0)
        return minx,miny,maxx,maxy

    def getFeatureInfo(self):
        if self.poly_points is None:
            self.compute()
        # 0    1 2 3 4 5  6   7   8   9   10  11  12  13  14     15      16    17
        #conf, x,y,r,h,w,tlx,tly,trx,try,brx,bry,blx,bly,r_left,r_right,std_r,classFeats

        return (self.getConf(), self.center_point[0], self.center_point[1], self.median_angle, self.height, self.width, *self.point_pairs[0][0], *self.point_pairs[-1][0], *self.point_pairs[-1][1], *self.point_pairs[0][1], self.r_left, self.r_right, self.std_r, self.getReadPosition(), *self.getCls())

    def getHeight(self):
        if self.poly_points is None:
            self.compute()
        return self.height
    def getWidth(self):
        if self.poly_points is None:
            self.compute()
        return self.width
    def getCenterPoint(self):
        if self.poly_points is None:
            self.compute()
        return self.center_point
    def medianAngle(self):
        if self.median_angle is None:
            self.compute()
        return self.median_angle
    def getCls(self):
        if self.cls is None:
            if self.all_cls is not None:
                self.cls = np.stack(self.all_cls,axis=0).mean(axis=0)
        #assert(type(self.cls) is np.ndarray)
        return self.cls
    def getConf(self):
        if self.conf is None:
            if self.all_conf is not None:
                self.conf = np.mean(self.all_conf)
        return self.conf.item()
    
    def getGrid(self,height,device):
        if self.point_pairs is None:
            self.compute()

        grid_line = []
        for i in range(len(self.point_pairs)-1):
            avg_h = (pointDistance(*self.point_pairs[i]) + pointDistance(*self.point_pairs[i+1]))/2
            avg_w = (pointDistance(self.point_pairs[i][0],self.point_pairs[i+1][0]) + pointDistance(self.point_pairs[i][1],self.point_pairs[i+1][1]))/2
            ratio = height/avg_h
            out_width = min(round(ratio*avg_w),5*height)
            #assert(out_width<5*height)

            t = ((np.arange(height) + 0.5) / float(height))[:,None].astype(np.float32)
            t = np.repeat(t,axis=1, repeats=out_width)
            t = torch.from_numpy(t)
            s = ((np.arange(out_width) + 0.5) / float(out_width))[:,None].astype(np.float32)
            s = np.repeat(s,axis=1, repeats=height)
            s = torch.from_numpy(s).t()
            
            t=t.to(device)
            s=s.to(device)
            #construct interpolation for the 4 points of the polygon to each pixel of output grid
            interpolations = torch.stack([
                (1-t)*(1-s), #tl*
                t*(1-s), #bl
                (1-t)*s, #tr
                t*s, #br*
            ], dim=2)
            points = torch.FloatTensor([*self.point_pairs[i],*self.point_pairs[i+1]]).to(device)
            #points=points[:,::-1] #flip x,y positions as torch.grid_sample() expects y,x
            #points = points.flip(dims=[1])
            grid = interpolations[:,:,:,None]*points[None,None,:,:] #add dimensions to allow broacast along xy and row, col dims
            grid=grid.sum(dim=2) #sum four points
            grid_line.append(grid)
        grid_line = torch.cat(grid_line, dim=1)
        return grid_line




### from https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

# Given three colinear points p, q, r, the function checks if  
# point q lies on line segment 'pr'  
def onSegment(p, q, r): 
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))): 
        return True
    return False
  
def orientation(p, q, r): 
    # to find the orientation of an ordered triplet (p,q,r) 
    # function returns the following values: 
    # 0 : Colinear points 
    # 1 : Clockwise points 
    # 2 : Counterclockwise 
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
    # for details of below formula.  
      
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1])) 
    if (val > 0): 
          
        # Clockwise orientation 
        return 1
    elif (val < 0): 
          
        # Counterclockwise orientation 
        return 2
    else: 
          
        # Colinear orientation 
        return 0
  
# The main function that returns true if  
# the line segment 'p1q1' and 'p2q2' intersect. 
def doIntersect(p1,q1,p2,q2): 
      
    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True
  
    # Special Cases 
  
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True
  
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True
  
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True
  
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True
  
    # If none of the cases 
    return False

##################################


