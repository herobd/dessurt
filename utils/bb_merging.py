import torch
import numpy as np
import math

#This assumes overget x1,y1,x2,y2,r prediciton
class ToMerge:
    def __init__(bb,include_bb_conf):
        self.include_bb_conf=include_bb_conf
        if include_bb_conf:
            self.locIdx=1
            self.classIdx=6
        else:
            self.locIdx=0
            self.classIdx=5
        #x0,y0,

    def merge(other):
        gertf

    def finalBB():
        #is this a horz, vert, or diagonal line?
        medAngle = np.median([bb[self.locIdx+2] for bb in self.bbs])

        #order points along angle. But we n

        if horz and upright:
            #get left most and right most
            pass

        #self.bbs.sort(key=lan...)
        return dfkgj


class TextLine:
    def __init__(self,pred_bb_info):
        pred_bb_info = pred_bb_info.cpu().detach()
        self.all_conf = [pred_bb_info[0]]
        self.all_cls = [pred_bb_info[6:]]

        self.median_angle = None
        self.poly_points=None

        self.all_primitive_rects = [ np.array([[pred_bb_info[1].item(),pred_bb_info[2].item()],[pred_bb_info[3].item(),pred_bb_info[2].item()],[pred_bb_info[3].item(),pred_bb_info[4].item()],[pred_bb_info[1].item(),pred_bb_info[4].item()]]) ] #tl, tr, bt, bl
        self.all_angles = [pred_bb_info[5]]

    def merge(self,other):
        self.all_primitive_rects+=other.all_primitive_rects
        self.all_angles+=other.all_angles
        self.all_conf+=other.all_conf
        self.all_cls+=other.all_cls
        self.median_angle = None
        self.poly_points=None

    def compute(self):
        self.median_angle = np.median(self.all_angles)
        primitive_rects = self.all_primitive_rects
        
        horz=upright=True

        if horz and upright:
            centriods = [rect.mean(axis=0) for rect in primitive_rects]
            combine = list(zip(centriods,primitive_rects))
            combine.sort(key=lambda c: c[0][0])
            centroids,primitive_rects = zip(* combine)
            top_points = [(rect[0]+rect[1])/2 for rect in primitive_rects]
            bot_points = [(rect[2]+rect[3])/2 for rect in primitive_rects]

            #average neighbors
            new_top_points = [(top_points[i]+top_points[i+1])/2 for i in range(len(top_points)-1)]
            new_bot_points = [(bot_points[i]+bot_points[i+1])/2 for i in range(len(bot_points)-1)]
            #average agian for points too close?
            height_based_thresh = (np.mean(new_bot_points,axis=0)[1] - np.mean(new_top_points,axis=0)[1])/5
            i=0
            while True:
                if np.linalg.norm(new_top_points[i]-new_top_points[i+1])<height_based_thresh:
                    new_top_points = new_top_points[:i-1]+[(new_top_points[i]+new_top_points[i+1])/2] + new_top_points[i+2:]
                i+=1
                if i >= len(new_top_points)-1:
                    break

            #now, calculate new endpoints based on angle created by last two points.
            ##TOP##

            #left endpoint calulation
            slope_at_left_end = (new_top_points[0][1]-new_top_points[1][1])/(new_top_points[0][0]-new_top_points[1][0])
            #assert(not math.isinf(slope_at_left_end))
            if abs(slope_at_left_end)>0.0001:
                print('cent :{}'.format(centroids[0]))
                print(primitive_rects[0])
                left_c = centroids[0][1] - slope_at_left_end*centroids[0][0]
                #identify corner lines
                #left_wall_interesction = (primitive_rects[0][0],slope*primitive_rects[0][0]+left_c)
                top_wall_intersection = (primitive_rects[0][0][1]-left_c)/slope_at_left_end
                bot_wall_intersection = (primitive_rects[0][3][1]-left_c)/slope_at_left_end
                print('top inter:{}, bot itner:{}, wall inter:{}'.format(top_wall_intersection,bot_wall_intersection,primitive_rects[0][0][0]))
                left_end_x = max(primitive_rects[0][0][0],min(top_wall_intersection,bot_wall_intersection))
            else: #if is left wall
                left_end_x = primitive_rects[0][0][0]
            #extrapolate to endpoint
            left_end_y = slope_at_left_end*left_end_x + (new_top_points[0][1]-slope_at_left_end*new_top_points[0][0])
            print(left_end_y)

            #right endpoint calulation
            slope_at_right_end = (new_top_points[-1][1]-new_top_points[-2][1])/(new_top_points[-1][0]-new_top_points[-2][0])
            if abs(slope_at_right_end)>0.0001:
                right_c = centroids[-1][1] - slope_at_right_end*centroids[-1][0]
                #identify corner lines
                #right_wall_interesction = (primitive_rects[0][0],slope*primitive_rects[0][0]+right_c)
                top_wall_intersection = (primitive_rects[-1][1][1]-right_c)/slope_at_right_end
                bot_wall_intersection = (primitive_rects[-1][2][1]-right_c)/slope_at_right_end
                right_end_x = min(primitive_rects[-1][1][0],max(top_wall_intersection,bot_wall_intersection))
            else: #if is right wall
                right_end_x = primitive_rects[-1][1][0]
            #extrapolate to endpoint
            right_end_y = slope_at_right_end*right_end_x + (new_top_points[-1][1]-slope_at_right_end*new_top_points[-1][0])

            new_top_points = [np.array([left_end_x,left_end_y])] + new_top_points + [np.array([right_end_x,right_end_y])]

            ###BOTTOM##

            #average agian for points too close?
            i=0
            while True:
                if np.linalg.norm(new_bot_points[i]-new_bot_points[i+1])<height_based_thresh:
                    new_bot_points = new_bot_points[:i-1]+[(new_bot_points[i]+new_bot_points[i+1])/2] + new_bot_points[i+2:]
                i+=1
                if i >= len(new_bot_points)-1:
                    break

            #left endpoint calulation
            slope_at_left_end = (new_bot_points[0][1]-new_bot_points[1][1])/(new_bot_points[0][0]-new_bot_points[1][0])
            if abs(slope_at_left_end)>0.0001:
                left_c = centroids[0][1] - slope_at_left_end*centroids[0][0]
                #identify corner lines
                #left_wall_interesction = (primitive_rects[0][0],slope*primitive_rects[0][0]+left_c)
                top_wall_intersection = (primitive_rects[0][0][1]-left_c)/slope_at_left_end
                bot_wall_intersection = (primitive_rects[0][3][1]-left_c)/slope_at_left_end
                left_end_x = max(primitive_rects[0][3][0],min(bot_wall_intersection,bot_wall_intersection))
            else: #if is left wall
                left_end_x = primitive_rects[0][3][0]
            #extrapolate to endpoint
            left_end_y = slope_at_left_end*left_end_x + (new_bot_points[0][1]-slope_at_left_end*new_bot_points[0][0])

            #right endpoint calulation
            slope_at_right_end = (new_bot_points[-1][1]-new_bot_points[-2][1])/(new_bot_points[-1][0]-new_bot_points[-2][0])
            if abs(slope_at_right_end)>0.0001:
                right_c = centroids[-1][1] - slope_at_right_end*centroids[-1][0]
                #identify corner lines
                #right_wall_interesction = (primitive_rects[0][0],slope*primitive_rects[0][0]+right_c)
                top_wall_intersection = (primitive_rects[-1][1][1]-right_c)/slope_at_right_end
                bot_wall_intersection = (primitive_rects[-1][2][1]-right_c)/slope_at_right_end
                right_end_x = min(primitive_rects[-1][2][0],max(bot_wall_intersection,bot_wall_intersection))
            else: #if is right wall
                right_end_x = primitive_rects[-1][2][0]
            #extrapolate to endpoint
            right_end_y = slope_at_right_end*right_end_x + (new_bot_points[-1][1]-slope_at_right_end*new_bot_points[-1][0])

            new_bot_points = [np.array([left_end_x,left_end_y])] + new_bot_points + [np.array([right_end_x,right_end_y])]

            new_bot_points.reverse()
            self.poly_points = new_top_points + new_bot_points

    def getReadPosition(self):
        if self.median_angle is None:
            self.compute()
        angle = self.median_angle
        slope = -math.tan(angle)
        xc = self.centriodX()
        yc = self.centriodY()

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

            if angle<math.pi/2 and angle>0:
                sign_x=1
                sign_y=1
            elif angle>math.pi/2 and angle<math.pi:
                sign_x=1
                sign_y=-1
            elif angle>-math.pi and angle<-math.pi/2:
                sign_x=-1
                sign_y=-1
            elif angle>-math.pi/2 and angle<0:
                sign_x=-1
                sign_y=1
            else:
                assert(False)



            t = b/(sign_y*math.sqrt(1/(1+slope**2))-slope*sign_x*math.sqrt(1/(1+(1/slope**2))))
            return t
        
    def polyYs(self):
        if self.poly_points is None:
            self.compute()
        return [p[1] for p in self.poly_points]
    def polyXs(self):
        if self.poly_points is None:
            self.compute()
        return [p[0] for p in self.poly_points]
    def polyPoints(self):
        if self.poly_points is None:
            self.compute()
        return self.poly_points
