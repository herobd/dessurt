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
        self.point_pairs=None

        self.all_primitive_rects = [ np.array([[pred_bb_info[1].item(),pred_bb_info[2].item()],[pred_bb_info[3].item(),pred_bb_info[2].item()],[pred_bb_info[3].item(),pred_bb_info[4].item()],[pred_bb_info[1].item(),pred_bb_info[4].item()]]) ] #tl, tr, bt, bl
        self.all_angles = [pred_bb_info[5]]

    def merge(self,other):
        self.all_primitive_rects+=other.all_primitive_rects
        self.all_angles+=other.all_angles
        self.all_conf+=other.all_conf
        self.all_cls+=other.all_cls
        self.median_angle = None
        self.poly_points=None
        self.point_pairs=None

    def compute(self):
        self.median_angle = np.median(self.all_angles)

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
            primitive_rects = [ [[r[2][1],r[2][0]],[r[1][1],r[1][0]],[r[0][1],r[0][0]],[r[3][1],r[3][0]]] for r in self.all_primitive_rects]

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
        top_points=[primitive_rects_top[0][0]]
        last_point_top=primitive_rects_top[0][0]
        for rect in primitive_rects_top:
            if last_point_top[0]>rect[1][0]: #the prior rectangle extendes beyond this one
                if rect[0][1]<last_point_top[1]:
                    top_points.append( [rect[0][0],last_point_top[1]] )
                    top_points.append( rect[0] )
                    top_points.append( rect[1] )
                    top_points.append( [rect[1][0],last_point_top[1]] )
                #otherwise it doesn't matter
            else:
                if rect[0][0]>=last_point_top[0]: #this rectangle starts after the ast
                    top_points.append(rect[0]) #just add it
                elif rect[0][1]<last_point_top[1]: #if the rect/line im adding is above
                    if top_points[-2][0]>rect[0][0]: #if prior line was clipped, we need to work with the intersection point top_points[-2]
                        if top_points[-3][1]>rect[0][1]:
                            intersect_point = [rect[0][0],top_points[-2][1]]
                            top_points[-3]=intersect_point
                            top_points[-2]=rect[0]
                            top_points.pop()
                        else:
                            intersect_point=[top_points[-3][0],rect[0][1]]
                            top_points[-2]=intersect_point
                            top_points.pop()
                    else:
                        intersect_point = [rect[0][0],last_point_top[1]]
                        top_points[-1]=intersect_point
                        top_points.append(rect[0])
                elif rect[0][1]>last_point_top[1]: #if the rect/line im adding is below
                    if top_points[-2][0]>rect[0][0]:
                        if top_points[-3][1]>rect[0][1]:
                            last_two_points = top_points[-2:]
                            intersect_point_3 = [rect[0][0],top_points[-3][1]]
                            top_points[-3]=intersect_point_3
                            top_points[-2]=rect[0]
                            intersect_point_2 = [top_points[-2][0],rect[0][1]]
                            top_points[-1]=intersect_point_2
                            top_points+=last_two_points
                        intersection_point_1 = [top_points[-1][0],rect[1][1]]
                        top_points.append(intersection_point_1)

                    else:
                        intersect_point = [last_point_top[0],rect[0][1]]
                        top_points.append(intersect_point)
                else:
                    #it's straight across, we'll just average the two overlapped points to provide a good midpoint
                    top_points[-1]=[(top_points[-1][0]+rect[0][0])/2,(top_points[-1][1]+rect[0][1])/2]
            top_points.append(rect[1])
            last_point_top = top_points[-1]

        primitive_rects_bot = list(primitive_rects)
        primitive_rects_bot.sort(key=lambda rect:rect[3][0])
        bot_points=[primitive_rects_bot[0][3]]
        last_point_bot=primitive_rects_bot[0][3]
        for rect in primitive_rects_bot:
            if last_point_bot[0]>rect[2][0]: #the prior rectangle extendes beyond this one
                if rect[3][1]<last_point_bot[1]:
                    bot_points.append( [rect[3][0],last_point_bot[1]] )
                    bot_points.append( rect[3] )
                    bot_points.append( rect[2] )
                    bot_points.append( [rect[2][0],last_point_bot[1]] )
                #otherwise it doesn't matter
            else:
                if rect[3][0]>=last_point_bot[0]:
                    bot_points.append(rect[3])
                elif rect[3][1]>last_point_bot[1]:
                    if bot_points[-2][0]>rect[3][0]: #if prior line was clipped, we need to work with the intersection point bot_points[-2]
                        if bot_points[-3][1]<rect[3][1]:
                            intersect_point = [rect[3][0],bot_points[-2][1]]
                            bot_points[-3]=intersect_point
                            bot_points[-2]=rect[3]
                            bot_points.pop()
                        else:
                            intersect_point=[bot_points[-3][0],rect[3][1]]
                            bot_points[-2]=intersect_point
                            bot_points.pop()
                    else:
                        intersect_point = [rect[3][0],last_point_bot[1]]
                        bot_points[-1]=intersect_point
                        bot_points.append(rect[3])
                elif rect[3][1]<last_point_bot[1]:
                    if bot_points[-2][0]>rect[3][0]:
                        if bot_points[-3][1]<rect[3][1]:
                            last_two_points = bot_points[-2:]
                            intersect_point_3 = [rect[3][0],bot_points[-3][1]]
                            bot_points[-3]=intersect_point_3

                            bot_points[-2]=rect[3]

                            intersect_point_2 = [bot_points[-2][0],rect[3][1]]
                            bot_points[-1]=intersect_point_2
                            bot_points+=last_two_points
                        intersection_point_1 = [bot_points[-1][0],rect[2][1]]
                        bot_points.append(intersection_point_1)

                    else:
                        intersect_point = [last_point_bot[0],rect[3][1]]
                        bot_points.append(intersect_point)
                else:
                    #it's straight across, we'll just average the two overlapped points to provide a good midpoint
                    bot_points[-1]=[(bot_points[-1][0]+rect[3][0])/2,(bot_points[-1][1]+rect[3][1])/2]

            bot_points.append(rect[2])
            assert( bot_points[-1][1]<118 or  bot_points[-1][1]>119) #This point shouldn't be added, it's above where we are (somehow)
            last_point_bot = bot_points[-1]
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
        print('top ps {}'.format(top_points))
        print('bot ps {}'.format(bot_points))
        top_points=np.array(top_points)
        bot_points=np.array(bot_points)

        #smooth
        #for i in range(1,len(top_points)-1):
        #    new_top_points.append((top_points[i-1]+top_points[i]+top_points
        top_total_distance=0
        for i in range(len(top_points)-1):
            top_total_distance += math.sqrt((top_points[i][0]-top_points[i-1][0])**2 + (top_points[i][1]-top_points[i-1][1])**2)
        bot_total_distance=0
        for i in range(len(bot_points)-1):
            bot_total_distance += math.sqrt((bot_points[i][0]-bot_points[i-1][0])**2 + (bot_points[i][1]-bot_points[i-1][1])**2)

        #step size is average "height"
        step_size = np.linalg.norm(top_points.mean(axis=0)-bot_points.mean(axis=0))

        step_size_bot = (2*step_size/top_total_distance)/(1/top_total_distance + 1/bot_total_distance)
        step_size_top = (2*step_size/bot_total_distance)/(1/top_total_distance + 1/bot_total_distance)

        assert(abs(step_size_bot+step_size_top-2*step_size)<0.001)

        final_points=[]
        
        t_i=1
        b_i=1
        while t_i<len(top_points) or b_i<len(bot_points):
            top_points_in_step=[top_points[t_i-1]]
            bot_points_in_step=[bot_points[b_i-1]]
            top_distance = 0
            while (top_distance<step_size_top or t_i>=len(top_points)-2) and t_i<len(top_points):
                last_top_distance=np.linalg.norm(top_points_in_step[-1]-top_points[t_i])
                top_distance+=last_top_distance
                top_points_in_step.append(top_points[t_i])
                #print('t_i:{}, added {},\tdistance:{}'.format(t_i,top_points[t_i],top_distance))
                t_i+=1
            if top_distance-step_size>-(top_distance-last_top_distance-step_size) and t_i<len(top_points)-2:
                top_distance-=last_top_distance
                t_i-=1
                r=top_points_in_step.pop()
                #print('t_i:{}, removed {},\tdistance:{}'.format(t_i,r,top_distance))
            top_distance = top_distance-step_size #reset the top_distance with the residual to keep the two sides in sync

            bot_distance = 0
            while (bot_distance<step_size_bot or b_i>=len(bot_points)-2) and b_i<len(bot_points):
                last_bot_distance=np.linalg.norm(bot_points_in_step[-1]-bot_points[b_i])
                bot_distance+=last_bot_distance
                bot_points_in_step.append(bot_points[b_i])
                #print('b_i:{}, added {}, distance:{}'.format(b_i,bot_points[b_i],bot_distance))
                b_i+=1
            if bot_distance-step_size>-(bot_distance-last_bot_distance-step_size) and b_i<len(bot_points)-2:
                bot_distance-=last_bot_distance
                b_i-=1
                r=bot_points_in_step.pop()
                #print('b_i:{}, removed {}, distance:{}'.format(b_i,r,bot_distance))
            bot_distance = bot_distance-step_size #reset the bot_distance with the residual to keep the two sides in sync


            assert(len(top_points_in_step)>0 and len(bot_points_in_step)>0)
            #if len(top_points_in_step)>0:
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
            final_points.append([top_point,bot_point])

        #now we'll add the other end point
        mean_end_point = (top_points_in_step[-1]+bot_points_in_step[-1])/2
        mag_top = (mean_end_point[0]+(mean_end_point[1]-top_new_c)*s)/(1+s**2)
        top_end_point = [mag_top,mag_top*s+top_new_c]
        mag_bot = (mean_end_point[0]+(mean_end_point[1]-bot_new_c)*s)/(1+s**2)
        bot_end_point = [mag_bot,mag_bot*s+bot_new_c]
        final_points.append([top_end_point,bot_end_point])

        print('final pair points: {}'.format(final_points))
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

        top_points,bot_points = zip(*final_points)
        top_points=list(top_points)
        bot_points=list(bot_points)
        bot_points.reverse()
        self.poly_points = top_points+bot_points



    def old_compute(self):
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
                left_c = centroids[0][1] - slope_at_left_end*centroids[0][0]
                #identify corner lines
                #left_wall_interesction = (primitive_rects[0][0],slope*primitive_rects[0][0]+left_c)
                top_wall_intersection = (primitive_rects[0][0][1]-left_c)/slope_at_left_end
                bot_wall_intersection = (primitive_rects[0][3][1]-left_c)/slope_at_left_end
                left_end_x = max(primitive_rects[0][0][0],min(top_wall_intersection,bot_wall_intersection))

                #calculate cs (y intercept)
                left_c = centroids[0][1] - slope_at_left_end*centroids[0][0] #center line
                left_top_c = new_top_points[0][1] - slope_at_left_end*new_top_points[0][0] #top line

                left_wall_c_interesction_x = primitive_rects[0][0][0]
                left_wall_c_interesction_y = slope_at_left_end*left_wall_c_interesction_x+left_c
                left_c_cutoff_x = (left_wall_c_interesction_y+(1/slope_at_left_end)*left_wall_c_interesction_x -left_top_c)/(slope_at_left_end+1/slope_at_left_end)

                #calcuate middle line intersections
                top_wall_c_intersection_x = (primitive_rects[0][0][1]-left_c)/slope_at_left_end
                top_wall_c_intersection_y = slope_at_left_end*top_wall_c_intersection_x + left_c
                #top line at same distance
                top_c_cutoff_x = (top_wall_c_intersection_y+(1/slope_at_left_end)*top_wall_c_intersection_x -left_top_c)/(slope_at_left_end+1/slope_at_left_end)

                bot_wall_c_intersection_x = (primitive_rects[0][3][1]-left_c)/slope_at_left_end
                bot_wall_c_intersection_y = slope_at_left_end*bot_wall_c_intersection_x + left_c
                bot_c_cutoff_x = (bot_wall_c_intersection_y+(1/slope_at_left_end)*bot_wall_c_intersection_x -left_top_c)/(slope_at_left_end+1/slope_at_left_end)

                #top line intersections
                top_wall_top_intersection_x = (primitive_rects[0][0][1]-left_top_c)/slope_at_left_end
                bot_wall_top_intersection_x = (primitive_rects[0][3][1]-left_top_c)/slope_at_left_end

                left_end_x = max(
                        min(left_c_cutoff_x,max(top_c_cutoff_x,bot_c_cutoff_x)),
                        min(primitive_rects[0][0][0],max(top_wall_top_intersection_x,bot_wall_top_intersection_x)))
            else: #if is left wall
                left_end_x = primitive_rects[0][0][0]
            #extrapolate to endpoint
            left_end_y = slope_at_left_end*left_end_x + (new_top_points[0][1]-slope_at_left_end*new_top_points[0][0])
            print(left_end_y)

            #right endpoint calulation
            slope_at_right_end = (new_top_points[-1][1]-new_top_points[-2][1])/(new_top_points[-1][0]-new_top_points[-2][0])
            if abs(slope_at_right_end)>0.0001:
                #We'll calculate a few different possible stopping points at take the furthest
                # + the right wall of the BB
                # + this top line's intersection with the top and bottom walls of the BB
                # + clipping the top line parallel-wise (same distance) with the center line's intersection with the top and bottom walls
                print('cent :{}'.format(centroids[-1]))
                print(primitive_rects[-1])
                #calculate cs (y intercept)
                right_c = centroids[-1][1] - slope_at_right_end*centroids[-1][0]
                right_top_c = new_top_points[-1][1] - slope_at_right_end*new_top_points[-1][0]

                right_wall_c_interesction_x = primitive_rects[-1][1][0]
                right_wall_c_interesction_y = slope_at_right_end*right_wall_c_interesction_x+right_c
                right_c_cutoff_x = (right_wall_c_interesction_y+(1/slope_at_right_end)*right_wall_c_interesction_x -right_top_c)/(slope_at_right_end+1/slope_at_right_end)

                #calcuate middle line intersections
                top_wall_c_intersection_x = (primitive_rects[-1][1][1]-right_c)/slope_at_right_end
                top_wall_c_intersection_y = slope_at_right_end*top_wall_c_intersection_x + right_c
                #top line at same distance
                top_c_cutoff_x = (top_wall_c_intersection_y+(1/slope_at_right_end)*top_wall_c_intersection_x -right_top_c)/(slope_at_right_end+1/slope_at_right_end)

                bot_wall_c_intersection_x = (primitive_rects[-1][2][1]-right_c)/slope_at_right_end
                bot_wall_c_intersection_y = slope_at_right_end*bot_wall_c_intersection_x + right_c
                bot_c_cutoff_x = (bot_wall_c_intersection_y+(1/slope_at_right_end)*bot_wall_c_intersection_x -right_top_c)/(slope_at_right_end+1/slope_at_right_end)

                #top line intersections
                top_wall_top_intersection_x = (primitive_rects[-1][1][1]-right_top_c)/slope_at_right_end
                bot_wall_top_intersection_x = (primitive_rects[-1][2][1]-right_top_c)/slope_at_right_end

                right_end_x = max(
                        min(right_c_cutoff_x,max(top_c_cutoff_x,bot_c_cutoff_x)),
                        min(primitive_rects[-1][1][0],max(top_wall_top_intersection_x,bot_wall_top_intersection_x)))
            else: 
                #this is a flat line, we want to just us the horiontal extent of the last BB
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
