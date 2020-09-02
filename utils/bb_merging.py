import torch
import numpy as np
import math

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
                assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
            else:
                if rect[0][0]>=last_point_top[0]: #this rectangle starts after the ast
                    top_points.append(rect[0]) #just add it
                    assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
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
                        assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
                    else:
                        intersect_point = [rect[0][0],last_point_top[1]]
                        top_points[-1]=intersect_point
                        top_points.append(rect[0])
                        assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
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
                            assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
                        intersection_point_1 = [top_points[-1][0],rect[1][1]]
                        top_points.append(intersection_point_1)
                        assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))

                    else:
                        intersect_point = [last_point_top[0],rect[0][1]]
                        top_points.append(intersect_point)
                        assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
                #else:
                #    #it's straight across, we'll just average the two outside points to provide a good midpoint
                #    #import pdb;pdb.set_trace()
                #    top_points[-1]=[(top_points[-2][0]+rect[1][0])/2,(top_points[-2][1]+rect[1][1])/2]
                #    assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
            top_points.append(rect[1])
            assert(all([top_points[i][0]<=top_points[i+1][0] for i in range(len(top_points)-1)]))
            last_point_top = top_points[-1]

        primitive_rects_bot = list(primitive_rects)
        primitive_rects_bot.sort(key=lambda rect:rect[3][0])
        #remove duplicates (or near duplicates)
        toremove = [i for i in range(len(primitive_rects_bot)-1) if abs(primitive_rects_bot[i][3][0]-primitive_rects_bot[i+1][3][0])<0.001 and abs(primitive_rects_bot[i][3][1]-primitive_rects_bot[i+1][3][1])<0.001 and abs(primitive_rects_bot[i][2][0]-primitive_rects_bot[i+1][2][0])<0.001 and abs(primitive_rects_bot[i][2][1]-primitive_rects_bot[i+1][2][1])<0.001]
        toremove.reverse()
        for r in toremove:
            del primitive_rects_bot[r]
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
                #else:
                #    #it's straight across, we'll just average the two overlapped points to provide a good midpoint
                #    bot_points[-1]=[(bot_points[-2][0]+rect[2][0])/2,(bot_points[-2][1]+rect[2][1])/2]

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

        #step size is average "height"
        top_points_np=np.array(top_points)
        bot_points_np=np.array(bot_points)
        step_size = np.linalg.norm(top_points_np.mean(axis=0)-bot_points_np.mean(axis=0))
        top_points_np= bot_points_np= None

        step_size_top= (2*step_size/top_total_distance)/(1/top_total_distance + 1/bot_total_distance)
        step_size_bot = (2*step_size/bot_total_distance)/(1/top_total_distance + 1/bot_total_distance)

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
                    toadd.append((i,[x,y]))
        toadd.reverse()
        #print('adding {} points to top'.format(len(toadd)))
        for i,p in toadd:
            top_points.insert(i,p)

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
                    toadd.append((i,[x,y]))
        toadd.reverse()
        #print('adding {} points to bot'.format(len(toadd)))
        for i,p in toadd:
            bot_points.insert(i,p)

        top_points=np.array(top_points)
        bot_points=np.array(bot_points)

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

        self.center_point = self.poly_points.mean(axis=0)
        self.conf = np.mean(self.all_conf)
        self.cls = np.mean(self.all_cls)
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


    def getGrid(self,height):
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

            T = cv.getPerspectiveTransform([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],[tl,tr,br,bl])

            ys = torch.arange(height).view(height,1,1).expand(height,width,1)
            xs = torch.arange(width).view(1,width,1).expand(height,width,1)
            orig_points = torch.cat([xs,ys],dim=2)

            new_points = T.mm(orig_points)
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

        return (self.conf, self.center_point[0], self.center_point[1], self.median_angle, self.height, self.width, *self.point_pairs[0][0], *self.point_pairs[-1][0], *self.point_pairs[-1][1], *self.point_pairs[0][1], self.r_left, self.r_right, self.std_r, self.cls)
