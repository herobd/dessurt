
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
        meanAngle = np.mean([bb[self.locIdx+2] for bb in self.bbs])

        self.bbs.sort(key=lan...)
        return dfkgj


class TextLine:
    def __init__(pred_bb_info):
        pred_bb_info = pred_bb_info.cpu().detach()
        self.conf = pred_bb_info[0]
        self.cls = pred_bb_info[...]

    def getReadPosition():
        angle = self.angle()
        slope = get mean angle (-180,180) and convert to slope
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
            if angle==0
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
        
    def polyYs():
        return np.array(...)
    def polyXs():
        return np.array(...)
