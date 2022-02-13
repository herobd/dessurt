import numpy as np
import math

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def lineIntersection(lineA, lineB, threshA_low=10, threshA_high=10, threshB_low=10, threshB_high=10, both=False):
    a1=lineA[0]
    a2=lineA[1]
    b1=lineB[0]
    b2=lineB[1]
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    point = (num / denom.astype(float))*db + b1
    #check if it is on atleast one line segment
    vecA = da/np.linalg.norm(da)
    p_A = np.dot(point,vecA)
    a1_A = np.dot(a1,vecA)
    a2_A = np.dot(a2,vecA)

    vecB = db/np.linalg.norm(db)
    p_B = np.dot(point,vecB)
    b1_B = np.dot(b1,vecB)
    b2_B = np.dot(b2,vecB)
    
    ###rint('A:{},  B:{}, int p:{}'.format(lineA,lineB,point))
    ###rint('{:.0f}>{:.0f} and {:.0f}<{:.0f}  and/or  {:.0f}>{:.0f} and {:.0f}<{:.0f} = {} {} {}'.format((p_A+threshA_low),(min(a1_A,a2_A)),(p_A-threshA_high),(max(a1_A,a2_A)),(p_B+threshB_low),(min(b1_B,b2_B)),(p_B-threshB_high),(max(b1_B,b2_B)),(p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)),'and' if both else 'or',(p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B))))
    if both:
        if ( (p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)) and
             (p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B)) ):
            return point
    else:
        if ( (p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)) or
             (p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B)) ):
            return point
    return None


def intersection(bb1,bb2):
    points = bb1['poly_points']
    if bb1['type']=='fieldRow':
        line1 = np.array([(points[0]+points[3])/2,(points[1]+points[2])/2])
    else:
        line1 = np.array([(points[0]+points[1])/2,(points[3]+points[2])/2])
    points = bb2['poly_points']
    if bb2['type']=='fieldRow':
        line2 = np.array([(points[0]+points[3])/2,(points[1]+points[2])/2])
    else:
        line2 = np.array([(points[0]+points[1])/2,(points[3]+points[2])/2])
    return lineIntersection(line1,line2) is not None

def getAngle(poly):
    p1 = (poly[0]+poly[3])/2
    p2 = (poly[1]+poly[2])/2
    return math.atan2(p2[1]-p1[1],p2[0]-p1[0])
def getHorzReadPosition(poly,angle=None):
    new_poly = np.array([poly[3],poly[0],poly[1],poly[2]])
    return getVertReadPosition(new_poly)
def getVertReadPosition(poly,angle=None):
    if angle is None:
        angle = getAngle(poly)
    angle *= -1
    slope = -math.tan(angle)
    #xc,yc = self.center_point
    xc = poly[:,0].mean()
    yc = poly[:,1].mean()
    #print(f'slope {slope}, x:{xc}, y:{yc}')

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
        elif angle>=math.pi/2: # and angle<=math.pi:
            sign_x=1
            sign_y=-1
        elif angle>-math.pi/2 and angle<0:
            sign_x=-1
            sign_y=1
        elif angle<=-math.pi/2:
            sign_x=-1
            sign_y=-1
        else:
            assert(False)



        t = b/(sign_y*math.sqrt(1/(1+slope**2))-slope*sign_x*math.sqrt(1/(1+(1/slope**2))))
        return t.item()
def getHeight(poly):
    p1 = (poly[0]+poly[1])/2
    p2 = (poly[3]+poly[2])/2
    return math.sqrt(np.power(p2-p1,2).sum())

def putInReadOrder(a,a_poly,b,b_poly,final=False,height=None):
    angle = (getAngle(a_poly) + getAngle(b_poly))/2

    pos_a = getVertReadPosition(a_poly, angle)
    pos_b = getVertReadPosition(b_poly, angle)

    if height is None:
        height_a = getHeight(a_poly)
        height_b = getHeight(b_poly)
        height = max(height_a,height_b)
    diff = pos_b-pos_a
    thresh = 0.4*height

    #for horz comparison, they need to be relatively close horizontally
    #a_x1 = (a_poly[0,0]+a_poly[3,0])/2
    #a_x2 = (a_poly[1,0]+a_poly[2,0])/2
    #b_x1 = (b_poly[0,0]+b_poly[3,0])/2
    #b_x2 = (b_poly[1,0]+b_poly[2,0])/2
    #h_diff = min(abs(a_x1-b_x2),abs(a_x2-b_x1))

    #print('angle: {}'.format(angle))
    #print('red {}: {}'.format('horz' if final else 'vert',pos_a))
    #print('grn {}: {}'.format('horz' if final else 'vert',pos_b))
    #print('  thresh: {},  diff: {}'.format(thresh,diff))
    if final or abs(diff)>thresh: # or h_diff>3*height:
        if diff>0:
            return [a,b]
        else:
            return [b,a]
    else:
        new_poly_a = np.array([a_poly[3],a_poly[0],a_poly[1],a_poly[2]])
        new_poly_b = np.array([b_poly[3],b_poly[0],b_poly[1],b_poly[2]])
        return putInReadOrder(a,new_poly_a,b,new_poly_b,final=True)

def sortReadOrder(to_sort):
    #insertion sort, will only have short lists
    new_list=[]
    height=0
    for item,box in to_sort:
        if len(box)==4 and not isinstance(box,np.ndarray):
            x1,y1,x2,y2 = box
            box = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        else:
            if not isinstance(box,np.ndarray):
                box=np.array(box)
            if box.shape!=(4,2):
                box = np.reshape(box[:8],(4,2))
        height = max(height,getHeight(box))
    for item,box in to_sort:
        if len(box)==4 and not isinstance(box,np.ndarray):
            x1,y1,x2,y2 = box
            box = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        else:
            if not isinstance(box,np.ndarray):
                box=np.array(box)
            if box.shape!=(4,2):
                box = np.reshape(box[:8],(4,2))
            
        added = False
        for i,(other_item,other_box) in reversed(list(enumerate(new_list))):
            if putInReadOrder(False,box,True,other_box,height=height)[0]:
                new_list = new_list[:i+1]+[(item,box)]+new_list[i+1:]
                added=True
                break
        if not added:
            new_list=[(item,box)]+new_list
    return [i[0] for i in new_list]

def sameLine(box1,box2):

    h1=getHeight(box1)
    h2=getHeight(box2)
    v1=getVertReadPosition(box1)
    v2=getVertReadPosition(box2)

    return abs(v1-v2)<0.4*max(h1,h2)
