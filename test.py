import numpy as np

def polyIntersect(poly1, poly2):
    prevPoint = poly1[-1]
    for point in poly1:
        perpVec = np.array([ -(point[1]-prevPoint[1]), point[0]-prevPoint[0] ])
        perpVec = perpVec/np.linalg.norm(perpVec)

        maxPoly1=np.dot(perpVec,poly1[0])
        minPoly1=maxPoly1
        for p in poly1:
            p_onLine = np.dot(perpVec,p)
            maxPoly1 = max(maxPoly1,p_onLine)
            minPoly1 = min(minPoly1,p_onLine)
        maxPoly2=np.dot(perpVec,poly2[0])
        minPoly2=maxPoly2
        for p in poly2:
            p_onLine = np.dot(perpVec,p)
            maxPoly2 = max(maxPoly2,p_onLine)
            minPoly2 = min(minPoly2,p_onLine)

        print('{}<{} or {}>{} : {} or {}'.format(maxPoly1,minPoly2, minPoly1,maxPoly2,maxPoly1<minPoly2 , minPoly1>maxPoly2))
        if (maxPoly1<minPoly2 or minPoly1>maxPoly2):
            return False
        prevPoint = point
    return True


a=[[1909, 802], [2224, 804], [2222, 2203], [1911, 2203]]
b=[[382, 806], [2220, 806], [2220, 878], [382, 878]]
c=[[2489, 3122], [4324, 3128], [4324, 3198], [2492, 3192]]

print(polyIntersect(a,b))
print(polyIntersect(a,c))
