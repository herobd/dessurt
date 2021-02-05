import torch
#from model.yolo_loss import bbox_iou
import math
import timeit
import numpy as np
from utils.forms_annotations import calcCornersTorch
from shapely.geometry import Polygon, LineString
import shapely
from utils.bb_merging import TextLine
from collections import defaultdict

def distancePoints(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
def distancePointLine(p,la,lb):
    return abs((la[1]-lb[1])*p[0]-(la[0]-lb[0])*p[1]+la[0]*lb[1]-la[1]*lb[0])/math.sqrt((la[1]-lb[1])**2 + (la[0]-lb[0])**2)

def non_max_sup_overseg(pred_boxes,thresh_conf=0.5, thresh_inter=0.5, hard_limit=300):
    return non_max_sup_(pred_boxes,thresh_conf, thresh_inter, verticle_bias_intersection, hard_limit)
def non_max_sup_iou(pred_boxes,thresh_conf=0.5, thresh_inter=0.5, hard_limit=300):
    return non_max_sup_(pred_boxes,thresh_conf, thresh_inter, max_intersection, hard_limit)
def non_max_sup_dist(pred_boxes,thresh_conf=0.5, thresh_dist=0.9, hard_limit=300):
    return non_max_sup_(pred_boxes,thresh_conf, thresh_dist*-1, dist_neg, hard_limit)
def non_max_sup_(pred_boxes,thresh_conf, thresh_loc, loc_metric, hard_limit):
    #rearr = [0,1,2,5,4,3]
    #for i in range(6,pred_boxes.shape[2]):
    #    rearr.append(i)
    #pred_boxes = pred_boxes[:,:,rearr]
    to_return=[]
    for b in range(pred_boxes.shape[0]):
        
        #allIOU = bbox_iou(
        above_thresh = []
        for i in range(pred_boxes.shape[1]):
            if pred_boxes[b,i,0]>thresh_conf:
                above_thresh.append( (pred_boxes[b,i,0], i) )
        above_thresh.sort(key=lambda a: a[0], reverse=True)
        above_thresh = above_thresh[:hard_limit]
        li = 0
        while li<len(above_thresh)-1:
            i=above_thresh[li][1]
            loc_measures = loc_metric(pred_boxes[b,i,1:6],pred_boxes[b,[x[1] for x in above_thresh[li+1:]],1:6])
            #ious = bbox_iou(pred_boxes[b,i:i+1,1:5],pred_boxes[b,[x[1] for x in above_thresh[li+1:]],1:5], x1y1x2y2=False)
            to_remove=[]
            for lj in range(len(above_thresh)-1,li,-1):
                j=above_thresh[lj][1]
                #if bbox_iou( pred_boxes[b,i:i+1,1:5], pred_boxes[b,j:j+1,1:5], x1y1x2y2=False) > thresh_iou:
                if loc_measures[lj-(li+1)] > thresh_loc:
                    to_remove.append(lj)
            #to_remove.reverse()
            for index in to_remove:
                del above_thresh[index]
            li+=1

        best = pred_boxes[b,[x[1] for x in above_thresh],:]
        to_return.append(best)#[:,rearr])
    return to_return

def non_max_sup_keep_overlap_iou(pred_boxes,thresh_conf, thresh_loc, hard_limit=9999):
    times_batch=[]
    times_above_thresh=[]
    times_all_iou=[]
    times_bridge_first=[]
    times_zip_sort=[]
    times_for_loc_measures=[]
    times_remove=[]
    times_bridge_second=[]

    to_return=[]
    for b in range(pred_boxes.shape[0]):
        tic_b=timeit.default_timer()

        
        #all_iou = ... or only on above_thresh?
        
        #TODO the creation of above_thresh could be faster
        above_thresh = []
        for i in range(pred_boxes.shape[1]):
            if pred_boxes[b,i,0]>thresh_conf:
                above_thresh.append( (pred_boxes[b,i,0], i) )
        above_thresh.sort(key=lambda a: a[0], reverse=True)
        above_thresh = above_thresh[:hard_limit]
        above_thresh = [x[1] for x in above_thresh]
        abv_thr_bbs = pred_boxes[b,above_thresh]

        times_above_thresh.append(timeit.default_timer()-tic_b)
        tic=timeit.default_timer()

        all_iou = allIOU(abv_thr_bbs,abv_thr_bbs,x1y1x2y2=True)

        times_all_iou.append(timeit.default_timer()-tic)
        tic=timeit.default_timer()

        adj_b = all_iou>0
        adj = adj_b.type(torch.IntTensor)
        bridged_to = torch.where(adj_b,torch.zeros_like(adj),torch.matmul(adj,adj)) #which bbs a bb is bridged to by another bb (but is not directly overlapping with)

        adj_b = None
        #this is some graph theory. adj^N=A is a matrix showing at A[i,j] how many paths there are from node i to node j
        #We use this to track which bbs are exclusive bridges, that is, they overlap two bbs and no other bb overlaps both of those bbs
        times_bridge_first.append(timeit.default_timer()-tic)

        li = 0
        to_remove_all = set()
        while li<len(above_thresh):
            #print('b:{}/{}, li:{}/{}'.format(b,pred_boxes.shape[0],li,len(above_thresh)))
            if li in to_remove_all:
                li+=1
                continue

            tic=timeit.default_timer()

            loc_measures = all_iou[li] #loc_metric(pred_boxes[b,i,1:6],pred_boxes[b,[x[1] for x in above_thresh[li+1:]],1:6])
            #loc_measures = list(enumerate(loc_measures)).sort(key=lambda a: a[1])
            loc_measures = list(zip(range(li+1,len(above_thresh)),loc_measures[li+1:],abv_thr_bbs[li+1:,0]))
            loc_measures.sort(key=lambda a: a[2])

            times_zip_sort.append(timeit.default_timer()-tic)
            tic=timeit.default_timer()

            to_remove=[]
            for loc_i,iou,loc_conf in loc_measures:
                if iou>thresh_loc:
                    #Is this an exclusive bridge?
                    bridging = (adj[loc_i]-adj[li])==1 #where this is 1, I am a bridge
                    if not ((bridged_to[li]==1) & bridging).any():
                        #loc_i is not an exlusive bridge
                        to_remove.append(loc_i)
                        bridged_to[li,bridging]-=1

            times_for_loc_measures.append(timeit.default_timer()-tic)
            tic=timeit.default_timer()
            
            if len(to_remove)>0:
                to_remove_all.update(to_remove)
                for index in to_remove:
                    #del above_thresh[index]
                    all_iou[index,:]=0
                    all_iou[:,index]=0

                times_remove.append(timeit.default_timer()-tic)
                tic=timeit.default_timer()

                #recompute bridging after removeal
                adj_b = all_iou>0
                adj = adj_b.type(torch.IntTensor)
                bridged_to = torch.where(adj_b,torch.zeros_like(adj),torch.matmul(adj,adj))
                adj_b = None

                times_bridge_second.append(timeit.default_timer()-tic)
            li+=1

    
        print('removing: {}.'.format(len(to_remove_all)))
        to_remove_all = list(to_remove_all)
        to_remove_all.sort(reverse=True)
        for index in to_remove_all:
            del above_thresh[index]
        best = pred_boxes[b,above_thresh,:]
        to_return.append(best)#[:,rearr])

        times_batch.append(timeit.default_timer()-tic_b)

    print('times_batch mean:{}, total:{}'.format(np.mean(times_batch),np.sum(times_batch)))
    print('times_above_thresh mean:{}, total:{}'.format(np.mean(times_above_thresh),np.sum(times_above_thresh)))
    print('times_all_iou mean:{}, total:{}'.format(np.mean(times_all_iou),np.sum(times_all_iou)))
    print('times_bridge_first mean:{}, total:{}'.format(np.mean(times_bridge_first),np.sum(times_bridge_first)))
    print('times_zip_sort mean:{}, total:{}'.format(np.mean(times_zip_sort),np.sum(times_zip_sort)))
    print('times_for_loc_measures mean:{}, total:{}'.format(np.mean(times_for_loc_measures),np.sum(times_for_loc_measures)))
    print('times_remove mean:{}, total:{}'.format(np.mean(times_remove),np.sum(times_remove)))
    print('times_bridge_second mean:{}, total:{}'.format(np.mean(times_bridge_second),np.sum(times_bridge_second)))

    return to_return

def non_max_sup_overseg(pred_boxes,thresh_iou=0.3,thresh_height_diff=0.2):
    ious = allIO_clipU(pred_boxes[:,1:],pred_boxes[:,1:],x1y1x2y2=True) #discard conf channel for iou
    heights = pred_boxes[:,4]-pred_boxes[:,2]
    heights1 = heights[None,:].expand(pred_boxes.size(0),pred_boxes.size(0))
    heights2 = heights[:,None].expand(pred_boxes.size(0),pred_boxes.size(0))
    diff_h = torch.abs(heights1-heights2)>thresh_height_diff*torch.min(heights1,heights2)
    ious = ious>thresh_iou
    in_conflict = torch.logical_and(diff_h,ious) #symetric
    # conf1>conf2... tensorize
    in_conflict = torch.triu(in_conflict,1) #not symetric
    to_remove = set()
    
    for a,b in torch.nonzero(in_conflict):
        if a not in to_remove and b not in to_remove:
            a_conf = pred_boxes[a,0]
            b_conf = pred_boxes[b,0]
            if a_conf>b_conf:
                to_remove.add(b.item())
            else:
                to_remove.add(a.item())
    keep = set(range(pred_boxes.size(0)))
    keep = keep-to_remove
    #import pdb;pdb.set_trace()

    #assert(len(keep) < pred_boxes.size(0))

    return pred_boxes[list(keep)]

#this is intended for the oversegmentation detector, where we care less about horizontal overlap (since these should be merged later on, and more about verticle overlap, since these should not be merged but discarded
def verticle_bias_intersection(query_box, candidate_boxes):
    q_x1, q_x2 = query_box[0]-query_box[4], query_box[0]+query_box[4]
    q_y1, q_y2 = query_box[1]-query_box[3], query_box[1]+query_box[3]
    q_yc = (q_y1+q_y2)/2
    q_h = q_y2-q_y1
    c_x1, c_x2 = candidate_boxes[:,0]-candidate_boxes[:,4], candidate_boxes[:,0]+candidate_boxes[:,4]
    c_y1, c_y2 = candidate_boxes[:,1]-candidate_boxes[:,3], candidate_boxes[:,1]+candidate_boxes[:,3]
    c_yc = (c_y1+c_y2)/2
    c_h = c_y2-c_y1

    inter_rect_x1 = torch.max(q_x1, c_x1)
    inter_rect_x2 = torch.min(q_x2, c_x2)
    inter_rect_y1 = torch.max(q_y1, c_y1)
    inter_rect_y2 = torch.min(q_y2, c_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0 )

    q_area = (q_x2 - q_x1 + 1) * (q_y2 - q_y1 + 1)
    c_area = (c_x2 - c_x1 + 1) * (c_y2 - c_y1 + 1)
    min_area = torch.min(q_area,c_area)
    #import pdb; pdb.set_trace()

    max_inter = inter_area/min_area
    #max_inter[(q_yc-c_yc).abs()>torch.min(
    #Apply bias
    mult=3
    max_inter *= (mult*(((q_yc-c_yc).abs()/torch.min(q_h,c_h))).pow(2)).clamp(min=0.5)
    return max_inter


def max_intersection(query_box, candidate_boxes):
    q_x1, q_x2 = query_box[0]-query_box[4], query_box[0]+query_box[4]
    q_y1, q_y2 = query_box[1]-query_box[3], query_box[1]+query_box[3]
    c_x1, c_x2 = candidate_boxes[:,0]-candidate_boxes[:,4], candidate_boxes[:,0]+candidate_boxes[:,4]
    c_y1, c_y2 = candidate_boxes[:,1]-candidate_boxes[:,3], candidate_boxes[:,1]+candidate_boxes[:,3]

    inter_rect_x1 = torch.max(q_x1, c_x1)
    inter_rect_x2 = torch.min(q_x2, c_x2)
    inter_rect_y1 = torch.max(q_y1, c_y1)
    inter_rect_y2 = torch.min(q_y2, c_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0 )

    q_area = (q_x2 - q_x1 + 1) * (q_y2 - q_y1 + 1)
    c_area = (c_x2 - c_x1 + 1) * (c_y2 - c_y1 + 1)
    min_area = torch.min(q_area,c_area)
    #import pdb; pdb.set_trace()

    return inter_area/min_area

def dist_neg(query_box, candidate_boxes):
    #convert boxes to points
    sin_r = torch.sin(query_box[2])
    cos_r = torch.cos(query_box[2])
    qlx = query_box[0] - cos_r*query_box[4]
    qly = query_box[1] + sin_r*query_box[3]
    qrx = query_box[0] + cos_r*query_box[4]
    qry = query_box[1] - sin_r*query_box[3]
    qtx = query_box[0] - cos_r*query_box[4]
    qty = query_box[1] - sin_r*query_box[3]
    qbx = query_box[0] + cos_r*query_box[4]
    qby = query_box[1] + sin_r*query_box[3]
    query_points = torch.tensor([[qlx,qly,qrx,qry,qtx,qty,qbx,qby]])
    queryHW = (query_box[4]+query_box[3])/2
    #queryHW = torch.min(query_box[3:5])

    query_points = query_points.expand(candidate_boxes.size(0),8)
    queryHW = queryHW.expand(candidate_boxes.size(0))

    sin_r = torch.sin(candidate_boxes[:,2])
    cos_r = torch.cos(candidate_boxes[:,2])
    clx = candidate_boxes[:,0] - cos_r*candidate_boxes[:,4]
    cly = candidate_boxes[:,1] + sin_r*candidate_boxes[:,3]
    crx = candidate_boxes[:,0] + cos_r*candidate_boxes[:,4]
    cry = candidate_boxes[:,1] - sin_r*candidate_boxes[:,3]
    ctx = candidate_boxes[:,0] - cos_r*candidate_boxes[:,4]
    cty = candidate_boxes[:,1] - sin_r*candidate_boxes[:,3]
    cbx = candidate_boxes[:,0] + cos_r*candidate_boxes[:,4]
    cby = candidate_boxes[:,1] + sin_r*candidate_boxes[:,3]
    cand_points = torch.stack([clx,cly,crx,cry,ctx,cty,cbx,cby],dim=1)
    candHW = (candidate_boxes[:,4]+candidate_boxes[:,3])/2
    #candHW,_ = torch.min(candidate_boxes[:,3:5],dim=1)
    #compute distances
    normalization = (queryHW+candHW)/2.0

    deltas = query_points - cand_points
    dist = ((
            torch.norm(deltas[:,0:2],2,1) +
            torch.norm(deltas[:,2:4],2,1) +
            torch.norm(deltas[:,4:6],2,1) +
            torch.norm(deltas[:,6:8],2,1)
           )/normalization)**2
    return dist*-1

def allIOU(boxes1,boxes2, boxes1XYWH=[0,1,4,3],x1y1x2y2=False):
    if x1y1x2y2:
        b1_x1=boxes1[:,0]
        b1_y1=boxes1[:,1]
        b1_x2=boxes1[:,2]
        b1_y2=boxes1[:,3]
        b2_x1=boxes2[:,0]
        b2_y1=boxes2[:,1]
        b2_x2=boxes2[:,2]
        b2_y2=boxes2[:,3]
    else:
        b1_x1, b1_x2 = boxes1[:,boxes1XYWH[0]]-boxes1[:,boxes1XYWH[2]], boxes1[:,boxes1XYWH[0]]+boxes1[:,boxes1XYWH[2]]
        b1_y1, b1_y2 = boxes1[:,boxes1XYWH[1]]-boxes1[:,boxes1XYWH[3]], boxes1[:,boxes1XYWH[1]]+boxes1[:,boxes1XYWH[3]]
        b2_x1, b2_x2 = boxes2[:,0]-boxes2[:,4], boxes2[:,0]+boxes2[:,4]
        b2_y1, b2_y2 = boxes2[:,1]-boxes2[:,3], boxes2[:,1]+boxes2[:,3]

    #expand to make two dimensional, allowing every instance of boxes1
    #to be compared with every intsance of boxes2
    b1_x1 = b1_x1[:,None].expand(boxes1.size(0), boxes2.size(0))
    b1_y1 = b1_y1[:,None].expand(boxes1.size(0), boxes2.size(0))
    b1_x2 = b1_x2[:,None].expand(boxes1.size(0), boxes2.size(0))
    b1_y2 = b1_y2[:,None].expand(boxes1.size(0), boxes2.size(0))
    b2_x1 = b2_x1[None,:].expand(boxes1.size(0), boxes2.size(0))
    b2_y1 = b2_y1[None,:].expand(boxes1.size(0), boxes2.size(0))
    b2_x2 = b2_x2[None,:].expand(boxes1.size(0), boxes2.size(0))
    b2_y2 = b2_y2[None,:].expand(boxes1.size(0), boxes2.size(0))

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0 )

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

def allIO_clipU(boxesT,boxesP, boxesPXYWH=[0,1,4,3],x1y1x2y2=False):
    if x1y1x2y2:
        bT_x1=boxesT[:,0]
        bT_y1=boxesT[:,1]
        bT_x2=boxesT[:,2]
        bT_y2=boxesT[:,3]
        bP_x1=boxesP[:,0]
        bP_y1=boxesP[:,1]
        bP_x2=boxesP[:,2]
        bP_y2=boxesP[:,3]
    else:
        bP_x1, bP_x2 = boxesP[:,boxesPXYWH[0]]-boxesP[:,boxesPXYWH[2]], boxesP[:,boxesPXYWH[0]]+boxesP[:,boxesPXYWH[2]]
        bP_y1, bP_y2 = boxesP[:,boxesPXYWH[1]]-boxesP[:,boxesPXYWH[3]], boxesP[:,boxesPXYWH[1]]+boxesP[:,boxesPXYWH[3]]
        bT_x1, bT_x2 = boxesT[:,0]-boxesT[:,4], boxesT[:,0]+boxesT[:,4]
        bT_y1, bT_y2 = boxesT[:,1]-boxesT[:,3], boxesT[:,1]+boxesT[:,3]

    #expand to make two dimensional, allowing every instance of boxesP
    #to be compared with every intsance of boxesT
    bT_x1 = bT_x1[:,None].expand(boxesT.size(0), boxesP.size(0))
    bT_y1 = bT_y1[:,None].expand(boxesT.size(0), boxesP.size(0))
    bT_x2 = bT_x2[:,None].expand(boxesT.size(0), boxesP.size(0))
    bT_y2 = bT_y2[:,None].expand(boxesT.size(0), boxesP.size(0))
    bP_x1 = bP_x1[None,:].expand(boxesT.size(0), boxesP.size(0))
    bP_y1 = bP_y1[None,:].expand(boxesT.size(0), boxesP.size(0))
    bP_x2 = bP_x2[None,:].expand(boxesT.size(0), boxesP.size(0))
    bP_y2 = bP_y2[None,:].expand(boxesT.size(0), boxesP.size(0))

    inter_rect_x1 = torch.max(bP_x1, bT_x1)
    inter_rect_x2 = torch.min(bP_x2, bT_x2)
    inter_rect_y1 = torch.max(bP_y1, bT_y1)
    inter_rect_y2 = torch.min(bP_y2, bT_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0 )

    bP_area = (bP_x2 - bP_x1 + 1) * (bP_y2 - bP_y1 + 1)
    #clip target region by pred region
    bT_clippedArea = (inter_rect_x2 - inter_rect_x1 + 1) * (bT_y2 - bT_y1 + 1)

    #iou = inter_area / (bP_area + bT_area - inter_area + 1e-16)
    io_clipped_u = inter_area / (bP_area + bT_clippedArea - inter_area + 1e-16)

    #gt_r = rboxes[:,None,2].expand(rboxes.size(0), len(bbs))
    #pr_allRs = pr_allRs[None,:].expand(rboxes.size(0), len(bbs))
    #angle_compatible = torch.abs(gt_r-pr_allRs)
    #angle_compatible[angle_compatible>math.pi]-=math.pi
    #angle_compatible[angle_compatible>math.pi]-=math.pi
    #angle_compatible = angle_compatible.abs()<math.pi/3
    #iou *= angle_compatible

    #
    #gt_cls_ind = torch.argmax(rboxes[:,13:],dim=1)
    #pr_allClss = torch.FloatTensor([bb.getCls() for bb in bbs])
    ##pr_allClss = torch.stack([bb.getCls() for bb in bbs],dim=0)
    #pr_cls_int = torch.argmax(pr_allClss,dim=1)
    #gt_cls_ind = gt_cls_ind[:,None].expand(rboxes.size(0), len(bbs))
    #pr_cls_int = pr_cls_int[None,:].expand(rboxes.size(0), len(bbs))
    #class_compatible = gt_cls_ind==pr_cls_int
    #iou *= class_compatible
    return io_clipped_u
def classIOU(boxesT,boxesP, num_classes, boxesPXYWH=[0,1,4,3]):
    bP_x1, bP_x2 = boxesP[:,boxesPXYWH[0]]-boxesP[:,boxesPXYWH[2]], boxesP[:,boxesPXYWH[0]]+boxesP[:,boxesPXYWH[2]]
    bP_y1, bP_y2 = boxesP[:,boxesPXYWH[1]]-boxesP[:,boxesPXYWH[3]], boxesP[:,boxesPXYWH[1]]+boxesP[:,boxesPXYWH[3]]
    bT_x1, bT_x2 = boxesT[:,0]-boxesT[:,4], boxesT[:,0]+boxesT[:,4]
    bT_y1, bT_y2 = boxesT[:,1]-boxesT[:,3], boxesT[:,1]+boxesT[:,3]

    #expand to make two dimensional, allowing every instance of boxesP
    #to be compared with every intsance of boxesT
    bT_x1 = bT_x1[:,None].expand(boxesT.size(0), boxesP.size(0))
    bT_y1 = bT_y1[:,None].expand(boxesT.size(0), boxesP.size(0))
    bT_x2 = bT_x2[:,None].expand(boxesT.size(0), boxesP.size(0))
    bT_y2 = bT_y2[:,None].expand(boxesT.size(0), boxesP.size(0))
    bP_x1 = bP_x1[None,:].expand(boxesT.size(0), boxesP.size(0))
    bP_y1 = bP_y1[None,:].expand(boxesT.size(0), boxesP.size(0))
    bP_x2 = bP_x2[None,:].expand(boxesT.size(0), boxesP.size(0))
    bP_y2 = bP_y2[None,:].expand(boxesT.size(0), boxesP.size(0))

    inter_rect_x1 = torch.max(bP_x1, bT_x1)
    inter_rect_x2 = torch.min(bP_x2, bT_x2)
    inter_rect_y1 = torch.max(bP_y1, bT_y1)
    inter_rect_y2 = torch.min(bP_y2, bT_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0 )

    bP_area = (bP_x2 - bP_x1 + 1) * (bP_y2 - bP_y1 + 1)
    bT_area = (bT_x2 - bT_x1 + 1) * (bT_y2 - bT_y1 + 1)

    iou = inter_area / (bP_area + bT_area - inter_area + 1e-16)

    #
    if num_classes>0:
        gt_cls_ind = torch.argmax(boxesT[:,13:13+num_classes],dim=1) #13:
        pr_cls_ind = torch.argmax(boxesP[:,5:5+num_classes],dim=1) #5+numN:
        gt_cls_ind = gt_cls_ind[:,None].expand(boxesT.size(0), boxesP.size(0))
        pr_cls_ind = pr_cls_ind[None,:].expand(boxesT.size(0), boxesP.size(0))
        class_compatible = gt_cls_ind==pr_cls_ind
        iou *= class_compatible
    #target[0] pred[8]?
    return iou


def allPolyIO_clipU(rboxes,bbs):

    #first we'll find encompassing boxes and compute IOU based on those (fast)
    #for intersections, we'll compute the intersection using polygons (slow)
    #rboxes = x,y,r,h,w
    gt_tlX,gt_tlY,gt_trX,gt_trY,gt_brX,gt_brY,gt_blX,gt_blY = calcCornersTorch(rboxes[:,0],rboxes[:,1],rboxes[:,2],rboxes[:,3],rboxes[:,4])
    gt_Xs = (gt_tlX,gt_tlX,gt_brX,gt_blX)
    gt_Ys = (gt_tlY,gt_tlY,gt_brY,gt_blY)
    gt_x1,_ = torch.stack(gt_Xs).min(dim=0)
    gt_x2,_ = torch.stack(gt_Xs).max(dim=0)
    gt_y1,_ = torch.stack(gt_Ys).min(dim=0)
    gt_y2,_ = torch.stack(gt_Ys).max(dim=0)

    pr_x1 = torch.FloatTensor([min(bb.polyXs()) for bb in bbs])
    pr_x2 = torch.FloatTensor([max(bb.polyXs()) for bb in bbs])
    pr_y1 = torch.FloatTensor([min(bb.polyYs()) for bb in bbs])
    pr_y2 = torch.FloatTensor([max(bb.polyYs()) for bb in bbs])

    pr_allRs = torch.FloatTensor([bb.medianAngle() for bb in bbs])

    #expand to make two dimensional, allowing every instance of boxesP
    #to be compared with every intsance of boxesT
    gt_x1 = gt_x1[:,None].expand(rboxes.size(0), len(bbs))
    gt_y1 = gt_y1[:,None].expand(rboxes.size(0), len(bbs))
    gt_x2 = gt_x2[:,None].expand(rboxes.size(0), len(bbs))
    gt_y2 = gt_y2[:,None].expand(rboxes.size(0), len(bbs))
    pr_x1 = pr_x1[None,:].expand(rboxes.size(0), len(bbs))
    pr_y1 = pr_y1[None,:].expand(rboxes.size(0), len(bbs))
    pr_x2 = pr_x2[None,:].expand(rboxes.size(0), len(bbs))
    pr_y2 = pr_y2[None,:].expand(rboxes.size(0), len(bbs))

    inter_rect_x1 = torch.max(pr_x1, gt_x1)
    inter_rect_x2 = torch.min(pr_x2, gt_x2)
    inter_rect_y1 = torch.max(pr_y1, gt_y1)
    inter_rect_y2 = torch.min(pr_y2, gt_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0 )

    pr_area = (pr_x2 - pr_x1 + 1) * (pr_y2 - pr_y1 + 1)
    gt_area = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)

    iou = inter_area / (pr_area + gt_area - inter_area + 1e-16)

    #we'll only match bbs with roughly the right orientation
    #we allow upsidedown orientation matches
    gt_r = rboxes[:,None,2].expand(rboxes.size(0), len(bbs))
    pr_allRs = pr_allRs[None,:].expand(rboxes.size(0), len(bbs))
    angle_compatible = torch.abs(gt_r-pr_allRs)
    angle_compatible[angle_compatible>math.pi]-=math.pi
    angle_compatible[angle_compatible>math.pi]-=math.pi
    angle_compatible = angle_compatible.abs()<math.pi/3
    iou *= angle_compatible


    #we compute clipped IOU. Here we "clip" the GT area horizontally (w.r.t. GT orientation) to match it's overlap with the prediction. Thus the IOU is not penalized for not coverving all the GT
    io_clipped_u = torch.zeros_like(iou)
        
    for i,j in torch.nonzero(iou>0.001,as_tuple=False):
        gt_poly = Polygon([[gt_tlX[i].item(),gt_tlY[i].item()],[gt_trX[i].item(),gt_trY[i].item()],[gt_brX[i].item(),gt_brY[i].item()],[gt_blX[i].item(),gt_blY[i].item()]])
        pr_poly = Polygon(bbs[j].polyPoints())
        
        #try:
        if pr_poly.is_valid:
            inter = gt_poly.intersection(pr_poly)
            #iou[i,j] = inter.area/(gt_poly.area+pr_poly.area-inter.area)
            
            if inter.area>0:
                length = getWidthWRT(inter,[(gt_tlX[i],gt_tlY[i]),(gt_trX[i],gt_trY[i])])

                #compute the area of the clipped gt
                clipped_gt_area = length*rboxes[i,3] #new width*height


                io_clipped_u[i,j] = inter.area/(clipped_gt_area+pr_poly.area-inter.area)
        #except shapely.errors.TopologicalError:
        #    print(bbs[j].pairPoints())
        #    import pdb;pdb.set_trace()
        #    io_clipped_u[i,j]=0
    return io_clipped_u
    #if ret_clip:
    #    return iou, io_clipped_u
    #else:
    #    return iou

#this finds the width of the polygon(s) w.r.t. to the given line (defines horizontal)
#Clips the width according to the line to give proper overlap region for intersection-over-clipped-union
def getWidthWRT(inter,line):
    line = np.array(line)
    p0=line[0]
    v=(line[1]-p0)[None,:]
    if type(inter) is shapely.geometry.multipolygon.MultiPolygon:
        points=[]
        for poly in inter:
            points += list(poly.exterior.coords)
        points = np.array(points)
    elif type(inter) is shapely.geometry.collection.GeometryCollection:
        points=[]
        for shape in inter:
            if type(shape) is shapely.geometry.linestring.LineString:
                points += list(shape.coords)
            elif type(shape) is shapely.geometry.point.Point:
                points.append(list(shape.coords)[0])
            else:
                points += list(shape.exterior.coords)
        points = np.array(points)
    else:
        points = np.array(inter.exterior.coords)
    #print('points: {}'.format(points))
    points -= p0
    projs=np.matmul(np.matmul(v.transpose(),v)/np.matmul(v,v.transpose()),points.transpose())
    ds=(projs**2).sum(axis=0)**0.5
    vd=np.matmul(v,v.transpose())**0.5
    #print('line: {}'.format(line))
    #print('length: {}'.format(min(ds.max(),vd)-max(ds.min(),0)))
    #import pdb;pdb.set_trace()
    return min(ds.max(),vd.item())-max(ds.min(),0)

def getWidth(inter):
    #We'll find points spanning the length of the intersection using distances
    #iterate over points of intersection poly and find closest to topline and botline
    #iterate from those points to the furthest from both (both iteratins same direction)
    #distance between these last two poings is length.
    #TODO this doesn't work. I'll average with the longest distance between any two vertices, but at somepoint I should fix this to be proper
    min_dist_top=min_dist_bot=9999999
    points = list(inter.exterior.coords)
    max_dist = 0
    for p_i,p in enumerate(points):
        #distance from top line to point x,y
        dist_top = distancePointLine(p,(gt_tlX[i],gt_tlY[i]),(gt_trX[i],gt_trY[i]))
        #dist_top = abs((gt_tlY[i]-gt_trY[i])*x-(gt_tlX[i]-gt_trX[i])*y+gt_tlX[i]*gt_trY[i]-gt_tlY[i]*gt_trX[i])/math.sqrt((gt_tlY[i]-gt_trY[i])**2 + (gt_tlX[i]-gt_trX[i])**2)
        if dist_top<min_dist_top:
            min_dist_top=dist_top
            top_point_i = p_i
        dist_bot = distancePointLine(p,(gt_blX[i],gt_blY[i]),(gt_brX[i],gt_brY[i]))
        #dist_bot = abs((gt_blY[i]-gt_brY[i])*x-(gt_blX[i]-gt_brX[i])*y+gt_blX[i]*gt_brY[i]-gt_blY[i]*gt_brX[i])/math.sqrt((gt_blY[i]-gt_brY[i])**2 + (gt_blX[i]-gt_brX[i])**2)
        if dist_bot<min_dist_bot:
            min_dist_bot=dist_bot
            bot_point_i = p_i

        for p2 in points[p_i+1:]:
            dist = distancePoints(p,p2)
            max_dist = max(max_dist,dist)
    min_dist_diff=9999999
    p_j = (top_point_i+1)%(len(points)-1) #-1 as Polygon repeats closure vertex
    if p_j == bot_point_i:
        side_point_A = ((points[top_point_i][0]+points[bot_point_i][0])/2,(points[top_point_i][1]+points[bot_point_i][1])/2)
    else:
        while p_j!=bot_point_i:
            dist_diff = abs(distancePointLine(points[p_j],(gt_tlX[i],gt_tlY[i]),(gt_trX[i],gt_trY[i]))-distancePointLine(points[p_j],(gt_blX[i],gt_blY[i]),(gt_brX[i],gt_brY[i])))
            dist_diff += abs(distancePoints(points[p_j],points[top_point_i])-distancePoints(points[p_j],points[bot_point_i]))*0.1
            if dist_diff < min_dist_diff:
                min_dist_diff = dist_diff
                side_point_A = points[p_j]
            p_j = (p_j+1)%(len(points)-1)
    min_dist_diff=9999999
    if p_j == top_point_i:
        side_point_B = ((points[top_point_i][0]+points[bot_point_i][0])/2,(points[top_point_i][1]+points[bot_point_i][1])/2)
    else:
        while p_j!=top_point_i:
            dist_diff = abs(distancePointLine(points[p_j],(gt_tlX[i],gt_tlY[i]),(gt_trX[i],gt_trY[i]))-distancePointLine(points[p_j],(gt_blX[i],gt_blY[i]),(gt_brX[i],gt_brY[i])))
            dist_diff += abs(distancePoints(points[p_j],points[top_point_i])-distancePoints(points[p_j],points[bot_point_i]))*0.1
            if dist_diff < min_dist_diff:
                min_dist_diff = dist_diff
                side_point_B = points[p_j]
            p_j = (p_j+1)%(len(points)-1)
    length_i = distancePoints(side_point_A,side_point_B)
    length = (length_i+max_dist)/2

    #print(points)
    #print('length: {}, length_i:{}, max_dist:{}, top: {}, bot: {}, A: {}, B: {}'.format(length,length_i,max_dist,top_point_i,bot_point_i,side_point_A,side_point_B))
    #import pdb;pdb.set_trace()

def classPolyIOU(rboxes,bbs,num_classes):

    #first we'll find encompassing boxes and compute IOU based on those (fast)
    #for intersections, we'll compute the intersection using polygons (slow)
    #rboxes = x,y,r,h,w
    gt_tlX,gt_tlY,gt_trX,gt_trY,gt_brX,gt_brY,gt_blX,gt_blY = calcCornersTorch(rboxes[:,0],rboxes[:,1],rboxes[:,2],rboxes[:,3],rboxes[:,4])
    gt_Xs = (gt_tlX,gt_tlX,gt_brX,gt_blX)
    gt_Ys = (gt_tlY,gt_tlY,gt_brY,gt_blY)
    gt_x1,_ = torch.stack(gt_Xs).min(dim=0)
    gt_x2,_ = torch.stack(gt_Xs).max(dim=0)
    gt_y1,_ = torch.stack(gt_Ys).min(dim=0)
    gt_y2,_ = torch.stack(gt_Ys).max(dim=0)

    pr_x1 = torch.FloatTensor([min(bb.polyXs()) for bb in bbs])
    pr_x2 = torch.FloatTensor([max(bb.polyXs()) for bb in bbs])
    pr_y1 = torch.FloatTensor([min(bb.polyYs()) for bb in bbs])
    pr_y2 = torch.FloatTensor([max(bb.polyYs()) for bb in bbs])

    pr_allRs = torch.FloatTensor([bb.medianAngle() for bb in bbs])

    #expand to make two dimensional, allowing every instance of boxesP
    #to be compared with every intsance of boxesT
    gt_x1 = gt_x1[:,None].expand(rboxes.size(0), len(bbs))
    gt_y1 = gt_y1[:,None].expand(rboxes.size(0), len(bbs))
    gt_x2 = gt_x2[:,None].expand(rboxes.size(0), len(bbs))
    gt_y2 = gt_y2[:,None].expand(rboxes.size(0), len(bbs))
    pr_x1 = pr_x1[None,:].expand(rboxes.size(0), len(bbs))
    pr_y1 = pr_y1[None,:].expand(rboxes.size(0), len(bbs))
    pr_x2 = pr_x2[None,:].expand(rboxes.size(0), len(bbs))
    pr_y2 = pr_y2[None,:].expand(rboxes.size(0), len(bbs))

    inter_rect_x1 = torch.max(pr_x1, gt_x1)
    inter_rect_x2 = torch.min(pr_x2, gt_x2)
    inter_rect_y1 = torch.max(pr_y1, gt_y1)
    inter_rect_y2 = torch.min(pr_y2, gt_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0 )

    pr_area = (pr_x2 - pr_x1 + 1) * (pr_y2 - pr_y1 + 1)
    gt_area = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)

    iou = inter_area / (pr_area + gt_area - inter_area + 1e-16)

    #we'll only match bbs with roughly the right orientation
    #we allow upsidedown orientation matches
    gt_r = rboxes[:,None,2].expand(rboxes.size(0), len(bbs))
    pr_allRs = pr_allRs[None,:].expand(rboxes.size(0), len(bbs))
    angle_compatible = torch.abs(gt_r-pr_allRs)
    angle_compatible[angle_compatible>math.pi]-=math.pi
    angle_compatible[angle_compatible>math.pi]-=math.pi
    angle_compatible = angle_compatible.abs()<math.pi/3
    iou *= angle_compatible

    if len(bbs)>0 and num_classes>0:
        gt_cls_ind = torch.argmax(rboxes[:,13:13+num_classes],dim=1)
        pr_allClss = torch.FloatTensor([bb.getCls()[:num_classes] for bb in bbs]) #discard extra classes
        #pr_allClss = torch.stack([bb.getCls() for bb in bbs],dim=0)
        pr_cls_int = torch.argmax(pr_allClss,dim=1)
        gt_cls_ind = gt_cls_ind[:,None].expand(rboxes.size(0), len(bbs))
        pr_cls_int = pr_cls_int[None,:].expand(rboxes.size(0), len(bbs))
        class_compatible = gt_cls_ind==pr_cls_int
        iou *= class_compatible

    for i,j in torch.nonzero(iou>0.001):
        gt_poly = Polygon([[gt_tlX[i].item(),gt_tlY[i].item()],[gt_trX[i].item(),gt_trY[i].item()],[gt_brX[i].item(),gt_brY[i].item()],[gt_blX[i].item(),gt_blY[i].item()]])
        pr_poly = Polygon(bbs[j].polyPoints())
        
        #try:
        if pr_poly.is_valid:
            inter = gt_poly.intersection(pr_poly)
            iou[i,j] = inter.area/(gt_poly.area+pr_poly.area-inter.area)
        else:
            iou[i,j]=0
        #except shapely.errors.TopologicalError:
    return iou

def allDist(boxes1,boxes2):
    b1_x = boxes1[:,0]
    b1_y = boxes1[:,1]
    b2_x = boxes2[:,0]
    b2_y = boxes2[:,1]

    #expand to make two dimensional, allowing every instance of boxes1
    #to be compared with every intsance of boxes2
    b1_x = b1_x[:,None].expand(boxes1.size(0), boxes2.size(0))
    b1_y = b1_y[:,None].expand(boxes1.size(0), boxes2.size(0))
    b2_x = b2_x[None,:].expand(boxes1.size(0), boxes2.size(0))
    b2_y = b2_y[None,:].expand(boxes1.size(0), boxes2.size(0))

    return torch.sqrt( torch.pow(b1_x-b2_x,2) + torch.pow(b1_y-b2_y,2) )

def allBoxDistNeg(boxes1,boxes2):
    #convert boxes to points
    sin_r = torch.sin(boxes1[:,2])
    cos_r = torch.cos(boxes1[:,2])
    clx = boxes1[:,0] - cos_r*boxes1[:,4]
    cly = boxes1[:,1] + sin_r*boxes1[:,3]
    crx = boxes1[:,0] + cos_r*boxes1[:,4]
    cry = boxes1[:,1] - sin_r*boxes1[:,3]
    ctx = boxes1[:,0] - cos_r*boxes1[:,4]
    cty = boxes1[:,1] - sin_r*boxes1[:,3]
    cbx = boxes1[:,0] + cos_r*boxes1[:,4]
    cby = boxes1[:,1] + sin_r*boxes1[:,3]
    boxes1_points = torch.stack([clx,cly,crx,cry,ctx,cty,cbx,cby],dim=1)
    boxes1HW = (boxes1[:,4]+boxes1[:,3])/2


    sin_r = torch.sin(boxes2[:,2])
    cos_r = torch.cos(boxes2[:,2])
    clx = boxes2[:,0] - cos_r*boxes2[:,4]
    cly = boxes2[:,1] + sin_r*boxes2[:,3]
    crx = boxes2[:,0] + cos_r*boxes2[:,4]
    cry = boxes2[:,1] - sin_r*boxes2[:,3]
    ctx = boxes2[:,0] - cos_r*boxes2[:,4]
    cty = boxes2[:,1] - sin_r*boxes2[:,3]
    cbx = boxes2[:,0] + cos_r*boxes2[:,4]
    cby = boxes2[:,1] + sin_r*boxes2[:,3]
    boxes2_points = torch.stack([clx,cly,crx,cry,ctx,cty,cbx,cby],dim=1)
    boxes2HW = (boxes2[:,4]+boxes2[:,3])/2
    #candHW,_ = torch.min(candidate_boxes[:,3:5],dim=1)
    #compute distances

    boxes1_points = boxes1_points[:,None,:].expand(boxes1.size(0),boxes2.size(0),8)
    boxes2_points = boxes2_points[None,:,:].expand(boxes1.size(0),boxes2.size(0),8)
    boxes1HW = boxes1HW[:,None].expand(boxes1.size(0),boxes2.size(0))
    boxes2HW = boxes2HW[None,:].expand(boxes1.size(0),boxes2.size(0))
    normalization = (boxes1HW+boxes2HW)/2.0

    deltas = boxes1_points - boxes2_points
    dist = ((
        torch.norm(deltas[:,:,0:2],2,2) +
        torch.norm(deltas[:,:,2:4],2,2) +
        torch.norm(deltas[:,:,4:6],2,2) +
        torch.norm(deltas[:,:,6:8],2,2)
           )/normalization)**2
    return dist*-1
 
#input is tensors of shape [instance,(conf,x,y,rot,h,w)]
def AP_iou(target,pred,iou_thresh,numClasses=2,ignoreClasses=False,beforeCls=0,getClassAP=False):
    return AP_(target,pred,iou_thresh,numClasses,ignoreClasses,beforeCls,allIOU,getClassAP)
def AP_dist(target,pred,dist_thresh,numClasses=2,ignoreClasses=False,beforeCls=0,getClassAP=False):
    return AP_(target,pred,-dist_thresh,numClasses,ignoreClasses,beforeCls,allBoxDistNeg,getClassAP)
def AP_(target,pred,iou_thresh,numClasses,ignoreClasses,beforeCls,getLoc,getClassAP):
    #mAP=0.0
    #aps=[]
    precisions=[]
    recalls=[]

    #how many classes are there?
    if ignoreClasses:
        numClasses=1
    if len(target.size())>1 and target.size(0)>0:
        #numClasses=target.size(1)-13
        pass
    elif pred is not None and len(pred.size())>1 and pred.size(0)>0:
        #if there are no targets, we shouldn't be pred anything
        if ignoreClasses:
            #aps.append(0)
            ap=0.0
            precisions.append(0.0)
            recalls.append(1.0)
        else:
            #numClasses=pred.size(1)-6
            ap=0
            class_ap=[]
            for cls in range(numClasses):
                if (torch.argmax(pred[:,beforeCls+6:],dim=1)==cls).any():
                    #aps.append(0) #but we did for this class :(
                    ap+=0.0
                    precisions.append(0.0)
                    class_ap.append(0.0)
                else:
                    #aps.append(1) #we didn't for this class :)
                    ap+=1.0
                    precisions.append(1.0)
                    class_ap.append(1.0)
                recalls.append(1.0)
        allPrec=0
        allRecall=1
        if getClassAP:
            return ap/numClasses, precisions, recalls, class_ap
        else:
            return ap/numClasses, precisions, recalls, allPrec,allRecall
    else:
        if getClassAP:
            return 1.0, [1.0]*numClasses, [1.0]*numClasses, [1.0]*numClasses #we didn't for all classes :)
        else:
            return 1.0, [1.0]*numClasses, [1.0]*numClasses, 1.0, 1.0

    allScores=[]
    classScores=[[] for i in range(numClasses)]
    allTruPos=0
    allPred=0
    allGT=0
    if pred is not None and len(pred.size())>1 and pred.size(0)>0:
        #This is an alternate metric that computes AP of all classes together
        #Your only a hit if you have the same class
        allIOUs = getLoc(target[:,0:],pred[:,1:])
        allHits = allIOUs>iou_thresh

        #evalute hits to see if they're valid (matching class)
        targetClasses_index = torch.argmax(target[:,13:13+numClasses],dim=1)
        predClasses = pred[:,beforeCls+6:beforeCls+6+numClasses]
        if predClasses.size(0)==0 or predClasses.size(1)==0:
            print('ERROR, zero sized predClasses: {}. pred is {}'.format(predClasses.size(),pred.size()))
        predClasses_index = torch.argmax(predClasses,dim=1)
        targetClasses_index_ex = targetClasses_index[:,None].expand(targetClasses_index.size(0),predClasses_index.size(0))
        predClasses_index_ex = predClasses_index[None,:].expand(targetClasses_index.size(0),predClasses_index.size(0))
        matchingClasses = targetClasses_index_ex==predClasses_index_ex
        validHits = allHits*matchingClasses

        #add all the preds that didn't have a hit
        hasHit,_ = validHits.max(dim=0) #which preds have hits
        notHitScores = pred[~hasHit,0]
        notHitClass = predClasses_index[~hasHit]
        for i in range(notHitScores.shape[0]):
            allScores.append( (notHitScores[i].item(), False) )
            cls = notHitClass[i]
            classScores[cls].append( (notHitScores[i].item(), False) )

        # if something has multiple hits, it gets paired to the closest (with matching class)
        allIOUs[~validHits] -= 9999999 #Force these to be smaller
        maxValidHitIndexes = torch.argmax(allIOUs,dim=0)
        for i in range(maxValidHitIndexes.size(0)):
            if validHits[maxValidHitIndexes[i],i]:
                allScores.append( (pred[i,0].item(),True) )
                #but now we've consumed this pred, so we'll zero its hit
                validHits[maxValidHitIndexes[i],i]=0
                cls = predClasses_index[i]
                classScores[cls].append( (pred[i,0].item(),True) )

        #add nan scores for missed targets
        gotHit,gotHitIndex = torch.max(validHits,dim=1)
        for i in range((gotHit==0).sum()):
            allScores.append( (float('nan'),True) )
            cls = targetClasses_index[i]
            classScores[cls].append( (float('nan'),True) )
    else:
        allScores.append( (float('nan'),True) )
        classScores=[[(float('nan'),True)]]*numClasses


    if ignoreClasses:
        numClasses=1
    totalTruPos=0
    totalPred=0
    totalGT=0
    #by class
    #import pdb; pdb.set_trace()
    for cls in range(numClasses):
        clsTargInd = target[:,cls+13]==1
        if pred is not None and len(pred.size())>1 and pred.size(0)>0:
            #print(pred.size())
            clsPredInd = torch.argmax(pred[:,beforeCls+6:beforeCls+6+numClasses],dim=1)==cls
        else:
            clsPredInd = torch.empty(0,dtype=torch.bool)
        if (ignoreClasses and pred.size(0)>0) or (clsTargInd.any() and clsPredInd.any()):
            if ignoreClasses:
                clsTarg=target
                clsPred=pred
            else:
                clsTarg = target[clsTargInd]
                clsPred = pred[clsPredInd]
            clsIOUs = getLoc(clsTarg[:,0:],clsPred[:,1:])
            hits = clsIOUs>iou_thresh

            clsIOUs *= hits.float()
            ps = torch.argmax(clsIOUs,dim=1)
            left_ps = torch.ones(clsPred.size(0),dtype=torch.bool)
            left_ps[ps]=0
            truePos=0
            for t in range(clsTarg.size(0)):
                p=ps[t]
                if hits[t,p]:
                    #scores.append( (clsPred[p,0],True) )
                    #hits[t,p]=0
                    truePos+=1
                #else:
                    #scores.append( (float('nan'),True) )
            
            left_conf = clsPred[left_ps,0]
            #for i in range(left_conf.size(0)):
                #scores.append( (left_conf[i],False) )
            
            #ap = computeAP(scores)
            #if ap is not None:
            #    aps.append(ap)

            precisions.append( truePos/max(clsPred.size(0),truePos) )
            if precisions[-1]>1:
                import pdb;pdb.set_trace()
            recalls.append( truePos/clsTarg.size(0) )
            totalTruPos+=truePos
            totalPred+=max(clsPred.size(0),truePos)
            totalGT+=clsTarg.size(0)
        else:
            totalPred+=clsPredInd.size(0)
            totalGT+=clsTargInd.size(0)
            if ignoreClasses:
                #no pred
                #aps.append(0)
                precisions.append(0)
                recalls.append(0)
            elif clsPredInd.any() or clsTargInd.any():
                #aps.append(0)
                if clsPredInd.any():
                    recalls.append(1)
                    precisions.append(0)
                else:
                    precisions.append(0)
                    recalls.append(0)
            else:
                #aps.append(1)
                precisions.append(1)
                recalls.append(1)
    
    allPrec = totalTruPos/totalPred if totalPred>0 else 1
    allRecall = totalTruPos/totalGT if totalGT>0 else 1
    if getClassAP:
        classAPs=[computeAP(scores) for scores in classScores]
        #for i in range(len(classAPs)):
        #    if classAPs[i] is None:
        #        classAPs[i]=1
        return computeAP(allScores), precisions, recalls, classAPs
    else:
        return computeAP(allScores), precisions, recalls, allPrec, allRecall

def AP_textLines(target,pred,iou_thresh,numClasses=2,ignoreClasses=False,beforeCls=0,getClassAP=False):
    #mAP=0.0
    #aps=[]
    precisions=[]
    recalls=[]

    #how many classes are there?
    if ignoreClasses:
        numClasses=1
    if len(target.size())>1 and target.size(0)>0:
        #numClasses=target.size(1)-13
        pass
    elif pred is not None and len(pred)>0:
        #if there are no targets, we shouldn't be pred anything
        if ignoreClasses:
            #aps.append(0)
            ap=0.0
            precisions.append(0.0)
            recalls.append(1.0)
        else:
            #numClasses=pred.size(1)-6
            ap=0
            class_ap=[]
            all_cls_pred = torch.FloatTensor([p.getCls() for p in pred])
            for cls in range(numClasses):
                if (torch.argmax(all_cls_pred,dim=1)==cls).any():
                    #aps.append(0) #but we did for this class :(
                    ap+=0.0
                    precisions.append(0.0)
                    class_ap.append(0.0)
                else:
                    #aps.append(1) #we didn't for this class :)
                    ap+=1.0
                    precisions.append(1.0)
                    class_ap.append(1.0)
                recalls.append(1.0)
        allPrec=0
        allRecall=1
        if getClassAP:
            return ap/numClasses, precisions, recalls, class_ap
        else:
            return ap/numClasses, precisions, recalls, allPrec, allRecall
    else:
        if getClassAP:
            return 1.0, [1.0]*numClasses, [1.0]*numClasses, [1.0]*numClasses #we didn't for all classes :)
        else:
            return 1.0, [1.0]*numClasses, [1.0]*numClasses, 1.0, 1.0

    allScores=[]
    classScores=[[] for i in range(numClasses)]
    targetClasses_index = torch.argmax(target[:,13:13+numClasses],dim=1)
    if pred is not None and len(pred)>0:
        #This is an alternate metric that computes AP of all classes together
        #Your only a hit if you have the same class
        allIOUs = classPolyIOU(target,pred,numClasses) 
        validHits = allHits = allIOUs>iou_thresh
        #evalute hits to see if they're valid (matching class)

        #add all the preds that didn't have a hit
        hasHit,_ = validHits.max(dim=0) #which preds have hits
        predConf=torch.FloatTensor([p.getConf() for p in pred])
        predClasses=torch.FloatTensor([p.getCls() for p in pred])
        predClasses_index = torch.argmax(predClasses[:,:numClasses],dim=1) #remove blank if present, get max class
        #targetClasses_index_ex = targetClasses_index[:,None].expand(targetClasses_index.size(0),predClasses_index.size(0))
        #predClasses_index_ex = predClasses_index[None,:].expand(targetClasses_index.size(0),predClasses_index.size(0))
        notHitScores = predConf[~hasHit]
        notHitClass = predClasses_index[~hasHit]
        for i in range(notHitScores.shape[0]):
            allScores.append( (notHitScores[i].item(), False) )
            cls = notHitClass[i]
            classScores[cls].append( (notHitScores[i].item(), False) )

        # if something has multiple hits, it gets paired to the closest (with matching class)
        allIOUs[~validHits] -= 9999999 #Force these to be smaller
        maxValidHitIndexes = torch.argmax(allIOUs,dim=0)
        for i in range(maxValidHitIndexes.size(0)):
            if validHits[maxValidHitIndexes[i],i]:
                allScores.append( (predConf[i].item(),True) )
                #but now we've consumed this pred, so we'll zero its hit
                validHits[maxValidHitIndexes[i],i]=0
                cls = predClasses_index[i]
                classScores[cls].append( (predConf[i].item(),True) )

        #add nan scores for missed targets
        gotHit,gotHitIndex = torch.max(validHits,dim=1)
        for i in range((gotHit==0).sum()):
            allScores.append( (float('nan'),True) )
            cls = targetClasses_index[i]
            classScores[cls].append( (float('nan'),True) )
    else:
        allScores.append( (float('nan'),True) )
        classScores=[[(float('nan'),True)]]*numClasses


    if ignoreClasses:
        numClasses=1
    
    totalTruPos=0
    totalPred=0
    totalGT=0
    #by class
    #import pdb; pdb.set_trace()
    for cls in range(numClasses):
        clsTargInd = target[:,cls+13]==1
        if pred is not None and len(pred)>0:
            #print(pred.size())
            clsPredInd = predClasses_index==cls
        else:
            clsPredInd = torch.empty(0,dtype=torch.bool)
        if (ignoreClasses and pred.size(0)>0) or (clsTargInd.any() and clsPredInd.any()):
            if ignoreClasses:
                clsTarg=target
                clsPredConf=torch.FloatTensor([p.get_conf() for p in pred])
            else:
                clsTarg = target[clsTargInd]
                clsPredConf=predConf[clsPredInd]#torch.FloatTensor([pred[i].get_conf() for i in torch.nonzero(clsPredInd)])
            #clsIOUs = getLoc(clsTarg[:,0:],clsPred[:,1:])
            clsIOUs = allIOUs[clsTargInd][:,clsPredInd]
            hits = clsIOUs>iou_thresh

            clsIOUs *= hits.float()
            ps = torch.argmax(clsIOUs,dim=1)
            left_ps = torch.ones(clsPredConf.size(0),dtype=torch.bool)
            left_ps[ps]=0
            truePos=0
            for t in range(clsTarg.size(0)):
                p=ps[t]
                if hits[t,p]:
                    #scores.append( (clsPred[p,0],True) )
                    #hits[t,p]=0
                    truePos+=1
                #else:
                    #scores.append( (float('nan'),True) )
            
            left_conf = clsPredConf[left_ps]
            #for i in range(left_conf.size(0)):
                #scores.append( (left_conf[i],False) )
            
            #ap = computeAP(scores)
            #if ap is not None:
            #    aps.append(ap)

            precisions.append( truePos/max(clsPredConf.size(0),truePos) )
            if precisions[-1]>1:
                import pdb;pdb.set_trace()
            recalls.append( truePos/clsTarg.size(0) )

            totalTruPos += truePos
            totalPred += max(clsPredConf.size(0),truePos)
            totalGT += clsTarg.size(0)
        else:
            totalPred+=clsPredInd.size(0)
            totalGT+=clsTargInd.size(0)
            if ignoreClasses:
                #no pred
                #aps.append(0)
                precisions.append(0)
                recalls.append(0)
            elif clsPredInd.any() or clsTargInd.any():
                #aps.append(0)
                if clsPredInd.any():
                    recalls.append(1)
                    precisions.append(0)
                else:
                    precisions.append(0)
                    recalls.append(0)
            else:
                #aps.append(1)
                precisions.append(1)
                recalls.append(1)
    
    allPrec = totalTruPos/totalPred if totalPred>0 else 1
    allRecall = totalTruPos/totalGT if totalGT>0 else 1
    if getClassAP:
        classAPs=[computeAP(scores) for scores in classScores]
        #for i in range(len(classAPs)):
        #    if classAPs[i] is None:
        #        classAPs[i]=1
        return computeAP(allScores), precisions, recalls, classAPs
    else:
        return computeAP(allScores), precisions, recalls, allPrec,allRecall

def getTargIndexForPreds_iou(target,pred,iou_thresh,numClasses,beforeCls=0,hard_thresh=True,fixed=True):
    return getTargIndexForPreds(target,pred,iou_thresh,numClasses,beforeCls,allIOU,hard_thresh,fixed)
def getTargIndexForPreds_dist(target,pred,iou_thresh,numClasses,beforeCls=0,hard_thresh=True,fixed=True):
    raise NotImplementedError('Checking if preds with no intersection not implemented for dist')
    return getTargIndexForPreds(target,pred,iou_thresh,numClasses,beforeCls,allBoxDistNeg,hard_thresh,fixed)

def getTargIndexForPreds(target,pred,iou_thresh,numClasses,beforeCls,getLoc, hard_thresh,fixed):
    targIndex = torch.LongTensor((pred.size(0)))
    targIndex[:] = -1
    #mAP=0.0
    aps=[]
    precisions=[]
    recalls=[]

    if len(target.size())<=1:
        return None, None

    #by class
    #import pdb; pdb.set_trace()
    #first get all IOUs, then process by class
    allIOUs = getLoc(target[:,0:],pred[:,1:])
    #This isn't going to work of dist as 0 is perfect
    maxIOUsForPred,_ = allIOUs.max(dim=0)
    predsWithNoIntersection=maxIOUsForPred==0

    hits = allIOUs>iou_thresh
    if hard_thresh:
        allIOUs *= hits.float()


    for cls in range(numClasses):
        scores=[]
        clsTargInd = target[:,cls+13]==1
        notClsTargInd = target[:,cls+13]!=1
        if len(pred.size())>1 and pred.size(0)>0:
            #print(pred.size())
            #clsPredInd = torch.argmax(pred[:,beforeCls+6:],dim=1)==cls
            clsPredInd = torch.argmax(pred[:,-numClasses:],dim=1)==cls
        else:
            clsPredInd = torch.empty(0,dtype=torch.uint8)
        if  clsPredInd.any():
            if notClsTargInd.any() and fixed:
                notClsTargIndX = notClsTargInd[:,None].expand(allIOUs.size())
                clsPredIndX = clsPredInd[None,:].expand(allIOUs.size())
                allIOUs[notClsTargIndX*clsPredIndX]=0 #set IOU for instances that are from different class than predicted to 0 (different class so no intersection)
                #allIOUs[notClsTargInd][:,clsPredInd]=0 this doesn't work for some reason
            val,targIndexes = torch.max(allIOUs[:,clsPredInd],dim=0)
            #targIndexes has the target indexes for the predictions of cls

            #assign -1 index to places that don't really have a match
            #targIndexes[:] = torch.where(val==0,-torch.ones_like(targIndexes),targIndexes)
            targIndexes[val==0] = -1
            #targIndexes[notClsTargInd] = -1
            #assert(notClsTargInd[targIndexes].sum()==0)
            targIndex[clsPredInd] =  targIndexes

    #debug
    #for i in range(targIndex.size(0)):
        #if targIndex[i]>=0:
    #         assert(torch.argmax(pred[i,-numClasses:],dim=0) == torch.argmax(target[targIndex[i],-numClasses:],dim=0))
            
    if hard_thresh:
        return targIndex, predsWithNoIntersection
    else:
        hits,_ = hits.max(dim=0) #since we always take max pred
        return targIndex, hits


#This also returns which pred BBs are oversegmentations of targets (horizontall)
def newGetTargIndexForPreds_iou(target,pred,iou_thresh,numClasses,train_targs):
    if pred is None: 
        return None

    if len(target.size())<=1 or target.size(0)==0 or pred.size(0)==0:
        return None

    #first get all IOUs. These are already filtered with angle and class
    #allIOUs, allIO_clippedU = allPolyIOU_andClip(target,pred,class_sensitive=not train_targs) #clippedUnion, target is clipped horizontally to match pred. This filters for class matching
    assert(pred.size(1)>=numClasses+6) #might have num neighbors
    if train_targs:
        allIOUs = allIO_clipU(target,pred[:,1:])
    else:
        allIOUs = classIOU(target,pred[:,1:],numClasses)
    hits = allIOUs>iou_thresh
    #overSeg_thresh = iou_thresh*1.05
    #overSegmented= (allIO_clippedU>overSeg_thresh)
    allIOUs *= hits.float()
    #allIO_clippedU *= overSegmented

    #if train_targs:
    #    val,targIndex = torch.max(allIO_clippedU,dim=0)
    #else:
    if allIOUs.size(0)>0 and allIOUs.size(1)>0:
        val,targIndex = torch.max(allIOUs,dim=0)
        targIndex[val==0]=-1 #These don't have a match
    else:
        targIndex=torch.IntTensor(0)


    return targIndex
    allIOUs, allIO_clippedU = getLoc(target[:,0:],pred[:,1:]) #clippedUnion, target is clipped horizontally to match pred

#This also returns which pred BBs are oversegmentations of targets (horizontall)
def newGetTargIndexForPreds_textLines(target,pred,iou_thresh,numClasses,train_targs,discard_bad_overlaps=False):
    if pred is None: 
        return None

    if len(target.size())<=1 or target.size(0)==0:
        return None

    #first get all IOUs. These are already filtered with angle and class
    #allIOUs, allIO_clippedU = allPolyIOU_andClip(target,pred,class_sensitive=not train_targs) #clippedUnion, target is clipped horizontally to match pred. This filters for class matching

    if train_targs:
        allIOUs = allPolyIO_clipU(target,pred)
    else:
        allIOUs = classPolyIOU(target,pred,numClasses)

    hits = allIOUs>iou_thresh
    #overSeg_thresh = iou_thresh*1.05
    #overSegmented= (allIO_clippedU>overSeg_thresh)
    allIOUs *= hits.float()
    #allIO_clippedU *= overSegmented

    #if train_targs:
    #    val,targIndex = torch.max(allIO_clippedU,dim=0)
    #else:
    if allIOUs.size(0)>0 and allIOUs.size(1)>0:
        val,targIndex = torch.max(allIOUs,dim=0)
        targIndex[val==0]=-1 #These don't have a match

        if discard_bad_overlaps:
            #Some pred bbs may claim the same target, but some may be better aligned that others
            #We'll simulate merging them and determine which help improve IOU
            #merge adjacent preds. Measure change in IclipU, bad merges should cause it to significantly incease
            # good merges should cause IclipU to stay about the same
            #For each pred in shared group
            #  calc IclipU for merging it's N neighbors
            #new groups based on the bi-direction maintence of IclipU
            #the group with best average individual IclipU is the right one
            #the other groups are not a match

            #for each pred in shared grou
            #  calc IOU
            #  calc IOU for mering it's N neighbors
            #new groups based on the bi-directionally improved IOU from merging
            for ti in range(target.size(0)):
                sharing = (targIndex==ti).nonzero(as_tuple=False)[:,0]
                sharing = sharing.tolist()
                if len(sharing)==2:
                    merged=TextLine(pred[sharing[0]],pred[sharing[1]])
                    IOUs = classPolyIOU(target[ti:ti+1],[pred[sharing[0]],pred[sharing[1]],merged],num_classes=0)
                    IOU_0 = IOUs[0,0]
                    IOU_1 = IOUs[0,1]
                    IOU_merged = IOUs[0,2]
                    if IOU_0>IOU_merged or IOU_1>IOU_merged:
                        if IOU_0>IOU_1:
                            targIndex[sharing[1]]=-1 #clear ti as a target
                        elif IOU_1>IOU_0:
                            targIndex[sharing[0]]=-1

                elif len(sharing)>2:
                    #N=3
                    #center_points = np.array([pred[pi].getCenterPoint() for pi in sharing])
                    #sq_distances = np.power(center_points[None,:]-center_points[:,None],2).sum(axis=2)
                    #to_IOU = [pred[pi] for pi in sharing]
                    #to_IOU_index={}
                    #pi_dists=[]
                    #for pi_i,pi in enumerate(sharing):
                    #    pi_dist = [(i,d) for i,d in enumerate(sq_distances[pi_i]) if i!=pi_i]
                    #    pi_dist.sort(key=lambda a:a[1])
                    #    for neighbor_i,d in pi_dist[:N]:
                    #        merged=TextLine(pred[pi],pred[sharing[neighbor_i]])
                    #        to_IOU_index[(pi,neighbor_i)]=len(to_IOU)
                    #        to_IOU.append(merged)
                    #    pi_dists.append(pi_dist)
                    #IOUs = classPolyIOU(target[ti:ti+1],to_IOU,num_classes=0)[0]

                    #i_want = defaultdict(list)
                    #for pi,pi_dist in zip(sharing,pi_dists):
                    #    for neighbor_i,d in pi_dist[:N]:
                    #        merged_IOU = IOUs[to_IOU_index[(pi,neighbor_i)]]
                    #        print('t{}: {} <m {} = {} >? {}'.format(ti,pi,sharing[neighbor_i],merged_IOU,IOUs[pi_i]))

                    #        if merged_IOU>IOUs[pi_i]:
                    #            i_want[pi].append(sharing[neighbor_i])

                    #bidirectional_wants = set()
                    #for pi,want_i in i_want.items():
                    #    for w_i in want_i:
                    #        if pi in i_want[w_i]:
                    #            bidirectional_wants.add((min(pi,w_i),max(pi,w_i)))
                    #print('t{}: bidirectional_wants = {}'.format(ti,bidirectional_wants))
                    #bidir_groups = {}
                    #bidir_map = {}
                    #bidir_group_index=0
                    #unused_pis = set(sharing)
                    #for p1i,p2i in bidirectional_wants:
                    #    try:
                    #        unused_pis.remove(p1i)
                    #    except KeyError:
                    #        pass
                    #    try:
                    #        unused_pis.remove(p2i)
                    #    except KeyError:
                    #        pass
                    #    if p1i in bidir_map and p2i in bidir_map:
                    #        group1 = bidir_map[p1i]
                    #        group2 = bidir_map[p2i]
                    #        if group1!=group2:
                    #            bidir_groups[group1]+=bidir_groups[group2]
                    #            for pi in bidir_groups[group2]:
                    #                bidir_map[pi] = group1
                    #            del bidir_groups[group2]
                    #    elif p1i in bidir_map:
                    #        bidir_groups[bidir_map[p1i]].append(p2i)
                    #        bidir_map[p2i] = bidir_map[p1i]
                    #    elif p2i in bidir_map:
                    #        bidir_groups[bidir_map[p2i]].append(p1i)
                    #        bidir_map[p1i] = bidir_map[p2i]
                    #    else:
                    #        bidir_groups[bidir_group_index]=[p1i,p2i]
                    #        bidir_map[p1i] = bidir_group_index
                    #        bidir_map[p2i] = bidir_group_index
                    #        bidir_group_index+=1
                    #    #print('t{}:   bidir_groups={}'.format(ti,bidir_groups))

                    #bidir_groups = list(bidir_groups.values())
                    #for pi in unused_pis:
                    #    bidir_groups.append([pi])
                    #print('t{}: bidir_groups = {}'.format(ti,bidir_groups))
                    #candidate_groups=bidir_groups
                    #if len(candidate_groups)>1:
                    #    #find the best group
                    #    
                    #    final_IOUs={}
                    #    best_IOU=0
                    #    best_IOU_i=0
                    #    for gi,group in enumerate(candidate_groups):
                    #        if len(group)==1:
                    #            index = sharing.index(group[0])
                    #            if IOUs[index] > best_IOU:
                    #                best_IOU=IOUs[index]
                    #                best_IOU_i=gi
                    #        else:
                    #            merged=TextLine(pred[group[0]],pred[group[1]])
                    #            for pi in group[2:]:
                    #                merged.merge(TextLine(pred[pi]))
                    #            IOU = classPolyIOU(target[ti:ti+1],[merged],num_classes=0)
                    #            if IOU>best_IOU:
                    #                best_IOU=IOU
                    #                best_IOU_i=gi
        
                    #    #clear ti as target for all except best
                    #    for gi,group in enumerate(candidate_groups):
                    #        if gi!=best_IOU_i:
                    #            for pi in group:
                    #                targIndex[pi]=-1

                    #divide matched predicitions into two groups based on size
                    heights = [pred[pi].getHeight() for pi in sharing]
                    widths = [pred[pi].getWidth() for pi in sharing]
                    heights.sort()
                    min_h = heights[0]
                    max_h = heights[-1]
                    small_group = []
                    big_group = []
                    for pi in sharing:
                        diff_min = pred[pi].getHeight()-min_h
                        diff_max = max_h-pred[pi].getHeight()
                        if diff_min<diff_max:
                            small_group.append(pi)
                        else:
                            big_group.append(pi)
                    
                    if len(small_group)==1:
                        merged_small = pred[small_group[0]]
                    else:
                        merged_small=TextLine(pred[small_group[0]],pred[small_group[1]])
                        for pi in small_group[2:]:
                            merged_small.merge(pred[pi])

                    if len(big_group)==1:
                        merged_big = pred[big_group[0]]
                    else:
                        merged_big=TextLine(pred[big_group[0]],pred[big_group[1]])
                        for pi in big_group[2:]:
                            merged_big.merge(pred[pi])

                    merged_all = TextLine(merged_small,merged_big)

                    IOUs = classPolyIOU(target[ti:ti+1],[merged_small,merged_big,merged_all],num_classes=0)[0]

                    if IOUs[0]>IOUs[1] and IOUs[0]>IOUs[2]: #small is best
                        for pi in big_group:
                            targIndex[pi]=-1
                    elif IOUs[1]>IOUs[0] and IOUs[1]>IOUs[2]: #big is best
                        for pi in small_group:
                            targIndex[pi]=-1
                    #elif IOUs[2]>IOUs[0] and IOUs[2]>IOUs[1]: #all is best
                    #    pass
                    #else:




                #else it's just 1
                
    else:
        targIndex=torch.IntTensor(0)


    return targIndex

def computeAP(scores):
    rank=[]
    missed=0
    for conf,rel in scores:
        if rel:
            if math.isnan(conf):
                missed+=1
            else:
                better=0
                equal=-1 # as we'll iterate over this instance here
                for conf2,rel2 in scores:
                    if conf2>conf:
                        better+=1
                    elif conf2==conf:
                        equal+=1
                rank.append(better+math.ceil(equal/2.0))
    if len(rank)==0:
        if missed>0:
            return 0
        return None
    rank.sort()
    ap=0.0
    for i in range(len(rank)):
        ap += float(i+1)/(rank[i]+1)
    ap/=(len(rank)+missed)
    if ap>1.0001:
        raise ValueError('ap greater than 1({}), from {}'.format(ap,scores))
    return ap
