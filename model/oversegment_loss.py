import timeit
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from utils.yolo_tools import allIOU, allDist
from matplotlib import pyplot as plt
from model.overseg_box_detector import MAX_H_PRED, MAX_W_PRED, NUM_ANCHORS
from utils.util import plotRect, xyrhwToCorners, inv_tanh
from shapely.geometry import Polygon
import shapely
import skimage.draw
import utils.img_f as img_f

UNMASK_CENT_DIST_THRESH=0.1
END_BOUNDARY_THRESH=0.4
END_UNMASK_THRESH=0.1
SCALE_DIFF_THRESH=0.251

def norm_angle(a):
    if a>np.pi:
        return a-2*np.pi
    elif a<-np.pi:
        return a+2*np.pi
    else:
        return a

def fraction_in_tile(left_side,tx,ty,li_x,ri_x):#,ti_y,bi_y):
    #if isHorz:
    if left_side:
        t_fraction_in_tile = (tx+1-li_x)/(ri_x-li_x)
    else:
        t_fraction_in_tile = (ri_x-tx)/(ri_x-li_x)
    #else:
    #    if top_side:
    #        t_fraction_in_tile = (ty+1-ti_y)/(bi_y-ti_y)
    #    else:
    #        t_fraction_in_tile = (bi_y-ty)/(bi_y-ti_y)

    return t_fraction_in_tile

def get_tiles(y1,x1,y2,x2):
    #this gets the cells/tiles intersected by a line passing from one point to the other
    #there are probably a lot faster implementations out there

    max_tx = max(int(x2),int(x1))
    min_tx = min(int(x2),int(x1))
    max_ty = max(int(y2),int(y1))
    min_ty = min(int(y2),int(y1))


    #tiles_x = [min_x,max_x]
    #tiles_y = [min_y,max_y]
    tiles = set([(int(x1),int(y1)),(int(x2),int(y2))])
    #print(tiles)
    
    if x1!=x2:
        slope = (y1-y2)/(x1-x2)
        c = y2 - x2*slope

        if abs(slope)<=1:
            for tx in range(min_tx+1,max_tx+1):
                yi = tx*slope + c
                #print('{}   {},{}'.format(slope,tx,yi))
                tiles.add((tx,int(yi)))
                tiles.add((tx-1,int(yi)))
                #print(tiles)
        else:
            for ty in range(min_ty+1,max_ty+1):
                xi = (ty-c)/slope
                #print('{}   {},{}'.format(slope,xi,ty))
                tiles.add((int(xi),ty))
                tiles.add((int(xi),ty-1))
                #print(tiles)

        xs,ys = zip(*tiles)
        return ys,xs
    else:
        ys = list(range(min_ty,max_ty+1))
        return ys,[int(x1)]*len(ys)


class MultiScaleOversegmentLoss (nn.Module):
    def __init__(self, num_classes, rotation, scale, anchors, bad_conf_weight=1.25, multiclass=False,tile_assign_mode='split',close_anchor_rule='unmask'):
        super(MultiScaleOversegmentLoss, self).__init__()
        self.num_classes=num_classes
        assert(rotation)
        assert(anchors is None)
        self.num_anchors=NUM_ANCHORS
        self.tile_assign_mode=tile_assign_mode
        self.close_anchor_rule=close_anchor_rule
        self.scales=scale
        self.bad_conf_weight=bad_conf_weight
        self.multiclass=multiclass
        self.mse_loss = nn.MSELoss(reduction='mean')  # Coordinate loss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')  # Class loss
        self.mse_loss = nn.MSELoss(reduction='mean')  # Num neighbor regression

        self.OPT_FULL=[]

    def forward(self,predictions, target, target_sizes, calc_stats=False):
        ticAll=timeit.default_timer()

        nA = self.num_anchors
        nHs=[]
        nWs=[]
        pred_boxes_scales=[]
        pred_conf_scales=[]
        pred_cls_scales=[]

        x1_scales=[]
        y1_scales=[]
        x2_scales=[]
        y2_scales=[]
        r_scales=[]
        for level,prediction in enumerate(predictions):
            #t#tic=timeit.default_timer()

            nB = prediction.size(0)
            nHs.append( prediction.size(2) )
            nH = prediction.size(2)
            nWs.append( prediction.size(3) )
            nW = prediction.size(3)

            FloatTensor = torch.cuda.FloatTensor if prediction.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if prediction.is_cuda else torch.LongTensor
            ByteTensor = torch.cuda.ByteTensor if prediction.is_cuda else torch.ByteTensor
            BoolTensor = torch.cuda.BoolTensor if prediction.is_cuda else torch.BoolTensor

            x1 = prediction[..., 1]  
            y1 = prediction[..., 2] 
            x2 = prediction[..., 3]
            y2 = prediction[..., 4]
            r = prediction[..., 5]  
            x1_scales.append(x1)
            y1_scales.append(y1)
            x2_scales.append(x2)
            y2_scales.append(y2)
            r_scales.append(r)
            pred_conf = prediction[..., 0]  # Conf 
            pred_cls = prediction[..., 6:]  # Cls pred.

            grid_x = torch.arange(nW).repeat(nH, 1).view([1, 1, nH, nW]).type(FloatTensor).to(prediction.device)
            grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, 1, nH, nW]).type(FloatTensor).to(prediction.device)

            # Add offset and scale with anchors
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = torch.tanh(x1.data)*MAX_W_PRED+0.5 + grid_x #level?
            pred_boxes[..., 1] = torch.tanh(y1.data)*MAX_H_PRED+0.5 + grid_y
            pred_boxes[..., 2] = torch.tanh(x2.data)*MAX_W_PRED+0.5 + grid_x
            pred_boxes[..., 3] = torch.tanh(y2.data)*MAX_H_PRED+0.5 + grid_y

            pred_boxes_scales.append(pred_boxes.cpu().data)
            pred_conf_scales.append(pred_conf.cpu().data)
            pred_cls_scales.append(pred_cls.cpu().data)

        #t#print('time setup A: '+str(timeit.default_timer()-tic))

        nGT, masks, conf_masks, t_Ls, t_Ts, t_Rs, t_Bs, t_rs, tconf_scales, tcls_scales, pred_covered, gt_covered, recall, precision, pred_covered_noclass, gt_covered_noclass, recall_noclass, precision_noclass = build_oversegmented_targets_multiscale(
            pred_boxes=pred_boxes_scales,
            pred_conf=pred_conf_scales,
            pred_cls=pred_cls_scales,
            target=target.cpu().data if target is not None else None,
            target_sizes=target_sizes,
            num_classes=self.num_classes,
            grid_sizesH=nHs,
            grid_sizesW=nWs,
            scale=self.scales,
            assign_mode = self.tile_assign_mode,
            close_anchor_rule = self.close_anchor_rule,
            calc_stats=calc_stats
        )
        #pred_boxes_scales=[]
        pred_conf_scales=[]
        pred_cls_scales=[]
        for level,prediction in enumerate(predictions):
            pred_conf = prediction[..., 0]  # Conf 
            pred_cls = prediction[..., 6:]  # Cls pred.
            #pred_boxes_scales.append(pred_boxes.cpu().data)
            pred_conf_scales.append(pred_conf)
            pred_cls_scales.append(pred_cls)

        #nProposals = int((pred_conf > 0).sum().item())
        #recall = float(nCorrect / nGT) if nGT else 1
        #if nProposals>0:
        #    precision = float(nCorrect / nProposals)
        #else:
        #    precision = 1

        # Handle masks
        masks = [mask.type(BoolTensor) for mask in masks]
        conf_masks = [conf_mask.type(BoolTensor) for conf_mask in conf_masks]
        # Handle target variables
        t_Ls = [t.type(FloatTensor).to(prediction.device) for t in t_Ls]
        t_Ts = [t.type(FloatTensor).to(prediction.device) for t in t_Ts]
        t_Rs = [t.type(FloatTensor).to(prediction.device) for t in t_Rs]
        t_Bs = [t.type(FloatTensor).to(prediction.device) for t in t_Bs]
        t_rs = [t.type(FloatTensor).to(prediction.device) for t in t_rs]
        tconf_scales = [t.type(FloatTensor).to(prediction.device) for t in tconf_scales]
        tcls_scales = [t.type(FloatTensor).to(prediction.device) for t in tcls_scales]

        # Get conf mask where gt and where there is no gt
        conf_masks_true = masks
        conf_masks_false = [conf_mask & ~mask  for conf_mask,mask in zip(conf_masks,masks)]

        #import pdb; pdb.set_trace()

        # Mask outputs to ignore non-existing objects
        loss_conf=0
        for pred_conf,tconf,conf_mask_falsel in zip(pred_conf_scales,tconf_scales,conf_masks_false):
            if conf_mask_falsel.any():
                loss_conf += self.bce_loss(pred_conf[conf_mask_falsel], tconf[conf_mask_falsel])
        loss_conf *= self.bad_conf_weight
        if target is not None and nGT>0:
            for pred_conf,tconf,conf_mask_truel in zip(pred_conf_scales,tconf_scales,conf_masks_true):
                if conf_mask_truel.any():
                    loss_conf += self.bce_loss(pred_conf[conf_mask_truel], tconf[conf_mask_truel])

            loss_L=0
            loss_T=0
            loss_R=0
            loss_B=0
            loss_r=0
            for level in range(len(t_Ls)):
                mask = masks[level]
                if mask.any():
                    loss_L += self.mse_loss(x1_scales[level][mask], t_Ls[level][mask])
                    loss_T += self.mse_loss(y1_scales[level][mask], t_Ts[level][mask])
                    loss_R += self.mse_loss(x2_scales[level][mask], t_Rs[level][mask])
                    loss_B += self.mse_loss(y2_scales[level][mask], t_Bs[level][mask])
                    loss_r += self.mse_loss(r_scales[level][mask], t_rs[level][mask])

            loss_cls=0
            for pred_cls,tcls,mask in zip(pred_cls_scales,tcls_scales,masks):
                if mask.any():
                    if self.multiclass:
                        loss_cls += self.bce_loss(pred_cls[mask], tcls[mask].float())
                    else:
                        loss_cls +=  self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1)) 

            loss = loss_L + loss_T + loss_R + loss_B + loss_r + loss_conf + loss_cls

            if False and build_gt_detections:
                ys = []
                for level in range(len(t_Ls)):
                    ys.append(torch.cat([torch.ones_like(t_Ls[level]),t_Ls[level],t_Ts[level],t_Rs[level], t_Bs[level],t_rs[level]],dim=1))
                gt_boxes = build_box_predictions(ys,self.scale,img.device,self.numAnchors,self.numBBParams,self.numBBTypes)

            #t#time = timeit.default_timer()-ticAll
            #t#print('time FULL: '+str(time))
            #t#self.OPT_FULL.append(time)
            #t#if len(self.OPT_FULL)>20:
            #t#    print('time mean FULL: {}'.format(np.mean(self.OPT_FULL))) #old 4.7, overlap removed down to 1
            #t#if len(self.OPT_FULL)>100:
            #t#    self.OPT_FULL=self.OPT_FULL[1:]
            if calc_stats:
                return (
                    loss,
                    (loss_L.item()+loss_T.item()+loss_R.item()+loss_B.item())/4,
                    loss_conf.item(),
                    loss_cls.item(),
                    loss_r.item(),
                    recall,
                    precision,
                    gt_covered,
                    pred_covered,
                    recall_noclass,
                    precision_noclass,
                    gt_covered_noclass,
                    pred_covered_noclass
                )
            else:
                return (
                    loss,
                    (loss_L.item()+loss_T.item()+loss_R.item()+loss_B.item())/4 if loss_L !=0 else 0,
                    loss_conf.item(),
                    loss_cls.item(),
                    loss_r.item(),
                    None,None,None,None,None,None,None,None
                    )
        else:
            #t#print('time FULL: '+str(timeit.default_timer()-ticAll))
            if calc_stats:
                return (
                    loss_conf,
                    0,
                    loss_conf.item(),
                    0,
                    0,
                    recall,
                    precision,
                    gt_covered,
                    pred_covered,
                    recall_noclass,
                    precision_noclass,
                    gt_covered_noclass,
                    pred_covered_noclass
                )
            else:
                return (
                    loss_conf,
                    0,
                    loss_conf.item(),
                    0,
                    0,
                    None,None,None,None,None,None,None,None
                    )

#This isn't totally anchor free, the model predicts horizontal and verticle text seperately.
#The model predicts the centerpoint offset (normalized to tile size), rotation and height (2X tile size) and width (1x tile size)
def build_oversegmented_targets_multiscale(
    pred_boxes, pred_conf, pred_cls, target, target_sizes, num_classes, grid_sizesH, grid_sizesW, scale, assign_mode='split', close_anchor_rule='unmask', calc_stats=False
):
    #t#tic=timeit.default_timer()

    VISUAL_DEBUG=False
    HIT_THRESH=0.5
    VIZ_SIZE=24
    use_rotation_aligned_predictions=False
    nC = num_classes
    nA = NUM_ANCHORS #4: primary horizonal (top), double horizonal (bot), primary verticle (left), double verticle (right)
    nHs = grid_sizesH
    nWs = grid_sizesW
    masks=[]
    shared_masks=[] #Do more then two (parallel) gts claim this cell?
    conf_masks=[]
    t_Ls=[]
    t_Rs=[]
    t_Ts=[]
    t_Bs=[]
    t_rs=[]
    t_confs=[]
    t_clss=[]

    #opt: could retain these so they don't need reallocated
    for level,(nH, nW) in enumerate(zip(nHs,nWs)):
        nB = pred_boxes[level].size(0)
        mask = torch.zeros(nB, nA, nH, nW)
        masks.append(mask)
        shared_mask = torch.zeros(nB, nA, nH, nW)
        shared_masks.append(shared_mask)
        conf_mask = torch.ones(nB, nA, nH, nW)
        conf_masks.append(conf_mask)
        targ_L = torch.zeros(nB, nA, nH, nW)
        targ_R = torch.zeros(nB, nA, nH, nW)
        targ_T = torch.zeros(nB, nA, nH, nW)
        targ_B = torch.zeros(nB, nA, nH, nW)
        targ_r = torch.zeros(nB, nA, nH, nW)
        targ_conf = torch.ByteTensor(nB, nA, nH, nW).fill_(0)
        targ_cls = torch.ByteTensor(nB, nA, nH, nW, nC).fill_(0)
        t_clss.append(targ_cls)
        t_confs.append(targ_conf)
        t_Ls.append(targ_L)
        t_Rs.append(targ_R)
        t_Ts.append(targ_T)
        t_Bs.append(targ_B)
        t_rs.append(targ_r)

    #t#print('time setup B: '+str(timeit.default_timer()-tic))

    nGT = 0
    nPred = 0
    covered_gt_area = 0
    on_pred_area = 0
    precision = 0
    recall = 0
    covered_gt_area_all = 0
    on_pred_area_all = 0
    precision_all = 0
    recall_all = 0
    #nCorrect = 0
    #import pdb; pdb.set_trace()
    for b in range(nB):
        #t#tic=timeit.default_timer()
        assignments=[]
        for nH,nW in zip(nHs,nWs):
            assignment = torch.IntTensor(nA, nH, nW).fill_(-1)
            assignments.append(assignment)
        if VISUAL_DEBUG:
            draw=[]
            for nH,nW in zip(nHs,nWs):
                drawl = np.zeros([nH*VIZ_SIZE,nW*VIZ_SIZE,3],np.uint8)
                for i in range(0,nW):
                    drawl[:,i*VIZ_SIZE,:]=60
                for j in range(0,nH):
                    drawl[j*VIZ_SIZE,:,:]=60
                #drawl[:,:,0]=90
                draw.append(drawl)

            draw_colors = [(255,0,0),(100,255,0),(0,0,255),(255,100,0),(0,255,0),(255,0,255),(200,200,0),(255,200,0),(200,255,0),(0,255,255)]
        on_pred_areaB=[]
        on_pred_areaB_all=[]
        covered_gt_areaB=torch.FloatTensor(target_sizes[b]).zero_()
        covered_gt_areaB_all=torch.FloatTensor(target_sizes[b]).zero_()
        for level in range(len(nHs)):
            on_pred_areaB.append( torch.FloatTensor(pred_boxes[level].shape[1:4]).zero_() )
            on_pred_areaB_all.append( torch.FloatTensor(pred_boxes[level].shape[1:4]).zero_() )

        #t#times_setup_and_level_select=[]
        #t#times_level_setup=[]
        #t#times_tile_hit=[]
        #t#times_lines=[]
        #t#times_border=[]
        #t#times_split_and_end=[]
        #t#times_handle_multi=[]
        #t#times_unmask_end=[]
        #t#times_assign=[]
        #t#times_overlaps=[]
        #For oversegmented, we need to identify all tiles (not just on) that correspon to gt
        #That brings up an interesting alternative: limit all predictions to their local tile (width). Proba not now...
        for t in range(target_sizes[b]): #range(target.shape[1]):
            #t#tic2=timeit.default_timer()
            #print('DEBUG t:{}'.format(t))
            num_assigned=0
            nGT += 1


            gr = target[b, t, 2]
            isHorz = (gr>-np.pi/4 and gr<np.pi/4) or (gr<-3*np.pi/4 or gr>3*np.pi/4)
            #TODO handle 45degree text better (predict on both anchors)

            #candidate_pos=[]
            #We need to decide which scale this is at.
            # I think the best is to simply choose the tile height that is closes to the (rotated) bb height
            closest_diff = 9999999999
            all_diff=[]
            for level in range(len(nHs)):

                gr = target[b, t, 2]
                gh = target[b, t, 3] / scale[level][1]

                #rh = gh*2
                if isHorz:
                    rh = abs(2*gh*math.cos(gr))
                else:
                    rh = abs(2*gh*math.sin(gr))
                diff = abs(rh-1)
                #print('{} diff level {}: {}  rh:{}'.format(t,level,diff,rh))
                if diff<closest_diff:
                    closest_diff = diff
                    closest = level
                all_diff.append(diff)
                    
            
            unmask_levels=[]
            if close_anchor_rule=='unmask':
                if closest>0 and abs(all_diff[closest-1]-all_diff[closest])<SCALE_DIFF_THRESH:
                    unmask_levels.append(closest-1)
                if closest<len(nHs)-1 and abs(all_diff[closest+1]-all_diff[closest])<SCALE_DIFF_THRESH:
                    unmask_levels.append(closest+1)

            #t#times_setup_and_level_select.append(timeit.default_timer()-tic2)

            for level in ([closest]+unmask_levels):
                #print('DEBUG    leve:{}'.format(level))
                #t#tic2=timeit.default_timer()
                only_unmask = level!=closest

                conf_mask = conf_masks[level]
                shared_mask = shared_masks[level]
                mask = masks[level]
                assignment = assignments[level]

                nH = nHs[level]
                nW = nWs[level]
                targ_L = t_Ls[level]
                targ_R = t_Rs[level]
                targ_T = t_Ts[level]
                targ_B = t_Bs[level]
                targ_r = t_rs[level]
                targ_conf = t_confs[level]
                targ_cls = t_clss[level]

                gx = target[b, t, 0] / scale[level][0]
                gy = target[b, t, 1] / scale[level][1]
                gw = target[b, t, 4] / scale[level][0]
                gh = target[b, t, 3] / scale[level][1]

                g_lx = target[b, t, 5] / scale[level][0]
                g_ly = target[b, t, 6] / scale[level][1]
                g_rx = target[b, t, 7] / scale[level][0]
                g_ry = target[b, t, 8] / scale[level][1]
                g_tx = target[b, t, 9] / scale[level][0]
                g_ty = target[b, t, 10] / scale[level][1]
                g_bx = target[b, t, 11] / scale[level][0]
                g_by = target[b, t, 12] / scale[level][1]

                tl,tr,br,bl = xyrhwToCorners(gx,gy,gr,gh,gw)
                g_min_x = min(tl[0],tr[0],br[0],bl[0])
                g_max_x = max(tl[0],tr[0],br[0],bl[0])
                g_min_y = min(tl[1],tr[1],br[1],bl[1])
                g_max_y = max(tl[1],tr[1],br[1],bl[1])


                if VISUAL_DEBUG:
                    #Draw GT

                    #draw_gy,draw_gx = skimage.draw.polygon([(g_lxI
                    #img_f.rectangle(draw[level],(gx,gy),(255,171,212),1
                    #print('{}'.format((gx,gy,gr,gh,gw)))
                    #plotRect(draw[level],(255,171,212),(gx,gy,gr,gh,gw))
                    #print('{} {} {} {}'.format(tl,tr,br,bl))
                    tl = (int(VIZ_SIZE*tl[0]),int(VIZ_SIZE*tl[1]))
                    tr = (int(VIZ_SIZE*tr[0]),int(VIZ_SIZE*tr[1]))
                    br = (int(VIZ_SIZE*br[0]),int(VIZ_SIZE*br[1]))
                    bl = (int(VIZ_SIZE*bl[0]),int(VIZ_SIZE*bl[1]))
                    colorKey=(112, 58, 99)
                    color=(92, 38, 79)
                    lineWidth=2
                    img_f.line(draw[level],tl,tr,color,lineWidth)
                    img_f.line(draw[level],tr,br,color,lineWidth)
                    img_f.line(draw[level],br,bl,color,lineWidth)
                    img_f.line(draw[level],bl,tl,colorKey,lineWidth)


                gt_area = gw*gh

                #gx1 = gx-gw
                #gx2 = gx+gw
                #gy1 = gy-gh
                #gy2 = gy+gh
            
                if gw==0 or gh==0:
                    #print('DEBUG: 0 sized bb')
                    continue
                #t#times_level_setup.append(timeit.default_timer()-tic2)
                #t#tic2=timeit.default_timer()
                #What tiles are relevant for predicting this? Just the center line? Do any tiles get an ignore (unmask)?
                #hit_tile_ys, hit_tile_xs = skimage.draw.line(int(g_ly),int(g_lx),int(g_ry),int(g_rx))
                #hit_tile_ys, hit_tile_xs = (y,x) for y,x in zip(hit_tile_ys, hit_tile_xs) if y>=0 and y<
                #ignore_tile_ys, ignore_tile_xs, weight_tile = skimage.draw.line_aa(int(g_ly),int(g_lx),int(g_ry),int(g_rx)) #This is nice, except I'm forced to pass integers in, perhaps this could be used for ignoring?

                hit_tile_ys, hit_tile_xs = get_tiles(g_ly,g_lx,g_ry,g_rx)

                hit_tile_ys = np.clip(hit_tile_ys,0,nH-1)
                hit_tile_xs = np.clip(hit_tile_xs,0,nW-1)
                hits = [(y,x) for y,x in zip(hit_tile_ys,hit_tile_xs) if (y>=0 and y<nH and x>=0 and x<nW)]
                hit_tile_ys,hit_tile_xs = zip(*hits)
                #ignore_tile_ys = np.clip(ignore_tile_ys,0,nH-1)
                #ignore_tile_xs = np.clip(ignore_tile_xs,0,nW-1)
                #t#times_tile_hit.append(timeit.default_timer()-tic2)
                #t#tic2=timeit.default_timer()

                #all_tile_ys, all_tile_xs = skimage.draw.polygon(r,c,(nH,nW))

                #we use the anti-ailiased to ignore close tiles
                #close_thresh=0.1
                #conf_mask[b,:,ignore_tile_ys,ignore_tile_xs]= torch.where(torch.from_numpy(weight_tile[None,...])>close_thresh,torch.zeros_like(conf_mask[b,:,ignore_tile_ys,ignore_tile_xs]),conf_mask[b,:,ignore_tile_ys,ignore_tile_xs])

                #shared_mask[b,0 if isHorz else 0,hit_tile_ys,hit_tile_xs] = torch.where(mask[b,0 if isHorz else 1,hit_tile_ys,hit_tile_xs]==1,mask[b,0 if isHorz else 1,hit_tile_ys,hit_tile_xs], shared_mask[b,0 if isHorz else 1,hit_tile_ys,hit_tile_xs]) #set to one everywhere that mask is already 1
                #mask[b,0 if isHorz else 1,hit_tile_ys,hit_tile_xs]= torch.where(shared_mask[b,0 if isHorz else 1,hit_tile_ys,hit_tile_xs]==1,torch.zeros_like(mask[b,0 if isHorz else 1,hit_tile_ys,hit_tile_xs]),torch.ones_like(mask[b,0 if isHorz else 1,hit_tile_ys,hit_tile_xs])) #set to 1 everywhere shared is 0, else set to 0
                #conf_mask[b,0 if isHorz else 1,hit_tile_ys,hit_tile_xs]=1
                #mask[b, best_n, gj, gi1:gi2+1] = 1
                #conf_mask[b, best_n, gj, gi1:gi2+1] = 1 #we ned to set this to 1 as we ignored it earylier
                # Coordinates

                # While along rotation is more "correct", I think axis aligned prediction may generalize better given there are few heavily skewed instances of text.
                if use_rotation_aligned_predictions:

                    #slope of topline
                    if (g_trx-g_tlx) !=0:
                        s_t = (g_try-g_tly)/(g_trx-g_tlx)
                    else:
                        s_t = float('inf')
                    c_t = g_tly-s_t*g_tlx
                    #slope of bottomline
                    if g_brx-g_blx !=0:
                        s_b = (g_bry-g_bly)/(g_brx-g_blx)
                    else:
                        s_b = float('inf')
                    c_b = g_bly-s_b*g_blx
                    s_p = math.tan(gr-np.pi/2) #slope, perpendicular
                    if s_p>9999999:
                        s_p = float('inf')
                    assert((s_b == float('inf') and s_t == float('inf')) or abs(s_b-s_t)<0.00001)
                    assert((s_b == float('inf') and s_p==0) or (s_p==float('inf') and s_b==0) or abs(s_b-(1/s_p))<0.00001)
                    if not np.isinf(s_p) and not np.isinf(s_t):
                        c_mid = tile_y-s_t*tile_x #assumes s_t and slope of mid line are the same
                    li_x = g_lx
                    li_y = g_ly
                    ri_x = g_rx
                    ri_y = g_ry
                    for index,(ty,tx) in enumerate(zip(hit_tile_ys,hit_tile_xs)):
                        targ_r[b, 0 if isHorz else 1, ty, tx] = math.asin(gr/np.pi)/np.pi

                        tile_x = tx+0.5
                        tile_y = ty+0.5

                        #Calculate T top and B bottom boudaries (from tile center)
                        if not np.isinf(s_p) and not np.isinf(s_t):
                            c_p = tile_y-s_p*tile_x
                            ti_x = (c_p-c_t)/(s_t-s_p)
                            ti_y = s_p*ti_x+c_p
                            bi_x = (c_p-c_b)/(s_b-s_p)
                            bi_y = s_p*bi_x+c_p
                        elif np.isinf(s_t):
                            ti_x = g_tlx
                            ti_y = tile_y
                            bi_x = g_blx
                            bi_y = tile_y
                        elif np.isinf(s_p):
                            ti_x = tile_x
                            ti_y = g_tly
                            bi_x = tile_x
                            bi_y = g_bly
                        else:
                            assert(False)

                        T = math.sqrt((ti_x-tile_x)**2 + (ti_y-tile_y)**2)
                        assert(T<MAX_H_PRED)
                        if abs(norm_angle(math.atan2(-ti_y+tile_y,ti_x-tile_x) - (gr+np.pi/2)))>0.01: #These should be identicle if top line above tile center
                            T *= -1
                        targ_T[b, 0 if isHorz else 1, ty, tx] = inv_tanh(T/MAX_H_PRED)

                        B = math.sqrt((bi_x-tile_x)**2 + (bi_y-tile_y)**2)
                        assert(B<MAX_H_PRED)
                        if abs(norm_angle(math.atan2(-bi_y+bile_y,bi_x-bile_x) - (gr-np.pi/2)))>0.01:
                            B *= -1
                        targ_B[b, 0 if isHorz else 1, ty, tx] = inv_tanh(B/MAX_H_PRED)

                        #same thing for left and right, but we cap L and R at 1 TODO
                        #Calculate L left and R right boudaries (from tile center)
                        L = math.sqrt((li_x-tile_x)**2 + (li_y-tile_y)**2)
                        L = min(MAX_W_PRED-0.01,L)
                        if abs(norm_angle(math.atan2(-li_y+tile_y,li_x-tile_x) - (gr+np.pi)))>0.01: #These should be identicle if top line above tile center
                            L *= -1
                        targ_L[b, 0 if isHorz else 1, ty, tx] = inv_tanh(L/MAX_W_PRED)

                        R = math.sqrt((ri_x-tile_x)**2 + (ri_y-tile_y)**2)
                        R = min(MAX_W_PRED-0.01,R)
                        if abs(norm_angle(math.atan2(-ri_y+tile_y,ri_x-tile_x) - gr))>0.01: #These should be identicle if top line above tile center
                            R *= -1
                        targ_R[b, 0 if isHorz else 1, ty, tx] = inv_tanh(R/MAX_W_PRED)

                        tcls[b, 0 if isHorz else 1, ty, tx] = target[b, t,13:]
                        tconf[b, 0 if isHorz else 1, ty, tx] = 1

                else:

                    ##AXIS ALIGNED
                    #slopes (s_) and y-intersections (c_)
                    if isHorz:
                        if (g_rx-g_lx) !=0:
                            s_len = (g_ry-g_ly)/(g_rx-g_lx)
                        else:
                            s_len = float('inf')
                        if (g_bx-g_tx) !=0:
                            s_perp = (g_by-g_ty)/(g_bx-g_tx)
                        else:
                            s_perp = float('inf')
                        if not math.isinf(s_perp) and not math.isinf(s_len):
                            s_len= (s_len-1/s_perp)/2
                            s_prep= -1/s_len

                        s_t=s_b=s_len
                        s_l=s_r=s_perp
                        if gr<np.pi/2 and gr>-np.pi/2:
                            c_t = g_ty-s_t*g_tx
                            c_b = g_by-s_b*g_bx
                            if math.isinf(s_perp):
                                c_l = g_lx
                                c_r = g_rx
                            else:
                                c_l = g_ly-s_l*g_lx
                                c_r = g_ry-s_r*g_rx
                            gt_left_x = g_lx
                            gt_left_y = g_ly
                            gt_right_x = g_rx
                            gt_right_y = g_ry
                        else:
                            c_t = g_by-s_t*g_bx
                            c_b = g_ty-s_b*g_tx
                            if math.isinf(s_perp):
                                c_l = g_rx
                                c_r = g_lx
                            else:
                                c_l = g_ry-s_l*g_rx
                                c_r = g_ly-s_r*g_lx
                            gt_left_x = g_rx
                            gt_left_y = g_ry
                            gt_right_x = g_lx
                            gt_right_y = g_ly
                    else:
                        #we're going to be inverting (x-y) everything to allow the same processing the horizontal lines use. 
                        if (g_ry-g_ly) !=0:
                            s_len = (g_rx-g_lx)/(g_ry-g_ly)
                        else:
                            s_len = float('inf')
                        if (g_by-g_ty) !=0:
                            s_perp = (g_bx-g_tx)/(g_by-g_ty)
                        else:
                            s_perp = float('inf')
                        if not math.isinf(s_perp) and not math.isinf(s_len):
                            s_len= (s_len-1/s_perp)/2
                            s_prep= -1/s_len
                        s_t=s_b=s_len
                        s_l=s_r=s_perp
                        if gr<0 or gr>np.pi:
                            c_t = g_bx-s_t*g_by
                            c_b = g_tx-s_b*g_ty
                            if math.isinf(s_perp):
                                c_l = g_ly
                                c_r = g_ry
                            else:
                                c_l = g_lx-s_l*g_ly
                                c_r = g_rx-s_r*g_ry
                            gt_left_x = g_ly
                            gt_left_y = g_lx
                            gt_right_x = g_ry
                            gt_right_y = g_rx
                        else:
                            c_t = g_tx-s_t*g_ty
                            c_b = g_bx-s_b*g_by
                            if math.isinf(s_perp):
                                c_l = g_ry
                                c_r = g_ly
                            else:
                                c_l = g_rx-s_l*g_ry
                                c_r = g_lx-s_r*g_ly
                            gt_left_x = g_ry
                            gt_left_y = g_rx
                            gt_right_x = g_ly
                            gt_right_y = g_lx

                        #old
                        #s_t=s_b=s_perp
                        #s_l=s_r=s_len
                        #if gr<0 and gr<np.pi:
                        #    c_t = g_ry-s_t*g_rx
                        #    c_b = g_ly-s_b*g_lx
                        #    if math.isinf(s_perp):
                        #        c_l = g_tx
                        #        c_r = g_bx
                        #    else:
                        #        c_l = g_ty-s_l*g_tx
                        #        c_r = g_by-s_r*g_bx
                        #else:
                        #    c_t = g_ly-s_t*g_lx
                        #    c_b = g_ry-s_b*g_rx
                        #    if math.isinf(s_perp):
                        #        c_l = g_bx
                        #        c_r = g_tx
                        #    else:
                        #        c_l = g_by-s_l*g_bx
                        #        c_r = g_ty-s_r*g_tx

                    
                    #print('level:{}, r:{:.3f},  {}'.format(level,gr,list(zip(hit_tile_xs,hit_tile_ys))))
                    #print('s_t/b:{}, c_t:{}, c_b:{},  s_l/r:{}, c_l:{}, c_r:{}'.format(s_t,c_t,c_b,s_l,c_l,c_r))
                    if isHorz:
                        DEBUG_max_x = max(hit_tile_xs)
                    else:
                        DEBUG_max_x = max(hit_tile_ys)
                    #t#times_lines.append(timeit.default_timer()-tic2)
                    for cell_index,(cell_y,cell_x) in enumerate(zip(hit_tile_ys,hit_tile_xs)): #kind of confusing, but here "tx ty" is for tile-i tile-j
                        #t#tic2=timeit.default_timer()
                        if isHorz:
                            tx = cell_x
                            ty = cell_y
                        else:
                            tx = cell_y
                            ty = cell_x

                        tile_x = tx+0.5
                        tile_y = ty+0.5

                        #top and bottom points (directly over tile center)
                        assert(not math.isinf(s_t))
                        ti_x = tile_x
                        ti_y = s_t*ti_x+c_t
                        bi_x = tile_x
                        bi_y = s_b*bi_x+c_b

                        if isHorz:
                            ti_y = max(ti_y,g_min_y)
                            bi_y = min(bi_y,g_max_y)
                        else:
                            ti_y = max(ti_y,g_min_x)
                            bi_y = min(bi_y,g_max_x)

                        #print('t:{},{}  ti:{},{}  bi:{},{}'.format(tx,ty,ti_x,ti_y,bi_x,bi_y))
                        

                        #left and right points (directly over/parallel to tile center)
                        li_y = tile_y
                        ri_y = tile_y

                        #new way
                        #The idea is to find the R (ri_x) that will maximize the IoU between the predicted box (axis aligned) and gt bb given the already selected ti_y and bi_y
                        #This requires looking at how the bb lines intersect eachother and the horizontal boundaries (ti_y/bi_y)
                        #We leverage the fact that at tile_x, the horizontal boundaries and the bb lines intersect (top and bottom)
                        #print('{},{} at ri_x comp  s_t:{}'.format(tx,ty,s_t))
                        if s_t==0 or math.isinf(s_l):
                            ri_x = c_r
                        else:
                            #find intersections
                            if s_t>0: #slope down
                                t_to_bh_inter_x = (bi_y-c_t)/s_t #intersection of gt bb top line to horizontal line at bi_y 
                                t_to_r_inter_x = (c_r-c_t)/(s_t-s_r) #intersection of gt bb top and right lines
                                ri_x = (t_to_bh_inter_x+tile_x)/2 #half way is optimal (iou type)
                                if t_to_bh_inter_x>t_to_r_inter_x: 
                                    #unless the optimal point is along the bb corner
                                    rtri_x = (c_t-c_r+(bi_y-ti_y)/2)/(s_r-s_t) #this is optimal along corner
                                    ri_x = min(ri_x,rtri_x)
                            else: #slope up
                                ###
                                #if cell_x==1 and cell_y==3:
                                #    bix = (bi_y-c_r)/s_r
                                #    tix = (ti_y-c_r)/s_r
                                #    print((bi_y,bix),(ti_y,tix))
                                #    img_f.line(draw[level],(int(VIZ_SIZE*bi_y),int(VIZ_SIZE*bix)),(int(VIZ_SIZE*ti_y),int(VIZ_SIZE*tix)),(255,255,255))
                                #    bix = (bi_y-c_b)/s_b
                                #    tix = (ti_y-c_b)/s_b
                                #    print((bi_y,bix),(ti_y,tix))
                                #    img_f.line(draw[level],(int(VIZ_SIZE*bi_y),int(VIZ_SIZE*bix)),(int(VIZ_SIZE*ti_y),int(VIZ_SIZE*tix)),(255,255,255))
                                ###
                                b_to_th_inter_x = (ti_y-c_b)/s_b #intersection of gt bb bot line to horizontal line at ti_y 
                                b_to_r_inter_x = (c_r-c_b)/(s_b-s_r) #intersection of gt bb bot and right lines (corner)
                                ri_x = (b_to_th_inter_x+tile_x)/2 #half way is optimal (iou type)
                                if b_to_th_inter_x>b_to_r_inter_x: 
                                    #unless the optimal point is along the bb corner
                                    rtri_x = (c_r-c_b+(bi_y-ti_y)/2)/(s_b-s_r) #this is optimal along corner
                                    #print('tri inside:{}, outside top:{}, bot:{}'.format(s_b*rtri_x+c_b-(s_r*rtri_x+c_r),s_r*rtri_x+c_r-ti_y,bi_y-(s_b*rtri_x+c_b)))
                                    ri_x = min(ri_x,rtri_x)
                                #print('b_to_th_inter_x:{}  b_to_r_inter_x:{} orri_x:{}  rtri_x:{}'.format(b_to_th_inter_x,b_to_r_inter_x,(b_to_th_inter_x+tile_x)/2,rtri_x if b_to_th_inter_x>b_to_r_inter_x else None))
                            if math.isinf(s_r):
                                ri_x = min(ri_x,c_r)
                            #else:
                            #    ri_x = min(ri_x,(ri_y-c_r)/s_r)

                        #compute li_x
                        if s_t==0 or math.isinf(s_l):
                            li_x = c_l
                        else:
                            #find intersections
                            #print('{},{} s_t:{}'.format(tx,ty,s_t))
                            if s_t<0: #slope down (left-ways)
                                t_to_bh_inter_x = (bi_y-c_t)/s_t #intersection of gt bb top line to horizontal line at bi_y 
                                t_to_l_inter_x = (c_l-c_t)/(s_t-s_l) #intersection of gt bb top and right lines
                                #print('t_to_bh_inter_x:{}  t_to_l_inter_x:{}'.format(t_to_bh_inter_x,t_to_l_inter_x))
                                li_x = (t_to_bh_inter_x+tile_x)/2 #half way is optimal (iou type)
                                if t_to_bh_inter_x<t_to_l_inter_x: 
                                    #unless the optimal point is along the bb corner
                                    ltri_x = (c_t-c_l+(bi_y-ti_y)/2)/(s_l-s_t) #this is optimal along corner
                                    li_x = max(li_x,ltri_x)
                            else: #slope up
                                b_to_th_inter_x = (ti_y-c_b)/s_b #intersection of gt bb bot line to horizontal line at ti_y 
                                b_to_l_inter_x = (c_l-c_b)/(s_b-s_l) #intersection of gt bb bot and right lines (corner)
                                #print('b_to_th_inter_x:{}  b_to_l_inter_x:{}'.format(b_to_th_inter_x,b_to_l_inter_x))
                                li_x = (b_to_th_inter_x+tile_x)/2 #half way is optimal (iou type)
                                if b_to_th_inter_x<b_to_l_inter_x: 
                                    #unless the optimal point is along the bb corner
                                    ltri_x = (c_l-c_b+(bi_y-ti_y)/2)/(s_b-s_l) #this is optimal along corner
                                    li_x = max(li_x,ltri_x)
                            if math.isinf(s_l):
                                li_x = max(li_x,c_l)
                            #else:
                            #    li_x = max(li_x,(li_y-c_l)/s_l)
                        #old way
                        #intersect_check_y = (max(ty,ti_y)+min(ty+1,bi_y))/2
                        #if math.isinf(s_l):
                        #    li_x = c_l
                        #else:
                        #    li_x = (li_y-c_l)/s_l
                        #if s_b!=0:
                        #    lbi_x = (intersect_check_y-c_b)/s_b
                        #    lti_x = (intersect_check_y-c_t)/s_t
                        #    li_x = max(li_x,
                        #            lbi_x if lbi_x<tile_x else -float('inf'),
                        #            lti_x if lti_x<tile_x else -float('inf'))

                        #if math.isinf(s_r):
                        #    ri_x = c_r
                        #else:
                        #    ri_x = (ri_y-c_r)/s_r
                        #if s_b!=0:
                        #    rbi_x = (intersect_check_y-c_b)/s_b
                        #    rti_x = (intersect_check_y-c_t)/s_t
                        #    #print('initial ri_x:{}, rbi_x:{}, rti_x:{}'.format(ri_x,rbi_x,rti_x))
                        #    ri_x = min(ri_x,
                        #            rbi_x if rbi_x>tile_x else float('inf'),
                        #            rti_x if rti_x>tile_x else float('inf'))

                        #print(cell_x,cell_y)
                        #if isHorz:
                        #    print('li_x,y:{},{}, ri_x,y:{},{}, ti_x,y:{},{}, bi_x,y:{},{}'.format(li_x,li_y,ri_x,ri_y,ti_x,ti_y,bi_x,bi_y))
                        #else:
                        #    print('V li_x,y:{},{}, ri_x,y:{},{}, ti_x,y:{},{}, bi_x,y:{},{}'.format(li_y,li_x,ri_y,ri_x,ti_y,ti_x,bi_y,bi_x))
                        #t#times_border.append(timeit.default_timer()-tic2)
                        #t#tic2=timeit.default_timer()

                        if assign_mode=='split':
                            #Predict based on position of the text line
                            #if isHorz:
                            if (ti_y+bi_y)/2<=tile_y:
                                assigned = 0 if isHorz else 2
                                not_assigned = 1 if isHorz else 3
                            else:
                                assigned = 1 if isHorz else 3
                                not_assigned = 0 if isHorz else 2

                            #isLeftEnd = li_x<=tx+1 and (ti_y+bi_y)/2<=ty+1 and (ti_y+bi_y)/2>=ty
                            #isRightEnd = tile_x>ri_x and ri_x>=tx and (ti_y+bi_y)/2<=ty+1 and (ti_y+bi_y)/2>=ty
                            isLeftEnd = li_x>tile_x 
                            isRightEnd = ri_x<tile_x 
                            #print('isLeftEnd: {} = {} > {}'.format(isLeftEnd,li_x,tile_x))
                            #print('isRightEnd: {} = {} < {}'.format(isRightEnd,ri_x,tile_x))

                            #evaluate if this tile is just barely contributing. skip it if it is
                            #print('{},{}:  l:{},{}   r:{},{}'.format(cell_x,cell_y,isLeftEnd,math.sqrt((tile_x-gt_left_x)**2 + (tile_y-gt_left_y)**2),isRightEnd,math.sqrt((tile_x-gt_right_x)**2 + (tile_y-gt_right_y)**2)))
                            #print('tile_x:{}, li_x:{}, ri_x:{}, gt_left_x:{}, gt_right_x:{}'.format(tile_x,li_x,ri_x,gt_left_x,gt_right_x))
                            #print('tile_x:{}, gt_left_x:{}, tile_y:{}, gt_left_y:{}'.format(tile_x,gt_left_x,tile_y,gt_left_y))
                            if ( len(hit_tile_ys)>1 and (
                                (isLeftEnd and math.sqrt((tile_x-gt_left_x)**2 + (tile_y-gt_left_y)**2)>END_BOUNDARY_THRESH) or
                                (isRightEnd and math.sqrt((tile_x-gt_right_x)**2 + (tile_y-gt_right_y)**2)>END_BOUNDARY_THRESH))):
                                #print('DEBUG: 1')
                                continue
                            #t#times_split_and_end.append(timeit.default_timer()-tic2)
                            #t#tic2=timeit.default_timer()

                            #if (( (li_x>tile_x and tx+1-li_x<END_UNMASK_THRESH) or 
                            #      (tile_x>ri_x and ri_x-tx<END_UNMASK_THRESH) ) and
                            #     len(hit_tile_ys)>1):
                            #    #it's a border tile, we'll just ignore it
                            #    continue
                            t_have_other_tiles = len(hit_tile_ys)-cell_index + (assignment==t).sum()>1
                            if conf_mask[b,assigned,cell_y,cell_x]==0 and t_have_other_tiles:
                                #print('DEBUG: 2')
                                continue
                                #print('{},{} skipped as already unmasked'.format(cell_x,cell_y))
                            elif assignment[assigned,cell_y,cell_x]!=-1:
                                if only_unmask:
                                    #print('DEBUG: 3')
                                    continue
                                #print('shared at {}, {}'.format(tx,ty))
                                shared_mask[b,assigned,cell_y,cell_x]=1
                                other_t = assignment[assigned,cell_y,cell_x]
                                #Uh oh, this half has been assigned already.
                                other_t_have_other_tiles = (assignment==other_t).sum()>1

                                #print('{},{}: me have other:{}  other have other:{}'.format(tx,ty,t_have_other_tiles,other_t_have_other_tiles))


                                if other_t_have_other_tiles and not t_have_other_tiles:
                                    #I get it!
                                    action='replace-other-t'
                                elif not other_t_have_other_tiles and t_have_other_tiles:
                                    #Keep everything unchanged. My other tile(s) will predict this
                                    action='skip-t'
                                elif other_t_have_other_tiles and t_have_other_tiles:
                                    #For which of us is this tile on the end of our BB?
                                    #if isHorz:
                                    t_end = (li_x>=tx and li_x<=tx+1) or (ri_x>=tx and ri_x<=tx+1)
                                    if isHorz:
                                        other_x1 = target[b, other_t, 5] / scale[level][0]
                                        other_x2 = target[b, other_t, 7] / scale[level][0]
                                    else:
                                        other_x1 = target[b, other_t, 6] / scale[level][1]
                                        other_x2 = target[b, other_t, 8] / scale[level][1]
                                    other_li_x = min(other_x1,other_x2)
                                    other_ri_x = max(other_x1,other_x2)
                                    other_t_end = (other_li_x>=tx and other_li_x<=tx+1) or (other_ri_x>=tx and other_ri_x<=tx+1)
                                    if t_end:
                                        left_side = li_x>=tx and li_x<=tx+1
                                    if other_t_end:
                                        other_left_side = other_li_x>=tx and other_li_x<=tx+1

                                    #other_ti_y = None #for convience
                                    #other_bi_y = None #for convience
                                    #else:
                                    #    t_end = (ti_y>=ty and ti_y<=ty+1) or (bi_y>=ty and bi_y<=ty+1)
                                    #    other_ti_y = target[b, other_t, 10] / scale[level][1]
                                    #    other_bi_y = target[b, other_t, 12] / scale[level][1]
                                    #    other_t_end = (other_ti_y>=ty and other_ti_y<=ty+1) or (other_bi_y>=ty and other_bi_y<=ty+1)
                                    #    if t_end:
                                    #        top_side = ti_y>=ty and ti_y<=ty+1
                                    #    if other_t_end:
                                    #        other_top_side = other_ti_y>=ty and other_ti_y<=ty+1

                                    #    other_li_y = None #for convience
                                    #    other_ri_y = None #for convience

                                    #print('{},{}: me end:{}  other end:{}'.format(tx,ty,t_end,other_t_end))
                                    if t_end and other_t_end:
                                        #Both ends? Great, if one of us has a large portion of our BB in this tile, we'll that one get predicted
                                        t_fraction_in_tile = fraction_in_tile(left_side,tx,ty,li_x,ri_x)#,ti_y,bi_y)
                                        other_t_fraction_in_tile = fraction_in_tile(other_left_side,tx,ty,other_li_x,other_ri_x)#,other_ti_y,other_bi_y)

                                        #print('{},{}: me frac:{}  other frac:{}'.format(tx,ty,t_fraction_in_tile,other_t_fraction_in_tile))
                                        if t_fraction_in_tile>0.4 and other_t_fraction_in_tile<0.33:
                                            #I get it!
                                            action='replace-other-t'
                                        elif t_fraction_in_tile<0.33 and other_t_fraction_in_tile>0.4:
                                            #let them get it
                                            action='skip-t'
                                        else:
                                            #We'll just unmask this. The behavoir isn't really important.
                                            action='unmask'
                                    elif not t_end and other_t_end:
                                        #I get it!
                                        action='replace-other-t'
                                    elif t_end and not other_t_end:
                                        #let them get it
                                        action='skip-t'
                                    else:
                                        #I don't know, unmask I guess
                                        #Or bump one up/down?
                                        action='unmask'
                                else:
                                    #Uuuhhhhhh, we both need this tile to predict us. Throw one to the other half. Hopefully this rarely happens.
                                    t_end = (li_x>=tx and li_x<=tx+1) or (ri_x>=tx and ri_x<=tx+1)
                                    if isHorz:
                                        other_x1 = target[b, other_t, 5] / scale[level][0]
                                        other_x2 = target[b, other_t, 7] / scale[level][0]
                                    else:
                                        other_x1 = target[b, other_t, 6] / scale[level][1]
                                        other_x2 = target[b, other_t, 8] / scale[level][1]
                                    other_li_x = min(other_x1,other_x2)
                                    other_ri_x = max(other_x1,other_x2)
                                    other_t_end = (other_li_x>=tx and other_li_x<=tx+1) or (other_ri_x>=tx and other_ri_x<=tx+1)
                                    if t_end:
                                        left_side = li_x>=tx and li_x<=tx+1
                                        t_fraction_in_tile = fraction_in_tile(left_side,tx,ty,li_x,ri_x)
                                    else:
                                        t_fraction_in_tile = 1/(ri_x-li_x)
                                    if other_t_end:
                                        other_left_side = other_li_x>=tx and other_li_x<=tx+1
                                        other_t_fraction_in_tile = fraction_in_tile(other_left_side,tx,ty,other_li_x,other_ri_x)
                                    else:
                                        other_t_fraction_in_tile = 1/(other_ri_x-other_li_x)

                                    if abs(tile_y-(ti_y+bi_y)/2)<UNMASK_CENT_DIST_THRESH and targ_conf[b,not_assigned,cell_y,cell_x]==0:
                                        #if I'm close, I'll just jump to othr
                                        tmp=assigned
                                        assigned=not_assigned
                                        not_assigned=tmp
                                        action='replace-other-t'
                                    elif t_fraction_in_tile>0.5 and other_t_fraction_in_tile<0.5:
                                        #I get it!
                                        action='replace-other-t'
                                    elif t_fraction_in_tile<0.5 and other_t_fraction_in_tile>0.5:
                                        #let them get it
                                        action='skip-t'
                                    else:
                                        #For now, just unmask
                                        action='unmask'
                                    print('WARNING: Tile is only tile for two BBs. Currently unmasking')
                                #print(action)
                                if action=='skip-t':
                                    #t#times_handle_multi.append(timeit.default_timer()-tic2)
                                    #print('DEBUG: 4')
                                    continue
                                elif action=='replace-other-t':
                                    pass #I'll just overwrite it
                                elif action=='unmask':
                                    mask[b,assigned,cell_y,cell_x]=0
                                    conf_mask[b,assigned,cell_y,cell_x]=0
                                    if assignment[assigned,cell_y,cell_x]!=-1:
                                        assignment[assigned,cell_y,cell_x]=-2

                                    #t#times_handle_multi.append(timeit.default_timer()-tic2)
                                    #print('DEBUG: 5')
                                    continue
                                else:
                                    raise NotImplementedError('Unknown sharing action: {}'.format(action))

                            #else:
                            #t#times_handle_multi.append(timeit.default_timer()-tic2)
                            #t#tic2=timeit.default_timer()
                            if only_unmask and targ_conf[b,assigned,cell_y,cell_x]==0:
                                conf_mask[b,assigned,cell_y,cell_x]= 0
                            else:
                                assignment[assigned,cell_y,cell_x]=t
                                mask[b,assigned,cell_y,cell_x]=1
                                targ_conf[b,assigned,cell_y,cell_x]=1
                                conf_mask[b,assigned,cell_y,cell_x]=1

                            if close_anchor_rule=='unmask':
                                #if ((isHorz and abs(tile_y-(ti_y+bi_y)/2)<UNMASK_CENT_DIST_THRESH) or (not isHorz and abs(tile_x-(li_x+ri_x)/2)<UNMASK_CENT_DIST_THRESH)) and targ_conf[b,not_assigned,cell_y,cell_x]==0:
                                if abs(tile_y-(ti_y+bi_y)/2)<UNMASK_CENT_DIST_THRESH and targ_conf[b,not_assigned,cell_y,cell_x]==0:
                                    conf_mask[b,not_assigned,cell_y,cell_x]= 0 
                                
                                if assigned==0 and ((ti_y+bi_y)/2)-ty<UNMASK_CENT_DIST_THRESH and cell_y>0 and targ_conf[b,not_assigned,cell_y-1,cell_x]==0:
                                    conf_mask[b,not_assigned,cell_y-1,cell_x]= 0
                                elif assigned==1 and (ty+1)-((ti_y+bi_y)/2)<UNMASK_CENT_DIST_THRESH and cell_y<nH-1 and targ_conf[b,not_assigned,cell_y+1,cell_x]==0:
                                    conf_mask[b,not_assigned,cell_y+1,cell_x]= 0
                                if assigned==2 and ((ti_y+bi_y)/2)-ty<UNMASK_CENT_DIST_THRESH and cell_x>0 and targ_conf[b,not_assigned,cell_y,cell_x-1]==0:
                                    conf_mask[b,not_assigned,cell_y,cell_x-1]= 0
                                elif assigned==3 and (ty+1)-((ti_y+bi_y)/2)<UNMASK_CENT_DIST_THRESH and cell_x<nW-1 and targ_conf[b,not_assigned,cell_y,cell_x+1]==0:
                                    conf_mask[b,not_assigned,cell_y,cell_x+1]= 0
                                #elif assigned==2 and ((li_x+ri_x)/2)-tx<UNMASK_CENT_DIST_THRESH and tx>0 and targ_conf[b,not_assigned,cell_y,cell_x-1]==0:
                                #    conf_mask[b,not_assigned,cell_y,cell_x-1]= 0
                                #elif assigned==3 and tx-((li_x+ri_x)/2)<UNMASK_CENT_DIST_THRESH and tx<nW-1 and targ_conf[b,not_assigned,cell_y,cell_x+1]==0:
                                #    conf_mask[b,not_assigned,cell_y,cell_x+1]= 0
                        else:
                            raise NotImplementedError('Uknown tile assignment mode: {}'.format(assign_mode))
                        if only_unmask:
                            #print('DEBUG: 6')
                            continue
                                    
                        #t#times_unmask_end.append(timeit.default_timer()-tic2)
                        #t#tic2=timeit.default_timer()


                        targ_cls[b, assigned, cell_y, cell_x] = target[b, t,13:]
                        targ_r[b, assigned, cell_y, cell_x] = math.asin(gr/np.pi)/np.pi
 
                        #assert(ti_y==ti_y and bi_y==bi_y and ri_x==ri_x and li_x ==li_x)
                        #assert(not (math.isinf(ti_y) or math.isinf(bi_y) or math.isinf(ri_x) or math.isinf(li_x)))
                        num_assigned +=1
                        if isHorz:
                            T=ti_y-tile_y #negative if above tile center (just add predcition to center)
                            #T = max(min(T,MAX_H_PRED-0.01),0.01-MAX_H_PRED)
                            assert(abs(T)<MAX_H_PRED)
                            targ_T[b, assigned, cell_y, cell_x] = inv_tanh(T/MAX_H_PRED)
                            B=bi_y-tile_y 
                            #B = max(min(B,MAX_H_PRED-0.01),0.01-MAX_H_PRED)
                            assert(abs(B)<MAX_H_PRED)
                            targ_B[b, assigned, cell_y, cell_x] = inv_tanh(B/MAX_H_PRED)
                            
                            L=li_x-tile_x #negative if left of tile center (just add predcition to center)
                            L = max(min(L,MAX_W_PRED-0.01),0.01-MAX_W_PRED)
                            targ_L[b, assigned, cell_y, cell_x] = inv_tanh(L/MAX_W_PRED)
                            R=ri_x-tile_x 
                            R = max(min(R,MAX_W_PRED-0.01),0.01-MAX_W_PRED)
                            targ_R[b, assigned, cell_y, cell_x] = inv_tanh(R/MAX_W_PRED)

                            assert(R>=L and T<=B)
                        else:
                            #T=ti_y-tile_y #negative if above tile center (just add predcition to center)
                            ##T = max(min(T,MAX_H_PRED-0.01),0.01-MAX_H_PRED)
                            #assert(abs(T)<MAX_H_PRED)
                            #targ_T[b, assigned, cell_y, cell_x] = inv_tanh(T/MAX_H_PRED)
                            #B=bi_y-tile_y 
                            ##B = max(min(B,MAX_H_PRED-0.01),0.01-MAX_H_PRED)
                            #assert(abs(B)<MAX_H_PRED)
                            #targ_B[b, assigned, cell_y, cell_x] = inv_tanh(B/MAX_H_PRED)
                            #L=ri_x-tile_x #negative if left of tile center (just add predcition to center)
                            #L = max(min(L,MAX_W_PRED-0.01),0.01-MAX_W_PRED)
                            ##print('L:{} {}-{}={}'.format(L,ri_x,tile_x,ri_x-tile_x))
                            #targ_L[b, assigned, cell_y, cell_x] = inv_tanh(L/MAX_W_PRED)
                            #R=li_x-tile_x 
                            #R = max(min(R,MAX_W_PRED-0.01),0.01-MAX_W_PRED)
                            #targ_R[b, assigned, cell_y, cell_x] = inv_tanh(R/MAX_W_PRED)
                            #old
                            T=li_x-tile_x #negative if above tile center (just add predcition to center)
                            T = max(min(T,MAX_W_PRED-0.01),0.01-MAX_W_PRED)
                            targ_T[b, assigned, cell_y, cell_x] = inv_tanh(T/MAX_W_PRED)
                            B=ri_x-tile_x
                            B = max(min(B,MAX_W_PRED-0.01),0.01-MAX_W_PRED)
                            targ_B[b, assigned, cell_y, cell_x] = inv_tanh(B/MAX_W_PRED)
                            
                            L=ti_y-tile_y #negative if left of tile center (just add predcition to center)
                            assert(abs(L)<MAX_H_PRED)
                            targ_L[b, assigned, cell_y, cell_x] = inv_tanh(L/MAX_H_PRED)
                            R=bi_y-tile_y 
                            assert(abs(R)<MAX_H_PRED)
                            targ_R[b, assigned, cell_y, cell_x] = inv_tanh(R/MAX_H_PRED)

                            assert(R>=L and T<=B)
                        #t#times_assign.append(timeit.default_timer()-tic2)




            #t#tic2=timeit.default_timer()
            # Calculate overlaps between ground truth and best matching prediction
            if calc_stats:
                for level in range(len(nHs)):
                    gx = target[b, t, 0] / scale[level][0]
                    gy = target[b, t, 1] / scale[level][1]
                    gw = target[b, t, 4] / scale[level][0]
                    gh = target[b, t, 3] / scale[level][1]

                    class_selector = torch.logical_and(pred_cls[level][b].argmax(dim=3)==torch.argmax(target[b,t,13:]), pred_conf[level][b]>0)
                    all_selector = pred_conf[level][b]>0
                    pred_right_label_boxes = pred_boxes[level][b][class_selector] #this is already normalized to tile space
                    pred_right_label_boxes_all = pred_boxes[level][b][all_selector] #this is already normalized to tile space
                    #convert?
                 
                    gt_area_covered, pred_area_covered = bbox_coverage_axis_rot((gx,gy,gr,gh,gw), pred_right_label_boxes)
                    gt_area_covered_all, pred_area_covered_all = bbox_coverage_axis_rot((gx,gy,gr,gh,gw), pred_right_label_boxes_all)
                    #print(pred_area_covered_all)
                    assert(len(pred_area_covered)==0 or max(pred_area_covered)<=1)
                    if gt_area_covered is not None:
                        #covered_gt_area += gt_area_covered
                        #covered_gt_area_all += gt_area_covered_all

                        covered_gt_areaB[t] = max(gt_area_covered,covered_gt_areaB[t])
                        covered_gt_areaB_all[t] = max(gt_area_covered_all,covered_gt_areaB_all[t])
                        #if gt_area_covered>0.5:
                        #    recall+=1
                        #if gt_area_covered_all>0.5:
                        #    recall_all+=1
                        on_pred_areaB[level][class_selector] = torch.max(on_pred_areaB[level][class_selector],torch.FloatTensor(pred_area_covered))
                        on_pred_areaB_all[level][all_selector] = torch.max(on_pred_areaB_all[level][all_selector],torch.FloatTensor(pred_area_covered_all))
                    else:
                        nGT-=1
                    #pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
                    #score = pred_conf[b, best_n, gj, gi]
                    #import pdb; pdb.set_trace()
                    #if iou > 0.5 and pred_label == torch.argmax(target[b,t,13:]) and score > 0:
                    #    nCorrect += 1
                #t#times_overlaps.append(timeit.default_timer()-tic2)


        if VISUAL_DEBUG:
            for level in range(len(nHs)):
                colorIndex=0
                print('draw level {}'.format(level))
                #draw[level][0,0,:]=255
                draw_level = draw[level]
                for ty in range(nHs[level]):
                    for tx in range(nWs[level]):
                        d_tile_x = (tx+0.5)*VIZ_SIZE
                        d_tile_y = (ty+0.5)*VIZ_SIZE
                        #for masks_m,ch in zip([t_confs],[2]):
                        #for masks_m,ch in zip([conf_masks,t_confs],[1,2]):
                        if conf_masks[level][b,0,ty,tx]:
                            draw_level[ty*VIZ_SIZE+1:int((ty+0.5)*VIZ_SIZE),tx*VIZ_SIZE+1:(tx+1)*VIZ_SIZE,1]+=50
                            if t_confs[level][b,0,ty,tx]:
                                draw_level[ty*VIZ_SIZE+1:int((ty+0.5)*VIZ_SIZE),tx*VIZ_SIZE+1:(tx+1)*VIZ_SIZE,2]+=50
                        if conf_masks[level][b,1,ty,tx]:
                            draw_level[int((ty+0.5)*VIZ_SIZE):(ty+1)*VIZ_SIZE,tx*VIZ_SIZE+1:(tx+1)*VIZ_SIZE,1]+=50
                            if t_confs[level][b,1,ty,tx]:
                                draw_level[int((ty+0.5)*VIZ_SIZE):(ty+1)*VIZ_SIZE,tx*VIZ_SIZE+1:(tx+1)*VIZ_SIZE,2]+=50
                        if conf_masks[level][b,2,ty,tx]:
                            draw_level[ty*VIZ_SIZE+1:(ty+1)*VIZ_SIZE,tx*VIZ_SIZE+1:int((tx+0.5)*VIZ_SIZE),1]+=50
                            if t_confs[level][b,2,ty,tx]:
                                draw_level[ty*VIZ_SIZE+1:(ty+1)*VIZ_SIZE,tx*VIZ_SIZE+1:int((tx+0.5)*VIZ_SIZE),2]+=50
                        if conf_masks[level][b,3,ty,tx]:
                            draw_level[ty*VIZ_SIZE+1:(ty+1)*VIZ_SIZE,int((tx+0.5)*VIZ_SIZE):(tx+1)*VIZ_SIZE,1]+=50
                            if t_confs[level][b,3,ty,tx]:
                                draw_level[ty*VIZ_SIZE+1:(ty+1)*VIZ_SIZE,int((tx+0.5)*VIZ_SIZE):(tx+1)*VIZ_SIZE,2]+=50
                        #if (shared_masks[level][b,:,ty,tx]==1).any():
                        #    draw_level[ty*VIZ_SIZE+1:(ty+1)*VIZ_SIZE,tx*VIZ_SIZE+1:(tx+1)*VIZ_SIZE,0]+=90
                        if shared_masks[level][b,0,ty,tx]:
                            draw_level[ty*VIZ_SIZE+1:int((ty+0.5)*VIZ_SIZE),tx*VIZ_SIZE+1:(tx+1)*VIZ_SIZE,0]+=50
                        if shared_masks[level][b,0,ty,tx]:
                            draw_level[int((ty+0.5)*VIZ_SIZE):(ty+1)*VIZ_SIZE,tx*VIZ_SIZE+1:(tx+1)*VIZ_SIZE,0]+=50
                        if shared_masks[level][b,2,ty,tx]:
                            draw_level[ty*VIZ_SIZE+1:(ty+1)*VIZ_SIZE,tx*VIZ_SIZE+1:int((tx+0.5)*VIZ_SIZE),0]+=50
                        if shared_masks[level][b,3,ty,tx]:
                            draw_level[ty*VIZ_SIZE+1:(ty+1)*VIZ_SIZE,int((tx+0.5)*VIZ_SIZE):(tx+1)*VIZ_SIZE,0]+=50
                for ty in range(nHs[level]):
                    for tx in range(nWs[level]):
                        d_tile_x = (tx+0.5)*VIZ_SIZE
                        d_tile_y = (ty+0.5)*VIZ_SIZE
                        for i,bright in zip(range(nA),[255,205,155,100]):
                            if masks[level][b,i,ty,tx]:
                                if i==0:
                                    coff_x = 0
                                    coff_y = -VIZ_SIZE*0.25
                                elif i==1:
                                    coff_x = 0
                                    coff_y = VIZ_SIZE*0.25
                                elif i==2:
                                    coff_x = -VIZ_SIZE*0.25
                                    coff_y = 0
                                elif i==3:
                                    coff_x = VIZ_SIZE*0.25
                                    coff_y = 0
                                if i==0 or i==1:
                                    T= torch.tanh(t_Ts[level][b,i,ty,tx])*MAX_H_PRED
                                    B= torch.tanh(t_Bs[level][b,i,ty,tx])*MAX_H_PRED
                                    L= torch.tanh(t_Ls[level][b,i,ty,tx])*MAX_W_PRED
                                    R= torch.tanh(t_Rs[level][b,i,ty,tx])*MAX_W_PRED
                                else:
                                    T= torch.tanh(t_Ts[level][b,i,ty,tx])*MAX_W_PRED
                                    B= torch.tanh(t_Bs[level][b,i,ty,tx])*MAX_W_PRED
                                    L= torch.tanh(t_Ls[level][b,i,ty,tx])*MAX_H_PRED
                                    R= torch.tanh(t_Rs[level][b,i,ty,tx])*MAX_H_PRED

                                #img_f.line(draw_level,(int(d_tile_x-1+coff_x),int(d_tile_y+coff_y)),(int(d_tile_x-1+coff_x),int(d_tile_y+(T*VIZ_SIZE))),(bright,bright,0),1)
                                #img_f.line(draw_level,(int(d_tile_x+coff_x),int(d_tile_y+coff_y)),(int(d_tile_x+coff_x),int(d_tile_y+(B*VIZ_SIZE))),(bright,0,0),1)
                                #img_f.line(draw_level,(int(d_tile_x+coff_x),int(d_tile_y-1+coff_y)),(int(d_tile_x+(L*VIZ_SIZE)),int(d_tile_y-1+coff_y)),(bright,0,bright),1)
                                #img_f.line(draw_level,(int(d_tile_x+coff_x),int(d_tile_y+coff_y)),(int(d_tile_x+(R*VIZ_SIZE)),int(d_tile_y+coff_y)),(0,bright,bright),1)
                                
                                draw_level[int(d_tile_y+coff_y)-1:int(d_tile_y+coff_y)+2,int(d_tile_x+coff_x)-1:int(d_tile_x+coff_x)+2]=draw_colors[colorIndex]
                                #if i==0 or i==1:
                                drawT = int(d_tile_y+T*VIZ_SIZE)
                                drawB = int(d_tile_y+B*VIZ_SIZE)
                                drawL = int(d_tile_x+L*VIZ_SIZE)
                                drawR = int(d_tile_x+R*VIZ_SIZE)
                                #elif i==2 or i==3:
                                #    drawL = int(d_tile_x+T*VIZ_SIZE)
                                #    drawR = int(d_tile_x+B*VIZ_SIZE)
                                #    drawB = int(d_tile_y+L*VIZ_SIZE)
                                #    drawT = int(d_tile_y+R*VIZ_SIZE)
                                #assert(drawL>=-VIZ_SIZE and drawL<=draw_level.shape[1]+VIZ_SIZE)
                                #assert(drawR>=-VIZ_SIZE and drawR<=draw_level.shape[1]+VIZ_SIZE)
                                #assert(drawT>=-VIZ_SIZE and drawT<=draw_level.shape[0]+VIZ_SIZE)
                                #assert(drawB>=-VIZ_SIZE and drawB<=draw_level.shape[0]+VIZ_SIZE)
                                img_f.line(draw_level,(drawL,drawT),(drawR,drawT),draw_colors[colorIndex])
                                img_f.line(draw_level,(drawR,drawT),(drawR,drawB),draw_colors[colorIndex])
                                img_f.line(draw_level,(drawR,drawB),(drawL,drawB),draw_colors[colorIndex])
                                img_f.line(draw_level,(drawL,drawB),(drawL,drawT),draw_colors[colorIndex])
                                colorIndex = (colorIndex+1)%len(draw_colors)
                                

                fig = plt.figure()
                axs=[]
                axs.append( plt.subplot() )
                axs[-1].set_axis_off()
                axs[-1].imshow(draw_level)
            plt.show()

        if calc_stats:
            for level in range(len(nHs)):
                on_pred_area += on_pred_areaB[level].sum().item()
                on_pred_area_all += on_pred_areaB_all[level].sum().item()
                #nPred += on_pred_areaB[level].size(0)
                precision += (on_pred_areaB[level]>HIT_THRESH).sum().item()
                precision_all += (on_pred_areaB_all[level]>HIT_THRESH).sum().item()
                nPred += (pred_conf[level][b]>0).sum().item()
            covered_gt_area += covered_gt_areaB.sum().item()
            recall += (covered_gt_areaB>HIT_THRESH).sum().item()
            covered_gt_area_all += covered_gt_areaB_all.sum().item()
            recall_all += (covered_gt_areaB_all>HIT_THRESH).sum().item()
        #t#print('time all batch{}: {}'.format(b,timeit.default_timer()-tic))
        #t#print('  times_setup_and_level_select: {}   std: {}, count{}'.format(np.mean(times_setup_and_level_select),np.std(times_setup_and_level_select),len(times_setup_and_level_select)))
        #t#print('  times_level_setup: {}   std: {}, count{}'.format(np.mean(times_level_setup),np.std(times_level_setup),len(times_level_setup)))
        #t#print('  times_tile_hit: {}   std: {}, count{}'.format(np.mean(times_tile_hit),np.std(times_tile_hit),len(times_tile_hit)))
        #t#print('  times_lines: {}   std: {}, count{}'.format(np.mean(times_lines),np.std(times_lines),len(times_lines)))
        #t#print('  times_border: {}   std: {}, count{}'.format(np.mean(times_border),np.std(times_border),len(times_border)))
        #t#print('  times_split_and_end: {}   std: {}, count{}'.format(np.mean(times_split_and_end),np.std(times_split_and_end),len(times_split_and_end)))
        #t#print('  times_handle_multi: {}   std: {}, count{}'.format(np.mean(times_handle_multi),np.std(times_handle_multi),len(times_handle_multi)))
        #t#print('  times_unmask_end: {}   std: {}, count{}'.format(np.mean(times_unmask_end),np.std(times_unmask_end),len(times_unmask_end)))
        #t#print('  times_assign: {}   std: {}, count{}'.format(np.mean(times_assign),np.std(times_assign),len(times_assign)))
        #t#print('  times_overlaps: {}   std: {}, count{}'.format(np.mean(times_overlaps),np.std(times_overlaps),len(times_overlaps)))

    #assert(False and 'TODO verify this works! Have you crafted corner cases?')
    for Ls,Rs,Ts,Bs in zip(t_Ls,t_Rs,t_Ts,t_Bs):
        assert((Ls<=Rs).all() and (Ts<=Bs).all())
    return (nGT, 
            masks,
            conf_masks, 
            t_Ls, 
            t_Ts, 
            t_Rs, 
            t_Bs, 
            t_rs, 
            t_confs, 
            t_clss, 
            on_pred_area/nPred if nPred>0 else 0, 
            covered_gt_area/nGT if nGT>0 else 0, 
            recall/nGT if nGT>0 else 1, 
            precision/nPred if nPred>0 else 1,
            on_pred_area_all/nPred if nPred>0 else 0, 
            covered_gt_area_all/nGT if nGT>0 else 0, 
            recall_all/nGT if nGT>0 else 1, 
            precision_all/nPred if nPred>0 else 1,
            )

def bbox_coverage_axis_rot(box_gt, pred_boxes):
    """
    Computes how much the gt is covered by predictions and how much of each prediction is covered by the gt
    """
    gt_poly =  Polygon(xyrhwToCorners(*box_gt[0:5]))
    pred_areas=[]
    agglom = None
    for i in range(pred_boxes.size(0)):
        pred_poly = Polygon([pred_boxes[i,0:2],(pred_boxes[i,2],pred_boxes[i,1]),pred_boxes[i,2:4],(pred_boxes[i,0],pred_boxes[i,3])])
        if gt_poly.intersects(pred_poly) and pred_poly.area>0:
            inter = gt_poly.intersection(pred_poly)
            pred_areas.append(inter.area/pred_poly.area)
            if agglom is None:
                agglom = inter
            else:
                agglom = agglom.union(inter)
        else:
            pred_areas.append(0)

    if agglom is not None:
        try:
            return agglom.intersection(gt_poly).area/gt_poly.area, pred_areas
        except shapely.errors.TopologicalError:
            print("Unknown shapely error, agglom points {}".format(len(agglom.exterior.coords)))
            return None, None
    else:
        return 0, pred_areas




def bbox_coverage(box1, box2, x1y1x2y2_1=True, x1y1x2y2_2=True):
    """
    Returns the covereage, how much of box1 is covered by the boxes (box2) and how much each box2 is covered by box1
    """
    assert(box1.size(0)==1)
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        #I assume H and W are half
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] , box1[:, 0] + box1[:, 2] 
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] , box1[:, 1] + box1[:, 3] 
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] , box2[:, 0] + box2[:, 2] 
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] , box2[:, 1] + box2[:, 3] 
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    #iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    #Remove pred boxes which do not intersect the GT in question
    inter_rect_x1=inter_rect_x1[inter_area>0]
    inter_rect_y1=inter_rect_y1[inter_area>0]
    inter_rect_x2=inter_rect_x2[inter_area>0]
    inter_rect_y2=inter_rect_y2[inter_area>0]

    #calculate the intersetions among the remaining pred-intersection boxes
    inter_rect_x1_R = inter_rect_x1[None,:].expand(inter_rect_x1.size(0),-1)
    inter_rect_x1_C = inter_rect_x1[:,None].expand(-1,inter_rect_x1.size(0))
    inter_rect_y1_R = inter_rect_y1[None,:].expand(inter_rect_y1.size(0),-1)
    inter_rect_y1_C = inter_rect_y1[:,None].expand(-1,inter_rect_y1.size(0))
    inter_rect_x2_R = inter_rect_x2[None,:].expand(inter_rect_x2.size(0),-1)
    inter_rect_x2_C = inter_rect_x2[:,None].expand(-1,inter_rect_x2.size(0))
    inter_rect_y2_R = inter_rect_y2[None,:].expand(inter_rect_y2.size(0),-1)
    inter_rect_y2_C = inter_rect_y2[:,None].expand(-1,inter_rect_y2.size(0))
    
    inter_inter_rect_x1 = torch.max(inter_rect_x1_R,inter_rect_x1_C)
    inter_inter_rect_y1 = torch.max(inter_rect_y1_R,inter_rect_y1_C)
    inter_inter_rect_x2 = torch.max(inter_rect_x2_R,inter_rect_x2_C)
    inter_inter_rect_y2 = torch.max(inter_rect_y2_R,inter_rect_y2_C)
    inter_inter_area = torch.clamp(inter_inter_rect_x2 - inter_inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_inter_rect_y2 - inter_inter_rect_y1 + 1, min=0
    )

    intersections_of_intersections_area = torch.triu(inter_inter_area,diagonal=1).sum()

    #We use the sum of pred-GT intersetions, minus the intersections intersecstions, as those would be counting double and shouldn't
    box1_coverage = ( inter_area.sum()-intersections_of_intersections_area )/b1_area
    box2_coverage = inter_area/b2_area

    return box1_coverage, box2_coverage

def build_oversegmented_targets(
    max_width, pred_boxes, pred_conf, pred_cls, target, target_sizes, anchors, num_anchors, num_classes, grid_sizeH, grid_sizeW, ignore_thresh, scale, calcIOUAndDist=False, target_num_neighbors=None
):
    VISUAL_DEBUG=True
    VIZ_SIZE=24
    nB = pred_boxes.size(0)
    nA = num_anchors
    nC = num_classes
    nH = grid_sizeH
    nW = grid_sizeW
    mask = torch.zeros(nB, nA, nH, nW)
    conf_mask = torch.ones(nB, nA, nH, nW)
    tx = torch.zeros(nB, nA, nH, nW)
    ty = torch.zeros(nB, nA, nH, nW)
    tw = torch.zeros(nB, nA, nH, nW)
    th = torch.zeros(nB, nA, nH, nW)
    tconf = torch.ByteTensor(nB, nA, nH, nW).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nH, nW, nC).fill_(0)
    if target_num_neighbors is not None:
        tneighbors = torch.FloatTensor(nB, nA, nH, nW).fill_(0)
    else:
        tneighbors=None


    assert(not calcIOUAndDist)

    nGT = 0
    nPred = 0
    covered_gt_area = 0
    on_pred_area = 0
    precision = 0
    recall = 0
    #nCorrect = 0
    #import pdb; pdb.set_trace()
    for b in range(nB):
        on_pred_areaB = torch.FloatTensor(pred_boxes.shape[1:4]).zero_()
        #For oversegmented, we need to identify all tiles (not just on) that correspon to gt
        #That brings up an interesting alternative: limit all predictions to their local tile (width). Proba not now...
        for t in range(target_sizes[b]): #range(target.shape[1]):

            if VISUAL_DEBUG:
                anchor_draw_size = int(VIZ_SIZE/nA)
                if anchor_draw_size==0:
                    VIZ_SIZE=nA
                    anchor_draw_size=1
                draw = np.zeros([nH*VIZ_SIZE,nW*VIZ_SIZE,3],np.uint)
                for i in range(0,nW):
                    draw[:,i*VIZ_SIZE,:]=60
                for j in range(0,nH):
                    draw[j*VIZ_SIZE,:,:]=60

                draw_colors = [(255,0,0),(100,255,0),(255,100,0),(0,255,0),(200,200,0),(255,200,0),(200,255,0)]

            #if target[b, t].sum() == 0:
            #    continue
            # Convert to position relative to box
            gx = target[b, t, 0] / scale[0]
            gy = target[b, t, 1] / scale[1]
            gw = target[b, t, 4] / scale[0]
            gh = target[b, t, 3] / scale[1]

            gx1 = gx-gw
            gx2 = gx+gw
            gy1 = gy-gh
            gy2 = gy+gh
        
            if gw==0 or gh==0:
                continue
            nGT += 1
            # Get grid box indices
            gi = max(min(int(gx),conf_mask.size(3)-1),0)
            gj = max(min(int(gy),conf_mask.size(2)-1),0)
            gi1 = max(min(int(gx1),conf_mask.size(3)-1),0)
            gj1 = max(min(int(gy1),conf_mask.size(2)-1),0)
            gi2 = max(min(int(gx2),conf_mask.size(3)-1),0)
            gj2 = max(min(int(gy2),conf_mask.size(2)-1),0)
            #We truncate with int() instead of rounding since each tile i is actually centered at i+0.5

            #We don't want to include a tile if the real box doesn't extend past it's centerpoint, these shouldn't predict (arguably unless the real box covers the whole tile we don't want to predict)
            if gx1>gi1+0.5:
                gi1+=1
            if gx2<gi2+0.5:
                gi2-=1

            #We need to handle the end points of the line differently (they probably need smaller anchor rectangles)
            if gi1>gi2: #uh oh, we have a really small box between two tiles
                gi1=gi2 = gi

            if gi1==gi2:
                over_seg_gws = [gw]
            else:
                
                #Get best matching anchor
                #Build oversegmented gt sizes for each i/tile: (gh,min(self.maxWidth,this_x-gx1,gx2-this_x))
                over_seg_gws = [min(max_width,this_i+0.5-gx1,gx2-(this_i+0.5)) for this_i in range(gi1,gi2+1)]
            best_ns,anch_ious = multi_get_closest_anchor_iou(anchors,gh,over_seg_gws)

            #best_n, anch_ious = get_closest_anchor_iou(anchors,gh,min(gw,self.maxWidth))
            # Where the overlap is larger than threshold set mask to zero (ignore)
            #conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            #  ignore_range, all, set to 1 later
            anch_x,ignore_anch = torch.where(anch_ious > ignore_thresh)
            #conf_mask[([b]*anch_x.size(0),ignore_anch,list(range(gj1:gj2+1)),anch_x)]=0
            for j in range(gj1,gj2+1):
                conf_mask[b,:,j,:][ignore_anch,anch_x]=0
                if VISUAL_DEBUG:
                    for iii in range( anch_x.size(0) ):
                        draw[j*VIZ_SIZE+1:(j+1)*VIZ_SIZE,anch_x[iii]*VIZ_SIZE+ignore_anch[iii]*anchor_draw_size:anch_x[iii]*VIZ_SIZE+(ignore_anch[iii]+1)*anchor_draw_size,1]=0
            #conf_mask[b, anch_ious > ignore_thres, gj1:gj2+1,gi1:gi2+1] = 0
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            gt_area = gw+gh
            # Get the best prediction
            #pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks

            mask[b,:,gj,:][(best_ns,list(range(gi1,gi2+1)))] = 1
            conf_mask[b,:,gj,:][(best_ns,list(range(gi1,gi2+1)))] = 1
            if VISUAL_DEBUG:
                for anchor_n, i in zip(best_ns,list(range(gi1,gi2+1))):
                    draw[gj*VIZ_SIZE+1:(gj+1)*VIZ_SIZE,i*VIZ_SIZE+anchor_n*anchor_draw_size:i*VIZ_SIZE+(anchor_n+1)*anchor_draw_size,0]=110
                    draw[gj*VIZ_SIZE+1:(gj+1)*VIZ_SIZE,i*VIZ_SIZE+anchor_n*anchor_draw_size:i*VIZ_SIZE+(anchor_n+1)*anchor_draw_size,1]=110
            #mask[b, best_n, gj, gi1:gi2+1] = 1
            #conf_mask[b, best_n, gj, gi1:gi2+1] = 1 #we ned to set this to 1 as we ignored it earylier
            # Coordinates
            #DO we first want to compute position and than the scaling based on that, or vice-versa? Always  trying to predict at the edge of a tile might lead to weird effects... What about random for each instance? Or halfway for each?
            #-> I think my final verdict will be to move and strech such that the side you're not matching stays in the same place. This should always maintain a constant distance form the center of the predicting tile to the max length for open/continuing sides
            #For X, the anchor was selected assuming a position centered on the tile. However, this won't work for some (end tiles). We can simply compute the best position given the selected anchors
            anchor_width = anchors[:,0]
            for index in range(len(best_ns)):
                i = index+gi1
                best_n = best_ns[index]
                diff1 = gx1-(i+0.5-anchor_width[best_n])
                diff2 = gx2-(i+0.5+anchor_width[best_n])

                if (anchor_width[best_n]>gw or gw-anchor_width[best_n]<=1.0) and abs(gx - (i+0.5))<0.55:
                    #We'll just fit the anchor box to the whole line, either becuase the line is smaller or very close to the anchor size (less than one tile)
                    offset = gx - (i+0.5)
                    assert(abs(offset.item())<1)
                    tx[b, best_n, gj, i] = inv_tanh(offset) 
                    scaleAnchor = gw / anchors[best_n][0]
                    tw[b, best_n, gj, i] = math.log(scaleAnchor + 1e-16)
                elif diff1>=-0.5:
                    #anchor box is close to left edge or past it, so lets move+strench left side to gt
                    scaleAnchor = 1 + diff1/(2*anchor_width[best_n])
                    offset = anchor_width[best_n]*(1-scaleAnchor)
                    assert(abs(offset.item())<1)
                    tx[b, best_n, gj, i] = inv_tanh(offset) 
                    tw[b, best_n, gj, i] = math.log(scaleAnchor + 1e-16)
                elif diff2<=0.5:
                    #anchor box is close to right edge or past it, so lets move+strench right side to gt
                    scaleAnchor = 1 + diff2/(2*anchor_width[best_n])
                    offset = anchor_width[best_n]*(scaleAnchor-1)
                    assert(abs(offset.item())<1)
                    tx[b, best_n, gj, i] = inv_tanh(offset) 
                    tw[b, best_n, gj, i] = math.log(scaleAnchor + 1e-16)
                elif diff1<=0 and diff2>=0:
                    #no change is needed, the real box extends well beyond the anchor
                    tx[b, best_n, gj, i] = 0 #TODO Should this actually be: No loss computed? seperate mask
                    tw[b, best_n, gj, gi] = 0
                    if VISUAL_DEBUG:
                        offset=torch.FloatTensor(1).zero_()
                        scaleAnchor=1
                else:
                    print("UNEXPECTED STATE")
                    import pdb;pdb.set_trace()
                th[b, best_n, gj, i] = math.log(gh / anchors[best_n][1] + 1e-16)
                if VISUAL_DEBUG:
                    #Draw loss boxes
                    offset_y = gy - (gj+0.5)
                    scaleAnchor_y = gh / anchors[best_n][1]
                    draw_xc = ((i+offset+0.5)*VIZ_SIZE).item()
                    draw_yc = ((gj+offset_y+0.5)*VIZ_SIZE).item()
                    draw_h = (anchors[best_n][1]*scaleAnchor_y).item()*VIZ_SIZE
                    draw_w = (anchors[best_n][0]*scaleAnchor).item()*VIZ_SIZE
                    draw_lx = max(round(draw_xc-draw_w),0)
                    draw_rx = min(round(draw_xc+draw_w),draw.shape[1]-1)
                    draw_ty = max(round(draw_yc-draw_h),0)
                    draw_by = min(round(draw_yc+draw_h),draw.shape[0]-1)
                    #draw[draw_ty,draw_lx:draw_rx+1,i%2]=150+(50*i%3)
                    #draw[draw_by,draw_lx:draw_rx+1,i%2]=150+(50*i%3)
                    #draw[draw_ty:draw_by+1,draw_lx,i%2]=150+(50*i%3)
                    #draw[draw_ty:draw_by+1,draw_rx,i%2]=150+(50*i%3)
                    #draw[round(draw_yc),round(draw_xc),i%2]=150+(50*i%3)
                    draw[draw_ty,draw_lx:draw_rx+1]=draw_colors[index%len(draw_colors)]
                    draw[draw_by,draw_lx:draw_rx+1]=draw_colors[index%len(draw_colors)]
                    draw[draw_ty:draw_by+1,draw_lx]=draw_colors[index%len(draw_colors)]
                    draw[draw_ty:draw_by+1,draw_rx]=draw_colors[index%len(draw_colors)]
                    draw[round(draw_yc),round(draw_xc)]=draw_colors[index%len(draw_colors)]
            ty[b, best_n, gj, gi1:gi2+1] = inv_tanh(gy - (gj+0.5))
            if VISUAL_DEBUG:
                #Draw GT
                draw_gy1 = max(round(gy1.item()*VIZ_SIZE)-1,0) #expand by one for visibility
                draw_gy2 = min(round(gy2.item()*VIZ_SIZE)+1,draw.shape[0]-1)
                draw_gx1 = max(round(gx1.item()*VIZ_SIZE)-1,0)
                draw_gx2 = min(round(gx2.item()*VIZ_SIZE)+1,draw.shape[1]-1)
                draw[draw_gy1,draw_gx1:draw_gx2+1,2]=255
                draw[draw_gy2,draw_gx1:draw_gx2+1,2]=255
                draw[draw_gy1:draw_gy2+1,draw_gx1,2]=255
                draw[draw_gy1:draw_gy2+1,draw_gx2,2]=255
            # Width and height
            # One-hot encoding of label
            #target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi1:gi2+1] = target[b, t,13:]
            if target_num_neighbors is not None:
                assert(False and 'Not really made for NN preds...')
                tneighbors[b, best_n, gj, gi1] = target_num_neighbors[b, t]+1
                tneighbors[b, best_n, gj, gi1+1:gi2] = target_num_neighbors[b, t]+2
                tneighbors[b, best_n, gj, gi2+1] = target_num_neighbors[b, t]+1
            tconf[b, best_n, gj, gi1:gi2+1] = 1

            # Calculate overlaps between ground truth and best matching prediction
            class_selector = torch.logical_and(pred_cls[b].argmax(dim=3)==torch.argmax(target[b,t,13:]), pred_conf[b]>0)
            pred_right_label_boxes = pred_boxes[b][class_selector] #this is already normalized to tile space
         
            gt_area_covered, pred_area_covered = bbox_coverage(gt_box, pred_right_label_boxes, x1y1x2y2=False)
            covered_gt_area += gt_area_covered/gt_area
            if gt_area_covered/gt_area>0.5:
                recall+=1
            on_pred_areaB[class_selector] = torch.max(on_pred_areaB[class_selector],pred_area_covered)
            #pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            #score = pred_conf[b, best_n, gj, gi]
            #import pdb; pdb.set_trace()
            #if iou > 0.5 and pred_label == torch.argmax(target[b,t,13:]) and score > 0:
            #    nCorrect += 1

        on_pred_area += on_pred_areaB.sum()
        nPred += on_pred_areaB.size(0)
        precision = (on_pred_areaB>0.5).sum()

        if VISUAL_DEBUG:
            fig = plt.figure()
            ax_im = plt.subplot()
            ax_im.set_axis_off()
            ax_im.imshow(draw)
            plt.show()


    assert(False and 'TODO verify this works!')
    return nGT, mask, conf_mask, tx, ty, tw, th, tconf, tcls, tneighbors, on_pred_area/nPred, covered_gt_area/nGT, recall/nGT, precision/nPred

class OversegmentLoss (nn.Module):
    def __init__(self, num_classes, rotation, scale, anchors, ignore_thresh=0.5,use_special_loss=False,bad_conf_weight=1.25, multiclass=False,max_width=100):
        super(OversegmentLoss, self).__init__()
        self.max_width=max_width
        self.ignore_thresh=ignore_thresh
        self.num_classes=num_classes
        self.rotation=rotation
        self.scale=scale
        self.use_special_loss=use_special_loss
        self.bad_conf_weight=bad_conf_weight
        self.multiclass=multiclass
        self.anchors=anchors
        self.num_anchors=len(anchors)
        self.mse_loss = nn.MSELoss(reduction='mean')  # Coordinate loss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')  # Class loss
        self.mse_loss = nn.MSELoss(reduction='mean')  # Num neighbor regression

    def forward(self,prediction, target, target_sizes, target_num_neighbors=None ):

        nA = self.num_anchors
        nB = prediction.size(0)
        nH = prediction.size(2)
        nW = prediction.size(3)
        stride=self.scale

        FloatTensor = torch.cuda.FloatTensor if prediction.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if prediction.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if prediction.is_cuda else torch.ByteTensor
        BoolTensor = torch.cuda.BoolTensor if prediction.is_cuda else torch.BoolTensor

        x = prediction[..., 1]  # Center x
        y = prediction[..., 2]  # Center y
        w = prediction[..., 5]  # Width
        h = prediction[..., 4]  # Height
        #r = prediction[..., 3]  # Rotation (not used here)
        pred_conf = prediction[..., 0]  # Conf 
        if target_num_neighbors is not None: #self.predNumNeighbors:
            pred_neighbors = 1+prediction[..., 6]  # num of neighbors, offset pred range so -1 is 0 neighbirs
            pred_cls = prediction[..., 7:]  # Cls pred.
        else:
            pred_cls = prediction[..., 6:]  # Cls pred.

        grid_x = torch.arange(nW).repeat(nH, 1).view([1, 1, nH, nW]).type(FloatTensor).to(prediction.device)
        grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, 1, nH, nW]).type(FloatTensor).to(prediction.device)
        scaled_anchors = FloatTensor([(a['width'] / stride[0], a['height']/ stride[1]) for a in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1)).to(prediction.device)
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1)).to(prediction.device)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = torch.tanh(x.data)+0.5 + grid_x
        pred_boxes[..., 1] = torch.tanh(y.data)+0.5 + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        #moved back into build_targets
        #if target is not None:
        #    target[:,:,[0,4]] /= self.scale[0]
        #    target[:,:,[1,3]] /= self.scale[1]

        nGT, mask, conf_mask, tx, ty, tw, th, tconf, tcls, tneighbors, distances, pred_covered, gt_covered, recall, precision = build_oversegmented_targets(
            self.max_width,
            pred_boxes=pred_boxes.cpu().data,
            pred_conf=pred_conf.cpu().data,
            pred_cls=pred_cls.cpu().data,
            target=target.cpu().data if target is not None else None,
            target_sizes=target_sizes,
            anchors=scaled_anchors.cpu().data,
            num_anchors=nA,
            num_classes=self.num_classes,
            grid_sizeH=nH,
            grid_sizeW=nW,
            ignore_thresh=self.ignore_thresh,
            scale=self.scale,
            calcIOUAndDist=self.use_special_loss,
            target_num_neighbors=target_num_neighbors
        )

        #nProposals = int((pred_conf > 0).sum().item())
        #recall = float(nCorrect / nGT) if nGT else 1
        #if nProposals>0:
        #    precision = float(nCorrect / nProposals)
        #else:
        #    precision = 1

        # Handle masks
        mask = (mask.type(BoolTensor))
        conf_mask = (conf_mask.type(BoolTensor))

        # Handle target variables
        tx = tx.type(FloatTensor).to(prediction.device)
        ty = ty.type(FloatTensor).to(prediction.device)
        tw = tw.type(FloatTensor).to(prediction.device)
        th = th.type(FloatTensor).to(prediction.device)
        tconf = tconf.type(FloatTensor).to(prediction.device)
        tcls = tcls.type(LongTensor).to(prediction.device)
        if target_num_neighbors is not None:
            tneighbors = tneighbors.type(FloatTensor).to(prediction.device)

        # Get conf mask where gt and where there is no gt
        conf_mask_true = mask
        conf_mask_false = conf_mask & ~mask #conf_mask - mask

        #import pdb; pdb.set_trace()

        # Mask outputs to ignore non-existing objects
        if self.use_special_loss:
            loss_conf = weighted_bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false],distances[conf_mask_false],ious[conf_mask_false],nB)
            distances=None
            ious=None
        else:
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false])
        loss_conf *= self.bad_conf_weight
        if target is not None and nGT>0:
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            if self.multiclass:
                loss_cls = self.bce_loss(pred_cls[mask], tcls[mask].float())
            else:
                loss_cls =  self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1)) *(1 / nB) #this multiply is erronous
            loss_conf += self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
            if target_num_neighbors is not None: #if self.predNumNeighbors:
                loss_nn = 0.1*self.mse_loss(pred_neighbors[mask],tneighbors[mask])
            else:
                loss_nn = 0
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls + loss_nn
            if target_num_neighbors is not None:
                loss_nn=loss_nn.item()
            return (
                loss,
                loss_x.item()+loss_y.item()+loss_w.item()+loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                loss_nn,
                recall,
                precision,
                gt_covered,
                pred_covered
            )
        else:
            return (
                loss_conf,
                0,
                loss_conf.item(),
                0,
                0,
                recall,
                precision,
                gt_covered,
                pred_covered
            )

