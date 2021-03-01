import numpy as np
import torch
from base import BaseTrainer
import timeit
from utils import util
from collections import defaultdict
from evaluators.draw_graph import draw_graph
import matplotlib.pyplot as plt
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist, AP_textLines, getTargIndexForPreds_iou, newGetTargIndexForPreds_iou, getTargIndexForPreds_dist, newGetTargIndexForPreds_textLines, computeAP, non_max_sup_overseg
from utils.group_pairing import getGTGroup, pure, purity
from datasets.testforms_graph_pair import display
import random, os, math

from model.oversegment_loss import build_oversegmented_targets_multiscale
from model.overseg_box_detector import build_box_predictions
try:
    from model.optimize import optimizeRelationships, optimizeRelationshipsSoft
except:
    pass

import torch.autograd.profiler as profile


def maxRelScoreIsHit(child_groups,parent_groups,edgeIndexes,edgePred):
    max_score=-1
    max_score_is_hit=False
    for ei,(pG0,pG1) in enumerate(edgeIndexes):
        score = edgePred[ei,-1,1]
        if score>max_score:
            if pG0 in child_groups:
                max_score = score
                if pG1 in parent_groups:
                    max_score_is_hit = True
                else:
                    max_score_is_hit = False
            elif pG1 in child_groups:
                max_score = score
                if pG0 in parent_groups:
                    max_score_is_hit = True
                else:
                    max_score_is_hit = False
    return max_score_is_hit

class GraphPairTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(GraphPairTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        if 'loss_params' in config:
            self.loss_params=config['loss_params']
        else:
            self.loss_params={}
        self.lossWeights = config['loss_weights'] if 'loss_weights' in config else {"box": 1, "rel":1}
        if 'box' in self.loss:
            self.loss['box'] = self.loss['box'](**self.loss_params['box'], 
                    num_classes=self.model_ref.numBBTypes, 
                    rotation=self.model_ref.rotation, 
                    scale=self.model_ref.scale,
                    anchors=self.model_ref.anchors)
        elif 'overseg' in self.loss:
            self.loss['overseg'] = self.loss['overseg'](**self.loss_params['overseg'],
                    num_classes=self.model_ref.numBBTypes,
                    rotation=self.model_ref.rotation,
                    scale=self.model_ref.scale,
                    anchors=self.model_ref.anchors)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        #for i in range(self.start_iteration,
        self.valid_data_loader = valid_data_loader
        #self.valid = True if self.valid_data_loader is not None else False
        #self.log_step = int(np.sqrt(self.batch_size))
        #lr schedule from "Attention is all you need"
        #base_lr=config['optimizer']['lr']


        self.mergeAndGroup = config['trainer']['mergeAndGroup']
        self.classMap = self.data_loader.dataset.classMap
        self.scoreClassMap = {k:v for k,v in self.data_loader.dataset.classMap.items() if k!='blank'}


        #default is unfrozen, can be frozen by setting 'start_froze' in the PairingGraph models params
        self.unfreeze_detector = config['trainer']['unfreeze_detector'] if 'unfreeze_detector' in config['trainer'] else None

        self.thresh_conf = config['trainer']['thresh_conf'] if 'thresh_conf' in config['trainer'] else 0.92
        self.thresh_intersect = config['trainer']['thresh_intersect'] if 'thresh_intersect' in config['trainer'] else 0.4
        self.thresh_rel = config['trainer']['thresh_rel'] if 'thresh_rel' in config['trainer'] else 0.5
        if self.mergeAndGroup:
            self.thresh_edge = self.model_ref.keepEdgeThresh
            self.thresh_overSeg = self.model_ref.mergeThresh
            self.thresh_group = self.model_ref.groupThresh
            self.thresh_rel = [self.thresh_rel]*len(self.thresh_group)
            self.thresh_error = config['trainer']['thresh_error'] if 'thresh_error' in config['trainer'] else [0.5]*len(self.thresh_group)

        self.gt_bb_align_IOcU_thresh = 0.4 if 'gt_bb_align_IOcU_thresh' not in config['trainer'] else config['trainer']['gt_bb_align_IOcU_thresh']
        self.final_bb_iou_thresh = config['trainer']['final_bb_iou_thresh'] if 'final_bb_iou_thresh' in config['trainer'] else (config['final_bb_iou_thresh'] if 'final_bb_iou_thresh' in config else 0.5)

        #we iniailly train the pairing using GT BBs, but eventually need to fine-tune the pairing using the networks performance
        self.stop_from_gt = config['trainer']['stop_from_gt'] if 'stop_from_gt' in config['trainer'] else None
        self.partial_from_gt = config['trainer']['partial_from_gt'] if 'partial_from_gt' in config['trainer'] else None
        self.max_use_pred = config['trainer']['max_use_pred'] if 'max_use_pred' in config['trainer'] else 0.9
        self.use_word_bbs_gt = config['trainer']['use_word_bbs_gt'] if 'use_word_bbs_gt' in config['trainer'] else -1
        self.valid_with_gt = config['trainer']['valid_with_gt'] if 'valid_with_gt'  in config['trainer'] else False

        self.conf_thresh_init = config['trainer']['conf_thresh_init'] if 'conf_thresh_init' in config['trainer'] else 0.9
        self.conf_thresh_change_iters = config['trainer']['conf_thresh_change_iters'] if 'conf_thresh_change_iters' in config['trainer'] else 5000

        self.train_hard_detect_limit = config['trainer']['train_hard_detect_limit'] if 'train_hard_detect_limit' in config['trainer'] else 300
        self.val_hard_detect_limit = config['trainer']['val_hard_detect_limit'] if 'val_hard_detect_limit' in config['trainer'] else 400

        self.useBadBBPredForRelLoss = config['trainer']['use_all_bb_pred_for_rel_loss'] if 'use_all_bb_pred_for_rel_loss' in config['trainer'] else False
        if self.useBadBBPredForRelLoss is True:
            self.useBadBBPredForRelLoss=1

        self.adaptLR = config['trainer']['adapt_lr'] if 'adapt_lr' in config['trainer'] else False
        self.adaptLR_base = config['trainer']['adapt_lr_base'] if 'adapt_lr_base' in config['trainer'] else 165 #roughly average number of rels
        self.adaptLR_ep = config['trainer']['adapt_lr_ep'] if 'adapt_lr_ep' in config['trainer'] else 15

        self.fixedAlign = config['trainer']['fixed_align'] if 'fixed_align' in config['trainer'] else False

        self.use_gt_trans = config['trainer']['use_gt_trans'] if 'use_gt_trans' in config['trainer'] else False

        self.merge_first_only_until = config['trainer']['merge_first_only_until'] if 'merge_first_only_until' in config['trainer'] else 100
        self.init_merge_rule = config['trainer']['init_merge_rule'] if 'init_merge_rule' in config['trainer'] else None
        self.picky_merging = 'picky' in self.init_merge_rule if self.init_merge_rule is not None else False

        self.num_node_error_class = 0
        self.final_class_bad_alignment = False
        self.final_class_bad_alignment = False
        self.final_class_inpure_group = False

        self.debug = 'DEBUG' in  config['trainer']
        self.save_images_every = config['trainer']['save_images_every'] if 'save_images_every' in config['trainer'] else 50
        self.save_images_dir = 'train_out'
        util.ensure_dir(self.save_images_dir)


        self.amp = config['trainer']['AMP'] if 'AMP' in config['trainer'] else False
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.accum_grad_steps = config['trainer']['accum_grad_steps'] if 'accum_grad_steps' in config['trainer'] else 1

        #Name change
        if 'edge' in self.lossWeights:
            self.lossWeights['rel'] = self.lossWeights['edge']
        if 'edge' in self.loss:
            self.loss['rel'] = self.loss['edge']

        #error fixing, eval special stuff
        self.remove_same_pairs = False if 'remove_same_pairs' not in config else config['remove_same_pairs']
        self.optimize = False if 'optimize' not in config else config['optimize']

        #t#self.opt_history = defaultdict(list)#t#
        self.do_characterization = config['characterization'] if 'characterization' in config else False
        if self.do_characterization:
            self.characterization_sum=defaultdict(int)
            self.characterization_form=defaultdict(list)
            self.characterization_hist=defaultdict(list)

    def _to_tensor(self, instance):
        image = instance['img']
        bbs = instance['bb_gt']
        adjaceny = instance['adj']
        num_neighbors = instance['num_neighbors']

        if self.with_cuda:
            image = image.to(self.gpu)
            if bbs is not None:
                bbs = bbs.to(self.gpu)
            if num_neighbors is not None:
                num_neighbors = num_neighbors.to(self.gpu)
            #adjacenyMatrix = adjacenyMatrix.to(self.gpu)
        return image, bbs, adjaceny, num_neighbors

    def _eval_metrics(self, typ,name,output, target):
        if len(self.metrics[typ])>0:
            #acc_metrics = np.zeros(len(self.metrics[typ]))
            met={}
            cpu_output=[]
            for pred in output:
                cpu_output.append(output.cpu().data.numpy())
            target = target.cpu().data.numpy()
            for i, metric in enumerate(self.metrics[typ]):
                met[name+metric.__name__] = metric(cpu_output, target)
            return acc_metrics
        else:
            #return np.zeros(0)
            return {}

    def useGT(self,iteration,force=False):
        if force:
            use=True
        elif self.stop_from_gt is not None and iteration>=self.stop_from_gt:
            use= random.random()>self.max_use_pred #I think it's best to always have some GT examples
        elif self.partial_from_gt is not None and iteration>=self.partial_from_gt:
            use= random.random()> self.max_use_pred*(iteration-self.partial_from_gt)/(self.stop_from_gt-self.partial_from_gt)
        else:
            use= True

        if use:
            ret='only_space'
            if random.random()<=self.use_word_bbs_gt:
                ret+=' word_bbs'
            return ret
        else:
            return False
    #NEW
    def _train_iteration(self, iteration):
        """
        Training logic for an iteration

        :param iteration: Current training iteration.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        if self.unfreeze_detector is not None and iteration>=self.unfreeze_detector:
            self.model_ref.unfreeze()
        self.model.train()
        #self.model.eval()
        #print("WARNING EVAL")

        #t#ticAll=timeit.default_timer()#t##t#
        batch_idx = (iteration-1) % len(self.data_loader)
        try:
            thisInstance = self.data_loader_iter.next()
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            thisInstance = self.data_loader_iter.next()
        if not self.model_ref.detector.predNumNeighbors:
            thisInstance['num_neighbors']=None
        ##toc=timeit.default_timer()
        #t#self.opt_history['get data'].append(timeit.default_timer()-ticAll)#t#
        
        #t#tic=timeit.default_timer()#t##t#
        if self.accum_grad_steps<2 or iteration%self.accum_grad_steps==1:
            self.optimizer.zero_grad()

        ##toc=timeit.default_timer()
        ##print('for: '+str(toc-tic))
        #loss = self.loss(output, target)
        index=0
        losses={}
        #t###tic=timeit.default_timer()#t#

        #if self.iteration % self.save_step == 0:
        #    targetPoints={}
        #    targetPixels=None
        #    _,lossC=FormsBoxPair_printer(None,thisInstance,self.model,self.gpu,self._eval_metrics,self.checkpoint_dir,self.iteration,self.loss['box'])
        #    loss, position_loss, conf_loss, class_loss, recall, precision = lossC
        #else:
        if self.conf_thresh_change_iters > iteration:
            threshIntur = 1 - iteration/self.conf_thresh_change_iters
        else:
            threshIntur = None
        useGT = self.useGT(iteration)
        #if self.bothGT and useGT and random.random()<0.9:
        #    useOnlyGTSpace = True
        #    useGT = False
        #else:
        #    useOnlyGTSpace = False
        #useOnlyGTSpace = self.useGT(iteration)
        #useGT = False

        #print('\t\t\t\t{} {}'.format(iteration,thisInstance['imgName']))
        if self.amp:
            with torch.cuda.amp.autocast():
                if self.mergeAndGroup:
                    losses, run_log, out = self.newRun(thisInstance,useGT,threshIntur)
                else:
                    useGT = useOnlyGTSpace
                    losses, run_log, out = self.run(thisInstance,useGT,threshIntur)
        else:
            if self.mergeAndGroup:
                losses, run_log, out = self.newRun(thisInstance,useGT,threshIntur)
            else:
                useGT = useOnlyGTSpace
                losses, run_log, out = self.run(thisInstance,useGT,threshIntur)
        #t#self.opt_history['full run'].append(timeit.default_timer()-tic)#t#

        #t#tic=timeit.default_timer()#t##t#
        loss=0
        for name in losses.keys():
            losses[name] *= self.lossWeights[name[:-4]]
            loss += losses[name]
            losses[name] = losses[name].item()
        if len(losses)>0:
            if self.accum_grad_steps>1:
                loss /= self.accum_grad_steps
            if self.amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        if self.accum_grad_steps<2 or iteration%self.accum_grad_steps==0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(),1)
            if self.amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        #t#self.opt_history['backprop'].append(timeit.default_timer()-tic)#t#
        meangrad=0
        count=0
        for m in self.model.parameters():
            if m.grad is None:
                continue
            count+=1
            meangrad+=m.grad.data.mean().cpu().item()
        if count!=0:
            meangrad/=count
        self.optimizer.step()
        if len(losses)>0:
            loss = loss.item()
        log = {
            'mean grad': meangrad,
            'loss': loss,
            **losses,
            #'edgePredLens':np.array([numEdgePred,numBoxPred,numEdgePred+numBoxPred,-1],dtype=np.float),
            
            **run_log

        }
        #t#self.opt_history['Full iteration'].append(timeit.default_timer()-ticAll)#t#
        #t#for name,times in self.opt_history.items():#t#
            #t#print('time {}({}): {}'.format(name,len(times),np.mean(times)))#t#
            #t#if len(times)>300: #t#
                #t#times.pop(0)#t#
                #t#if len(times)>600:#t#
                    #t#times.pop(0)#t#
        #t#print('--------------------------')#t#
        return log

    def _minor_log(self, log):
        ls=''
        for i,(key,val) in enumerate(log.items()):
            ls += key
            if type(val) is float or type(val) is np.float64:

                this_data=': {:.3f},'.format(val)
            else:
                this_data=': {},'.format(val)
            ls+=this_data
            this_len=len(this_data)
            if i%2==0 and this_len>0:
                #ls+='\t\t'
                ls+=' '*(20-this_len)
            else:
                ls+='\n      '
        self.logger.info('Train '+ls)
    
    #New
    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        val_metrics = {}#defaultdict(lambda: 0.0)
        val_count = defaultdict(lambda: 1)

        useGT = self.useGT(self.iteration,True) if self.valid_with_gt else False
        prefix = 'valGT_' if self.valid_with_gt else 'val_'


        with torch.no_grad():
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.model_ref.detector.predNumNeighbors:
                    instance['num_neighbors']=None
                if not self.logged:
                    print('iter:{} valid batch: {}/{}'.format(self.iteration,batch_idx,len(self.valid_data_loader)), end='\r')
                if self.mergeAndGroup:
                    losses,log_run, out = self.newRun(instance,useGT,get=['bb_stats','nn_acc'])
                else:
                    losses,log_run, out = self.run(instance,useGT,get=['bb_stats','nn_acc'])

                for name,value in log_run.items():
                    if value is not None:
                        val_name = prefix+name
                        if val_name in val_metrics:
                            val_metrics[val_name]+=value
                            val_count[val_name]+=1
                        else:
                            val_metrics[val_name]=value
                for name,value in losses.items():
                    if value is not None:
                        value = value.item()
                        val_name = prefix+name
                        if val_name in val_metrics:
                            val_metrics[val_name]+=value
                            val_count[val_name]+=1
                        else:
                            val_metrics[val_name]=value



                #total_val_metrics += self._eval_metrics(output, target)

        for val_name in val_metrics:
            if val_count[val_name]>0:
                val_metrics[val_name] /= val_count[val_name]
        return val_metrics

    def alignEdgePred(self,targetBoxes,adj,outputBoxes,relPred,relIndexes,rel_prop_pred):
        if relPred is None or targetBoxes is None:
            if targetBoxes is None:
                if relPred is not None and (relPred>self.thresh_rel).any():
                    prec=0
                    ap=0
                else:
                    prec=1
                    ap=1
                recall=1
                targIndex = -torch.ones(outputBoxes.size(0)).int()
            elif relPred is None:
                if targetBoxes is not None:
                    recall=0
                    ap=0
                else:
                    recall=1
                    ap=1
                prec=1
                targIndex = None

            return torch.tensor([]),torch.tensor([]),recall,prec,prec,ap, targIndex, torch.ones(outputBoxes.size(0)), None, recall, prec
        targetBoxes = targetBoxes.cpu()
        #decide which predicted boxes belong to which target boxes
        #should this be the same as AP_?
        numClasses = 2

        if self.model_ref.rotation:
            targIndex, fullHit = getTargIndexForPreds_dist(targetBoxes[0],outputBoxes,1.1,numClasses,hard_thresh=False)
        else:
            targIndex, fullHit = getTargIndexForPreds_iou(targetBoxes[0],outputBoxes,0.4,numClasses,hard_thresh=False,fixed=self.fixedAlign)
        #else:
        #    if self.model_ref.rotation:
        #        targIndex, predsWithNoIntersection = getTargIndexForPreds_dist(targetBoxes[0],outputBoxes,1.1,numClasses)
        #    else:
        #        targIndex, predsWithNoIntersection = getTargIndexForPreds_iou(targetBoxes[0],outputBoxes,0.4,numClasses)

        #Create gt vector to match relPred.values()

        rels = relIndexes #relPred._indices().cpu()
        predsAll = relPred #relPred._values()
        sigPredsAll = torch.sigmoid(predsAll[:,-1])
        predsPos = []
        predsNeg = []
        scores = []
        matches=0
        truePred=falsePred=badPred=0
        for i,(n0,n1) in enumerate(rels):
            t0 = targIndex[n0].item()
            t1 = targIndex[n1].item()
            if t0>=0 and t1>=0:
                if (min(t0,t1),max(t0,t1)) in adj:
                    #if self.useBadBBPredForRelLoss!='fixed' or (fullHit[n0] and fullHit[n1]):
                    if fullHit[n0] and fullHit[n1]:
                        matches+=1
                        predsPos.append(predsAll[i])
                        scores.append( (sigPredsAll[i],True) )
                        if sigPredsAll[i]>self.thresh_rel:
                            truePred+=1
                    else:
                        scores.append( (sigPredsAll[i],False) ) #for the sake of scoring, this is a bad relationship
                else:
                    predsNeg.append(predsAll[i])
                    scores.append( (sigPredsAll[i],False) )
                    if sigPredsAll[i]>self.thresh_rel:
                        falsePred+=1

            else:
                #if self.useBadBBPredForRelLoss=='fixed' or (self.useBadBBPredForRelLoss and (predsWithNoIntersection[n0] or predsWithNoIntersection[n1])):
                if self.useBadBBPredForRelLoss:
                    if self.useBadBBPredForRelLoss=='full' or np.random.rand()<self.useBadBBPredForRelLoss:
                        predsNeg.append(predsAll[i])
                scores.append( (sigPredsAll[i],False) )
                if sigPredsAll[i]>self.thresh_rel:
                    badPred+=1
        #Add score 0 for instances we didn't predict
        for i in range(len(adj)-matches):
            scores.append( (float('nan'),True) )
        if len(adj)>0:
            final_prop_rel_recall = matches/len(adj)
        else:
            final_prop_rel_recall = 1
        if len(rels)>0:
            final_prop_rel_prec = matches/len(rels)
        else:
            final_prop_rel_prec = 1
    
        if len(predsPos)>0:
            predsPos = torch.stack(predsPos).to(relPred.device)
        else:
            predsPos = None
        if len(predsNeg)>0:
            predsNeg = torch.stack(predsNeg).to(relPred.device)
        else:
            predsNeg = None

        if len(adj)>0:
            recall = truePred/len(adj)
        else:
            recall = 1
        if falsePred>0:
            prec = truePred/(truePred+falsePred)
        else:
            prec = 1
        if falsePred+badPred>0:
            fullPrec = truePred/(truePred+falsePred+badPred)
        else:
            fullPrec = 1


        if rel_prop_pred is not None:
            relPropScores,relPropIds, threshPropRel = rel_prop_pred
            truePropPred=falsePropPred=badPropPred=0
            propPredsPos=[]
            propPredsNeg=[]
            for i,(n0,n1) in enumerate(relPropIds):
                t0 = targIndex[n0].item()
                t1 = targIndex[n1].item()
                if t0>=0 and t1>=0:
                    if (min(t0,t1),max(t0,t1)) in adj:
                        #if self.useBadBBPredForRelLoss!='fixed' or (fullHit[n0] and fullHit[n1]):
                        if fullHit[n0] and fullHit[n1]:
                            #matches+=1
                            propPredsPos.append(relPropScores[i])
                            #scores.append( (sigPredsAll[i],True) )
                            if relPropScores[i]>threshPropRel:
                                truePropPred+=1
                        #else:
                        #    scores.append( (sigPredsAll[i],False) ) #for the sake of scoring, this is a bad relationship
                    else:
                        propPredsNeg.append(relPropScores[i])
                        #scores.append( (sigPredsAll[i],False) )
                        if relPropScores[i]>threshPropRel:
                            falsePropPred+=1
                else:
                    if self.useBadBBPredForRelLoss:
                        if self.useBadBBPredForRelLoss=='full' or np.random.rand()<self.useBadBBPredForRelLoss:
                            propPredsNeg.append(relPropScores[i])
                    #scores.append( (sigPredsAll[i],False) )
                    if relPropScores[i]>threshPropRel:
                        badPropPred+=1
            #Add score 0 for instances we didn't predict
            #for i in range(len(adj)-matches):
            #    scores.append( (float('nan'),True) )
        
            if len(propPredsPos)>0:
                propPredsPos = torch.stack(propPredsPos).to(relPred.device)
            else:
                propPredsPos = None
            if len(propPredsNeg)>0:
                propPredsNeg = torch.stack(propPredsNeg).to(relPred.device)
            else:
                propPredsNeg = None

            if len(adj)>0:
                propRecall = truePropPred/len(adj)
            else:
                propRecall = 1
            #if falsePropPred>0:
            #    propPrec = truePropPred/(truePropPred+falsePropPred)
            #else:
            #    propPrec = 1
            if falsePropPred+badPropPred>0:
                propFullPrec = truePropPred/(truePropPred+falsePropPred+badPropPred)
            else:
                propFullPrec = 1

            proposedInfo = (propPredsPos,propPredsNeg, propRecall, propFullPrec)
        else:
            proposedInfo = None


        return predsPos,predsNeg, recall, prec ,fullPrec, computeAP(scores), targIndex, fullHit, proposedInfo, final_prop_rel_recall, final_prop_rel_prec



    def simplerAlignEdgePred(self,targetBoxes,targetIndexToGroup,gtGroupAdj,outputBoxes,edgePred,edgePredIndexes,predGroups,rel_prop_pred,thresh_edge,thresh_rel,thresh_overSeg,thresh_group,thresh_error,merge_only=False):
        assert(self.useBadBBPredForRelLoss=='full' or self.useBadBBPredForRelLoss==1)
        if edgePred is None:
            if targetBoxes is None:
                prec=1
                ap=1
                recall=1
                targIndex = -torch.ones(len(outputBoxes)).int()
                missed_rels = set()
            else:
                recall=0
                ap=0
                prec=1
                targIndex = None
                missed_rels = gtGroupAdj
            Fm=2*recall*prec/(recall+prec) if recall+prec>0 else 0
            log = {
                'recallRel' : recall,
                'precRel' : prec,
                'FmRel' : Fm,
                'recallOverSeg' : recall,
                'precOverSeg' : prec,
                'FmOverSeg' : Fm,
                'recallGroup' : recall,
                'precGroup' : prec,
                'FmGroup' : Fm,
                'recallError' : recall,
                'precError' : prec,
                'FmError' : Fm
                }
            
            #return torch.tensor([]),torch.tensor([]), targIndex, torch.ones(outputBoxes.size(0)), None, log
            predsGTYes = torch.tensor([])
            predsGTNo = torch.tensor([])
            matches=0
            predTypes = None
            missed_rels = gtGroupAdj
        else:

            #decide which predicted boxes belong to which target boxes
            #should this be the same as AP_?
            numClasses = self.model_ref.numBBTypes

            #t#tic=timeit.default_timer()#t##t#
            
            if targetBoxes is not None:
                targetBoxes = targetBoxes.cpu()
                #what I want:
                # alignment from pred to target (-1 if none), each GT has only one pred
                # targIndex = alginment from pred to target (-1 if none) based on IO_clippedU thresh, not class
                if self.model_ref.useCurvedBBs:
                    targIndex = newGetTargIndexForPreds_textLines(targetBoxes[0],outputBoxes,self.gt_bb_align_IOcU_thresh,numClasses,True,self.picky_merging and not merge_only)
                elif self.model_ref.rotation:
                    assert(False and 'untested and should be changed to reflect new newGetTargIndexForPreds_s')
                    targIndex, fullHit, overSegmented = newGetTargIndexForPreds_dist(targetBoxes[0],outputBoxes,1.1,numClasses,hard_thresh=False)
                else:
                    targIndex = newGetTargIndexForPreds_iou(targetBoxes[0],outputBoxes,0.4,numClasses,True)
                targIndex = targIndex.numpy()
            else:
                #targIndex=torch.LongTensor(len(outputBoxes)).fill_(-1)
                targIndex = [-1]*len(outputBoxes)
            
            #t#self.opt_history['simplerAlign newGetTargIndexForPreds'].append(timeit.default_timer()-tic)#t#
            #t#tic=timeit.default_timer()#t##t#

            #Create gt vector to match edgePred.values()
            num_internal_iters = edgePred.size(-2)
            predsEdge = edgePred[...,0] 
            assert(not torch.isnan(predsEdge).any())
            predsGTEdge = []
            predsGTNoEdge = []
            truePosEdge=falsePosEdge=trueNegEdge=falseNegEdge=0
            saveEdgePred={}
            if not merge_only:
                if not self.model_ref.legacy:
                    predsRel = edgePred[...,1] 
                    predsOverSeg = edgePred[...,2] 
                    predsGroup = edgePred[...,3] 
                    predsError = edgePred[...,4] 
                else:
                    predsRel = predsEdge
                    predsOverSeg = edgePred[...,1] 
                    predsGroup = edgePred[...,2] 
                    predsError = edgePred[...,3]
                    predsEdge,_ = torch.max(torch.stack((predsRel,predsOverSeg,predsGroup),dim=0),dim=0)

                predsGTRel = []
                predsGTNoRel = []
                predsGTOverSeg = []
                predsGTNotOverSeg = []
                predsGTGroup = []
                predsGTNoGroup = []
                predsGTNoError = []
                predsGTError = []

                truePosRel=falsePosRel=trueNegRel=falseNegRel=0
                truePosOverSeg=falsePosOverSeg=trueNegOverSeg=falseNegOverSeg=0
                truePosGroup=falsePosGroup=trueNegGroup=falseNegGroup=0
                truePosError=falsePosError=trueNegError=falseNegError=0

                saveRelPred={}
                saveOverSegPred={}
                saveGroupPred={}
                saveErrorPred={}

            predGroupsT={}
            predGroupsTNear={}
            for node in range(len(predGroups)):
                predGroupsT[node] = [targIndex[bb] for bb in predGroups[node] if targIndex[bb]>=0]
            shouldBeEdge={}

            #gtGroupToPred = 

            #t#self.opt_history['simplerAlign edge setup'].append(timeit.default_timer()-tic)#t#
            #t#tic=timeit.default_timer()#t#

            ##new vectorization
            #We only operate over the subset of nodes that have an edge
            nodeLs = [p[0] for p in edgePredIndexes]
            nodeRs = [p[1] for p in edgePredIndexes]
            nodes_with_edges = list(set(nodeLs+nodeRs))
            node_to_nwe_index = {n:i for i,n in enumerate(nodes_with_edges)}
            nweLs = [node_to_nwe_index[n] for n in nodeLs]
            nweRs = [node_to_nwe_index[n] for n in nodeRs]

            #get information for that subset of nodes
            compute = [(
                len(predGroupsT[n0]),
                len(predGroups[n0]),
                getGTGroup(predGroupsT[n0],targetIndexToGroup),
                purity(predGroupsT[n0],targetIndexToGroup),
                predGroupsT[n0][0] if len(predGroupsT[n0])>0 else -1
                ) for n0 in nodes_with_edges]


            edge_loss_device = predsEdge.device #is it best to have this on gpu?
            
            #expand information into row and column matrices to allow all-to-all node comparisons
            g_target_len, g_len, GTGroups, purities, ts_0 = zip(*compute)
            gtNE = [(GTGroups[node_to_nwe_index[n0]],GTGroups[node_to_nwe_index[n1]])  for n0,n1 in edgePredIndexes]
            #gtGroupToNWE = {gt:nwe for nwe,gt in enumerate(GTGroups)}

            g_target_len = torch.IntTensor(g_target_len).to(edge_loss_device)
            g_len = torch.IntTensor(g_len).to(edge_loss_device)
            purities = torch.FloatTensor(purities).to(edge_loss_device)
            ts_0 = torch.IntTensor(ts_0).to(edge_loss_device)
            GTGroups = torch.IntTensor(GTGroups).to(edge_loss_device)

            g_target_len_L = g_target_len[nweLs]
            g_target_len_R = g_target_len[nweRs]
            g_len_R = g_len[nweRs]
            g_len_L = g_len[nweLs]
            purity_R = purities[nweRs]
            purity_L = purities[nweLs]
            ts_0_R = ts_0[nweRs]
            ts_0_L = ts_0[nweLs]
            same_ts_0 = (ts_0_R==ts_0_L) * (ts_0_R>=0) * (ts_0_L>=0)
            GTGroups_R = GTGroups[nweRs]
            GTGroups_L = GTGroups[nweLs]
            same_GTGroups = (GTGroups_R==GTGroups_L) * (GTGroups_R>=0) * (GTGroups_L>=0) #have to account for -1 being unaligned

            #g_target_len_R = g_target_len[:,None].expand(len(nodes_with_edges),len(nodes_with_edges))
            #g_target_len_C = g_target_len[None,:].expand(len(nodes_with_edges),len(nodes_with_edges))
            #g_len_R = g_len[:,None].expand(len(nodes_with_edges),len(nodes_with_edges))
            #g_len_C = g_len[None,:].expand(len(nodes_with_edges),len(nodes_with_edges))
            #purity_R = purities[:,None].expand(len(nodes_with_edges),len(nodes_with_edges))
            #purity_C = purities[None,:].expand(len(nodes_with_edges),len(nodes_with_edges))
            #ts_0_R = ts_0[:,None].expand(len(nodes_with_edges),len(nodes_with_edges))
            #ts_0_C = ts_0[None,:].expand(len(nodes_with_edges),len(nodes_with_edges))
            #same_ts_0 = ts_0_R==ts_0_C
            #GTGroups_R = GTGroups[:,None].expand(len(nodes_with_edges),len(nodes_with_edges))
            #GTGroups_L = GTGroups[None,:].expand(len(nodes_with_edges),len(nodes_with_edges))
            #same_GTGroups = GTGroups_R==GTGroups_L


            ##which (of all edges) are actually ones we predicted
            #predEdgesMat = torch.BoolTensor(len(nodes_with_edges),len(nodes_with_edges)).zero_()
            #N0, N1 = zip(*edgePredIndexes)
            #N0 = [node_to_nwe_index[n] for n in N0]
            #N1 = [node_to_nwe_index[n] for n in N1]
            #predEdgesMat[N0+N1,N1+N0]=True

            #which edges are GT ones
            gtNE = [(min(gtGroup0,gtGroup1),max(gtGroup0,gtGroup1)) for gtGroup0,gtGroup1 in gtNE]
            gtGroupAdjMat = [pair in gtGroupAdj for pair in gtNE]
            gtGroupAdjMat = torch.BoolTensor(gtGroupAdjMat).to(edge_loss_device)

            hit_rels = set(gtNE)
            missed_rels = gtGroupAdj.difference(hit_rels)

            

            #common conditions
            bothTarged = (g_target_len_R>0)*(g_target_len_L>0)
            badTarged = (g_target_len_R==0)+(g_target_len_L==0)
            bothPure = (purity_R>0.8)*(purity_L>0.8)
            
            #Actually start determining GT/training scenarios
            wasRel = bothTarged*((g_len_L>1)+(g_len_R>1)+~same_ts_0)*bothPure*~same_GTGroups*gtGroupAdjMat
            wasNoRel = (badTarged+(bothTarged*((g_len_L>1)+(g_len_R>1)+~same_ts_0)*bothPure*~same_GTGroups*~gtGroupAdjMat))
            
            wasNoOverSeg = (badTarged+(bothTarged*bothPure*~((g_len_R==1)*(g_len_L==1)*same_ts_0)))
            wasOverSeg = bothTarged*(g_len_R==1)*(g_len_L==1)*same_ts_0
            if not merge_only or self.init_merge_rule is None:
                pass
            elif 'adjacent' in self.init_merge_rule or self.picky_merging:
                tx,ty,bx,by = zip(* [outputBoxes[n].boundingRect() for n in nodes_with_edges])
                tx = torch.FloatTensor(tx).to(edge_loss_device)
                ty = torch.FloatTensor(ty).to(edge_loss_device)
                bx = torch.FloatTensor(bx).to(edge_loss_device)
                by = torch.FloatTensor(by).to(edge_loss_device)
                tx_R = tx[nweRs]
                tx_L = tx[nweLs]
                ty_R = ty[nweRs]
                ty_L = ty[nweLs]
                bx_R = bx[nweRs]
                bx_L = bx[nweLs]
                by_R = by[nweRs]
                by_L = by[nweLs]
                #tx_R = tx[:,None].expand(len(nodes_with_edges),len(nodes_with_edges))
                #tx_L = tx[None,:].expand(len(nodes_with_edges),len(nodes_with_edges))
                #ty_R = ty[:,None].expand(len(nodes_with_edges),len(nodes_with_edges))
                #ty_L = ty[None,:].expand(len(nodes_with_edges),len(nodes_with_edges))
                #bx_R = bx[:,None].expand(len(nodes_with_edges),len(nodes_with_edges))
                #bx_L = bx[None,:].expand(len(nodes_with_edges),len(nodes_with_edges))
                #by_R = by[:,None].expand(len(nodes_with_edges),len(nodes_with_edges))
                #by_L = by[None,:].expand(len(nodes_with_edges),len(nodes_with_edges))

                #roughly see if the two polygons are overlapping
                overlapping = ((torch.min(bx_R,bx_L)-torch.max(tx_R,tx_L))>0) * ((torch.min(by_R,by_L)-torch.max(ty_R,ty_L))>0)

                #define adjacency as having the end points being close. Only calculate where needed (not overlapping, but is overSeg candidate
                #close_ends = torch.BoolTensor(len(nodes_with_edges),len(nodes_with_edges)).zero_()
                close_ends = torch.BoolTensor(len(edgePredIndexes)).zero_()
                #for nwe0,nwe1 in torch.triu(torch.nonzero(~overlapping*wasOverSeg),1):
                for i in torch.nonzero(~overlapping*wasOverSeg):
                    n0,n1 = edgePredIndexes[i]
                    #n0 = nodes_with_edges[nwe0]
                    #n1 = nodes_with_edges[nwe1]
                    point_pairs_0 = outputBoxes[n0].pairPoints()
                    point_pairs_1 = outputBoxes[n1].pairPoints()
                    dist = min(util.pointDistance(point_pairs_0[0][0],point_pairs_1[-1][0]),util.pointDistance(point_pairs_0[-1][0],point_pairs_1[0][0]),util.pointDistance(point_pairs_0[0][1],point_pairs_1[-1][1]),util.pointDistance(point_pairs_0[-1][1],point_pairs_1[0][1]))
                    mean_h = (outputBoxes[n0].getHeight()+outputBoxes[n1].getHeight())/2
                    if dist<mean_h:
                        #close_ends[nwe0,nwe1]=True
                        #close_ends[nwe1,nwe0]=True
                        close_ends[i]=True


                wasOverSeg *= overlapping+close_ends.to(edge_loss_device) #refine candidates by rule

                if self.picky_merging:
                    #Only merge things of matching height
                    h_L = by_L-ty_L
                    h_R = by_R-ty_R
                    h_ratio = torch.min(h_L,h_R)/torch.max(h_L,h_R)
                    matching_height = h_ratio>0.8#self.height_ratio_thresh
                    wasOverSeg *= matching_height

            else:
                raise NotImplementedError('Unknown init merge rule: {}'.format(self.init_merge_rule))
            
            wasGroup = bothTarged*((g_len_L>1)+(g_len_R>1)+~same_ts_0)*bothPure*same_GTGroups
            wasNoGroup = badTarged+(bothTarged*((g_len_L>1)+(g_len_R>1)+~same_ts_0)*bothPure*~same_GTGroups)

            wasError = bothTarged*((purity_R<1)+(purity_L<1))
            wasNoError = bothTarged*~((purity_R<1)+(purity_L<1))

            #build vectors for loss, score results, and indicate each edges case (saveEdgePred)
            saveIndex={'TP':1,'TN':2,'FP':3,'FN':4,'UP':5,'UN':6}
            revSaveIndex={v:k for k,v in saveIndex.items()}

            predsEdgeAboveThresh = torch.sigmoid(predsEdge[:,-1])>thresh_edge

            saveEdgePredMat = torch.IntTensor(len(edgePredIndexes)).zero_()
            if not merge_only:

                shouldBeEdge = wasRel+wasOverSeg+wasGroup

                #TP,FP,FN,TN moved to below as reused by merge_only

                #unknownEdge = wasError*~shouldBeEdge
                #UP = unknownEdge*predsEdgeAboveThresh
                #saveEdgePredMat[UP]=saveIndex['UP']
                #UN = unknownEdge*~predsEdgeAboveThresh
                #saveEdgePredMat[UN]=saveIndex['UN']
                #saveEdgePred ={i:revSaveIndex[saveEdgePredMat[i].item()] for i in range(len(edgePredIndexes)) if saveEdgePredMat[i]>0}

                predsRelAboveThresh = torch.sigmoid(predsRel[:,-1])>thresh_rel
                saveRelPredMat = torch.IntTensor(len(edgePredIndexes)).zero_()

                predsGTRel = predsRel[wasRel]
                predsGTNoRel = predsRel[wasNoRel]
                TP = wasRel*predsRelAboveThresh
                truePosRel = TP.sum().item()
                saveRelPredMat[TP]=saveIndex['TP']
                FP = wasNoRel*predsRelAboveThresh
                falsePosRel = FP.sum().item()
                saveRelPredMat[FP]=saveIndex['FP']
                FN = wasRel*~predsRelAboveThresh
                falseNegRel = FN.sum().item()
                saveRelPredMat[FN]=saveIndex['FN']
                TN = wasNoRel*~predsRelAboveThresh
                trueNegRel = TN.sum().item()
                saveRelPredMat[TN]=saveIndex['TN']
                unk = ~wasRel*~wasNoRel
                UP = unk*predsRelAboveThresh
                saveRelPredMat[UP]=saveIndex['UP']
                UN = unk*~predsRelAboveThresh
                saveRelPredMat[UN]=saveIndex['UN']
                saveRelPred ={i:revSaveIndex[saveRelPredMat[i].item()] for i in range(len(edgePredIndexes)) if saveRelPredMat[i]>0}


                predsGroupAboveThresh = torch.sigmoid(predsGroup[:,-1])>thresh_group
                saveGroupPredMat = torch.IntTensor(len(edgePredIndexes)).zero_()

                predsGTGroup = predsGroup[wasGroup]
                predsGTNoGroup = predsGroup[wasNoGroup]
                TP = wasGroup*predsGroupAboveThresh
                truePosGroup = TP.sum().item()
                saveGroupPredMat[TP]=saveIndex['TP']
                FP = wasNoGroup*predsGroupAboveThresh
                falsePosGroup = FP.sum().item()
                saveGroupPredMat[FP]=saveIndex['FP']
                FN = wasGroup*~predsGroupAboveThresh
                falseNegGroup = FN.sum().item()
                saveGroupPredMat[FN]=saveIndex['FN']
                TN = wasNoGroup*~predsGroupAboveThresh
                trueNegGroup = TN.sum().item()
                saveGroupPredMat[TN]=saveIndex['TN']
                unk = ~wasGroup*~wasNoGroup
                UP = unk*predsGroupAboveThresh
                saveGroupPredMat[UP]=saveIndex['UP']
                UN = unk*~predsGroupAboveThresh
                saveGroupPredMat[UN]=saveIndex['UN']
                saveGroupPred ={i:revSaveIndex[saveGroupPredMat[i].item()] for i in range(len(edgePredIndexes)) if saveGroupPredMat[i]>0}


                predsOverSegAboveThresh = torch.sigmoid(predsOverSeg[:,-1])>thresh_overSeg
                saveOverSegPredMat = torch.IntTensor(len(edgePredIndexes)).zero_()

                d_indexesOverSeg = torch.nonzero(wasOverSeg)
                d_indexesNoOverSeg = torch.nonzero(wasNoOverSeg)
                d_indexesOverSeg = set([a.item() for a in d_indexesOverSeg])
                d_indexesNoOverSeg = set([a.item() for a in d_indexesNoOverSeg])
                predsGTOverSeg = predsOverSeg[wasOverSeg]
                predsGTNotOverSeg = predsOverSeg[wasNoOverSeg]
                TP = wasOverSeg*predsOverSegAboveThresh
                truePosOverSeg = TP.sum().item()
                saveOverSegPredMat[TP]=saveIndex['TP']
                FP = wasNoOverSeg*predsOverSegAboveThresh
                falsePosOverSeg = FP.sum().item()
                saveOverSegPredMat[FP]=saveIndex['FP']
                FN = wasOverSeg*~predsOverSegAboveThresh
                falseNegOverSeg = FN.sum().item()
                saveOverSegPredMat[FN]=saveIndex['FN']
                TN = wasNoOverSeg*~predsOverSegAboveThresh
                trueNegOverSeg = TN.sum().item()
                saveOverSegPredMat[TN]=saveIndex['TN']
                unk = ~wasOverSeg*~wasNoOverSeg
                UP = unk*predsOverSegAboveThresh
                saveOverSegPredMat[UP]=saveIndex['UP']
                UN = unk*~predsOverSegAboveThresh
                saveOverSegPredMat[UN]=saveIndex['UN']
                saveOverSegPred ={i:revSaveIndex[saveOverSegPredMat[i].item()] for i in range(len(edgePredIndexes)) if saveOverSegPredMat[i]>0}


                predsErrorAboveThresh = torch.sigmoid(predsError[:,-1])>thresh_error
                saveErrorPredMat = torch.IntTensor(len(edgePredIndexes)).zero_()

                predsGTError = predsError[wasError]
                predsGTNoError = predsError[wasNoError]
                TP = wasError*predsErrorAboveThresh
                truePosError = TP.sum().item()
                saveErrorPredMat[TP]=saveIndex['TP']
                FP = wasNoError*predsErrorAboveThresh
                falsePosError = FP.sum().item()
                saveErrorPredMat[FP]=saveIndex['FP']
                FN = wasError*~predsErrorAboveThresh
                falseNegError = FN.sum().item()
                saveErrorPredMat[FN]=saveIndex['FN']
                TN = wasNoError*~predsErrorAboveThresh
                trueNegError = TN.sum().item()
                saveErrorPredMat[TN]=saveIndex['TN']
                unk = ~wasError*~wasNoError
                UP = unk*predsErrorAboveThresh
                saveErrorPredMat[UP]=saveIndex['UP']
                UN = unk*~predsErrorAboveThresh
                saveErrorPredMat[UN]=saveIndex['UN']
                saveErrorPred ={i:revSaveIndex[saveErrorPredMat[i].item()] for i in range(len(edgePredIndexes)) if saveErrorPredMat[i]>0}

            else:
                shouldBeEdge = wasOverSeg
                d_indexesRel=d_indexesNoRel=d_indexesOverSeg=d_indexesNoOverSeg=None
            predsGTEdge = predsEdge[shouldBeEdge]
            TP=shouldBeEdge*predsEdgeAboveThresh
            truePosEdge = TP.sum().item()
            saveEdgePredMat[TP]=saveIndex['TP']
            FN=shouldBeEdge*~predsEdgeAboveThresh
            falseNegEdge = FN.sum().item()
            saveEdgePredMat[FN]=saveIndex['FN']
            #missedGTRel TODO
            for ind in torch.nonzero(FN):
                missed_rels.add(gtNE[ind.item()])

            shouldNotBeEdge = ~wasError*~shouldBeEdge
            predsGTNoEdge = predsEdge[shouldNotBeEdge]
            FP = shouldNotBeEdge*predsEdgeAboveThresh
            falsePosEdge = FP.sum().item()
            saveEdgePredMat[FP]=saveIndex['FP']
            TN = shouldNotBeEdge*~predsEdgeAboveThresh
            trueNegEdge = TN.sum().item()
            saveEdgePredMat[TN]=saveIndex['TN']
            unk = ~shouldBeEdge*~shouldNotBeEdge
            UP = unk*predsEdgeAboveThresh
            saveEdgePredMat[UP]=saveIndex['UP']
            UN = unk*~predsEdgeAboveThresh
            saveEdgePredMat[UN]=saveIndex['UN']
            saveEdgePred ={i:revSaveIndex[saveEdgePredMat[i].item()] for i in range(len(edgePredIndexes)) if saveEdgePredMat[i]>0}

            #t#time = timeit.default_timer()-tic#t#
            #t#self.opt_history['simplerAlign edge loop'].append(time)#t#
            #t#self.opt_history['simplerAlign edge loop per'].append(time/len(edgePredIndexes))#t#
            #t#tic=timeit.default_timer()#t#
            
            if not merge_only:
                predsGTYes = torch.cat((predsGTEdge,predsGTRel,predsGTOverSeg,predsGTGroup,predsGTError),dim=0)
                predsGTNo = torch.cat((predsGTNoEdge,predsGTNoRel,predsGTNotOverSeg,predsGTNoGroup,predsGTNoError),dim=0)

            else: #merge_only
                predsGTYes=predsGTEdge
                predsGTNo=predsGTNoEdge
            recallEdge = truePosEdge/(truePosEdge+falseNegEdge) if truePosEdge+falseNegEdge>0 else 1
            precEdge = truePosEdge/(truePosEdge+falsePosEdge) if truePosEdge+falsePosEdge>0 else 1


            if merge_only:
                log = {
                    'recallMergeFirst' : recallEdge, 
                    'precMergeFirst' : precEdge, 
                    'FmMergeFirst' : 2*(precEdge*recallEdge)/(recallEdge+precEdge) if recallEdge+precEdge>0 else 0
                    }
                predTypes=[saveEdgePred]
            else:

                recallRel = truePosRel/(truePosRel+falseNegRel) if truePosRel+falseNegRel>0 else 1
                precRel = truePosRel/(truePosRel+falsePosRel) if truePosRel+falsePosRel>0 else 1
                recallOverSeg = truePosOverSeg/(truePosOverSeg+falseNegOverSeg) if truePosOverSeg+falseNegOverSeg>0 else 1
                precOverSeg = truePosOverSeg/(truePosOverSeg+falsePosOverSeg) if truePosOverSeg+falsePosOverSeg>0 else 1
                recallGroup = truePosGroup/(truePosGroup+falseNegGroup) if truePosGroup+falseNegGroup>0 else 1
                assert(falsePosGroup>=0 and truePosGroup>=0)
                precGroup = truePosGroup/(truePosGroup+falsePosGroup) if truePosGroup+falsePosGroup>0 else 1
                recallError = truePosError/(truePosError+falseNegError) if truePosError+falseNegError>0 else 1
                precError = truePosError/(truePosError+falsePosError) if truePosError+falsePosError>0 else 1


                log = {
                    'recallEdge' : recallEdge, 
                    'precEdge' : precEdge, 
                    'FmEdge' : 2*(precEdge*recallEdge)/(recallEdge+precEdge) if recallEdge+precEdge>0 else 0,
                    'recallRel' : recallRel, 
                    'precRel' : precRel, 
                    'FmRel' : 2*(precRel*recallRel)/(recallRel+precRel) if recallRel+precRel>0 else 0,
                    'recallOverSeg' : recallOverSeg,
                    'precOverSeg' : precOverSeg,
                    'FmOverSeg' : 2*(precOverSeg*recallOverSeg)/(recallOverSeg+precOverSeg) if recallOverSeg+precOverSeg>0 else 0,
                    'recallGroup' : recallGroup,
                    'precGroup' : precGroup, 
                    'FmGroup' : 2*(precGroup*recallGroup)/(recallGroup+precGroup) if recallGroup+precGroup>0 else 0,
                    'recallError' : recallError,
                    'precError' : precError,
                    'FmError' : 2*(precError*recallError)/(recallError+precError) if recallError+precError>0 else 0,
                    }
                predTypes = [saveEdgePred,saveRelPred,saveOverSegPred,saveGroupPred,saveErrorPred]

            #t#self.opt_history['simplerAlign edge resolution'].append(timeit.default_timer()-tic)#t#


        if rel_prop_pred is not None:

            relPropScores,relPropIds, threshPropRel = rel_prop_pred

            #print('\tcount rel prop: {}'.format(len(relPropIds)))
            #t#tic=timeit.default_timer()#t##t#
            truePropPred=falsePropPred=falseNegProp=0
            propPredsPos=[]
            propPredsNeg=[]
            if not merge_only:
                #edgesWithTargetGroup = [(targIndex[n0],targIndex[n1],targetIndexToGroup[t0]
                #isEdge = torch.BoolTensor([
                #        targIndex[n0]>=0 and targIndex[n1]>=0 and 
                #    (   targIndex[n0]==targIndex[n1] or
                #         
                #    ) for n0,n1 in relPropIds]).to(relPropScores.device)
                for i,(n0,n1) in enumerate(relPropIds):
                    #if (n0,n1) in shouldBeEdge: #unsure if this really saves time
                    #    isEdge = shouldBeEdge[(n0,n1)]
                    #else:
                    t0 = targIndex[n0]
                    t1 = targIndex[n1]
                    #ts0=predGroupsT[n0]
                    #ts1=predGroupsT[n1]
                    #assert(len(predGroupsT[n0])<=1 and len(predGroupsT[n1])<=1) removing for efficiency
                    isEdge=False
                    if t0>=0 and t1>=0:
                        if t0==t1:
                            isEdge=True
                        else:
                            gtGroup0 = targetIndexToGroup[t0]#getGTGroup([t0],gtGroups)
                            gtGroup1 = targetIndexToGroup[t1]#getGTGroup([t1],gtGroups)
                            
                            if gtGroup0==gtGroup1:
                                isEdge=True
                            else:
                                if (min(gtGroup0,gtGroup1),max(gtGroup0,gtGroup1)) in gtGroupAdj:
                                    isEdge=True
                    if isEdge:
                        propPredsPos.append(relPropScores[i])
                        if relPropScores[i]>threshPropRel:
                            truePropPred+=1
                        else:
                            falseNegProp+=1
                    else:
                        propPredsNeg.append(relPropScores[i])
                        if relPropScores[i]>threshPropRel:
                            falsePropPred+=1
                if len(propPredsPos)>0:
                    propPredsPos = torch.stack(propPredsPos,dim=0)
                else:
                    propPredsPos=None
                if len(propPredsNeg)>0:
                    propPredsNeg = torch.stack(propPredsNeg,dim=0)
                else:
                    propPredsNeg=None
                #t#time = timeit.default_timer()-tic#t##t
                #t#self.opt_history['simplerAlign proposal'].append(time)#t#
                #t#self.opt_history['simplerAlign proposal per edge'].append(time/len(relPropIds))#t#
            else:
                #vectorized, this has a lot of instances
                isEdge = torch.BoolTensor([(targIndex[n0]==targIndex[n1] and targIndex[n0]>=0) for n0,n1 in relPropIds]).to(relPropScores.device)
                relPropScoresAboveThresh = relPropScores>threshPropRel
                truePropPred= (isEdge*relPropScoresAboveThresh).sum().item()
                falseNegProp= (isEdge*~relPropScoresAboveThresh).sum().item()
                falsePropPred= (~isEdge*relPropScoresAboveThresh).sum().item()
                propPredsPos = relPropScores[isEdge]
                propPredsNeg = relPropScores[~isEdge]
                if len(propPredsPos)==0:
                    propPredsPos = None
                if len(propPredsNeg)==0:
                    propPredsNeg = None

                #t#time = timeit.default_timer()-tic#t##t
                #t#self.opt_history['simplerAlign proposal m1st'].append(time)#t#
                #t#self.opt_history['simplerAlign proposal m1st per edge'].append(time/len(relPropIds))#t#


        


        
            propRecall=truePropPred/(truePropPred+falseNegProp) if truePropPred+falseNegProp>0 else 1
            propPrec=truePropPred/(truePropPred+falsePropPred) if truePropPred+falsePropPred>0 else 1
            log['edgePropRecall']=propRecall
            log['edgePropPrec']=propPrec

            proposedInfo = (propPredsPos,propPredsNeg, propRecall, propPrec)
            #self.timing{
        else:
            proposedInfo = None

        return predsGTYes, predsGTNo, targIndex,  proposedInfo, log, predTypes, missed_rels


    def prealignedEdgePred(self,adj,relPred,relIndexes,rel_prop_pred):
        if relPred is None:
            #assert(adj is None or len(adj)==0) this is a failure of the heuristic pairing
            if adj is not None and len(adj)>0:
                recall=0
                ap=0
            else:
                recall=1
                ap=1
            prec=1

            return torch.tensor([]),torch.tensor([]),recall,prec,prec,ap, None
        rels = relIndexes #relPred._indices().cpu().t()
        predsAll = relPred
        sigPredsAll = torch.sigmoid(predsAll[:,-1])

        #gt = torch.empty(len(rels))#rels.size(0))
        predsPos = []
        predsNeg = []
        scores = []
        truePred=falsePred=0
        for i,(n0,n1) in enumerate(rels):
            #n0 = rels[i,0]
            #n1 = rels[i,1]
            #gt[i] = int((n0,n1) in adj) #(adjM[ n0, n1 ])
            if (n0,n1) in adj:
                predsPos.append(predsAll[i])
                scores.append( (sigPredsAll[i],True) )
                if sigPredsAll[i]>self.thresh_rel:
                    truePred+=1
            else:
                predsNeg.append(predsAll[i])
                scores.append( (sigPredsAll[i],False) )
                if sigPredsAll[i]>self.thresh_rel:
                    falsePred+=1
    
        #return gt.to(relPred.device), relPred._values().view(-1).view(-1)
        #return gt.to(relPred[1].device), relPred[1].view(-1)
        if len(predsPos)>0:
            predsPos = torch.stack(predsPos).to(relPred.device)
        else:
            predsPos = None
        if len(predsNeg)>0:
            predsNeg = torch.stack(predsNeg).to(relPred.device)
        else:
            predsNeg = None
        if len(adj)>0:
            recall = truePred/len(adj)
        else:
            recall = 1
        if falsePred>0:
            prec = truePred/(truePred+falsePred)
        else:
            prec = 1


        if rel_prop_pred is not None:
            relPropScores,relPropIds, threshPropRel = rel_prop_pred
            propPredsPos = []
            propPredsNeg = []
            scores = []
            truePropPred=falsePropPred=0
            for i,(n0,n1) in enumerate(rels):
                #n0 = rels[i,0]
                #n1 = rels[i,1]
                #gt[i] = int((n0,n1) in adj) #(adjM[ n0, n1 ])
                if (n0,n1) in adj:
                    propPredsPos.append(relPropScores[i])
                    #scores.append( (sigPredsAll[i],True) )
                    if relPropScores[i]>threshPropRel:
                        truePropPred+=1
                else:
                    propPredsNeg.append(relPropScores[i])
                    #scores.append( (relPropScores[i],False) )
                    if relPropScores[i]>threshPropRel:
                        falsePropPred+=1
        
            #return gt.to(relPred.device), relPred._values().view(-1).view(-1)
            #return gt.to(relPred[1].device), relPred[1].view(-1)
            if len(propPredsPos)>0:
                propPredsPos = torch.stack(propPredsPos).to(relPred.device)
            else:
                propPredsPos = None
            if len(propPredsNeg)>0:
                propPredsNeg = torch.stack(propPredsNeg).to(relPred.device)
            else:
                propPredsNeg = None
            if len(adj)>0:
                propRecall = truePropPred/len(adj)
            else:
                propRecall = 1
            if falsePropPred>0:
                propPrec = truePropPred/(truePropPred+falsePropPred)
            else:
                propPrec = 1
            proposedInfo = (propPredsPos,propPredsNeg, propRecall, propPrec )
        else:
            proposedInfo = None


        return predsPos,predsNeg, recall, prec, prec, computeAP(scores), proposedInfo

    #old
    def run(self,instance,useGT,threshIntur=None,get=[]):
        numClasses = self.model_ref.numBBTypes
        if 'no_blanks' in self.config['validation'] and not self.config['data_loader']['no_blanks']:
            numClasses-=1
        image, targetBoxes, adj, target_num_neighbors = self._to_tensor(instance)
        if useGT:
            outputBoxes, outputOffsets, relPred, relIndexes, bbPred, rel_prop_pred = self.model(image,targetBoxes,target_num_neighbors,True,
                    otherThresh=self.conf_thresh_init, otherThreshIntur=threshIntur, hard_detect_limit=self.train_hard_detect_limit)
            #_=None
            #gtPairing,predPairing = self.prealignedEdgePred(adj,relPred)
            predPairingShouldBeTrue,predPairingShouldBeFalse, eRecall,ePrec,fullPrec,ap,proposedInfo = self.prealignedEdgePred(adj,relPred,relIndexes,rel_prop_pred)
            if bbPred is not None:
                if self.model_ref.predNN or self.model_ref.predClass:
                    if target_num_neighbors is not None:
                        alignedNN_use = target_num_neighbors[0]
                    bbPredNN_use = bbPred[:,:,0]
                    start=1
                else:
                    start=0
                if self.model_ref.predClass:
                    if targetBoxes is not None:
                        alignedClass_use =  targetBoxes[0,:,13:13+self.model_ref.numBBTypes]
                    bbPredClass_use = bbPred[:,:,start:start+self.model_ref.numBBTypes]
            else:
                bbPredNN_use=None
                bbPredClass_use=None
            final_prop_rel_recall = final_prop_rel_prec = None
        else:
            outputBoxes, outputOffsets, relPred, relIndexes, bbPred, rel_prop_pred = self.model(image,
                    otherThresh=self.conf_thresh_init, otherThreshIntur=threshIntur, hard_detect_limit=self.train_hard_detect_limit)

            if self.optimize:
                #This changes relPred
                self.optimizeF(relPred,relIndexes,bbPred[:,-1,0])

            #gtPairing,predPairing = self.alignEdgePred(targetBoxes,adj,outputBoxes,relPred)
            predPairingShouldBeTrue,predPairingShouldBeFalse, eRecall,ePrec,fullPrec,ap, bbAlignment, bbFullHit, proposedInfo, final_prop_rel_recall, final_prop_rel_prec = self.alignEdgePred(targetBoxes,adj,outputBoxes,relPred,relIndexes, rel_prop_pred)
            if bbPred is not None and bbPred.size(0)>0:
                #create aligned GT
                #this was wrong...
                    #first, remove unmatched predicitons that didn't overlap (weren't close) to any targets
                    #toKeep = 1-((bbNoIntersections==1) * (bbAlignment==-1))
                #remove predictions that overlapped with GT, but not enough
                if self.model_ref.predNN:
                    start=1
                    toKeep = ~((bbFullHit==0) * (bbAlignment!=-1)) #toKeep = not (incomplete_overlap and did_overlap)
                    if toKeep.any():
                        bbPredNN_use = bbPred[toKeep][:,:,0]
                        bbAlignment_use = bbAlignment[toKeep]
                        #becuase we used -1 to indicate no match (in bbAlignment), we add 0 as the last position in the GT, as unmatched 
                        if target_num_neighbors is not None:
                            target_num_neighbors_use = torch.cat((target_num_neighbors[0].float(),torch.zeros(1).to(target_num_neighbors.device)),dim=0)
                        else:
                            target_num_neighbors_use = torch.zeros(1).to(bbPred.device)
                        alignedNN_use = target_num_neighbors_use[bbAlignment_use.long()]

                    else:
                        bbPredNN_use=None
                        alignedNN_use=None
                else:
                    start=0
                if self.model_ref.predClass:
                    #We really don't care about the class of non-overlapping instances
                    if targetBoxes is not None:
                        toKeep = bbFullHit==1
                        if toKeep.any():
                            bbPredClass_use = bbPred[toKeep][:,:,start:start+self.model_ref.numBBTypes]
                            bbAlignment_use = bbAlignment[toKeep]
                            alignedClass_use =  targetBoxes[0][bbAlignment_use.long()][:,13:13+self.model_ref.numBBTypes] #There should be no -1 indexes in hereS
                        else:
                            alignedClass_use = None
                            bbPredClass_use = None
                    else:
                        alignedClass_use = None
                        bbPredClass_use = None
            else:
                bbPredNN_use = None
                bbPredClass_use = None
        #if relPred is not None:
        #    numEdgePred = relPred.size(0)
        #    if predPairingShouldBeTrue is not None:
        #        lenTrue = predPairingShouldBeTrue.size(0)
        #    else:
        #        lenTrue = 0
        #    if predPairingShouldBeFalse is not None:
        #        lenFalse = predPairingShouldBeFalse.size(0)
        #    else:
        #        lenFalse = 0
        #else:
        #    numEdgePred = lenTrue = lenFalse = 0
        numBoxPred = outputBoxes.size(0)
        #if len(predPairing.size())>0 and predPairing.size(0)>0:
        #    relLoss = self.loss['rel'](predPairing,gtPairing)
        #else:
        #    relLoss = torch.tensor(0.0,requires_grad=True).to(image.device)
        #relLoss = torch.tensor(0.0).to(image.device)
        relLoss = None
        #seperating the loss into true and false portions is not only convienint, it balances the loss between true/false examples
        if predPairingShouldBeTrue is not None and predPairingShouldBeTrue.size(0)>0:
            ones = torch.ones_like(predPairingShouldBeTrue).to(image.device)
            relLoss = self.loss['rel'](predPairingShouldBeTrue,ones)
            debug_avg_relTrue = predPairingShouldBeTrue.mean().item()
        else:
            debug_avg_relTrue =0 
        if predPairingShouldBeFalse is not None and predPairingShouldBeFalse.size(0)>0:
            zeros = torch.zeros_like(predPairingShouldBeFalse).to(image.device)
            relLossFalse = self.loss['rel'](predPairingShouldBeFalse,zeros)
            if relLoss is None:
                relLoss=relLossFalse
            else:
                relLoss+=relLossFalse
            debug_avg_relFalse = predPairingShouldBeFalse.mean().item()
        else:
            debug_avg_relFalse = 0
        losses={}
        if relLoss is not None:
            #relLoss *= self.lossWeights['rel']
            losses['relLoss']=relLoss

        if proposedInfo is not None:
            propPredPairingShouldBeTrue,propPredPairingShouldBeFalse= proposedInfo[0:2]
            propRelLoss = None
            #seperating the loss into true and false portions is not only convienint, it balances the loss between true/false examples
            if propPredPairingShouldBeTrue is not None and propPredPairingShouldBeTrue.size(0)>0:
                ones = torch.ones_like(propPredPairingShouldBeTrue).to(image.device)
                propRelLoss = self.loss['propRel'](propPredPairingShouldBeTrue,ones)
            if propPredPairingShouldBeFalse is not None and propPredPairingShouldBeFalse.size(0)>0:
                zeros = torch.zeros_like(propPredPairingShouldBeFalse).to(image.device)
                propRelLossFalse = self.loss['propRel'](propPredPairingShouldBeFalse,zeros)
                if propRelLoss is None:
                    propRelLoss=propRelLossFalse
                else:
                    propRelLoss+=propRelLossFalse
            if propRelLoss is not None:
                losses['propRelLoss']=propRelLoss



        if not self.model_ref.detector_frozen:
            if targetBoxes is not None:
                targSize = targetBoxes.size(1)
            else:
                targSize =0 

            if 'box' in self.loss:
                boxLoss, position_loss, conf_loss, class_loss, nn_loss, recall, precision = self.loss['box'](outputOffsets,targetBoxes,[targSize],target_num_neighbors)
                losses['boxLoss'] = boxLoss
            else:
                oversegLoss, position_loss, conf_loss, class_loss, rot_loss, recall, precision, gt_covered, pred_covered, recall_noclass, precision_noclass, gt_covered_noclass, pred_covered_noclass = self.loss['overseg'](outputOffsets,targetBoxes,[targSize],calc_stats='bb_stats' in get)
                losses['oversegLoss'] = oversegLoss
            #boxLoss *= self.lossWeights['box']
            #if relLoss is not None:
            #    loss = relLoss + boxLoss
            #else:
            #    loss = boxLoss
        #else:
        #    loss = relLoss


        if self.model_ref.predNN and bbPredNN_use is not None and bbPredNN_use.size(0)>0 and 'nnFinal' in self.loss:
            alignedNN_use = alignedNN_use[:,None] #introduce "time" dimension to broadcast
            nn_loss_final = self.loss['nnFinal'](bbPredNN_use,alignedNN_use)
            losses['nnFinalLoss']=nn_loss_final
            #nn_loss_final *= self.lossWeights['nn']
            
            #if loss is not None:
            #    loss += nn_loss_final
            #else:
            #    loss = nn_loss_final
            #nn_loss_final = nn_loss_final.item()
        #else:
            #nn_loss_final=0

        if self.model_ref.predClass and bbPredClass_use is not None and bbPredClass_use.size(0)>0:
            alignedClass_use = alignedClass_use[:,None] #introduce "time" dimension to broadcast
            class_loss_final = self.loss['classFinal'](bbPredClass_use,alignedClass_use)
            losses['classFinalLoss'] = class_loss_final
            #class_loss_final *= self.lossWeights['class']
            #loss += class_loss_final
            #class_loss_final = class_loss_final.item()
        #else:
            #class_loss_final = 0
        
        log={
                'rel_prec': fullPrec,
                'rel_recall': eRecall,
                'rel_Fm': 2*(fullPrec*eRecall)/(eRecall+fullPrec) if eRecall+fullPrec>0 else 0
                }

        gt_hit = [False]*targetBoxes.size(1)
        if self.model_ref.predNN:
            start=7
        else:
            start=6
        ed_true_pos=0
        for ni in range(outputBoxes.size(0)):
            if bbAlignment[ni]>-1 and bbFullHit[ni] and not gt_hit[bbAlignment[ni]]:
                p_cls = outputBoxes[ni,start:start+self.model_ref.numBBTypes].argmax().item()
                if targetBoxes[0,bbAlignment[ni],13+p_cls]==1:
                    ed_true_pos+=1
                    gt_hit[bbAlignment[ni]]=True
        if targetBoxes.size(1)>0:
            log['ED_recall'] = ed_true_pos/targetBoxes.size(1)
        else:
            log['ED_recall'] = 1
        if outputBoxes.size(0)>0:
            log['ED_prec'] = ed_true_pos/outputBoxes.size(0)
        else:
            log['ED_prec'] = 1
        if log['ED_recall']+log['ED_prec']>0:
            log['ED_F1']=2*log['ED_prec']*log['ED_recall']/(log['ED_recall']+log['ED_prec'])
        else:
            log['ED_F1']=0

        if ap is not None:
            log['rel_AP']=ap
        if not self.model_ref.detector_frozen:
            if 'nnFinalLoss' in losses:
                log['nn loss improvement (neg is good)'] = losses['nnFinalLoss'].item()-nn_loss
            if 'classFinalLoss' in losses:
                log['class loss improvement (neg is good)'] = losses['classFinalLoss'].item()-class_loss

        if 'bb_stats' in get:
            if self.model_ref.detector.predNumNeighbors:
                outputBoxes=torch.cat((outputBoxes[:,0:6],outputBoxes[:,7:]),dim=1) #throw away NN pred
            if targetBoxes is not None:
                targetBoxes = targetBoxes.cpu()
                target_for_b = targetBoxes[0]
            else:
                target_for_b = torch.empty(0)
            if self.model_ref.useCurvedBBs:
                ap_5, prec_5, recall_5, allPrec, allRecall =AP_textLines(target_for_b,outputBoxes,0.5,numClasses)
            elif self.model_ref.rotation:
                ap_5, prec_5, recall_5, allPrec, allRecall =AP_dist(target_for_b,outputBoxes,0.9,numClasses)
            else:
                ap_5, prec_5, recall_5, allPrec, allRecall =AP_iou(target_for_b,outputBoxes,0.5,numClasses)
            prec_5 = np.array(prec_5)
            recall_5 = np.array(recall_5)
            log['bb_AP']=ap_5
            log['bb_prec']=prec_5
            log['bb_recall']=recall_5
            #Fm=2*(prec_5*recall_5)/(prec_5+recall_5)
            #Fm[np.isnan(Fm)]=0
            #log['bb_Fm_avg']=Fm.mean()
            log['bb_allPrec']=allPrec
            log['bb_allRecall']=allRecall
            log['bb_allFm']= 2*allPrec*allRecall/(allPrec+allRecall) if allPrec+allRecall>0 else 0


        if 'nn_acc' in get:
            if self.model_ref.predNN and bbPred is not None:
                predNN_p=bbPred[:,-1,0]
                diffs=torch.abs(predNN_p-target_num_neighbors[0][bbAlignment].float())
                nn_acc = (diffs<0.5).float().mean().item()
                log['nn_acc']=nn_acc

        if proposedInfo is not None:
            propRecall,propPrec = proposedInfo[2:4]
            log['prop_rel_recall'] = propRecall
            log['prop_rel_prec'] = propPrec
        if final_prop_rel_recall is not None:
            log['final_prop_rel_recall']=final_prop_rel_recall
        if final_prop_rel_prec is not None:
            log['final_prop_rel_prec']=final_prop_rel_prec

        got={}#outputBoxes, outputOffsets, relPred, relIndexes, bbPred, rel_prop_pred
        for name in get:
            if name=='relPred':
                got['relPred'] = relPred.detach().cpu()
            elif name=='outputBoxes':
                if useGT:
                    got['outputBoxes'] = targetBoxes.cpu()
                else:
                    got['outputBoxes'] = outputBoxes.detach().cpu()
            elif name=='outputOffsets':
                got['outputOffsets'] = outputOffsets.detach().cpu()
            elif name=='relIndexes':
                got['relIndexes'] = relIndexes
            elif name=='bbPred':
                got['bbPred'] = bbPred.detach().cpu()

        return losses, log, got






    def newRun(self,instance,useGT,threshIntur=None,get=[]):#,useOnlyGTSpace=False,useGTGroups=False):
        assert(not self.model_ref.predNN)
        numClasses = len(self.classMap)
        image, targetBoxes, adj, target_num_neighbors = self._to_tensor(instance)
        gtGroups = instance['gt_groups']
        gtGroupAdj = instance['gt_groups_adj']
        targetIndexToGroup = instance['targetIndexToGroup']
        targetIndexToGroup = instance['targetIndexToGroup']
        if self.use_gt_trans:
            gtTrans = instance['transcription']
            if (gtTrans)==0:
                gtTrans=None
        else:
            gtTrans = None
        #t#tic=timeit.default_timer()#t##t#
        if useGT and len(useGT)>0:
            numBBTypes = self.model_ref.numBBTypes
            if 'word_bbs' in useGT: #useOnlyGTSpace and self.use_word_bbs_gt:
                word_boxes = instance['form_metadata']['word_boxes'][None,:,:,].to(targetBoxes.device) #I can change this as it isn't used later
                if self.model_ref.useCurvedBBs:
                    word_boxes = instance['form_metadata']['word_boxes']
                    x1 = word_boxes[:,:,0]-word_boxes[:,:,4]
                    x2 = word_boxes[:,:,0]+word_boxes[:,:,4]
                    y1 = word_boxes[:,:,1]-word_boxes[:,:,3]
                    y2 = word_boxes[:,:,1]+word_boxes[:,:,3]
                    r = word_boxes[:,:,2]
                    targetBoxes_changed = torch.stack((x1,y1,x2,y2,r),dim=2) #leave out class information
                    if self.model.training:
                        targetBoxes_changed[:,:,0] += torch.randn_like(targetBoxes_changed[:,:,0])
                        targetBoxes_changed[:,:,1] += torch.randn_like(targetBoxes_changed[:,:,1])
                        targetBoxes_changed[:,:,2] += torch.randn_like(targetBoxes_changed[:,:,2])
                        targetBoxes_changed[:,:,3] += torch.randn_like(targetBoxes_changed[:,:,2])
                else:
                    targetBoxes_changed=word_boxes
                    if self.model.training:
                        targetBoxes_changed[:,:,0] += torch.randn_like(targetBoxes_changed[:,:,0])
                        targetBoxes_changed[:,:,1] += torch.randn_like(targetBoxes_changed[:,:,1])
                        if self.model_ref.rotation:
                            targetBoxes_changed[:,:,2] += torch.randn_like(targetBoxes_changed[:,:,2])*0.01
                        targetBoxes_changed[:,:,3] += torch.randn_like(targetBoxes_changed[:,:,3])
                        targetBoxes_changed[:,:,4] += torch.randn_like(targetBoxes_changed[:,:,4])
                        targetBoxes_changed[:,:,3][targetBoxes_changed[:,:,3]<1]=1
                        targetBoxes_changed[:,:,4][targetBoxes_changed[:,:,4]<1]=1

            elif self.model_ref.useCurvedBBs:# and 'only_space' not in useGT:#not useOnlyGTSpace:
                #build targets of GT to pass as detections
                ph_boxes = [torch.zeros(1,1,1,1,1)]*3
                ph_cls = [torch.zeros(1,1,1,1,1)]*3
                ph_conf = [torch.zeros(1,1,1,1)]*3
                scale = self.model_ref.detector.scale
                numAnchors = self.model_ref.detector.numAnchors
                numBBParams = self.model_ref.detector.numBBParams
                numBBParams = self.model_ref.detector.numBBParams
                grid_sizesH=[image.size(2)//s[0] for s in scale]
                grid_sizesW=[image.size(3)//s[0] for s in scale]


                nGT, masks, conf_masks, t_Ls, t_Ts, t_Rs, t_Bs, t_rs, tconf_scales, tcls_scales, pred_covered, gt_covered, recall, precision, pred_covered_noclass, gt_covered_noclass, recall_noclass, precision_noclass = build_oversegmented_targets_multiscale(ph_boxes, ph_conf, ph_cls, targetBoxes, [targetBoxes.size(1)], numBBTypes, grid_sizesH, grid_sizesW,scale=scale, assign_mode='split', close_anchor_rule='unmask')
                
                for i in range(len(t_Ls)):
                    assert((t_Ls[i]<=t_Rs[i]).all() and (t_Ts[i]<=t_Bs[i]).all())

                #add some jitter
                if self.model.training:
                    t_Ls = [t.type(torch.FloatTensor) + torch.FloatTensor(t.size()).normal_() for t in t_Ls]
                    t_Ts = [t.type(torch.FloatTensor) + torch.FloatTensor(t.size()).normal_() for t in t_Ts]
                    t_Rs = [t.type(torch.FloatTensor) + torch.FloatTensor(t.size()).normal_() for t in t_Rs]
                    t_Bs = [t.type(torch.FloatTensor) + torch.FloatTensor(t.size()).normal_() for t in t_Bs]
                    t_rs = [t.type(torch.FloatTensor) + torch.FloatTensor(t.size()).normal_(std=0.01) for t in t_rs]
                    #fix bad BBs introduced by jitter
                    t_Ls[t_Ls>=t_Rs]+=1
                    t_Ts[t_Ts>=t_Bs]+=1

                tconf_scales = [t.type(torch.FloatTensor) for t in tconf_scales]
                tcls_scales = [t.type(torch.FloatTensor) for t in tcls_scales]

                ys = []
                for level in range(len(t_Ls)):
                    level_y = torch.cat([ torch.stack([2*tconf_scales[level]-1,t_Ls[level],t_Ts[level],t_Rs[level], t_Bs[level],t_rs[level]],dim=2), 2*tcls_scales[level].permute(0,1,4,2,3)-1], dim=2)
                    ###
                    #for r in range(level_y.size(3)):
                    #    for c in range(level_y.size(4)):
                    #        for bo in range(level_y.size(1)):
                    #            conf = level_y[0,bo,0,r,c]
                    #            x = level_y[0,bo,1,r,c]
                    #            y = level_y[0,bo,2,r,c]
                    #            if x<46 and x<95 and y>102 and y<130 and conf>0.5:
                    #                print('{} level_y[0,{},:,{},{}] = {}'.format(level,bo,r,c,level_y[0,bo,:,r,c]))
                    ys.append(level_y.view(level_y.size(0),level_y.size(1)*level_y.size(2),level_y.size(3),level_y.size(4)))
                targetBoxes_changed = build_box_predictions(ys,scale,ys[0].device,numAnchors,numBBParams,numBBTypes)
                #only take good predictions
                targetBoxes_changed = targetBoxes_changed[:,targetBoxes_changed[0,:,0]>0.5]

                #remove conf and class information
                targetBoxes_changed = targetBoxes_changed[:,:,1:6]

            #elif self.model_ref.useCurvedBBs and 'only_space' in useGT: #useOnlyGTSpace:
            #    #convert target boxes to x1y1x2y2r, as that's what TextLine expects
            #    x1 = targetBoxes[:,:,0]-targetBoxes[:,:,4]
            #    x2 = targetBoxes[:,:,0]+targetBoxes[:,:,4]
            #    y1 = targetBoxes[:,:,1]-targetBoxes[:,:,3]
            #    y2 = targetBoxes[:,:,1]+targetBoxes[:,:,3]
            #    r = targetBoxes[:,:,2]
            #    targetBoxes_changed = torch.stack((x1,y1,x2,y2,r),dim=2) #leave out class information
            else:
                targetBoxes_changed=targetBoxes.clone()
                if self.model.training:
                    targetBoxes_changed[:,:,0] += torch.randn_like(targetBoxes_changed[:,:,0])
                    targetBoxes_changed[:,:,1] += torch.randn_like(targetBoxes_changed[:,:,1])
                    if self.model_ref.rotation:
                        targetBoxes_changed[:,:,2] += torch.randn_like(targetBoxes_changed[:,:,2])*0.01
                    targetBoxes_changed[:,:,3] += torch.randn_like(targetBoxes_changed[:,:,3])
                    targetBoxes_changed[:,:,4] += torch.randn_like(targetBoxes_changed[:,:,4])
                    targetBoxes_changed[:,:,3][targetBoxes_changed[:,:,3]<1]=1
                    targetBoxes_changed[:,:,4][targetBoxes_changed[:,:,4]<1]=1
                    #we tweak the classes in the model

            if 'only_space' in useGT and not self.model_ref.useCurvedBBs:
                targetBoxes_changed[:,:,5:]=0 #zero out other information to ensure results aren't contaminated
                #useCurved doesnt include class



            allOutputBoxes, outputOffsets, allEdgePred, allEdgeIndexes, allNodePred, allPredGroups, rel_prop_pred,merge_prop_scores, final = self.model(
                                image,
                                targetBoxes_changed,
                                target_num_neighbors,
                                useGT,
                                #useOnlyGTSpace=useOnlyGTSpace,
                                otherThresh=self.conf_thresh_init, 
                                otherThreshIntur=threshIntur, 
                                hard_detect_limit=self.train_hard_detect_limit,
                                gtTrans = gtTrans,
                                merge_first_only = self.iteration<self.merge_first_only_until,
                                gtGroups = gtGroups if 'groups' in useGT else None)
            #TODO
            #predPairingShouldBeTrue,predPairingShouldBeFalse, eRecall,ePrec,fullPrec,ap,proposedInfo = self.prealignedEdgePred(adj,relPred,relIndexes,rel_prop_pred)
            #if bbPred is not None:
            #    if self.model_ref.predNN or self.model_ref.predClass:
            #        if target_num_neighbors is not None:
            #            alignedNN_use = target_num_neighbors[0]
            #        bbPredNN_use = bbPred[:,:,0]
            #        start=1
            #    else:
            #        start=0
            #    if self.model_ref.predClass:
            #        if targetBoxes is not None:
            #            alignedClass_use =  targetBoxes[0,:,13:13+self.model_ref.numBBTypes]
            #        bbPredClass_use = bbPred[:,:,start:start+self.model_ref.numBBTypes]
            #else:
            #    bbPredNN_use=None
            #    bbPredClass_use=None
            #final_prop_rel_recall = final_prop_rel_prec = None
        else:
            #outputBoxes, outputOffsets: one, predicted at the begining
            #relPred, relIndexes, bbPred, predGroups: multiple, for each step in graph prediction. relIndexes indexes into predGroups, which indexes to outputBoxes
            #rel_prop_pred: if we use prop, one for begining
            allOutputBoxes, outputOffsets, allEdgePred, allEdgeIndexes, allNodePred, allPredGroups, rel_prop_pred,merge_prop_scores, final = self.model(image,
                    targetBoxes if gtTrans is not None else None,
                    otherThresh=self.conf_thresh_init, 
                    otherThreshIntur=threshIntur, 
                    hard_detect_limit=self.train_hard_detect_limit,
                    gtTrans = gtTrans,
                    merge_first_only = self.iteration<self.merge_first_only_until)
            #gtPairing,predPairing = self.alignEdgePred(targetBoxes,adj,outputBoxes,relPred)
        #t#self.opt_history['run model'].append(timeit.default_timer()-tic)#t#
        #t#tic=timeit.default_timer()#t##t#
        ### TODO code prealigned
        losses=defaultdict(lambda:0)
        log={}
        #for graphIteration in range(len(allEdgePred)):
        allEdgePredTypes=[]
        allMissedRels=[]
        allBBAlignment=[]
        proposedInfo=None
        mergeProposedInfo=None

       # print('effective prop thresh: {:.3f}, raw: {:.3f}'.format(torch.sigmoid(torch.FloatTensor([rel_prop_pred[-1]])).item(),rel_prop_pred[-1]))

        merged_first = self.model_ref.merge_first and not useOnlyGTSpace
        if allEdgePred is not None:
            for graphIteration,(outputBoxes,edgePred,nodePred,edgeIndexes,predGroups) in enumerate(zip(allOutputBoxes,allEdgePred,allNodePred,allEdgeIndexes,allPredGroups)):

                if self.model_ref.merge_first and useOnlyGTSpace:
                    graphIteration+=1

                #t#tic2=timeit.default_timer()#t##t#
                predEdgeShouldBeTrue,predEdgeShouldBeFalse, bbAlignment, proposedInfoI, logIter, edgePredTypes, missedRels = self.simplerAlignEdgePred(
                        targetBoxes,
                        targetIndexToGroup,
                        gtGroupAdj,
                        outputBoxes,
                        edgePred,
                        edgeIndexes,
                        predGroups, 
                        rel_prop_pred if (graphIteration==0 and not merged_first) or (graphIteration==1 and merged_first) else (merge_prop_scores if graphIteration==0 and merged_first else None),
                        self.thresh_edge[graphIteration],
                        self.thresh_rel[graphIteration],
                        self.thresh_overSeg[graphIteration],
                        self.thresh_group[graphIteration],
                        self.thresh_error[graphIteration],
                        merge_only= graphIteration==0 and merged_first
                        )
                #t#self.opt_history['newAlignEdgePred gI{}'.format(graphIteration)].append(timeit.default_timer()-tic2)#t#

                allEdgePredTypes.append(edgePredTypes)
                allMissedRels.append(missedRels)
                allBBAlignment.append(bbAlignment)
                if graphIteration==0 and merged_first:
                    mergeProposedInfo=proposedInfoI
                elif (graphIteration==0 and not merged_first) or (graphIteration==1 and merged_first):
                    proposedInfo=proposedInfoI

                if self.model_ref.predClass and nodePred is not None:
                    node_pred_use_index=[]
                    node_gt_use_class_indexes=[]
                    node_pred_use_index_sp=[]
                    alignedClass_use_sp=[]

                    node_conf_use_index=[]
                    node_conf_gt=[]#torch.FloatTensor(len(predGroups))

                    for i,predGroup in enumerate(predGroups):
                        ts=[bbAlignment[pId] for pId in predGroup]
                        classes=defaultdict(lambda:0)
                        classesIndexes={-2:-2}
                        hits=misses=0
                        for tId in ts:
                            if tId>=0:
                                #this is unfortunate. It's here since we use multiple classes
                                clsRep = ','.join([str(int(targetBoxes[0][tId,13+clasI])) for clasI in range(len(self.classMap))])
                                classes[clsRep] += 1
                                classesIndexes[clsRep] = tId
                                hits+=1
                            else:
                                classes[-1]+=1
                                classesIndexes[-1]=-1
                                misses+=1
                        targetClass=-2
                        for cls,count in classes.items():
                            if count/len(ts)>0.8:
                                targetClass = cls
                                break

                        if type(targetClass) is str:
                            node_pred_use_index.append(i)
                            node_gt_use_class_indexes.append(classesIndexes[targetClass])
                        elif targetClass==-1 and self.final_class_bad_alignment:
                            node_pred_use_index_sp.append(i)
                            error_class = torch.FloatTensor(1,len(self.classMap)+self.num_node_error_class).zero_()
                            error_class[0,self.final_class_bad_alignment_index]=1
                            alignedClass_use_sp.append(error_class)
                        elif targetClass==-2 and self.final_class_inpure_group:
                            node_pred_use_index_sp.append(i)
                            error_class = torch.FloatTensor(1,len(self.classMap)+self.num_node_error_class).zero_()
                            error_class[0,self.final_class_inpure_group_index]=1
                            alignedClass_use_sp.append(error_class)

                        if hits==0:
                            node_conf_use_index.append(i)
                            node_conf_gt.append(0)
                        elif misses==0 or hits/misses>0.5:
                            node_conf_use_index.append(i)
                            node_conf_gt.append(1)
                        #if 0 in predGroup:
                            #import pdb;pdb.set_trace()

                    node_pred_use_index += node_pred_use_index_sp

                    if len(node_pred_use_index)>0:
                        nodePredClass_use = nodePred[node_pred_use_index][:,:,self.model_ref.nodeIdxClass:self.model_ref.nodeIdxClassEnd]
                        alignedClass_use = targetBoxes[0][node_gt_use_class_indexes,13:13+len(self.classMap)]
                        if self.num_node_error_class>0:
                            alignedClass_use = torch.cat((alignedClass_use,torch.FloatTensor(alignedClass_use.size(0),self.num_bb_error_class).zero_().to(alignedClass_use.device)),dim=1)
                            if len(alignedClass_use_sp)>0:
                                alignedClass_use_sp = torch.cat(alignedClass_use_sp,dim=0).to(alignedClass_use.device)
                                alignedClass_use = torch.cat((alignedClass_use,alignedClass_use_sp),dim=0)
                    else:
                        nodePredClass_use = None
                        alignedClass_use = None

                    if len(node_conf_use_index)>0:
                        nodePredConf_use = nodePred[node_conf_use_index][:,:,self.model_ref.nodeIdxConf]
                        nodeGTConf_use = torch.FloatTensor(node_conf_gt).to(nodePred.device)
                else:
                    nodePredClass_use = None
                    alignedClass_use = None
                    nodePredConf_use = None
                    nodeGTConf_use = None

                ####


                #if edgePred is not None:
                #    numEdgePred = edgePred.size(0)
                #    if predEdgeShouldBeTrue is not None:
                #        lenTrue = predEdgeShouldBeTrue.size(0)
                #    else:
                #        lenTrue = 0
                #    if predEdgeShouldBeFalse is not None:
                #        lenFalse = predEdgeShouldBeFalse.size(0)
                #    else:
                #        lenFalse = 0
                #else:
                #    numEdgePred = lenTrue = lenFalse = 0

                relLoss = None
                #separating the loss into true and false portions is not only convienint, it balances the loss between true/false examples
                if predEdgeShouldBeTrue is not None and predEdgeShouldBeTrue.size(0)>0 and predEdgeShouldBeTrue.size(1)>0:
                    ones = torch.ones_like(predEdgeShouldBeTrue).to(image.device)
                    relLoss = self.loss['rel'](predEdgeShouldBeTrue,ones)
                    assert(not torch.isnan(relLoss))
                    debug_avg_relTrue = predEdgeShouldBeTrue.mean().item()
                else:
                    debug_avg_relTrue =0 
                if predEdgeShouldBeFalse is not None and predEdgeShouldBeFalse.size(0)>0 and predEdgeShouldBeFalse.size(1)>0:
                    zeros = torch.zeros_like(predEdgeShouldBeFalse).to(image.device)
                    relLossFalse = self.loss['rel'](predEdgeShouldBeFalse,zeros)
                    assert(not torch.isnan(relLossFalse))
                    if relLoss is None:
                        relLoss=relLossFalse
                    else:
                        relLoss+=relLossFalse
                    debug_avg_relFalse = predEdgeShouldBeFalse.mean().item()
                else:
                    debug_avg_relFalse = 0
                if relLoss is not None:
                    #relLoss *= self.lossWeights['rel']
                    losses['relLoss']+=relLoss

                if proposedInfoI is not None:
                    propPredPairingShouldBeTrue,propPredPairingShouldBeFalse= proposedInfoI[0:2]
                    propRelLoss = None
                    #seperating the loss into true and false portions is not only convienint, it balances the loss between true/false examples
                    if propPredPairingShouldBeTrue is not None and propPredPairingShouldBeTrue.size(0)>0:
                        ones = torch.ones_like(propPredPairingShouldBeTrue).to(image.device)
                        propRelLoss = self.loss['propRel'](propPredPairingShouldBeTrue,ones)
                    if propPredPairingShouldBeFalse is not None and propPredPairingShouldBeFalse.size(0)>0:
                        zeros = torch.zeros_like(propPredPairingShouldBeFalse).to(image.device)
                        propRelLossFalse = self.loss['propRel'](propPredPairingShouldBeFalse,zeros)
                        if propRelLoss is None:
                            propRelLoss=propRelLossFalse
                        else:
                            propRelLoss+=propRelLossFalse
                    if propRelLoss is not None:
                        losses['propRelLoss']+=propRelLoss



                


                    #t#self.opt_history['box_loss'].append(timeit.default_timer()-tic2)#t#

                    #boxLoss *= self.lossWeights['box']
                    #if relLoss is not None:
                    #    loss = relLoss + boxLoss
                    #else:
                    #    loss = boxLoss
                #else:
                #    loss = relLoss



                if self.model_ref.predClass and nodePredClass_use is not None and nodePredClass_use.size(0)>0:
                    alignedClass_use = alignedClass_use[:,None] #introduce "time" dimension to broadcast
                    class_loss_final = self.loss['classFinal'](nodePredClass_use,alignedClass_use)
                    losses['classFinalLoss'] += class_loss_final

                if nodePredConf_use is not None and nodePredConf_use.size(0)>0:
                    if len(nodeGTConf_use.size())<len(nodePredConf_use.size()):
                        nodeGTConf_use = nodeGTConf_use[:,None] #introduce "time" dimension to broadcast
                    conf_loss_final = self.loss['classFinal'](nodePredConf_use,nodeGTConf_use)
                    losses['confFinalLoss'] += conf_loss_final
                    #class_loss_final *= self.lossWeights['class']
                    #loss += class_loss_final
                    #class_loss_final = class_loss_final.item()
                #else:
                    #class_loss_final = 0

                for name,stat in logIter.items():
                    log['{}_{}'.format(name,graphIteration)]=stat

                if self.save_images_every>0 and self.iteration%self.save_images_every==0:
                    path = os.path.join(self.save_images_dir,'{}_{}.png'.format('b',graphIteration))#instance['name'],graphIteration))
                    
                    draw_graph(
                            outputBoxes,
                            self.model_ref.used_threshConf,
                            torch.sigmoid(nodePred).cpu().detach() if nodePred is not None else None,
                            torch.sigmoid(edgePred).cpu().detach() if edgePred is not None else None,
                            edgeIndexes,
                            predGroups,
                            image,
                            edgePredTypes,
                            missedRels,
                            None,
                            targetBoxes,
                            self.classMap,
                            path,
                            useTextLines=self.model_ref.useCurvedBBs,
                            targetGroups=instance['gt_groups'],
                            targetPairs=instance['gt_groups_adj'])
                    print('saved {}'.format(path))

                if 'bb_stats' in get:

                    #if self.model_ref.detector.predNumNeighbors:
                    #    beforeCls=1
                    #    #outputBoxesM=torch.cat((outputBoxes[:,0:6],outputBoxes[:,7:]),dim=1) #throw away NN pred
                    #else:
                    #    beforeCls=0
                    #if targetBoxes is not None:
                    #    targetBoxes = targetBoxes.cpu()
                    if targetBoxes is not None:
                        target_for_b = targetBoxes[0].cpu()
                    else:
                        target_for_b = torch.empty(0)
                    if self.model_ref.useCurvedBBs:
                        ap_5, prec_5, recall_5, allPrec, allRecall =AP_textLines(target_for_b,outputBoxes,0.5,numClasses)
                    elif self.model_ref.rotation:
                        ap_5, prec_5, recall_5 =AP_dist(target_for_b,outputBoxes,0.9,numClasses)
                    else:
                        ap_5, prec_5, recall_5, allPrec, allRecall =AP_iou(target_for_b,outputBoxes,0.5,numClasses)
                    prec_5 = np.array(prec_5)
                    recall_5 = np.array(recall_5)
                    log['bb_AP_{}'.format(graphIteration)]=ap_5
                    log['bb_prec_{}'.format(graphIteration)]=prec_5
                    log['bb_recall_{}'.format(graphIteration)]=recall_5
                    log['bb_allPrec_{}'.format(graphIteration)]=allPrec
                    log['bb_allRecall_{}'.format(graphIteration)]=allRecall
                    log['bb_allFm_{}'.format(graphIteration)]= 2*allPrec*allRecall/(allPrec+allRecall) if allPrec+allRecall>0 else 0

        #Fine tuning detector. Should only happed once
        if not self.model_ref.detector_frozen:
            if targetBoxes is not None:
                targSize = targetBoxes.size(1)
            else:
                targSize =0 

            tic2=timeit.default_timer()
            if 'box' in self.loss:
                boxLoss, position_loss, conf_loss, class_loss, nn_loss, recall, precision = self.loss['box'](outputOffsets,targetBoxes,[targSize],target_num_neighbors)
                losses['boxLoss'] += boxLoss
                logIter['bb_position_loss'] = position_loss
                logIter['bb_conf_loss'] = conf_loss
                logIter['bb_class_loss'] = class_loss
                logIter['bb_nn_loss'] = nn_loss
            else:
                oversegLoss, position_loss, conf_loss, class_loss, rot_loss, recall, precision, gt_covered, pred_covered, recall_noclass, precision_noclass, gt_covered_noclass, pred_covered_noclass = self.loss['overseg'](outputOffsets,targetBoxes,[targSize],calc_stats='bb_stats' in get)
                losses['oversegLoss'] = oversegLoss
                logIter['bb_position_loss'] = position_loss
                logIter['bb_conf_loss'] = conf_loss
                logIter['bb_class_loss'] = class_loss
                if 'bb_stats' in get:
                    logIter['bb_recall_noclass']=recall_noclass
                    logIter['bb_precision_noclass']=precision_noclass
                    logIter['bb_gt_covered_noclass']=gt_covered_noclass
                    logIter['bb_pred_covered_noclass']=pred_covered_noclass

        #We'll use information from the final prediction before the final pruning
        if 'DocStruct' in get:
            predToGTGroup={}
            #gtGroupToPred={}
            for node in range(len(predGroups)):
                predTargGroup = [bbAlignment[bb] for bb in predGroups[node] if bbAlignment[bb]>=0]
                if len(predTargGroup)>0:
                    gtGroup = getGTGroup(predTargGroup,targetIndexToGroup)
                    predToGTGroup[node]=gtGroup
                    #assert gtGroup not in gtGroupToPred #shouldn't happend with gt detections and grouping
                    #if it has split a group over two nodes, we'll just use both of them, as this is supposed to query from gt
                    #if useGTGroups and gtGroup in gtGroupToPred:
                    #    print('WARNING WARNING')
                    #    print('> pred groups NOT aligned to GT groups')
                    #    predTargGroupOther = [bbAlignment[bb] for bb in predGroups[gtGroupToPred[gtGroup]] if bbAlignment[bb]>=0]
                    #    print('> gt group:{}, me:{}, other:{}'.format(gtGroups[gtGroup],predTargGroup,predTargGroupOther))
                    #    print('WARNING WARNING')
                    #if gtGroup not in gtGroupToPred:
                    #    gtGroupToPred[gtGroup]=node
                    #else:
                    #    #this is missed
                    #    del predToGTGroup[gtGroupToPred[gtGroup]]
                    #    del predToGTGroup[node]
                    #    del gtGroupToPred[gtGroup]
            
            classMap = self.scoreClassMap
            num_class = len(self.scoreClassMap)
            minI = min(classMap.values())
            classMap = {v-minI:k for k,v in classMap.items()}
            classIs = nodePred[:,-1,1:1+num_class].argmax(dim=1).tolist()
            gt_classIs = targetBoxes[0,:,13:13+num_class].argmax(dim=1).tolist()
            #unused_gt_adj = set(gtGroupAdj)
            #candidate_lists=defaultdict(list)
            #for ei,(n0,n1) in enumerate(edgeIndexes):
            #    if n0 not in predToGTGroup or n1 not in predToGTGroup:
            #        continue
            #    gtG0 = predToGTGroup[n0]
            #    gtG1 = predToGTGroup[n1]
            #    class0 = classMap[classIs[n0]]
            #    class1 = classMap[classIs[n1]]
            #    true_pos = (min(gtG0,gtG1),max(gtG0,gtG1)) in gtGroupAdj
            #    if true_pos:
            #        try:
            #            unused_gt_adj.remove((min(gtG0,gtG1),max(gtG0,gtG1)))
            #        except KeyError as e:
            #            if useGTGroups:
            #                raise e
            #    
            #    if (class0=='header' and class1=='question') or (class0=='question' and class1=='answer'):
            #        candidate_lists[gtG1].append((edgePred[ei,-1,1],true_pos))
            #    elif (class1=='header' and class0=='question') or (class1=='question' and class0=='answer'):
            #        candidate_lists[gtG0].append((edgePred[ei,-1,1],true_pos))
            #    else:
            #        candidate_lists[gtG0].append((edgePred[ei,-1,1],False))
            #        candidate_lists[gtG1].append((edgePred[ei,-1,1],False))
            #for gtG0,gtG1 in unused_gt_adj:
            #    gtBB0=gtGroups[gtG0][0]
            #    gtBB1=gtGroups[gtG1][0]
            #    class0=classMap[gt_classIs[gtBB0]]
            #    class1=classMap[gt_classIs[gtBB1]]
            #    if (class0=='header' and class1=='question') or (class0=='question' and class1=='answer'):
            #        candidate_lists[gtG1].append((-1,False))
            #    elif (class1=='header' and class0=='question') or (class1=='question' and class0=='answer'):
            #        candidate_lists[gtG0].append((-1,False))
            #    #elif class0=='question'  and class1=='question':
            #    #    #IDK
            #    #    candidate_lists[gtG0].append((-1,False))
            #    else:
            #        #there are some lableling annomalies in the FUNSD test set
            #        candidate_lists[gtG0].append((-1,False))
            #        #assert False

            #sum_ap=0
            #hit_at_1=0
            #hit_at_2=0
            #hit_at_5=0

            #for gtG, candidate_list in candidate_lists.items():
            #    #if len(candidate_list)>1 or candidate_list[0][0]>=0:
            #    #    sum_ap += computeAP(candidate_list)
            #    #else:
            #    #    sum_ap += 0 #total miss

            #    candidate_list.sort(key=lambda a:a[0],reverse=True)
            #    if candidate_list[0][1]:
            #        hit_at_1 +=1
            #    if any(a[1] for a in candidate_list[:2]):
            #        hit_at_2 +=1
            #    if any(a[1] for a in candidate_list[:5]):
            #        hit_at_5 +=1
            ##log['DocStruct mAP'] = sum_ap/len(candidate_lists) #not quite right. Need to add every other possible rel. A
            #if len(candidate_lists)>0:
            #    log['DocStruct hit@1'] = hit_at_1/len(candidate_lists)
            #    log['DocStruct hit@2'] = hit_at_2/len(candidate_lists)
            #    log['DocStruct hit@5'] = hit_at_5/len(candidate_lists)


            sum_hit=0
            gtGroupToPred=defaultdict(list) #list as we could have a gt group split between two pred groups
            for pG,gtG in predToGTGroup.items():
                gtGroupToPred[gtG].append(pG)
            for gg0,gg1 in gtGroupAdj:
                gtBB0=gtGroups[gg0][0]
                gtBB1=gtGroups[gg1][0]
                class0=classMap[gt_classIs[gtBB0]]
                class1=classMap[gt_classIs[gtBB1]]
                if (class0=='header' and class1=='question') or (class0=='question' and class1=='answer'):
                    parent=gg0
                    child=gg1
                    parent_groups = gtGroupToPred[parent]
                    child_groups = gtGroupToPred[child]

                    if maxRelScoreIsHit(child_groups,parent_groups,edgeIndexes,edgePred):
                        sum_hit+=1
                elif (class1=='header' and class0=='question') or (class1=='question' and class0=='answer'):
                    parent=gg1
                    child=gg0
                    parent_groups = gtGroupToPred[parent]
                    child_groups = gtGroupToPred[child]

                    if maxRelScoreIsHit(child_groups,parent_groups,edgeIndexes,edgePred):
                        sum_hit+=1
                else: #labeling annomally, check both
                    parent_groups = gtGroupToPred[gg0]
                    child_groups = gtGroupToPred[gg1]
                    if maxRelScoreIsHit(child_groups,parent_groups,edgeIndexes,edgePred):
                        sum_hit+=0.5
                    parent_groups = gtGroupToPred[gg1]
                    child_groups = gtGroupToPred[gg0]
                    if maxRelScoreIsHit(child_groups,parent_groups,edgeIndexes,edgePred):
                        sum_hit+=0.5
            if len(gtGroupAdj)>0:
                log['DocStruct redid hit@1'] = sum_hit/len(gtGroupAdj)

            
            

            #TODO missed rels (nodes)

        #t#self.opt_history['all losses'].append(timeit.default_timer()-tic)#t#
        #t#tic=timeit.default_timer()#t##t#
        #print final state of graph
        ###
        gt_groups_adj = instance['gt_groups_adj']
        if final is not None:
            if self.remove_same_pairs:
                final = self.removeSamePairs(final)
            finalLog, finalRelTypes, finalMissedRels, finalMissedGroups = self.finalEval(targetBoxes.cpu() if targetBoxes is not None else None,gtGroups,gt_groups_adj,targetIndexToGroup,*final,bb_iou_thresh=self.final_bb_iou_thresh)
            log.update(finalLog)
            if self.save_images_every>0 and self.iteration%self.save_images_every==0:
                path = os.path.join(self.save_images_dir,'{}_{}.png'.format('b','final'))#instance['name'],graphIteration))
                finalOutputBoxes, finalPredGroups, finalEdgeIndexes, finalBBTrans = final
                draw_graph(
                        finalOutputBoxes,
                        self.model_ref.used_threshConf,
                        None,
                        None,
                        finalEdgeIndexes,
                        finalPredGroups,
                        image,
                        finalRelTypes,
                        finalMissedRels,
                        finalMissedGroups,
                        targetBoxes,
                        self.classMap,
                        path,
                        bbTrans=finalBBTrans,
                        useTextLines=self.model_ref.useCurvedBBs,
                        targetGroups=instance['gt_groups'],
                        targetPairs=instance['gt_groups_adj'])
                #print('saved {}'.format(path))
            finalOutputBoxes, finalPredGroups, finalEdgeIndexes, finalBBTrans = final
            if self.do_characterization:
                self.characterization_eval(
                            allOutputBoxes,
                            allEdgePred,
                            allNodePred,
                            allEdgeIndexes,
                            allPredGroups,
                            finalOutputBoxes,
                            finalEdgeIndexes,
                            finalPredGroups,
                            targetBoxes,
                            targetIndexToGroup,
                            gtGroups,
                            gtGroupAdj)
            #print('DEBUG final num node:{}, num edges: {}'.format(len(finalOutputBoxes) if finalOutputBoxes is not None else 0,len(finalEdgeIndexes) if finalEdgeIndexes is not None else 0))
        ###
        
        #log['rel_prec']= fullPrec
        #log['rel_recall']= eRecall
        #log['rel_Fm']= 2*(fullPrec*eRecall)/(eRecall+fullPrec) if eRecall+fullPrec>0 else 0
        #t#self.opt_history['final eval'].append(timeit.default_timer()-tic)#t#




        if proposedInfo is not None:
            propRecall,propPrec = proposedInfo[2:4]
            log['prop_rel_recall'] = propRecall
            log['prop_rel_prec'] = propPrec
        if mergeProposedInfo is not None:
            propRecall,propPrec = mergeProposedInfo[2:4]
            log['prop_merge_recall'] = propRecall
            log['prop_merge_prec'] = propPrec
        #if final_prop_rel_recall is not None:
        #    log['final_prop_rel_recall']=final_prop_rel_recall
        #if final_prop_rel_prec is not None:
        #    log['final_prop_rel_prec']=final_prop_rel_prec





        got={}#outputBoxes, outputOffsets, relPred, relIndexes, bbPred, rel_prop_pred
        for name in get:
            if name=='edgePred':
                got['edgePred'] = edgePred.detach().cpu()
            elif name=='outputBoxes':
                if useGT:
                    got['outputBoxes'] = targetBoxes.cpu()
                else:
                    got['outputBoxes'] = outputBoxes.detach().cpu()
            elif name=='outputOffsets':
                got['outputOffsets'] = outputOffsets.detach().cpu()
            elif name=='edgeIndexes':
                got['edgeIndexes'] = edgeIndexes
            elif name=='nodePred':
                 got['nodePred'] = nodePred.detach().cpu()
            elif name=='allNodePred':
                 got[name] = [n.detach().cpu() if n is not None else None for n in allNodePred] if allNodePred is not None else None
            elif name=='allEdgePred':
                 got[name] = [n.detach().cpu() if n is not None else None for n in allEdgePred] if allEdgePred is not None else None
            elif name=='allEdgeIndexes':
                 got[name] = allEdgeIndexes
            elif name=='allPredGroups':
                 got[name] = allPredGroups
            elif name=='allOutputBoxes':
                 got[name] = allOutputBoxes
            elif name=='allEdgePredTypes':
                 got[name] = allEdgePredTypes
            elif name=='allMissedRels':
                 got[name] = allMissedRels
            elif name=='allBBAlignment':
                 got[name] = allBBAlignment
            elif name=='final':
                 got[name] = final
            elif name=='final_edgePredTypes':
                 got[name] = finalRelTypes
            elif name=='final_missedRels':
                 got[name] = finalMissedRels
                 got['final_missedGroups'] = finalMissedGroups
            elif name=='DocStruct':
                if 'DocStruct redid hit@1' in log:
                    got[name]=log['DocStruct redid hit@1']
            elif name != 'bb_stats' and name != 'nn_acc':
                raise NotImplementedError('Cannot get [{}], unknown'.format(name))
        return losses, log, got

    def removeSamePairs(self,final):
        outputBoxes,predGroups,predPairs,predTrans = final
        new_pairs = []
        for g0,g1 in predPairs:
            assert len(predGroups[g0])==1
            assert len(predGroups[g1])==1
            #assert 'blank' not in self.classMap

            num_classes = len(self.scoreClassMap)
            class0 = outputBoxes[predGroups[g0][0],6:6+num_classes].argmax().item()
            class1 = outputBoxes[predGroups[g1][0],6:6+num_classes].argmax().item()

            if class0!=class1:
                new_pairs.append((g0,g1))
        return outputBoxes,predGroups,new_pairs,predTrans
            

    def finalEval(self,targetBoxes,gtGroups,gt_groups_adj,targetIndexToGroup,outputBoxes,predGroups,predPairs,predTrans=None, bb_iou_thresh=0.5):
        log={}
        numClasses = len(self.scoreClassMap)

        #Remove blanks
        if 'blank' in self.classMap:
            blank_index = self.classMap['blank']
            if targetBoxes is not None:
                gtNotBlanks = targetBoxes[0,:,blank_index]<0.5
                if not gtNotBlanks.all():
                    #rewrite all the GT to not include blanks
                    targetBoxes=targetBoxes[:,gtNotBlanks]
                    gtOldToNewBBs = {oi.item():ni for ni,oi in enumerate(torch.nonzero(gtNotBlanks)[:,0])}
                    newGTGroups=[[gtOldToNewBBs[bb] for bb in group if bb in gtOldToNewBBs] for group in gtGroups]
                    gtOldToNewGroups={}
                    gtGroups=[]
                    for i,group in enumerate(newGTGroups):
                        if len(group)>0:
                            gtOldToNewGroups[i]=len(gtGroups)
                            gtGroups.append(group)
                    newToOldGTGroups = {n:o for o,n in gtOldToNewGroups.items()}
                    gt_groups_adj = set((gtOldToNewGroups[g1],gtOldToNewGroups[g2]) for g1,g2 in gt_groups_adj if g1 in gtOldToNewGroups and g2 in gtOldToNewGroups)
                    targetIndexToGroup={}
                    for groupId,bbIds in enumerate(gtGroups):
                        targetIndexToGroup.update({bbId:groupId for bbId in bbIds})
                else:
                    newToOldGTGroups = list(range(len(gtGroups)))

            if outputBoxes is not None and len(outputBoxes)>0:
                if self.model_ref.useCurvedBBs:
                    outputBoxesNotBlanks=torch.FloatTensor([box.getCls() for box in outputBoxes])
                    outputBoxesNotBlanks=outputBoxesNotBlanks[:,blank_index-13]<0.5
                    outputBoxes = [box for i,box in enumerate(outputBoxes) if outputBoxesNotBlanks[i]]
                else:
                    outputBoxesNotBlanks=outputBoxes[:,1+blank_index-8]<0.5
                    outputBoxes = outputBoxes[outputBoxesNotBlanks]
                newToOldOutputBoxes = torch.arange(0,len(outputBoxesNotBlanks),dtype=torch.int64)[outputBoxesNotBlanks]
                oldToNewOutputBoxes = {o.item():n for n,o in enumerate(newToOldOutputBoxes)}
                if predGroups is not None:
                    predGroups = [[oldToNewOutputBoxes[bId] for bId in group if bId in oldToNewOutputBoxes] for group in predGroups]
                    newToOldGroups = []
                    newGroups = []
                    for gId,group in enumerate(predGroups):
                        if len(group)>0:
                            newGroups.append(group)
                            newToOldGroups.append(gId)
                    predGroups = newGroups
                    oldToNewGroups = {o:n for n,o in enumerate(newToOldGroups)}
                    num_pred_pairs_with_blanks = len(predPairs)
                    predPairs = [(i,(oldToNewGroups[g1],oldToNewGroups[g2])) for i,(g1,g2) in enumerate(predPairs) if g1 in oldToNewGroups and g2 in oldToNewGroups]
                    if len(predPairs)>0:
                        newToOldPredPairs,predPairs = zip(*predPairs)
                    else:
                        newToOldPredPairs=[]
                    for a,b in predPairs:
                        assert(a < len(predGroups))
                        assert(b < len(predGroups))
                if predTrans is not None:
                    predTrans = [predTrans[newToOldOutputBoxes[n]] for n in range(len(newToOldOutputBoxes))]

        if targetBoxes is not None:
            targetBoxes = targetBoxes.cpu()
            if self.model_ref.useCurvedBBs:
                targIndex = newGetTargIndexForPreds_textLines(targetBoxes[0],outputBoxes,bb_iou_thresh,numClasses,False)
            elif self.model_ref.rotation:
                raise NotImplementedError('newGetTargIndexForPreds_dist should be modified to reflect the behavoir or newGetTargIndexForPreds_textLines')
                targIndex, fullHit, overSegmented = newGetTargIndexForPreds_dist(targetBoxes[0],outputBoxes,1.1,numClasses,hard_thresh=False)
            else:
                targIndex = newGetTargIndexForPreds_iou(targetBoxes[0],outputBoxes,bb_iou_thresh,numClasses,False)
                
        elif outputBoxes is not None:
            targIndex=torch.LongTensor(len(outputBoxes)).fill_(-1)

        #if self.model_ref.detector.predNumNeighbors:
        #    beforeCls=1
        #    #outputBoxesM=torch.cat((outputBoxes[:,0:6],outputBoxes[:,7:]),dim=1) #throw away NN pred
        #else:
        #    beforeCls=0
        #if targetBoxes is not None:
        #    targetBoxes = targetBoxes.cpu()
        if targetBoxes is not None:
            target_for_b = targetBoxes[0].cpu()
        else:
            target_for_b = torch.empty(0)
        if self.model_ref.useCurvedBBs:
            ap_5, prec_5, recall_5, allPrec, allRecall =AP_textLines(target_for_b,outputBoxes,bb_iou_thresh,numClasses)
        elif self.model_ref.rotation:
            ap_5, prec_5, recall_5, allPrec, allRecall =AP_dist(target_for_b,outputBoxes,0.9,numClasses)
        else:
            ap_5, prec_5, recall_5, allPrec, allRecall =AP_iou(target_for_b,outputBoxes,bb_iou_thresh,numClasses)
        prec_5 = np.array(prec_5)
        recall_5 = np.array(recall_5)
        log['final_bb_AP']=ap_5
        log['final_bb_prec']=prec_5
        log['final_bb_recall']=recall_5
        log['final_bb_allPrec']=allPrec
        log['final_bb_allRecall']=allRecall
        log['final_bb_allFm']= 2*allPrec*allRecall/(allPrec+allRecall) if allPrec+allRecall>0 else 0


        predGroupsT={}
        if predGroups is not None and targIndex is not None:
            for node in range(len(predGroups)):
                predGroupsT[node] = [targIndex[bb].item() for bb in predGroups[node] if targIndex[bb].item()>=0]
        elif  predGroups is not None:
            for node in range(len(predGroups)):
                predGroupsT[node] = []


        entity_detection_TP=0
        BROS_head_to_group = {group[0]:i for i,group in enumerate(gtGroups)}

        
        gtGroupHit=[False]*len(gtGroups)
        gtGroupHit_pure=[False]*len(gtGroups)
        groupCompleteness={}
        groupPurity={}
        predToGTGroup={}
        predToGTGroup_BROS={}
        offId = -1
        for node,predGroupT in predGroupsT.items():
            gtGroupId = getGTGroup(predGroupT,targetIndexToGroup)
            predToGTGroup[node]=gtGroupId
            if gtGroupId<0:
                purity=0
                gtGroupId=offId
                offId-=1
            else:
                if gtGroupHit[gtGroupId]:
                    purity=sum([tId in gtGroups[gtGroupId] for tId in predGroupT])
                    purity/=len(predGroups[node])
                    if purity<groupPurity[gtGroupId]:
                        gtGroupId=offId
                        offId-=1
                        purity=0
                    else:
                        groupPurity[offId]=0
                        offId-=1
                else:
                    purity=sum([tId in gtGroups[gtGroupId] for tId in predGroupT])
                    purity/=len(predGroups[node])
                    gtGroupHit[gtGroupId]=True
            groupPurity[gtGroupId]=purity

            if gtGroupId>=0:
                completeness=sum([gtId in predGroupT for gtId in gtGroups[gtGroupId]])
                completeness/=len(gtGroups[gtGroupId])
                groupCompleteness[node]=completeness
                
                if completeness==1 and purity==1:
                    entity_detection_TP+=1
                    gtGroupHit_pure[gtGroupId]=True
            
            #FOR BROS EVAL
            hit=False
            for predT in predGroupT:
                if predT in BROS_head_to_group:
                    if node not in predToGTGroup_BROS:
                        predToGTGroup_BROS[node] = BROS_head_to_group[predT]
                        hit=True
                    else:
                        predToGTGroup_BROS[node]=-1
                        hit=False
                        break
            if not hit:
                predToGTGroup_BROS[node]=-1

        log['final_group_XX_TP']=entity_detection_TP
        log['final_group_XX_gtCount']=len(gtGroups)
        log['final_group_XX_predCount']=len(predGroupsT)

        if len(gtGroups)>0:
            log['final_group_ED_recall']=entity_detection_TP/len(gtGroups)
        else:
            log['final_group_ED_recall']=1
        if len(predGroupsT)>0:
            log['final_group_ED_precision']=entity_detection_TP/len(predGroupsT)
        else:
            log['final_group_ED_precision']=1
        if log['final_group_ED_recall']+log['final_group_ED_precision']>0:
            log['final_group_ED_F1'] = 2*log['final_group_ED_precision']*log['final_group_ED_recall']/(log['final_group_ED_recall']+log['final_group_ED_precision'])
        else:
            log['final_group_ED_F1'] = 0

        groupCompleteness_list = list(groupCompleteness.values())
        for hit in gtGroupHit:
            if not hit:
                groupCompleteness_list.append(0)
        

        log['final_groupCompleteness']=np.mean(groupCompleteness_list)
        log['final_groupPurity']=np.mean([v for k,v in groupPurity.items()])



        gtRelHit=set()
        gtRelHit_BROS=set()
        gtRelHit_strict=set()
        relPrec=0
        relPrec_BROS=0
        relPrec_strict=0
        if predPairs is None:
            predPairs=[]
        if 'blank' in self.classMap and predGroups is not None:
            rel_types=['UP']*num_pred_pairs_with_blanks
        else:
            rel_types=[]
        for pi,(n0,n1) in enumerate(predPairs):
            BROS_gtG0 = predToGTGroup_BROS[n0]
            BROS_gtG1 = predToGTGroup_BROS[n1]
            hit=False
            if BROS_gtG0>=0 and BROS_gtG1>=0:
                pair_id = (min(BROS_gtG0,BROS_gtG1),max(BROS_gtG0,BROS_gtG1))
                if pair_id in gt_groups_adj:
                    hit=True
                    relPrec_BROS+=1
                    gtRelHit_BROS.add((min(BROS_gtG0,BROS_gtG1),max(BROS_gtG0,BROS_gtG1)))
                    if 'blank' in self.classMap:
                        old_pi = newToOldPredPairs[pi]
                        rel_types[old_pi] = 'TP'
                    else:
                        rel_types.append('TP')
            if not hit:
                if 'blank' in self.classMap:
                    old_pi = newToOldPredPairs[pi]
                    rel_types[old_pi] = 'FP'
                else:
                    rel_types.append('FP')
            if n0 not in predToGTGroup or n1 not in predToGTGroup:
                print('ERROR, pair ({},{}) not foundi n predToGTGroup'.format(n0,n1))
                print('predToGTGroup {}: {}'.format(len(predToGTGroup),predToGTGroup))
                print('predGroups {}: {}'.format(len(predGroups),predGroups))
                print('outputBoxesNotBlanks: {}'.format(outputBoxesNotBlanks))
            gtG0=predToGTGroup[n0]
            gtG1=predToGTGroup[n1]
            if gtG0>=0 and gtG1>=0:
                pair_id = (min(gtG0,gtG1),max(gtG0,gtG1))
                if pair_id in gt_groups_adj:
                    relPrec+=1
                    gtRelHit.add((min(gtG0,gtG1),max(gtG0,gtG1)))
                    if groupPurity[gtG0]==1 and groupPurity[gtG1]==1 and n0 in groupCompleteness and groupCompleteness[n0]==1 and n1 in groupCompleteness and groupCompleteness[n1]==1:
                        relPrec_strict+=1
                        gtRelHit_strict.add((min(gtG0,gtG1),max(gtG0,gtG1)))
                        #TODO failed in training
                        #assert BROS_gtG0==gtG0
                        #assert BROS_gtG1==gtG1
                        #assert (min(gtG0,gtG1),max(gtG0,gtG1)) in gtRelHit_BROS
                    #if 'blank' in self.classMap:
                    #    old_pi = newToOldPredPairs[pi]
                    #    rel_types[old_pi] = 'TP'
                    #else:
                    #    rel_types.append('TP')
                    continue
            #if 'blank' in self.classMap:
            #    old_pi = newToOldPredPairs[pi]
            #    rel_types[old_pi] = 'FP'
            #else:
            #    rel_types.append('FP')

        #print('DEBUG true positives={}'.format(len(gtRelHit)))
        #print('DEBUG false positives={}'.format(len(predPairs)-len(gtRelHit)))
        #print('DEBUG false negatives={}'.format(len(gt_groups_adj)-len(gtRelHit)))
        #log['final_rel_TP']=relPrec

        #TODO, these failed in training
        #assert relPrec_strict==len(gtRelHit_strict)
        #assert relPrec_BROS==len(gtRelHit_BROS)
        log['final_rel_XX_strict_TP']=relPrec_strict
        log['final_rel_XX_BROS_TP']=relPrec_BROS
        log['final_rel_XX_predCount']=len(predPairs)
        log['final_rel_XX_gtCount']=len(gt_groups_adj)
        if len(predPairs)>0:
            relPrec /= len(predPairs)
            relPrec_strict /= len(predPairs)
            relPrec_BROS /= len(predPairs)
        else:
            relPrec = 1
            relPrec_strict = 1
            relPrec_BROS = 1
        if len(gt_groups_adj)>0:
            relRecall = len(gtRelHit)/len(gt_groups_adj)
            relRecall_strict = len(gtRelHit_strict)/len(gt_groups_adj)
            relRecall_BROS = len(gtRelHit_BROS)/len(gt_groups_adj)
        else:
            relRecall = 1
            relRecall_strict = 1
            relRecall_BROS = 1


        log['final_rel_prec']=relPrec
        log['final_rel_recall']=relRecall
        log['final_rel_strict_prec']=relPrec_strict
        log['final_rel_strict_recall']=relRecall_strict
        log['final_rel_BROS_prec']=relPrec_BROS
        log['final_rel_BROS_recall']=relRecall_BROS
        if relPrec+relRecall>0:
            log['final_rel_Fm']=(2*(relPrec*relRecall)/(relPrec+relRecall))
        else:
            log['final_rel_Fm']=0
        if relPrec_strict+relRecall_strict>0:
            log['final_rel_strict_Fm']=(2*(relPrec_strict*relRecall_strict)/(relPrec_strict+relRecall_strict))
        else:
            log['final_rel_strict_Fm']=0
        if relPrec_BROS+relRecall_BROS>0:
            log['final_rel_BROS_Fm']=(2*(relPrec_BROS*relRecall_BROS)/(relPrec_BROS+relRecall_BROS))
        else:
            log['final_rel_BROS_Fm']=0


        missed_rels = gt_groups_adj.difference(gtRelHit_BROS)
        if 'blank' in self.classMap:
            missed_rels = set((newToOldGTGroups[g1],newToOldGTGroups[g2]) for g1,g2 in missed_rels)

        missed_groups = [i for i in range(len(gtGroups)) if not gtGroupHit_pure[i]]
        if 'blank' in self.classMap:
            missed_groups = [newToOldGTGroups[gi] for gi in missed_groups]
        return log, [rel_types], missed_rels, missed_groups

    def characterization_eval(self,
                        allOutputBoxes,
                        allEdgePred,
                        allNodePred,
                        allEdgeIndexes,
                        allPredGroups,
                        finalOutputBoxes,
                        finalEdgeIndexes,
                        finalPredGroups,
                        targetBoxes,
                        targetIndexToGroup,
                        gtGroups,
                        gtGroupAdj,
                        ):

        numClassesFull = self.model_ref.numBBTypes
        numClasses = len(self.scoreClassMap)

        #Remove blanks
        if 'blank' in self.classMap:
            blank_index = self.classMap['blank']
            if targetBoxes is not None:
                gtNotBlanks = targetBoxes[0,:,blank_index]<0.5
                newToOldBBs = [i for i in range(targetBoxes.size(1)) if gtNotBlanks[i]]
                oldToNewBBs = {o:n for n,o in newToOldBBs}
                targetBoxes=targetBoxes[:,gtNotBlanks]
                gtGroups = [ [oldToNewBBs[bb] for bb in group] for group in gtGroups ]
                newGroups=[]
                for gId,group in enumerate(gtGroups):
                    if len(group)>0:
                        newGroups.append(group)
                        newToOldGroups.append(gId)
                gtGroups=newGroups
                oldToNewGroups = {o:n for n,o in enumerate(newToOldGroups)}
                gtGroupAdj = [(oldToNewGroups[g1],oldToNewGroups[g2]) for g1,g2 in gtGroupAdj if g1 in oldToNewGroups and g2 in oldToNewGroups]
            if finalOutputBoxes is not None and len(finalOutputBoxes)>0:
                if self.model_ref.useCurvedBBs:
                    finalOutputBoxesNotBlanks=torch.FloatTensor([box.getCls() for box in finalOutputBoxes])
                    finalOutputBoxesNotBlanks=finalOutputBoxesNotBlanks[:,blank_index-13]<0.5
                    finalOutputBoxes = [box for i,box in enumerate(finalOutputBoxes) if finalOutputBoxesNotBlanks[i]]
                else:
                    finalOutputBoxesNotBlanks=finalOutputBoxes[:,1+blank_index-8]<0.5
                    finalOutputBoxes = finalOutputBoxes[finalOutputBoxesNotBlanks]
                newToOldOutputBoxes = torch.arange(0,len(finalOutputBoxesNotBlanks),dtype=torch.int64)[finalOutputBoxesNotBlanks]
                oldToNewOutputBoxes = {o.item():n for n,o in enumerate(newToOldOutputBoxes)}
                
                if finalPredGroups is not None:
                    finalPredGroups = [[oldToNewOutputBoxes[bId] for bId in group if bId in oldToNewOutputBoxes] for group in finalPredGroups]
                    newToOldGroups = []
                    newGroups = []
                    for gId,group in enumerate(finalPredGroups):
                        if len(group)>0:
                            newGroups.append(group)
                            newToOldGroups.append(gId)
                    oldToNewGroups = {o:n for n,o in enumerate(newToOldGroups)}
                    finalEdgeIndexes = [(oldToNewGroups[g1],oldToNewGroups[g2]) for g1,g2 in finalEdgeIndexes if g1 in oldToNewGroups and g2 in oldToNewGroups]
                    finalPredGroups = newGroups

                    for a,b in finalEdgeIndexes:
                        assert(a < len(finalPredGroups))
                        assert(b < len(finalPredGroups))
        targetBoxes=targetBoxes[0]

        #go to last
        #find alignment
        #if rel is missing, 
        false_pos_is_single=0
        false_pos_group_involved=0
        false_pos_inpure_group=0
        false_pos_from_bad_class=0
        false_pos_bad_node=0
        false_pos_with_good_nodes=0
        false_pos_with_misclassed_nodes=0
        inconsistent_edges=0
        false_pos_consistent_header_rels=0
        false_pos_consistent_question_rels=0

        double_rel_pred=0
        missed_rel_from_bad_detection=0
        missed_rel_from_bad_merge=0
        missed_rel_from_missed_prop=0
        missed_rel_from_bad_group=0
        missed_rel_from_poor_alignement=0
        missed_rel_from_misclass=0
        missed_rel_was_single=0
        missed_rel_from_pruned_edge=defaultdict(lambda: 0)
        missed_header_rels=0
        missed_question_rels=0
        missed_misc_rels=0
        hit_header_rels=0
        hit_question_rels=0
        hit_misc_rels=0

        false_pos_keep_scores=[]
        true_pos_keep_scores=[]
        false_pos_rel_scores=[]
        true_pos_rel_scores=[]

        #for each gt node, look through history. Was it found? Incorrectly merged? Grouped?
        #   found: a partail detection
        #   merge: was there are merge which made it not a match?
        #   was it grouped incorrectly? was it supposed to be a group, but isn't?
        #for missing rel, was it every present? when did it disapear, were any nodes bad?
        #for false pos rel, are either nodes bad detection or group?
        gtBBs2Pred=[]
        gtGroups2Pred=[]
        allEdgeScores=[]
        allRelScores=[]
        allMergeScores=[]
        allGroupScores=[]
        num_giter = len(allOutputBoxes)
        for graphIteration,(outputBoxes,edgePred,nodePred,edgeIndexes,predGroups) in enumerate(zip(allOutputBoxes,allEdgePred,allNodePred,allEdgeIndexes,allPredGroups)):
            if self.model_ref.useCurvedBBs:
                targIndex = newGetTargIndexForPreds_textLines(targetBoxes.cpu(),outputBoxes,self.gt_bb_align_IOcU_thresh,numClasses,True,self.picky_merging and (graphIteration>0 or not self.model.merge_first))
            elif self.model_ref.rotation:
                assert(False and 'untested and should be changed to reflect new newGetTargIndexForPreds_s')
                targIndex, fullHit, overSegmented = newGetTargIndexForPreds_dist(targetBoxes,outputBoxes,1.1,numClasses,hard_thresh=False)
            else:
                targIndex = newGetTargIndexForPreds_iou(targetBoxes,outputBoxes,0.5,numClasses,True)
            targIndex = targIndex.numpy()

            predGroup2GT={}
            for node in range(len(predGroups)):
                predGroupT = [targIndex[bb] for bb in predGroups[node] if targIndex[bb]>=0]
                predGroup2GT[node] = getGTGroup(predGroupT,targetIndexToGroup)
            #predGroups2GT.append(predGroup2GT)

            gtBB2Pred = [-1]*len(targetBoxes)
            for i,tInd in enumerate(targIndex):
                if tInd>=0:
                    gtBB2Pred[tInd]=i
            gtBBs2Pred.append(gtBB2Pred)
            gtGroup2Pred = defaultdict(list)
            for predG,gtG in predGroup2GT.items():
                if gtG!=-1:
                    gtGroup2Pred[gtG].append(predG)
            gtGroups2Pred.append(gtGroup2Pred)
            
            if graphIteration>0 or not self.model.merge_first:
                if not self.model_ref.legacy:
                    edgeScores = torch.sigmoid(edgePred[:,-1,0])
                    relScores = torch.sigmoid(edgePred[:,-1,1])
                    mergeScores = torch.sigmoid(edgePred[:,-1,2])
                    groupScores = torch.sigmoid(edgePred[:,-1,3])
                else:
                    relScores = torch.sigmoid(edgePred[:,-1,0])
                    mergeScores = torch.sigmoid(edgePred[:,-1,1])
                    groupScores = torch.sigmoid(edgePred[:,-1,2])
                    edgeScores = torch.max(torch.sigmoid(edgePred[:,-1,0:3]),dim=1)
            else:
                edgeScores=relScores=groupScores=None
                mergeScores = torch.sigmoid(edgePred[:,-1,0])
            allEdgeScores.append(edgeScores)
            allRelScores.append(relScores)
            allMergeScores.append(mergeScores)
            allGroupScores.append(groupScores)

        #final, we'll count it as additional graph iteration
        if self.model_ref.useCurvedBBs:
            targIndex = newGetTargIndexForPreds_textLines(targetBoxes.cpu(),finalOutputBoxes,0.5,numClasses,False)
            noClassTargIndex = newGetTargIndexForPreds_textLines(targetBoxes.cpu(),finalOutputBoxes,0.5,0,False)
        elif self.model_ref.rotation:
            assert(False and 'untested and should be changed to reflect new newGetTargIndexForPreds_s')
            targIndex, fullHit, overSegmented = newGetTargIndexForPreds_dist(targetBoxes,outputBoxes,1.1,numClasses,hard_thresh=False)
        else:
            targIndex = newGetTargIndexForPreds_iou(targetBoxes,finalOutputBoxes,0.4,numClasses,False)
            noClassTargIndex = newGetTargIndexForPreds_iou(targetBoxes,finalOutputBoxes,0.4,0,False)
        targIndex = targIndex.numpy()
        noClassTargIndex = noClassTargIndex.numpy()

        #cacluate which pred bbs are close to eachother (for density measurement)
        if self.model_ref.useCurvedBBs:
            bb_centers = np.array([bb.getCenterPoint() for bb in finalOutputBoxes])
            bb_lefts = np.array([bb.pairPoints()[0] for bb in finalOutputBoxes]).mean(axis=1)
            bb_rights = np.array([bb.pairPoints()[1] for bb in finalOutputBoxes]).mean(axis=1)
        else:
            bb_centers = finalOutputBoxes[:,1:3]
            assert(not self.model_ref.rotation)
            bb_lefts = bb_centers.clone()
            bb_lefts[:,0]-=finalOutputBoxes[:,5]
            bb_rights = bb_centers.clone()
            bb_rights[:,0]+=finalOutputBoxes[:,5]
            
        dist_center_center = np.power(np.power(bb_centers[None,:,:]-bb_centers[:,None,:],2).sum(axis=2),0.5)
        dist_center_left = np.power(np.power(bb_centers[None,:,:]-bb_lefts[:,None,:],2).sum(axis=2),0.5)
        dist_center_right = np.power(np.power(bb_centers[None,:,:]-bb_rights[:,None,:],2).sum(axis=2),0.5)
        dist_left_right = np.power(np.power(bb_lefts[None,:,:]-bb_rights[:,None,:],2).sum(axis=2),0.5)
        dist_right_right = np.power(np.power(bb_rights[None,:,:]-bb_rights[:,None,:],2).sum(axis=2),0.5)
        dist_left_left = np.power(np.power(bb_lefts[None,:,:]-bb_lefts[:,None,:],2).sum(axis=2),0.5)
        d_thresh=50
        bb_close = (dist_center_center<d_thresh)+(dist_center_left<d_thresh)+(dist_center_right<d_thresh)+(dist_left_right<d_thresh)+(dist_right_right<d_thresh)+(dist_left_left<d_thresh)

        predGroup2GT={}
        noClassPredGroup2GT={}
        finalPurity=[]
        finalNoClassPurity=[]
        final_density=[]
        final_node_center=[]
        for node in range(len(finalPredGroups)):
            predGroupT = [targIndex[bb] for bb in finalPredGroups[node] if targIndex[bb]>=0]
            predGroup2GT[node] = getGTGroup(predGroupT,targetIndexToGroup) if purity(predGroupT,targetIndexToGroup)>0.25 else -1
            finalPurity.append(purity(predGroupT,targetIndexToGroup))

            noClassPredGroupT = [noClassTargIndex[bb] for bb in finalPredGroups[node] if noClassTargIndex[bb]>=0]
            noClassPredGroup2GT[node] = getGTGroup(noClassPredGroupT,targetIndexToGroup)
            finalNoClassPurity.append(purity(noClassPredGroupT,targetIndexToGroup))
            
            all_close_bbs = set()
            for bb in finalPredGroups[node]:
                nonzero = bb_close[bb].nonzero()
                if type(nonzero) is tuple:
                    close_bbs = bb_close[bb].nonzero()[0]
                else:
                    assert(bb_close[bb].nonzero().shape[1]==1)
                    close_bbs = bb_close[bb].nonzero()[:,0]
                all_close_bbs.update(obb for obb in close_bbs if obb not in finalPredGroups[node])
            final_density.append(len(all_close_bbs))
            final_node_center.append(bb_centers[finalPredGroups[node]].mean(axis=0))

        gtBB2Pred = [-1]*len(targetBoxes)
        for i,tInd in enumerate(targIndex):
            if tInd>=0:
                gtBB2Pred[tInd]=i
        finalNoClassGtBB2Pred = [-1]*len(targetBoxes)
        for i,tInd in enumerate(noClassTargIndex):
            if tInd>=0:
                finalNoClassGtBB2Pred[tInd]=i
        gtBBs2Pred.append(gtBB2Pred)
        gtGroup2Pred = defaultdict(list)
        for predG,gtG in predGroup2GT.items():
            if gtG!=-1:
                gtGroup2Pred[gtG].append(predG)
        gtGroups2Pred.append(gtGroup2Pred)

        allEdgeIndexes.append(finalEdgeIndexes)

        ###
        num_final_merge = len(allOutputBoxes[-1]) - len(finalOutputBoxes)
        #map from final bbs to last bbs
        finalBB2LastBB={}
        unmatched_finals=[]
        last_used=[False]*len(allOutputBoxes[-1])
        for finalI, bbF in enumerate(finalOutputBoxes):
            if self.model_ref.useCurvedBBs:
                ppF = bbF.polyPoints()
            else:
                ppF = bbF[1:6]
            match_found=False
            for lastI, bbL in enumerate(allOutputBoxes[-1]):
                
                if self.model_ref.useCurvedBBs:
                    ppL = bbL.polyPoints()
                else:
                    ppL = bbL[1:6]
                if len(ppF)==len(ppL):
                    max_diff = np.abs(ppF-ppL).max()
                    if max_diff<0.01:
                        match_found=True
                        #assert(not last_used[lastI]) #this actually occurs with gt (perfect overlap)
                        last_used[lastI]=True
                        finalBB2LastBB[finalI]=lastI
                        break
            if not match_found:
                unmatched_finals.append(finalI)
        #if num_final_merge>0:
        for finalI in unmatched_finals:
            best_diff=99999999
            best_lastI=None
            bbF = finalOutputBoxes[finalI]
            if self.model_ref.useCurvedBBs:
                ppF = bbF.polyPoints()
            else:
                ppF = bbF[1:6]
            match_found=False
            for lastI, bbL in enumerate(allOutputBoxes[-1]):
                if not last_used[lastI]:
                    
                    if self.model_ref.useCurvedBBs:
                        ppL = bbL.polyPoints()
                        diff = np.abs(ppF.mean(axis=0)-ppL.mean(axis=0)).sum()
                    else:
                        ppL = bbL[1:6]
                        diff = p.abs(ppF-ppL).sum()

                    if diff<best_diff:
                        best_diff=diff
                        best_lastI=lastI

            finalBB2LastBB[finalI]=best_lastI


        gtEdge2Pred = {}
        badPredEdges = []
        false_pos_edges = []
        false_pos_distances = []
        true_pos_distances = []
        true_pos_all_densities=[]
        false_pos_all_densities=[]
        for ei,(n1,n2) in enumerate(finalEdgeIndexes):
            gtGId1 = predGroup2GT[n1]
            gtGId2 = predGroup2GT[n2]
            edge = (min(gtGId1,gtGId2),max(gtGId1,gtGId2))
            distance = math.sqrt(np.power(final_node_center[n1]-final_node_center[n2],2).sum())
            is_false_pos=False
            if edge in gtEdge2Pred:
                #two predicted edges claiming the same gt edge
                #this can occur if one of both of the GT groups is still split
                double_rel_pred+=1
            elif edge in gtGroupAdj:
                gtEdge2Pred[edge] = ei
                true_pos_distances.append(distance)
                true_pos_all_densities.append(max(final_density[n1],final_density[n2]))
            else:
                false_pos_edges.append((ei,n1,n2)+edge)
                false_pos_distances.append(distance)
                false_pos_all_densities.append(max(final_density[n1],final_density[n2]))
                is_false_pos=True

            #is this relationship consistent with classes
            if self.model_ref.useCurvedBBs:
                classIdx1 = finalOutputBoxes[finalPredGroups[n1][0]].getCls()[:numClasses].argmax()
                classIdx2 = finalOutputBoxes[finalPredGroups[n2][0]].getCls()[:numClasses].argmax()
            else:
                classIdx1 = finalOutputBoxes[finalPredGroups[n1][0],6:7+numClasses].argmax()
                classIdx2 = finalOutputBoxes[finalPredGroups[n2][0],6:7+numClasses].argmax()
            tClass = min(classIdx1,classIdx2)
            bClass = max(classIdx1,classIdx2)
            is_consistent=True
            if not (classIdx1!=classIdx2 and ((tClass==0 and bClass==1) or (tClass==1 and bClass==2))):
                inconsistent_edges+=1
                is_consistent=False
            #else:
            #    #check neighbor edges
            #    for aei,(an1,an2) in enumerate(finalEdgeIndexes):
            #        if aei!=ei and (an1==n1  or an2==n1 or an1==n2 or an2==n2):
            #            #this is a shared edge. Check each consistent scenario
            #            an1_class=finalOutputBoxes[finalPredGroups[an1][0]].getCls()[:numClasses].argmax()
            #            an2_class=finalOutputBoxes[finalPredGroups[an2][0]].getCls()[:numClasses].argmax()
            #            if an1_class==tClass and an1_class==1 and an2_class==0:
            #                continue
            #            elif an1_class==tClass and an1_class==0 and an2_class==1:
            #                continue
            #            elif an1_class==bClass and an1_class==1 and an2_class==2:
            #                continue
            #            elif an1_class==bClass and an1_class==0 and an2_class==1:
            #                continue
            #            elif an2_class==tClass and an2_class==1 and an1_class==0:
            #                continue
            #            elif an2_class==tClass and an2_class==0 and an1_class==1:
            #                continue
            #            elif an2_class==bClass and an2_class==1 and an1_class==2:
            #                continue
            #            elif an2_class==bClass and an2_class==0 and an1_class==1:
            #                continue

            #            inconsistent_edges+=1
            #            is_consistent=False
            #            break
            if is_false_pos and is_consistent:
                if tClass==0 and bClass==1:
                    false_pos_consistent_header_rels+=1
                elif tClass==1 and bClass==2:
                    false_pos_consistent_question_rels+=1


            #find the confidence value (keep or rel?)
            lastBBs1 = set(finalBB2LastBB[bb] for bb in finalPredGroups[n1])
            lastBBs2 = set(finalBB2LastBB[bb] for bb in finalPredGroups[n2])
            #find most consistent last group
            lastNode1=lastNode2=None
            for lgi, bbs in enumerate(allPredGroups[-1]):
                bbs=set(bbs)
                same1 = len(bbs.intersection(lastBBs1))
                if same1/max(len(bbs),len(lastBBs1))>0.5:
                    lastNode1=lgi

                same2 = len(bbs.intersection(lastBBs2))
                if same2/max(len(bbs),len(lastBBs2))>0.5:
                    lastNode2=lgi
            
            if lastNode1 is not None and lastNode2 is not None:
                try:
                    ei = allEdgeIndexes[-1].index((min(lastNode1,lastNode2),max(lastNode1,lastNode2)))
                    keep = allEdgeScores[-1][ei] #allEdgePred[-1][ei,0]
                    rel = allRelScores[-1][ei]
                    if is_false_pos:
                        false_pos_keep_scores.append(keep)
                        false_pos_rel_scores.append(rel)
                    else:
                        true_pos_keep_scores.append(keep)
                        true_pos_rel_scores.append(rel)
                except ValueError:
                    pass
                    
                    

        missed_rels=gtGroupAdj.difference(set(gtEdge2Pred.keys()))


        for gtGId1,gtGId2 in gtEdge2Pred.keys():
            classIdx1 = targetBoxes[gtGroups[gtGId1][0],13:13+numClasses].argmax()
            classIdx2 = targetBoxes[gtGroups[gtGId2][0],13:13+numClasses].argmax()
            #assert(classIdx1!=classIdx2)
            minIdx = min(classIdx1,classIdx2)
            maxIdx = max(classIdx1,classIdx2)
            if minIdx==0 and maxIdx==1:
                hit_header_rels+=1
            elif minIdx==1 and maxIdx==2:
                hit_question_rels+=1
            elif minIdx==maxIdx:
                hit_misc_rels+=1
            else:
                assert(False)
        

        
        for gtGId1,gtGId2 in missed_rels:
            classIdx1 = targetBoxes[gtGroups[gtGId1][0],13:13+numClasses].argmax()
            classIdx2 = targetBoxes[gtGroups[gtGId2][0],13:13+numClasses].argmax()
            #assert(classIdx1!=classIdx2)
            minIdx = min(classIdx1,classIdx2)
            maxIdx = max(classIdx1,classIdx2)
            if minIdx==0 and maxIdx==1:
                missed_header_rels+=1
            elif minIdx==1 and maxIdx==2:
                missed_question_rels+=1
            elif minIdx==maxIdx:
                missed_misc_rels+=1
            else:
                assert(False)

            #is this a single/isolated relationship?
            is_single = True
            for (gn1,gn2) in gtGroupAdj:
                if (gn1==gtGId1 and gn2!=gtGId2) or (gn1==gtGId2 and gn2!=gtGId1) or (gn2==gtGId1 and gn1!=gtGId2) or (gn2==gtGId2 and gn1!=gtGId1):
                    is_single=False
                    break
            if is_single:
                missed_rel_was_single+=1


            found1 = any([gtBBs2Pred[0][gtbb]!=-1 for gtbb in gtGroups[gtGId1]])
            found2 = any([gtBBs2Pred[0][gtbb]!=-1 for gtbb in gtGroups[gtGId2]])
            if not found1 or not found2:
                missed_rel_from_bad_detection+=1
            else:
                for giter in range(num_giter+1): #+1 for final
                    found1 = any([gtBBs2Pred[giter][gtbb]!=-1 for gtbb in gtGroups[gtGId1]])
                    found2 = any([gtBBs2Pred[giter][gtbb]!=-1 for gtbb in gtGroups[gtGId2]])
                    if giter == num_giter:
                        noClassFound1 = any([finalNoClassGtBB2Pred[gtbb]!=-1 for gtbb in gtGroups[gtGId1]])
                        noClassFound2 = any([finalNoClassGtBB2Pred[gtbb]!=-1 for gtbb in gtGroups[gtGId2]])
                    if not found1:
                        was_merge=0
                        prevPredNodeIds = gtGroups2Pred[giter-1][gtGId1]
                        for prevPredNodeId in prevPredNodeIds:
                            #find all edges to see if it was merged
                            for ei,(n1,n2) in enumerate(allEdgeIndexes[giter-1]):
                                if n1==prevPredNodeId or n2==prevPredNodeId:
                                    if allMergeScores[giter-1][ei]>self.model.mergeThresh[giter-1]:
                                        was_merge+=1
                                        break
                        if giter == num_giter:
                            if was_merge>len(prevPredNodeIds):
                                missed_rel_from_bad_merge+=1
                            else:
                                #was it wrong class?
                                if noClassFound1 and (found2 or noClassFound2):
                                    missed_rel_from_misclass+=1
                                else:
                                    missed_rel_from_poor_alignement+=1
                        else:
                            assert(was_merge==len(prevPredNodeIds))
                            missed_rel_from_bad_merge+=1
                        break
                    if not found2:
                        was_merge=0
                        prevPredNodeIds = gtGroups2Pred[giter-1][gtGId2]
                        for prevPredNodeId in prevPredNodeIds:
                            #find all edges to see if it was merged
                            for ei,(n1,n2) in enumerate(allEdgeIndexes[giter-1]):
                                if n1==prevPredNodeId or n2==prevPredNodeId:
                                    if allMergeScores[giter-1][ei]>self.model.mergeThresh[giter-1]:
                                        was_merge+=1
                                        break
                        if giter == num_giter:
                            if was_merge>len(prevPredNodeIds):
                                missed_rel_from_bad_merge+=1
                            else:
                                #was it wrong class?
                                if noClassFound2 and (found1 or noClassFound1):
                                    missed_rel_from_misclass+=1
                                else:
                                    missed_rel_from_poor_alignement+=1
                        else:
                            assert(was_merge==len(prevPredNodeIds))
                            missed_rel_from_bad_merge+=1
                        break

                    predGroups1 = gtGroups2Pred[giter][gtGId1]
                    predGroups2 = gtGroups2Pred[giter][gtGId2]

                    if len(predGroups1)==0:
                        #bad grouping on prev iter. Let's double check that
                        was_group=0
                        prevPredNodeIds = gtGroups2Pred[giter-1][gtGId1]
                        for prevPredNodeId in prevPredNodeIds:
                            #find all edges to see if it was groupd
                            for ei,(n1,n2) in enumerate(allEdgeIndexes[giter-1]):
                                if n1==prevPredNodeId or n2==prevPredNodeId:
                                    if allGroupScores[giter-1][ei]>self.model.groupThresh[giter-1]:
                                        was_group+=1
                                        break
                        assert(was_group==len(prevPredNodeIds))
                        missed_rel_from_bad_group+=1
                        break
                    if len(predGroups2)==0:
                        #bad grouping on prev iter. Let's double check that
                        was_group=0
                        prevPredNodeIds = gtGroups2Pred[giter-1][gtGId2]
                        for prevPredNodeId in prevPredNodeIds:
                            #find all edges to see if it was groupd
                            for ei,(n1,n2) in enumerate(allEdgeIndexes[giter-1]):
                                if n1==prevPredNodeId or n2==prevPredNodeId:
                                    if allGroupScores[giter-1][ei]>self.model.groupThresh[giter-1]:
                                        was_group+=1
                                        break
                        assert(was_group==len(prevPredNodeIds))
                        missed_rel_from_bad_group+=1
                        break


                    if not self.model.merge_first or giter>0: #the merge-first is not doing relationships
                        edge_present=False
                        for ei,(n1,n2) in enumerate(allEdgeIndexes[giter]):
                            if (n1 in predGroups1 and n2 in predGroups2) or (n2 in predGroups1 and n1 in predGroups2):
                                edge_present=True
                                break
                        if not edge_present:
                            #we have it's nodes, so it must have been dropped as an edge
                            if (not self.model.merge_first and giter==0) or (self.model.merge_first and giter==1):
                                #import pdb;pdb.set_trace()
                                missed_rel_from_missed_prop+=1
                            else:
                                #it must have been dropped in the previous iteration. Double check this is right
                                prevEdgePred=[]
                                #prevRelPred=[]
                                prevPredGroups1 = gtGroups2Pred[giter-1][gtGId1]
                                prevPredGroups2 = gtGroups2Pred[giter-1][gtGId2]
                                for ei,(n1,n2) in enumerate(allEdgeIndexes[giter-1]):
                                    if (n1 in prevPredGroups1 and n2 in prevPredGroups2) or (n2 in prevPredGroups1 and n1 in prevPredGroups2):
                                        prevEdgePred.append(allEdgeScores[giter-1][ei]>self.model.keepEdgeThresh[giter-1])
                                        #prevRelPred.append(allRelScores[giter-1][ei]>self.model.keepRelThresh)
                                assert(len(prevEdgePred)>0)
                                #assert(not any(prevEdgePred))
                                missed_rel_from_pruned_edge[giter-1]+=1

                            break

        for pei,pn1,pn2,gtGId1,gtGId2 in false_pos_edges:
            #is it single/isolated? (are there any connected predicted edges?)
            is_single=True
            for n1,n2 in finalEdgeIndexes:
                if (pn1==n1 and pn2!=n2) or (pn1==n2 and pn2!=n1) or (pn2==n1 and pn1!=n2) or (pn2==n2 and pn1!=n1):
                    is_single=False
                    break
            if is_single:
                false_pos_is_single+=1


            impure_group=False
            if len(finalPredGroups[pn1])>1 or len(finalPredGroups[pn2])>1 or (gtGId1!=-1 and len(gtGroups[gtGId1])>1) or (gtGId2!=-1 and len(gtGroups[gtGId2])>1):
                false_pos_group_involved+=1

                if (finalNoClassPurity[pn1]<0.99 and len(finalPredGroups[pn1])>1) or (finalNoClassPurity[pn2]<0.99 and len(finalPredGroups[pn2])>1):
                    false_pos_inpure_group+=1
                    impure_group=True
                #print('fp group {}({}) -- {}({})'.format(bb_centers[finalPredGroups[pn1][0]],finalNoClassPurity[pn1],bb_centers[finalPredGroups[pn2][0]],finalNoClassPurity[pn2]))
                #import pdb;pdb.set_trace()


            #is there a bad detection? or merge? These are hard to tell apart
            #It's a bad merge if it would fine without a given detection
            #it's a bad detection if there is no way to have combine detected elements to match
            if gtGId1==-1 or gtGId2==-2:
                new_gt1 = noClassPredGroup2GT[pn1]
                new_gt2 = noClassPredGroup2GT[pn2]
                if new_gt1!=-1 and new_gt2!=-1 and (min(new_gt1,new_gt2),max(new_gt1,new_gt2)) in gtGroupAdj:
                    false_pos_from_bad_class+=1
                elif not impure_group and (new_gt1==-1 or new_gt2==-1):
                    false_pos_bad_node+=1
                    #print('fp group {} -- {}'.format(bb_centers[finalPredGroups[pn1][0]],bb_centers[finalPredGroups[pn2][0]]))
                elif not impure_group:
                    false_pos_with_misclassed_nodes+=1
            elif not impure_group:
                false_pos_with_good_nodes+=1
                
            #TODO, chickening out and just saying "bad node"

        num_true_pos = len(true_pos_distances)
        num_false_pos = len(false_pos_distances)
        num_false_neg = len(missed_rels)
        num_pos = num_true_pos+num_false_pos


        self.characterization_sum['num_true_pos']+=num_true_pos 
        self.characterization_sum['num_false_pos']+=num_false_pos
        self.characterization_sum['num_false_neg']+=num_false_neg

        self.characterization_sum['num_header_rel_true_pos']+=hit_header_rels
        self.characterization_sum['num_header_rel_false_pos']+=false_pos_consistent_header_rels
        self.characterization_sum['num_header_rel_false_neg']+=missed_header_rels
        self.characterization_sum['num_question_rel_true_pos']+=hit_question_rels
        self.characterization_sum['num_misc_rel_true_pos']+=hit_question_rels
        self.characterization_sum['num_question_rel_false_pos']+=false_pos_consistent_question_rels
        self.characterization_sum['num_question_rel_false_neg']+=missed_question_rels
        self.characterization_sum['num_misc_rel_false_neg']+=missed_misc_rels

        self.characterization_sum['false_pos_is_single']+=false_pos_is_single
        self.characterization_sum['false_pos_group_involved']+=false_pos_group_involved
        self.characterization_sum['false_pos_inpure_group']+=false_pos_inpure_group
        self.characterization_sum['false_pos_from_bad_class']+=false_pos_from_bad_class
        self.characterization_sum['false_pos_bad_node']+=false_pos_bad_node
        self.characterization_sum['false_pos_with_good_nodes']+=false_pos_with_good_nodes
        self.characterization_sum['false_pos_with_misclassed_nodes']+=false_pos_with_misclassed_nodes

        self.characterization_sum['inconsistent_edges']+=inconsistent_edges
        self.characterization_sum['false_pos_consistent_header_rels']+=false_pos_consistent_header_rels
        self.characterization_sum['false_pos_consistent_question_rels']+=false_pos_consistent_question_rels
        #self.characterization_sum['missed_header_rels']+=missed_header_rels
        #self.characterization_sum['missed_question_rels']+=missed_question_rels
        #self.characterization_sum['hit_header_rels']+=hit_header_rels
        #self.characterization_sum['hit_question_rels']+=hit_question_rels

        self.characterization_sum['double_rel_pred']+=double_rel_pred
        self.characterization_sum['missed_rel_was_single']+=missed_rel_was_single


        self.characterization_sum['missed_rel_from_bad_detection']+=missed_rel_from_bad_detection
        self.characterization_sum['missed_rel_from_bad_merge']+=missed_rel_from_bad_merge
        self.characterization_sum['missed_rel_from_missed_prop']+=missed_rel_from_missed_prop
        self.characterization_sum['missed_rel_from_bad_group']+=missed_rel_from_bad_group
        self.characterization_sum['missed_rel_from_poor_alignement']+=missed_rel_from_poor_alignement
        self.characterization_sum['missed_rel_from_misclass']+=missed_rel_from_misclass

        self.characterization_sum['total_merges']+=self.model.merges_performed

        
        self.characterization_form['false_pos_is_single'].append(false_pos_is_single/num_false_pos if num_false_pos>0 else 0)
        self.characterization_form['false_pos_group_involved'].append(false_pos_group_involved/num_false_pos if num_false_pos>0 else 0)
        self.characterization_form['false_pos_inpure_group'].append(false_pos_inpure_group/num_false_pos if num_false_pos>0 else 0)
        self.characterization_form['false_pos_from_bad_class'].append(false_pos_from_bad_class/num_false_pos if num_false_pos>0 else 0)
        self.characterization_form['false_pos_bad_node'].append(false_pos_bad_node/num_false_pos if num_false_pos>0 else 0)
        self.characterization_form['false_pos_with_good_nodes'].append(false_pos_with_good_nodes/num_false_pos if num_false_pos>0 else 0)
        self.characterization_form['false_pos_with_misclassed_nodes'].append(false_pos_with_misclassed_nodes/num_false_pos if num_false_pos>0 else 0)
        assert(inconsistent_edges<=num_pos)
        self.characterization_form['inconsistent_edges'].append(inconsistent_edges/(num_pos) if (num_pos)>0 else 0)
        self.characterization_form['false_pos_consistent_header_rels'].append(false_pos_consistent_header_rels/num_false_pos if num_false_pos>0 else 0)
        self.characterization_form['false_pos_consistent_question_rels'].append(false_pos_consistent_question_rels/num_false_pos if num_false_pos>0 else 0)
        self.characterization_form['missed_header_rels'].append(missed_header_rels/num_false_neg if num_false_neg>0 else 0)
        self.characterization_form['missed_question_rels'].append(missed_question_rels/num_false_neg if num_false_neg>0 else 0)
        self.characterization_form['missed_misc_rels'].append(missed_misc_rels/num_false_neg if num_false_neg>0 else 0)
        self.characterization_form['hit_header_rels'].append(hit_header_rels/num_true_pos if num_true_pos>0 else 0)
        self.characterization_form['hit_question_rels'].append(hit_question_rels/num_true_pos if num_true_pos>0 else 0)
        self.characterization_form['hit_misc_rels'].append(hit_misc_rels/num_true_pos if num_true_pos>0 else 0)

        self.characterization_form['double_rel_pred'].append(double_rel_pred/num_pos if num_pos>0 else 0)
        self.characterization_form['missed_rel_was_single'].append(missed_rel_was_single/num_false_neg if num_false_neg>0 else 0)


        self.characterization_form['missed_rel_from_bad_detection'].append(missed_rel_from_bad_detection/num_false_neg if num_false_neg>0 else 0)
        self.characterization_form['missed_rel_from_bad_merge'].append(missed_rel_from_bad_merge/num_false_neg if num_false_neg>0 else 0)
        self.characterization_form['missed_rel_from_missed_prop'].append(missed_rel_from_missed_prop/num_false_neg if num_false_neg>0 else 0)
        self.characterization_form['missed_rel_from_bad_group'].append(missed_rel_from_bad_group/num_false_neg if num_false_neg>0 else 0)
        self.characterization_form['missed_rel_from_poor_alignement'].append(missed_rel_from_poor_alignement/num_false_neg if num_false_neg>0 else 0)
        self.characterization_form['missed_rel_from_misclass'].append(missed_rel_from_misclass/num_false_neg if num_false_neg>0 else 0)

        all_sum=0
        for giter,missed in missed_rel_from_pruned_edge.items():
            self.characterization_sum['missed_rel_from_pruned_edge_{}'.format(giter)]+=missed
            self.characterization_form['missed_rel_from_pruned_edge_{}'.format(giter)].append(missed/num_false_neg)
            all_sum+=missed
        self.characterization_sum['missed_rel_from_pruned_edge_all']+=all_sum
        self.characterization_form['missed_rel_from_pruned_edge_all'].append(all_sum/num_false_neg if num_false_neg>0 else 0)

        self.characterization_hist['true_pos_distances']+=true_pos_distances
        self.characterization_hist['false_pos_distances']+=false_pos_distances
        self.characterization_hist['true_pos_all_densities']+=true_pos_all_densities
        self.characterization_hist['false_pos_all_densities']+=false_pos_all_densities
        self.characterization_hist['true_pos_keep_scores']+=true_pos_keep_scores
        self.characterization_hist['false_pos_keep_scores']+=false_pos_keep_scores
        self.characterization_hist['true_pos_rel_scores']+=true_pos_rel_scores
        self.characterization_hist['false_pos_rel_scores']+=false_pos_rel_scores
        #@#####
        return
        #######

        print('true positives = {}'.format(num_true_pos))
        print('false positives = {}'.format(num_false_pos))
        print('false negatives = {}'.format(num_false_neg))

        print('false_pos_is_single={}'.format(false_pos_is_single))
        print('false_pos_group_involved={}'.format(false_pos_group_involved))
        print('false_pos_inpure_group={}'.format(false_pos_inpure_group))
        print('false_pos_from_bad_class={}'.format(false_pos_from_bad_class))
        print('false_pos_bad_node={}'.format(false_pos_bad_node))
        print('false_pos_with_good_nodes={}'.format(false_pos_with_good_nodes))
        print('false_pos_with_misclassed_nodes={}'.format(false_pos_with_misclassed_nodes))

        print('inconsistent_edges={}'.format(inconsistent_edges))
        print('false_pos_consistent_header_rels={}'.format(false_pos_consistent_header_rels))
        print('false_pos_consistent_question_rels={}'.format(false_pos_consistent_question_rels))
        print('missed_header_rels={}'.format(missed_header_rels))
        print('missed_question_rels={}'.format(missed_question_rels))
        print('hit_header_rels={}'.format(hit_header_rels))
        print('hit_question_rels={}'.format(hit_question_rels))

        print('double_rel_pred={}'.format(double_rel_pred))
        print('missed_rel_was_single={}'.format(missed_rel_was_single))

        all_sum=0
        for giter,missed in missed_rel_from_pruned_edge.items():
            print('missed_rel_from_pruned_edge[{}]={}'.format(giter,missed))
            all_sum+=missed
        print('missed_rel_from_pruned_edge[all]={}'.format(all_sum))
        all_sum+=missed_rel_from_bad_detection+missed_rel_from_bad_merge+missed_rel_from_missed_prop+missed_rel_from_bad_group+missed_rel_from_poor_alignement+missed_rel_from_misclass
        #missed_rel_from_bad_detection/=len(missed_rels)
        print('missed_rel_from_bad_detection={}'.format(missed_rel_from_bad_detection))
        #missed_rel_from_bad_merge/=len(missed_rels)
        print('missed_rel_from_bad_merge={}'.format(missed_rel_from_bad_merge))
        #missed_rel_from_missed_prop/=len(missed_rels)
        print('missed_rel_from_missed_prop={}'.format(missed_rel_from_missed_prop))
        #missed_rel_from_bad_group/=len(missed_rels)
        print('missed_rel_from_bad_group={}'.format(missed_rel_from_bad_group))
        print('missed_rel_from_poor_alignement={}'.format(missed_rel_from_poor_alignement))
        print('missed_rel_from_misclass={}'.format(missed_rel_from_misclass))
        
        print('accounted missed={}, unaccounted missed={}'.format(all_sum,len(missed_rels)-all_sum))

        plt.figure(1)
        #max_distance = max(true_pos_distances+false_pos_distances)
        #bins = np.linspace(0, max_distance, 10)
        plt.hist([true_pos_distances,false_pos_distances],bins=10,label=['true_pos','false_pos'])
        plt.xlabel('distance')
        plt.ylabel('count')
        plt.title('rel_distances')
        plt.legend(loc='upper right')

        plt.figure(2)
        #max_den = max(true_pos_all_densities+false_pos_all_densities)
        #bins = np.linspace(0, max_den, 5)
        plt.hist([true_pos_all_densities,false_pos_all_densities],bins=5,label=['true_pos','false_pos'])
        plt.xlabel('density')
        plt.ylabel('count')
        plt.title('densities')
        plt.legend(loc='upper right')

        plt.figure(3)
        plt.hist([true_pos_keep_scores,false_pos_keep_scores],bins=5,label=['true_pos','false_pos'])
        plt.xlabel('score')
        plt.ylabel('count')
        plt.title('keep edge')
        plt.legend(loc='upper right')

        plt.figure(4)
        plt.hist([true_pos_rel_scores,false_pos_rel_scores],bins=5,label=['true_pos','false_pos'])
        plt.xlabel('score')
        plt.ylabel('count')
        plt.title('rel edge')
        plt.legend(loc='upper right')


        plt.show()


    def displayCharacterization(self):
        print('\n==============')
        print('Avg by form')
        print('==============')
        for name,values in self.characterization_form.items():
            print('{}:\t{:.3f}'.format(name,np.mean(values)))


        print('\n==============')
        print('Total count')
        print('==============')
        for name,value in self.characterization_sum.items():
            print('{}:\t{:.3f}'.format(name,value))

        print('\n==============')
        print('Total portions')
        print('==============')
        num_true_pos = self.characterization_sum['num_true_pos']
        num_false_pos = self.characterization_sum['num_false_pos']
        num_false_neg = self.characterization_sum['num_false_neg']
        num_pos = num_true_pos+num_false_pos
        print('precision:\t{:.3f}'.format(num_true_pos/num_pos))
        print('recall:\t{:.3f}'.format(num_true_pos/(num_true_pos+num_false_neg)))
        del self.characterization_sum['num_true_pos']
        del self.characterization_sum['num_false_pos']
        del self.characterization_sum['num_false_neg']
        for a in ['header','question']:
            a_num_true_pos = self.characterization_sum['num_{}_rel_true_pos'.format(a)]
            a_num_false_pos = self.characterization_sum['num_{}_rel_false_pos'.format(a)]
            a_num_false_neg = self.characterization_sum['num_{}_rel_false_neg'.format(a)]
            a_num_pos = a_num_true_pos+a_num_false_pos
            print('{}_rel_precision:\t{:.3f}'.format(a,a_num_true_pos/a_num_pos))
            print('{}_rel_recall:\t{:.3f}'.format(a, a_num_true_pos/(a_num_true_pos+a_num_false_neg)))
            del self.characterization_sum['num_{}_rel_true_pos'.format(a)]
            del self.characterization_sum['num_{}_rel_false_pos'.format(a)]
            del self.characterization_sum['num_{}_rel_false_neg'.format(a)]
        for name,value in self.characterization_sum.items():
            if 'false_neg' in name or 'missed' in name:
                divide = num_false_neg
            elif 'false_pos' in name:
                divide = num_false_pos
            elif 'hit' in name:
                divide = num_true_pos
            else:
                divide = num_pos
            if divide!=0:
                print('{}:\t{:.3f}'.format(name,value/divide))
            else:
                print('{}:\t{}/{}'.format(name,value,divide))

        plt.figure(1)
        #max_distance = max(true_pos_distances+false_pos_distances)
        #bins = np.linspace(0, max_distance, 10)
        plt.hist([self.characterization_hist['true_pos_distances'],self.characterization_hist['false_pos_distances']],bins=15,label=['true_pos','false_pos'])
        plt.xlabel('distance')
        plt.ylabel('count')
        plt.title('rel_distances')
        plt.legend(loc='upper right')

        plt.savefig('characterization_distance.png')

        plt.figure(2)
        #max_den = max(true_pos_all_densities+false_pos_all_densities)
        #bins = np.linspace(0, max_den, 5)
        plt.hist([self.characterization_hist['true_pos_all_densities'],self.characterization_hist['false_pos_all_densities']],bins=10,label=['true_pos','false_pos'])
        plt.xlabel('density')
        plt.ylabel('count')
        plt.title('densities')
        plt.legend(loc='upper right')

        plt.savefig('characterization_density.png')

        plt.figure(3)
        plt.hist([self.characterization_hist['true_pos_keep_scores'],self.characterization_hist['false_pos_keep_scores']],bins=10,label=['true_pos','false_pos'])
        plt.xlabel('score')
        plt.ylabel('count')
        plt.title('keep edge')
        plt.legend(loc='upper right')

        plt.savefig('characterization_keep_score.png')

        plt.figure(4)
        plt.hist([self.characterization_hist['true_pos_rel_scores'],self.characterization_hist['false_pos_rel_scores']],bins=10,label=['true_pos','false_pos'])
        plt.xlabel('score')
        plt.ylabel('count')
        plt.title('rel edge')
        plt.legend(loc='upper right')

        fignum=5
        for name,values in self.characterization_form.items():
            plt.figure(fignum)
            fignum+=1
            plt.hist(values,bins=10)
            plt.xlabel(name)
            plt.ylabel('count')
            plt.title(name)

            plt.savefig('characterization_{}.png'.format(name))

        #plt.show()

    def optimizeF(self,relPred,relCand,predNN,rel_threshold=0.7):
        penalty = 0.25
        #print('optimizing with penalty {}'.format(penalty))
        thresh=0.15
        while thresh<0.45:
            keep = relPred>thresh
            newRelPred = relPred[keep].cpu()
            if newRelPred.size(0)<700:
                break
        if newRelPred.size(0)>0:
            #newRelCand = [ cand for i,cand in enumerate(relCand) if keep[i] ]
            usePredNN= True# predNN is not None and config['optimize']!='gt'
            idMap={}
            newId=0
            newRelCand=[]
            numNeighbors=[]
            for index,(id1,id2) in enumerate(relCand):
                if keep[index]:
                    if id1 not in idMap:
                        idMap[id1]=newId
                        if not usePredNN:
                            numNeighbors.append(target_num_neighbors[0,bbAlignment[id1]])
                        else:
                            numNeighbors.append(predNN[id1].item())
                        newId+=1
                    if id2 not in idMap:
                        idMap[id2]=newId
                        if not usePredNN:
                            numNeighbors.append(target_num_neighbors[0,bbAlignment[id2]])
                        else:
                            numNeighbors.append(predNN[id2].item())
                        newId+=1
                    newRelCand.append( [idMap[id1],idMap[id2]] )            


            #if not usePredNN:
                #    decision = optimizeRelationships(newRelPred,newRelCand,numNeighbors,penalty)
            #else:
            decision= optimizeRelationshipsSoft(newRelPred,newRelCand,numNeighbors,penalty, rel_threshold)
            decision= torch.from_numpy( np.round_(decision).astype(int) )
            decision=decision.to(relPred.device)
            relPred[keep] = torch.where(0==decision,relPred[keep]-1,relPred[keep]+self.thresh_rel)
            relPred[~keep] -=1
