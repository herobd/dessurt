import numpy as np
import torch
from base import BaseTrainer
import timeit
from utils import util
from collections import defaultdict
from evaluators.draw_graph import draw_graph
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist, AP_textLines, getTargIndexForPreds_iou, newGetTargIndexForPreds_iou, getTargIndexForPreds_dist, newGetTargIndexForPreds_textLines, computeAP
from utils.group_pairing import getGTGroup, pure, purity
from datasets.testforms_graph_pair import display
import random, os

from model.oversegment_loss import build_oversegmented_targets_multiscale
from model.overseg_box_detector import build_box_predictions

import torch.autograd.profiler as profile

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
        self.thresh_edge = self.model_ref.keepEdgeThresh
        self.thresh_overSeg = self.model_ref.mergeThresh
        self.thresh_group = self.model_ref.groupThresh
        self.thresh_rel = [self.thresh_rel]*len(self.thresh_group)
        self.thresh_error = config['trainer']['thresh_error'] if 'thresh_error' in config['trainer'] else [0.5]*len(self.thresh_group)

        self.gt_bb_align_IOcU_thresh = 0.4 if 'gt_bb_align_IOcU_thresh' not in config['trainer'] else config['trainer']['gt_bb_align_IOcU_thresh']

        #we iniailly train the pairing using GT BBs, but eventually need to fine-tune the pairing using the networks performance
        self.stop_from_gt = config['trainer']['stop_from_gt'] if 'stop_from_gt' in config['trainer'] else None
        self.partial_from_gt = config['trainer']['partial_from_gt'] if 'partial_from_gt' in config['trainer'] else None
        self.max_use_pred = config['trainer']['max_use_pred'] if 'max_use_pred' in config['trainer'] else 0.9

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

        self.num_node_error_class = 0
        self.final_class_bad_alignment = False
        self.final_class_bad_alignment = False
        self.final_class_inpure_group = False

        self.debug = 'DEBUG' in  config['trainer']
        self.save_images_every = config['trainer']['save_images_every'] if 'save_images_every' in config['trainer'] else 50
        self.save_images_dir = 'train_out'
        util.ensure_dir(self.save_images_dir)

        #Name change
        if 'edge' in self.lossWeights:
            self.lossWeights['rel'] = self.lossWeights['edge']
        if 'edge' in self.loss:
            self.loss['rel'] = self.loss['edge']

        #t#self.opt_history = defaultdict(list)#t#

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

    def useGT(self,iteration):
        if self.stop_from_gt is not None and iteration>=self.stop_from_gt:
            return random.random()>self.max_use_pred #I think it's best to always have some GT examples
        elif self.partial_from_gt is not None and iteration>=self.partial_from_gt:
            return random.random()> self.max_use_pred*(iteration-self.partial_from_gt)/(self.stop_from_gt-self.partial_from_gt)
        else:
            return True
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
        if self.mergeAndGroup:
            losses, run_log, out = self.newRun(thisInstance,useGT,threshIntur)
        else:
            losses, run_log, out = self.run(thisInstance,useGT,threshIntur)
        #t#self.opt_history['full run'].append(timeit.default_timer()-tic)#t#

        #t#tic=timeit.default_timer()#t##t#
        loss=0
        for name in losses.keys():
            losses[name] *= self.lossWeights[name[:-4]]
            loss += losses[name]
            losses[name] = losses[name].item()
        if len(losses)>0:
            loss.backward()

        torch.nn.utils.clip_grad_value_(self.model.parameters(),1)
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


        with torch.no_grad():
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.model_ref.detector.predNumNeighbors:
                    instance['num_neighbors']=None
                if not self.logged:
                    print('iter:{} valid batch: {}/{}'.format(self.iteration,batch_idx,len(self.valid_data_loader)), end='\r')
                if self.mergeAndGroup:
                    losses,log_run, out = self.newRun(instance,False,get=['bb_stats','nn_acc'])
                else:
                    losses,log_run, out = self.run(instance,False,get=['bb_stats','nn_acc'])

                for name,value in log_run.items():
                    if value is not None:
                        val_name = 'val_'+name
                        if val_name in val_metrics:
                            val_metrics[val_name]+=value
                            val_count[val_name]+=1
                        else:
                            val_metrics[val_name]=value
                for name,value in losses.items():
                    if value is not None:
                        value = value.item()
                        val_name = 'val_'+name
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
            else:
                recall=0
                ap=0
                prec=1
                targIndex = None
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
                    targIndex = newGetTargIndexForPreds_textLines(targetBoxes[0],outputBoxes,self.gt_bb_align_IOcU_thresh,numClasses,True)
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
            gtGroupAdjMat = [(min(gtGroup0,gtGroup1),max(gtGroup0,gtGroup1)) in gtGroupAdj for gtGroup0,gtGroup1 in gtNE]
            gtGroupAdjMat = torch.BoolTensor(gtGroupAdjMat).to(edge_loss_device)

            

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
            elif self.init_merge_rule=='adjacent':
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

            else:
                raise NotImplementedError('Unknown merge rule: {}'.format(self.merge_rule))
            
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

        return predsGTYes, predsGTNo, targIndex,  proposedInfo, log, predTypes


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
                    toKeep = 1-((bbFullHit==0) * (bbAlignment!=-1)) #toKeep = not (incomplete_overlap and did_overlap)
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


        if self.model_ref.predNN and bbPredNN_use is not None and bbPredNN_use.size(0)>0:
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






    def newRun(self,instance,useGT,threshIntur=None,get=[]):
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
        if useGT and targetBoxes is not None:
            if self.model_ref.useCurvedBBs:
                #build targets of GT to pass as detections
                ph_boxes = [torch.zeros(1,1,1,1,1)]*3
                ph_cls = [torch.zeros(1,1,1,1,1)]*3
                ph_conf = [torch.zeros(1,1,1,1)]*3
                scale = self.model_ref.detector.scale
                numAnchors = self.model_ref.detector.numAnchors
                numBBParams = self.model_ref.detector.numBBParams
                numBBParams = self.model_ref.detector.numBBParams
                numBBTypes = self.model_ref.numBBTypes
                grid_sizesH=[image.size(2)//s[0] for s in scale]
                grid_sizesW=[image.size(3)//s[0] for s in scale]


                nGT, masks, conf_masks, t_Ls, t_Ts, t_Rs, t_Bs, t_rs, tconf_scales, tcls_scales, pred_covered, gt_covered, recall, precision, pred_covered_noclass, gt_covered_noclass, recall_noclass, precision_noclass = build_oversegmented_targets_multiscale(ph_boxes, ph_conf, ph_cls, targetBoxes, [targetBoxes.size(1)], numBBTypes, grid_sizesH, grid_sizesW,scale=scale, assign_mode='split', close_anchor_rule='unmask')
                
                for i in range(len(t_Ls)):
                    assert((t_Ls[i]<=t_Rs[i]).all() and (t_Ts[i]<=t_Bs[i]).all())

                #add some jitter
                jitter_std=0.001
                t_Ls = [t.type(torch.FloatTensor) + torch.FloatTensor(t.size()).normal_(std=jitter_std) for t in t_Ls]
                t_Ts = [t.type(torch.FloatTensor) + torch.FloatTensor(t.size()).normal_(std=jitter_std) for t in t_Ts]
                t_Rs = [t.type(torch.FloatTensor) + torch.FloatTensor(t.size()).normal_(std=jitter_std) for t in t_Rs]
                t_Bs = [t.type(torch.FloatTensor) + torch.FloatTensor(t.size()).normal_(std=jitter_std) for t in t_Bs]
                t_rs = [t.type(torch.FloatTensor) + torch.FloatTensor(t.size()).normal_(std=jitter_std) for t in t_rs]

                tconf_scales = [t.type(torch.FloatTensor) for t in tconf_scales]
                tcls_scales = [t.type(torch.FloatTensor) for t in tcls_scales]

                ys = []
                for level in range(len(t_Ls)):
                    level_y = torch.cat([ torch.stack([2*tconf_scales[level]-1,t_Ls[level],t_Ts[level],t_Rs[level], t_Bs[level],t_rs[level]],dim=2), 2*tcls_scales[level].permute(0,1,4,2,3)-1], dim=2)
                    ys.append(level_y.view(level_y.size(0),level_y.size(1)*level_y.size(2),level_y.size(3),level_y.size(4)))
                targetBoxes_changed = build_box_predictions(ys,scale,ys[0].device,numAnchors,numBBParams,numBBTypes)
            else:
                targetBoxes_changed=targetBoxes

            #if self.iteration>=self.merge_first_only_until:
            #    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
            #        allOutputBoxes, outputOffsets, allEdgePred, allEdgeIndexes, allNodePred, allPredGroups, rel_prop_pred,merge_prop_scores, final = self.model(
            #                            image,
            #                            targetBoxes_changed,
            #                            target_num_neighbors,
            #                            True,
            #                            otherThresh=self.conf_thresh_init, 
            #                            otherThreshIntur=threshIntur, 
            #                            hard_detect_limit=self.train_hard_detect_limit,
            #                            gtTrans = gtTrans,
            #                            dont_merge = self.iteration<self.merge_first_only_until)
            #        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
            #else:
            allOutputBoxes, outputOffsets, allEdgePred, allEdgeIndexes, allNodePred, allPredGroups, rel_prop_pred,merge_prop_scores, final = self.model(
                                image,
                                targetBoxes_changed,
                                target_num_neighbors,
                                True,
                                otherThresh=self.conf_thresh_init, 
                                otherThreshIntur=threshIntur, 
                                hard_detect_limit=self.train_hard_detect_limit,
                                gtTrans = gtTrans,
                                merge_first_only = self.iteration<self.merge_first_only_until)
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
        proposedInfo=None
        mergeProposedInfo=None
        if allEdgePred is not None:
            for graphIteration,(outputBoxes,edgePred,nodePred,edgeIndexes,predGroups) in enumerate(zip(allOutputBoxes,allEdgePred,allNodePred,allEdgeIndexes,allPredGroups)):

                #t#tic2=timeit.default_timer()#t##t#
                predEdgeShouldBeTrue,predEdgeShouldBeFalse, bbAlignment, proposedInfoI, logIter, edgePredTypes = self.simplerAlignEdgePred(
                        targetBoxes,
                        targetIndexToGroup,
                        gtGroupAdj,
                        outputBoxes,
                        edgePred,
                        edgeIndexes,
                        predGroups, 
                        rel_prop_pred if (graphIteration==0 and not self.model_ref.merge_first) or (graphIteration==1 and self.model_ref.merge_first) else (merge_prop_scores if graphIteration==0 and self.model_ref.merge_first else None),
                        self.thresh_edge[graphIteration],
                        self.thresh_rel[graphIteration],
                        self.thresh_overSeg[graphIteration],
                        self.thresh_group[graphIteration],
                        self.thresh_error[graphIteration],
                        merge_only= graphIteration==0 and self.model_ref.merge_first
                        )
                #t#self.opt_history['newAlignEdgePred gI{}'.format(graphIteration)].append(timeit.default_timer()-tic2)#t#
                allEdgePredTypes.append(edgePredTypes)
                if graphIteration==0 and self.model_ref.merge_first:
                    mergeProposedInfo=proposedInfoI
                elif (graphIteration==0 and not self.model_ref.merge_first) or (graphIteration==1 and self.model_ref.merge_first):
                    proposedInfo=proposedInfoI

                assert(not self.model_ref.predNN)
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



                
                #Fine tuning detector. Should only happed once
                if not self.model_ref.detector_frozen and graphIteration==0:
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


                    #t#self.opt_history['box_loss'].append(timeit.default_timer()-tic2)#t#

                    #boxLoss *= self.lossWeights['box']
                    #if relLoss is not None:
                    #    loss = relLoss + boxLoss
                    #else:
                    #    loss = boxLoss
                #else:
                #    loss = relLoss


                if self.model_ref.predNN and nodePredNN_use is not None and nodePredNN_use.size(0)>0:
                    alignedNN_use = alignedNN_use[:,None] #introduce "time" dimension to broadcast
                    nn_loss_final = self.loss['nnFinal'](nodePredNN_use,alignedNN_use)
                    losses['nnFinalLoss']+=nn_loss_final
                    #nn_loss_final *= self.lossWeights['nn']
                    
                    #if loss is not None:
                    #    loss += nn_loss_final
                    #else:
                    #    loss = nn_loss_final
                    #nn_loss_final = nn_loss_final.item()
                #else:
                    #nn_loss_final=0

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
                            targetBoxes,
                            self.classMap,
                            path,
                            useTextLines=self.model_ref.useCurvedBBs,
                            targetGroups=instance['gt_groups'],
                            targetPairs=instance['gt_groups_adj'])
                    print('saved {}'.format(path))

                if 'bb_stats' in get:

                    if self.model_ref.detector.predNumNeighbors:
                        beforeCls=1
                        #outputBoxesM=torch.cat((outputBoxes[:,0:6],outputBoxes[:,7:]),dim=1) #throw away NN pred
                    else:
                        beforeCls=0
                    #if targetBoxes is not None:
                    #    targetBoxes = targetBoxes.cpu()
                    if targetBoxes is not None:
                        target_for_b = targetBoxes[0].cpu()
                    else:
                        target_for_b = torch.empty(0)
                    if self.model_ref.useCurvedBBs:
                        ap_5, prec_5, recall_5, allPrec, allRecall =AP_textLines(target_for_b,outputBoxes,0.5,numClasses)
                    elif self.model_ref.rotation:
                        ap_5, prec_5, recall_5 =AP_dist(target_for_b,outputBoxes,0.9,numClasses, beforeCls=beforeCls)
                    else:
                        ap_5, prec_5, recall_5, allPrec, allRecall =AP_iou(target_for_b,outputBoxes,0.5,numClasses, beforeCls=beforeCls)
                    prec_5 = np.array(prec_5)
                    recall_5 = np.array(recall_5)
                    log['bb_AP_{}'.format(graphIteration)]=ap_5
                    log['bb_prec_{}'.format(graphIteration)]=prec_5
                    log['bb_recall_{}'.format(graphIteration)]=recall_5
                    log['bb_allPrec_{}'.format(graphIteration)]=allPrec
                    log['bb_allRecall_{}'.format(graphIteration)]=allRecall
                    log['bb_allFm_{}'.format(graphIteration)]= 2*allPrec*allRecall/(allPrec+allRecall) if allPrec+allRecall>0 else 0

        #t#self.opt_history['all losses'].append(timeit.default_timer()-tic)#t#
        #t#tic=timeit.default_timer()#t##t#
        #print final state of graph
        ###
        if final is not None:
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
                        None,
                        targetBoxes,
                        self.classMap,
                        path,
                        bbTrans=finalBBTrans,
                        useTextLines=self.model_ref.useCurvedBBs,
                        targetGroups=instance['gt_groups'],
                        targetPairs=instance['gt_groups_adj'])
                #print('saved {}'.format(path))
            finalOutputBoxes, finalPredGroups, finalEdgeIndexes, finalBBTrans = final
            #print('DEBUG final num node:{}, num edges: {}'.format(len(finalOutputBoxes) if finalOutputBoxes is not None else 0,len(finalEdgeIndexes) if finalEdgeIndexes is not None else 0))
        ###
        
        #log['rel_prec']= fullPrec
        #log['rel_recall']= eRecall
        #log['rel_Fm']= 2*(fullPrec*eRecall)/(eRecall+fullPrec) if eRecall+fullPrec>0 else 0
        #t#self.opt_history['final eval'].append(timeit.default_timer()-tic)#t#

        if not self.model_ref.detector_frozen:
            if 'nnFinalLoss' in losses:
                log['nn loss improvement (neg is good)'] = losses['nnFinalLoss'].item()-nn_loss
            if 'classFinalLoss' in losses:
                log['class loss improvement (neg is good)'] = losses['classFinalLoss'].item()-class_loss


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
        if mergeProposedInfo is not None:
            propRecall,propPrec = mergeProposedInfo[2:4]
            log['prop_merge_recall'] = propRecall
            log['prop_merge_prec'] = propPrec
        #if final_prop_rel_recall is not None:
        #    log['final_prop_rel_recall']=final_prop_rel_recall
        #if final_prop_rel_prec is not None:
        #    log['final_prop_rel_prec']=final_prop_rel_prec

        gt_groups_adj = instance['gt_groups_adj']
        if final is not None:
            finalLog = self.final_eval(targetBoxes.cpu() if targetBoxes is not None else None,gtGroups,gt_groups_adj,targetIndexToGroup,*final)
            log.update(finalLog)

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
            elif name=='final':
                 got[name] = final
            elif name != 'bb_stats' and name != 'nn_acc':
                raise NotImplementedError('Cannot get [{}], unknown'.format(name))
        return losses, log, got



    def final_eval(self,targetBoxes,gtGroups,gt_groups_adj,targetIndexToGroup,outputBoxes,predGroups,predPairs,predTrans=None):
        log={}
        numClasses = len(self.scoreClassMap)

        #Remove blanks
        if 'blank' in self.classMap:
            blank_index = self.classMap['blank']
            if targetBoxes is not None:
                gtNotBlanks = targetBoxes[0,:,blank_index]<0.5
                targetBoxes=targetBoxes[:,gtNotBlanks]
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
                    oldToNewGroups = {o:n for n,o in enumerate(newToOldGroups)}
                    predPairs = [(oldToNewGroups[g1],oldToNewGroups[g2]) for g1,g2 in predPairs if g1 in oldToNewGroups and g2 in oldToNewGroups]
                    for a,b in predPairs:
                        assert(a < len(predGroups))
                        assert(b < len(predGroups))
                if predTrans is not None:
                    predTrans = [predTrans[newToOldOutputBoxes[n]] for n in range(len(newToOldOutputBoxes))]

        if targetBoxes is not None:
            targetBoxes = targetBoxes.cpu()
            if self.model_ref.useCurvedBBs:
                targIndex = newGetTargIndexForPreds_textLines(targetBoxes[0],outputBoxes,0.5,numClasses,False)
            elif self.model_ref.rotation:
                raise NotImplementedError('newGetTargIndexForPreds_dist should be modified to reflect the behavoir or newGetTargIndexForPreds_textLines')
                targIndex, fullHit, overSegmented = newGetTargIndexForPreds_dist(targetBoxes[0],outputBoxes,1.1,numClasses,hard_thresh=False)
            else:
                targIndex = newGetTargIndexForPreds_iou(targetBoxes[0],outputBoxes,0.5,numClasses,False)
        elif outputBoxes is not None:
            targIndex=torch.LongTensor(len(outputBoxes)).fill_(-1)

        if self.model_ref.detector.predNumNeighbors:
            beforeCls=1
            #outputBoxesM=torch.cat((outputBoxes[:,0:6],outputBoxes[:,7:]),dim=1) #throw away NN pred
        else:
            beforeCls=0
        #if targetBoxes is not None:
        #    targetBoxes = targetBoxes.cpu()
        if targetBoxes is not None:
            target_for_b = targetBoxes[0].cpu()
        else:
            target_for_b = torch.empty(0)
        if self.model_ref.useCurvedBBs:
            ap_5, prec_5, recall_5, allPrec, allRecall =AP_textLines(target_for_b,outputBoxes,0.5,numClasses)
        elif self.model_ref.rotation:
            ap_5, prec_5, recall_5, allPrec, allRecall =AP_dist(target_for_b,outputBoxes,0.9,numClasses, beforeCls=beforeCls)
        else:
            ap_5, prec_5, recall_5, allPrec, allRecall =AP_iou(target_for_b,outputBoxes,0.5,numClasses, beforeCls=beforeCls)
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

        
        gtGroupHit=[False]*len(gtGroups)
        groupCompleteness=[]
        groupPurity={}
        predToGTGroup={}
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
                groupCompleteness.append(completeness)

        for hit in gtGroupHit:
            if not hit:
                groupCompleteness.append(0)
        

        log['final_groupCompleteness']=np.mean(groupCompleteness)
        log['final_groupPurity']=np.mean([v for k,v in groupPurity.items()])

        gtRelHit=set()
        relPrec=0
        if predPairs is None:
            predPairs=[]
        for n0,n1 in predPairs:
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
        if len(predPairs)>0:
            relPrec /= len(predPairs)
        else:
            relPrec = 1
        if len(gt_groups_adj)>0:
            relRecall = len(gtRelHit)/len(gt_groups_adj)
        else:
            relRecall = 1

        log['final_rel_prec']=relPrec
        log['final_rel_recall']=relRecall
        if relPrec+relRecall>0:
            log['final_rel_Fm']=(2*(relPrec*relRecall)/(relPrec+relRecall))
        else:
            log['final_rel_Fm']=0

        return log
