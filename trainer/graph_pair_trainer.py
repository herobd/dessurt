import numpy as np
import torch
from base import BaseTrainer
import timeit
from utils import util
from collections import defaultdict
from evaluators.draw_graph import draw_graph
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist, getTargIndexForPreds_iou, newGetTargIndexForPreds_iou, getTargIndexForPreds_dist, computeAP
from utils.group_pairing import getGTGroup, pure
from datasets.testforms_graph_pair import display
import random, os


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
        self.loss['box'] = self.loss['box'](**self.loss_params['box'], 
                num_classes=model.numBBTypes, 
                rotation=model.rotation, 
                scale=model.scale,
                anchors=model.anchors)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        #for i in range(self.start_iteration,
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        #self.log_step = int(np.sqrt(self.batch_size))
        #lr schedule from "Attention is all you need"
        #base_lr=config['optimizer']['lr']


        self.mergeAndGroup = config['trainer']['mergeAndGroup']
        self.classMap = self.data_loader.dataset.classMap


        #default is unfrozen, can be frozen by setting 'start_froze' in the PairingGraph models params
        self.unfreeze_detector = config['trainer']['unfreeze_detector'] if 'unfreeze_detector' in config['trainer'] else None

        self.thresh_conf = config['trainer']['thresh_conf'] if 'thresh_conf' in config['trainer'] else 0.92
        self.thresh_intersect = config['trainer']['thresh_intersect'] if 'thresh_intersect' in config['trainer'] else 0.4
        self.thresh_rel = config['trainer']['thresh_rel'] if 'thresh_rel' in config['trainer'] else 0.5
        self.thresh_overSeg = self.model.mergeThresh
        self.thresh_group = self.model.groupThresh
        self.thresh_error = config['trainer']['thresh_error'] if 'thresh_error' in config['trainer'] else 0.5

        #we iniailly train the pairing using GT BBs, but eventually need to fine-tune the pairing using the networks performance
        self.stop_from_gt = config['trainer']['stop_from_gt'] if 'stop_from_gt' in config['trainer'] else None
        self.partial_from_gt = config['trainer']['partial_from_gt'] if 'partial_from_gt' in config['trainer'] else None
        self.max_use_pred = config['trainer']['max_use_pred'] if 'max_use_pred' in config['trainer'] else 0.9

        self.conf_thresh_init = config['trainer']['conf_thresh_init'] if 'conf_thresh_init' in config['trainer'] else 0.9
        self.conf_thresh_change_iters = config['trainer']['conf_thresh_change_iters'] if 'conf_thresh_change_iters' in config['trainer'] else 5000

        self.train_hard_detect_limit = config['trainer']['train_hard_detect_limit'] if 'train_hard_detect_limit' in config['trainer'] else 100
        self.val_hard_detect_limit = config['trainer']['val_hard_detect_limit'] if 'val_hard_detect_limit' in config['trainer'] else 300

        self.useBadBBPredForRelLoss = config['trainer']['use_all_bb_pred_for_rel_loss'] if 'use_all_bb_pred_for_rel_loss' in config['trainer'] else False
        if self.useBadBBPredForRelLoss is True:
            self.useBadBBPredForRelLoss=1

        self.adaptLR = config['trainer']['adapt_lr'] if 'adapt_lr' in config['trainer'] else False
        self.adaptLR_base = config['trainer']['adapt_lr_base'] if 'adapt_lr_base' in config['trainer'] else 165 #roughly average number of rels
        self.adaptLR_ep = config['trainer']['adapt_lr_ep'] if 'adapt_lr_ep' in config['trainer'] else 15

        self.fixedAlign = config['trainer']['fixed_align'] if 'fixed_align' in config['trainer'] else False

        self.num_node_error_class = 0
        self.final_class_bad_alignment = False
        self.final_class_bad_alignment = False
        self.final_class_inpure_group = False

        self.debug = 'DEBUG' in  config['trainer']
        self.save_images_every = 50
        self.save_images_dir = 'train_out'
        util.ensure_dir(self.save_images_dir)

        #Name change
        if 'edge' in self.lossWeights:
            self.lossWeights['rel'] = self.lossWeights['edge']
        if 'edge' in self.loss:
            self.loss['rel'] = self.loss['edge']

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
            self.model.unfreeze()
        self.model.train()
        #self.model.eval()
        #print("WARNING EVAL")

        ##tic=timeit.default_timer()
        batch_idx = (iteration-1) % len(self.data_loader)
        try:
            thisInstance = self.data_loader_iter.next()
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            thisInstance = self.data_loader_iter.next()
        if not self.model.detector.predNumNeighbors:
            thisInstance['num_neighbors']=None
        ##toc=timeit.default_timer()
        ##print('data: '+str(toc-tic))
        
        ##tic=timeit.default_timer()

        self.optimizer.zero_grad()

        ##toc=timeit.default_timer()
        ##print('for: '+str(toc-tic))
        #loss = self.loss(output, target)
        index=0
        losses={}
        ##tic=timeit.default_timer()

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
        loss=0
        for name in losses.keys():
            losses[name] *= self.lossWeights[name[:-4]]
            loss += losses[name]
            losses[name] = losses[name].item()
        if len(losses)>0:
            loss.backward()

        torch.nn.utils.clip_grad_value_(self.model.parameters(),1)
        self.optimizer.step()
        meangrad=0
        count=0
        for m in self.model.parameters():
            if m.grad is None:
                continue
            count+=1
            meangrad+=m.grad.data.mean()
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
        return log

    #Old
    #def _train_iteration(self, iteration):
    #    """
    #    Training logic for an iteration

    #    :param iteration: Current training iteration.
    #    :return: A log that contains all information you want to save.

    #    Note:
    #        If you have additional information to record, for example:
    #            > additional_log = {"x": x, "y": y}
    #        merge it with log before return. i.e.
    #            > log = {**log, **additional_log}
    #            > return log

    #        The metrics in log must have the key 'metrics'.
    #    """
    #    if self.unfreeze_detector is not None and iteration>=self.unfreeze_detector:
    #        self.model.unfreeze()
    #    self.model.train()
    #    #self.model.eval()
    #    #print("WARNING EVAL")

    #    ##tic=timeit.default_timer()
    #    batch_idx = (iteration-1) % len(self.data_loader)
    #    try:
    #        thisInstance = self.data_loader_iter.next()
    #    except StopIteration:
    #        self.data_loader_iter = iter(self.data_loader)
    #        thisInstance = self.data_loader_iter.next()
    #    if not self.model.detector.predNumNeighbors:
    #        thisInstance['num_neighbors']=None
    #    ##toc=timeit.default_timer()
    #    ##print('data: '+str(toc-tic))
    #    
    #    ##tic=timeit.default_timer()

    #    self.optimizer.zero_grad()

    #    ##toc=timeit.default_timer()
    #    ##print('for: '+str(toc-tic))
    #    #loss = self.loss(output, target)
    #    index=0
    #    losses={}
    #    ##tic=timeit.default_timer()

    #    #if self.iteration % self.save_step == 0:
    #    #    targetPoints={}
    #    #    targetPixels=None
    #    #    _,lossC=FormsBoxPair_printer(None,thisInstance,self.model,self.gpu,self._eval_metrics,self.checkpoint_dir,self.iteration,self.loss['box'])
    #    #    loss, position_loss, conf_loss, class_loss, recall, precision = lossC
    #    #else:
    #    if self.conf_thresh_change_iters > iteration:
    #        threshIntur = 1 - iteration/self.conf_thresh_change_iters
    #    else:
    #        threshIntur = None
    #    image, targetBoxes, adj, target_num_neighbors = self._to_tensor(thisInstance)
    #    useGT = self.useGT(iteration)
    #    if useGT:
    #        outputBoxes, outputOffsets, relPred, relIndexes, bbPred = self.model(image,targetBoxes,target_num_neighbors,True,
    #                otherThresh=self.conf_thresh_init, otherThreshIntur=threshIntur, hard_detect_limit=self.train_hard_detect_limit)
    #        #_=None
    #        #gtPairing,predPairing = self.prealignedEdgePred(adj,relPred)
    #        predPairingShouldBeTrue,predPairingShouldBeFalse, eRecall,ePrec,fullPrec,ap = self.prealignedEdgePred(adj,relPred,relIndexes)
    #        if bbPred is not None:
    #            if self.model.predNN or self.model.predClass:
    #                if target_num_neighbors is not None:
    #                    alignedNN_use = target_num_neighbors[0]
    #                bbPredNN_use = bbPred[:,:,0]
    #                start=1
    #            else:
    #                start=0
    #            if self.model.predClass:
    #                if targetBoxes is not None:
    #                    alignedClass_use =  targetBoxes[0,:,13:13+self.model.numBBTypes]
    #                bbPredClass_use = bbPred[:,:,start:start+self.model.numBBTypes]
    #        else:
    #            bbPredNN_use=None
    #            bbPredClass_use=None
    #    else:
    #        outputBoxes, outputOffsets, relPred, relIndexes, bbPred = self.model(image,
    #                otherThresh=self.conf_thresh_init, otherThreshIntur=threshIntur, hard_detect_limit=self.train_hard_detect_limit)
    #        #gtPairing,predPairing = self.alignEdgePred(targetBoxes,adj,outputBoxes,relPred)
    #        predPairingShouldBeTrue,predPairingShouldBeFalse, eRecall,ePrec,fullPrec,ap, bbAlignment, bbFullHit = self.alignEdgePred(targetBoxes,adj,outputBoxes,relPred,relIndexes)
    #        if bbPred is not None and bbPred.size(0)>0:
    #            #create aligned GT
    #            #this was wrong...
    #                #first, remove unmatched predicitons that didn't overlap (weren't close) to any targets
    #                #toKeep = 1-((bbNoIntersections==1) * (bbAlignment==-1))
    #            #remove predictions that overlapped with GT, but not enough
    #            if self.model.predNN:
    #                start=1
    #                toKeep = 1-((bbFullHit==0) * (bbAlignment!=-1)) #toKeep = not (incomplete_overlap and did_overlap)
    #                if toKeep.any():
    #                    bbPredNN_use = bbPred[toKeep][:,:,0]
    #                    bbAlignment_use = bbAlignment[toKeep]
    #                    #becuase we used -1 to indicate no match (in bbAlignment), we add 0 as the last position in the GT, as unmatched 
    #                    if target_num_neighbors is not None:
    #                        target_num_neighbors_use = torch.cat((target_num_neighbors[0].float(),torch.zeros(1).to(target_num_neighbors.device)),dim=0)
    #                    else:
    #                        target_num_neighbors_use = torch.zeros(1).to(bbPred.device)
    #                    alignedNN_use = target_num_neighbors_use[bbAlignment_use.long()]

    #                else:
    #                    bbPredNN_use=None
    #                    alignedNN_use=None
    #            else:
    #                start=0
    #            if self.model.predClass:
    #                #We really don't care about the class of non-overlapping instances
    #                if targetBoxes is not None:
    #                    toKeep = bbFullHit==1
    #                    if toKeep.any():
    #                        bbPredClass_use = bbPred[toKeep][:,:,start:start+self.model.numBBTypes]
    #                        bbAlignment_use = bbAlignment[toKeep]
    #                        alignedClass_use =  targetBoxes[0][bbAlignment_use.long()][:,13:13+self.model.numBBTypes] #There should be no -1 indexes in hereS
    #                    else:
    #                        alignedClass_use = None
    #                        bbPredClass_use = None
    #                else:
    #                    alignedClass_use = None
    #                    bbPredClass_use = None
    #        else:
    #            bbPredNN_use = None
    #            bbPredClass_use = None
    #    if relPred is not None:
    #        numEdgePred = relPred.size(0)
    #        if predPairingShouldBeTrue is not None:
    #            lenTrue = predPairingShouldBeTrue.size(0)
    #        else:
    #            lenTrue = 0
    #        if predPairingShouldBeFalse is not None:
    #            lenFalse = predPairingShouldBeFalse.size(0)
    #        else:
    #            lenFalse = 0
    #    else:
    #        numEdgePred = lenTrue = lenFalse = 0
    #    numBoxPred = outputBoxes.size(0)
    #    #if iteration>25:
    #    #    import pdb;pdb.set_trace()
    #    #if len(predPairing.size())>0 and predPairing.size(0)>0:
    #    #    relLoss = self.loss['rel'](predPairing,gtPairing)
    #    #else:
    #    #    relLoss = torch.tensor(0.0,requires_grad=True).to(image.device)
    #    #relLoss = torch.tensor(0.0).to(image.device)
    #    relLoss = None
    #    #seperating the loss into true and false portions is not only convienint, it balances the loss between true/false examples
    #    if predPairingShouldBeTrue is not None and predPairingShouldBeTrue.size(0)>0:
    #        ones = torch.ones_like(predPairingShouldBeTrue).to(image.device)
    #        relLoss = self.loss['rel'](predPairingShouldBeTrue,ones)
    #        debug_avg_relTrue = predPairingShouldBeTrue.mean().item()
    #    else:
    #        debug_avg_relTrue =0 
    #    if predPairingShouldBeFalse is not None and predPairingShouldBeFalse.size(0)>0:
    #        zeros = torch.zeros_like(predPairingShouldBeFalse).to(image.device)
    #        relLossFalse = self.loss['rel'](predPairingShouldBeFalse,zeros)
    #        if relLoss is None:
    #            relLoss=relLossFalse
    #        else:
    #            relLoss+=relLossFalse
    #        debug_avg_relFalse = predPairingShouldBeFalse.mean().item()
    #    else:
    #        debug_avg_relFalse = 0
    #    if relLoss is not None:
    #        relLoss *= self.lossWeights['rel']



    #    if not self.model.detector_frozen:
    #        if targetBoxes is not None:
    #            targSize = targetBoxes.size(1)
    #        else:
    #            targSize =0 
    #        #import pdb;pdb.set_trace()
    #        boxLoss, position_loss, conf_loss, class_loss, nn_loss, recall, precision = self.loss['box'](outputOffsets,targetBoxes,[targSize],target_num_neighbors)
    #        boxLoss *= self.lossWeights['box']
    #        if relLoss is not None:
    #            loss = relLoss + boxLoss
    #        else:
    #            loss = boxLoss
    #    else:
    #        loss = relLoss


    #    if self.model.predNN and bbPredNN_use is not None and bbPredNN_use.size(0)>0:
    #        alignedNN_use = alignedNN_use[:,None] #introduce "time" dimension to broadcast
    #        nn_loss_final = self.loss['nn'](bbPredNN_use,alignedNN_use)
    #        nn_loss_final *= self.lossWeights['nn']
    #        
    #        if loss is not None:
    #            loss += nn_loss_final
    #        else:
    #            loss = nn_loss_final
    #        nn_loss_final = nn_loss_final.item()
    #    else:
    #        nn_loss_final=0

    #    if self.model.predClass and bbPredClass_use is not None and bbPredClass_use.size(0)>0:
    #        alignedClass_use = alignedClass_use[:,None] #introduce "time" dimension to broadcast
    #        class_loss_final = self.loss['class'](bbPredClass_use,alignedClass_use)
    #        class_loss_final *= self.lossWeights['class']
    #        loss += class_loss_final
    #        class_loss_final = class_loss_final.item()
    #    else:
    #        class_loss_final = 0
    #        
    #    ##toc=timeit.default_timer()
    #    ##print('loss: '+str(toc-tic))
    #    ##tic=timeit.default_timer()
    #    if not self.debug:
    #        predPairingShouldBeTrue= predPairingShouldBeFalse=outputBoxes=outputOffsets=relPred=image=targetBoxes=relLossFalse=None
    #    if relLoss is not None:
    #        relLoss = relLoss.item()
    #    else:
    #        relLoss = 0
    #    if not self.model.detector_frozen:
    #        boxLoss = boxLoss.item()
    #    else:
    #        boxLoss = 0
    #    if loss is not None:
    #        if self.adaptLR:
    #            #if we only have a few relationship preds, step smaller so that we don't skew with a bad bias
    #            #This effects the box loss too so that it doesn't yank the detector/backbone features around
    #            #we actually just scale the loss, but its all the same :)
    #            scale = (numEdgePred+self.adaptLR_ep)/(self.adaptLR_ep+self.adaptLR_base)
    #            loss *= scale
    #        loss.backward()

    #        torch.nn.utils.clip_grad_value_(self.model.parameters(),1)
    #        self.optimizer.step()

    #        loss = loss.item()
    #    else:
    #        loss=0

    #    ##toc=timeit.default_timer()
    #    ##print('bac: '+str(toc-tic))

    #    #tic=timeit.default_timer()
    #    metrics={}
    #    #index=0
    #    #for name, target in targetBoxes.items():
    #    #    metrics = {**metrics, **self._eval_metrics('box',name,output, target)}
    #    #for name, target in targetPoints.items():
    #    #    metrics = {**metrics, **self._eval_metrics('point',name,output, target)}
    #    #    metrics = self._eval_metrics(name,output, target)
    #    #toc=timeit.default_timer()
    #    #print('metric: '+str(toc-tic))

    #    #perAnchor={}
    #    #for i in range(avg_conf_per_anchor.size(0)):
    #    #    perAnchor['anchor{}'.format(i)]=avg_conf_per_anchor[i]

    #    log = {
    #        'loss': loss,
    #        'boxLoss': boxLoss,
    #        'relLoss': relLoss,
    #        'edgePredLens':np.array([numEdgePred,numBoxPred,numEdgePred+numBoxPred,-1],dtype=np.float),
    #        'rel_recall':eRecall,
    #        #'rel_prec': ePrec,
    #        'rel_fullPrec':fullPrec,
    #        'rel_F': (eRecall+fullPrec)/2,
    #        #'debug_avg_relTrue': debug_avg_relTrue,
    #        #'debug_avg_relFalse': debug_avg_relFalse,

    #        **metrics,
    #    }
    #    if self.model.predNN:
    #        log['nn_loss_final'] = nn_loss_final
    #        if not self.model.detector_frozen:
    #            log['nn_loss_diff'] = nn_loss_final-nn_loss
    #    if self.model.predClass:
    #        log['class_loss_final'] = class_loss_final
    #        if not self.model.detector_frozen:
    #            log['class_loss_diff'] = class_loss_final-class_loss
    #    if ap is not None:
    #        log['rel_AP']=ap

    #    #if iteration%10==0:
    #    #image=None
    #    #queryMask=None
    #    #targetBoxes=None
    #    #outputBoxes=None
    #    #outputOffsets=None
    #    #loss=None
    #    #torch.cuda.empty_cache()


    #    return log#
    def _minor_log(self, log):
        ls=''
        for key,val in log.items():
            ls += key
            if type(val) is float:
                ls +=': {:.6f},\t'.format(val)
            else:
                ls +=': {},\t'.format(val)
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
        val_count = defaultdict(lambda: 0)


        with torch.no_grad():
            losses = defaultdict(lambda: 0)
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.model.detector.predNumNeighbors:
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
    #def _valid_epoch(self):
    #    """
    #    Validate after training an epoch

    #    :return: A log that contains information about validation

    #    Note:
    #        The validation metrics in log must have the key 'val_metrics'.
    #    """
    #    self.model.eval()
    #    total_val_loss = 0
    #    total_box_loss =0
    #    total_rel_loss =0
    #    total_rel_recall=0
    #    total_rel_prec=0
    #    total_rel_fullPrec=0
    #    total_AP=0
    #    AP_count=0
    #    total_val_metrics = np.zeros(len(self.metrics))
    #    nn_loss_final_total=0
    #    nn_acc_total=0
    #    nn_loss_diff_total=0
    #    class_loss_final_total=0
    #    class_loss_diff_total=0

    #    numClasses = self.model.numBBTypes
    #    if 'no_blanks' in self.config['validation'] and not self.config['data_loader']['no_blanks']:
    #        numClasses-=1
    #    mAP = 0
    #    mAP_count = 0
    #    mRecall = np.zeros(numClasses)
    #    mPrecision = np.zeros(numClasses)

    #    with torch.no_grad():
    #        losses = defaultdict(lambda: 0)
    #        for batch_idx, instance in enumerate(self.valid_data_loader):
    #            if not self.model.detector.predNumNeighbors:
    #                instance['num_neighbors']=None
    #            if not self.logged:
    #                print('iter:{} valid batch: {}/{}'.format(self.iteration,batch_idx,len(self.valid_data_loader)), end='\r')

    #            image, targetBoxes, adjM, target_num_neighbors = self._to_tensor(instance)

    #            outputBoxes, outputOffsets, relPred, relIndexes, bbPred, = self.model(image, hard_detect_limit=self.val_hard_detect_limit)
    #            #loss = self.loss(output, target)
    #            loss = 0
    #            index=0
    #            
    #            predPairingShouldBeTrue,predPairingShouldBeFalse, recall,prec,fullPrec, ap, bbAlignment, bbFullHit = self.alignEdgePred(targetBoxes,adjM,outputBoxes,relPred,relIndexes)
    #            total_rel_recall+=recall
    #            total_rel_prec+=prec
    #            total_rel_fullPrec+=fullPrec
    #            if ap is not None:
    #                total_AP+=ap
    #                AP_count+=1
    #            #relLoss = torch.tensor(0.0,requires_grad=True).to(image.device)
    #            relLoss=None
    #            if predPairingShouldBeTrue is not None and predPairingShouldBeTrue.size(0)>0:
    #                relLoss = self.loss['rel'](predPairingShouldBeTrue,torch.ones_like(predPairingShouldBeTrue).to(image.device))
    #            if predPairingShouldBeFalse is not None  and predPairingShouldBeFalse.size(0)>0:
    #                relFalseLoss = self.loss['rel'](predPairingShouldBeFalse,torch.zeros_like(predPairingShouldBeFalse).to(image.device))
    #                if relLoss is not None:
    #                    relLoss += relFalseLoss
    #                else:
    #                    relLoss = relFalseLoss
    #            if relLoss is None:
    #                relLoss = torch.tensor(0.0).to(image.device)
    #            #else:
    #            #    relLoss = relLoss.cpu()
    #            if not self.model.detector_frozen:
    #                boxLoss, position_loss, conf_loss, class_loss, nn_loss, recallX, precisionX = self.loss['box'](outputOffsets,targetBoxes,[targetBoxes.size(1)],target_num_neighbors)
    #                loss = relLoss*self.lossWeights['rel'] + boxLoss*self.lossWeights['box']
    #            else:
    #                boxLoss=torch.tensor(0.0)
    #                loss = relLoss*self.lossWeights['rel']
    #            total_box_loss+=boxLoss.item()
    #            total_rel_loss+=relLoss.item()

    #            if bbPred is not None:
    #                #create aligned GT
    #                #this was wrong...
    #                    #first, remove unmatched predicitons that didn't overlap (weren't close) to any targets
    #                    #toKeep = 1-((bbNoIntersections==1) * (bbAlignment==-1))
    #                #remove predictions that overlapped with GT, but not enough
    #                if self.model.predNN:
    #                    start=1
    #                    toKeep = 1-((bbFullHit==0) * (bbAlignment!=-1)) #toKeep = not (incomplete_overlap and did_overlap)
    #                    if toKeep.any():
    #                        bbPredNN_use = bbPred[toKeep][:,0]
    #                        bbAlignment_use = bbAlignment[toKeep]
    #                        #becuase we used -1 to indicate no match (in bbAlignment), we add 0 as the last position in the GT, as unmatched 
    #                        if target_num_neighbors is not None:
    #                            target_num_neighbors_use = torch.cat((target_num_neighbors[0].float(),torch.zeros(1).to(target_num_neighbors.device)),dim=0)
    #                        else:
    #                            target_num_neighbors_use = torch.zeros(1).to(bbPred.device)
    #                        alignedNN_use = target_num_neighbors_use[bbAlignment_use]
    #                    else:
    #                        bbAlignment_use=None
    #                        alignedNN_use=None
    #                else:
    #                    start=0
    #                if self.model.predClass:
    #                    #We really don't care about the class of non-overlapping instances
    #                    if targetBoxes is not None:
    #                        toKeep = bbFullHit==1
    #                        bbPredClass_use = bbPred[toKeep][:,:,start:start+self.model.numBBTypes]
    #                        bbAlignment_use = bbAlignment[toKeep]
    #                        alignedClass_use =  targetBoxes[0][bbAlignment_use][:,13:13+self.model.numBBTypes] #There should be no -1 indexes in hereS
    #                    else:
    #                        alignedClass_use = None
    #            else:
    #                bbPredNN_use = None
    #                bbPredClass_use = None

    #            if self.model.predNN and bbPredNN_use is not None and bbPredNN_use.size(0)>0:
    #                alignedNN_use = alignedNN_use[:,None] #introduce "time" dimension to broadcast
    #                nn_loss_final = self.loss['nn'](bbPredNN_use,alignedNN_use)
    #                nn_loss_final *= self.lossWeights['nn']

    #                loss += nn_loss_final.to(loss.device)
    #                nn_loss_final = nn_loss_final.item()
    #            else:
    #                nn_loss_final=0
    #            nn_loss_final_total += nn_loss_final
    #            nn_acc=-1
    #            if self.model.predNN and bbPred is not None:
    #                predNN_p=bbPred[:,-1,0]
    #                diffs=torch.abs(predNN_p-target_num_neighbors[0][bbAlignment].float())
    #                nn_acc = (diffs<0.5).float().mean().item()
    #            nn_acc_total += nn_acc

    #            if self.model.predClass and bbPredClass_use is not None and bbPredClass_use.size(0)>0:
    #                alignedClass_use = alignedClass_use[:,None] #introduce "time" dimension to broadcast
    #                class_loss_final = self.loss['class'](bbPredClass_use,alignedClass_use)
    #                class_loss_final *= self.lossWeights['class']
    #                loss += class_loss_final
    #                class_loss_final = class_loss_final.item()
    #            else:
    #                class_loss_final = 0
    #            class_loss_final_total += class_loss_final

    #            if not self.model.detector_frozen:
    #                nn_loss_diff_total += nn_loss_final-nn_loss
    #                class_loss_diff_total += class_loss_final-class_loss
    #            
    #            if self.model.detector.predNumNeighbors and outputBoxes.size(0)>0:
    #                outputBoxes=torch.cat((outputBoxes[:,0:6],outputBoxes[:,7:]),dim=1) #throw away NN pred
    #            if targetBoxes is not None:
    #                targetBoxes = targetBoxes.cpu()
    #            if targetBoxes is not None:
    #                target_for_b = targetBoxes[0]
    #            else:
    #                target_for_b = torch.empty(0)
    #            if self.model.rotation:
    #                ap_5, prec_5, recall_5 =AP_dist(target_for_b,outputBoxes,0.9,numClasses)
    #            else:
    #                ap_5, prec_5, recall_5 =AP_iou(target_for_b,outputBoxes,0.5,numClasses)

    #            #import pdb;pdb.set_trace()
    #            if ap_5 is not None:
    #                mAP+=ap_5
    #                mAP_count+=1
    #            mRecall += np.array(recall_5)
    #            mPrecision += np.array(prec_5)

    #            total_val_loss += loss.item()
    #            loss=relFalseLoss=relLoss=boxLoss=None
    #            instance=predPairingShouldBeTrue= predPairingShouldBeFalse=outputBoxes=outputOffsets=relPred=image=targetBoxes=relLossFalse=None
    #            #total_val_metrics += self._eval_metrics(output, target)
    #    if mAP_count==0:
    #        mAP_count=1
    #    total_rel_prec/=len(self.valid_data_loader)
    #    total_rel_recall/=len(self.valid_data_loader)
    #    mRecall/=len(self.valid_data_loader)
    #    mPrecision/=len(self.valid_data_loader)

    #    toRet= {
    #        'val_loss': total_val_loss / len(self.valid_data_loader),
    #        'val_box_loss': total_box_loss / len(self.valid_data_loader),
    #        'val_rel_loss': total_rel_loss / len(self.valid_data_loader),
    #        'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist(),
    #        'val_bb_recall':(mRecall).tolist(),
    #        'val_bb_precision':(mPrecision).tolist(),
    #        #'val_bb_F':(( (mRecall+mPrecision)/2 )/len(self.valid_data_loader)).tolist(),
    #        'val_bb_F_avg':( 2*(mRecall*mPrecision)/(mRecall+mPrecision) ).mean(),
    #        'val_bb_mAP':(mAP/mAP_count),
    #        'val_rel_recall':total_rel_recall,
    #        'val_rel_prec':total_rel_prec,
    #        'val_rel_F':2*(total_rel_prec*total_rel_recall)/(total_rel_prec+total_rel_recall),
    #        'val_rel_fullPrec':total_rel_fullPrec/len(self.valid_data_loader),
    #        'val_rel_mAP': total_AP/AP_count
    #        #'val_position_loss':total_position_loss / len(self.valid_data_loader),
    #        #'val_conf_loss':total_conf_loss / len(self.valid_data_loader),
    #        #'val_class_loss':tota_class_loss / len(self.valid_data_loader),
    #    }
    #    if self.model.predNN:
    #        toRet['val_nn_loss_final']=nn_loss_final_total/len(self.valid_data_loader)
    #        toRet['val_nn_loss_diff']=nn_loss_diff_total/len(self.valid_data_loader)
    #        toRet['val_nn_acc'] = nn_acc_total/len(self.valid_data_loader)
    #    if self.model.predClass:
    #        toRet['val_class_loss_final']=class_loss_final_total/len(self.valid_data_loader)
    #        toRet['val_class_loss_diff']=class_loss_diff_total/len(self.valid_data_loader)
    #    return toRet


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

        if self.model.rotation:
            targIndex, fullHit = getTargIndexForPreds_dist(targetBoxes[0],outputBoxes,1.1,numClasses,hard_thresh=False)
        else:
            targIndex, fullHit = getTargIndexForPreds_iou(targetBoxes[0],outputBoxes,0.4,numClasses,hard_thresh=False,fixed=self.fixedAlign)
        #else:
        #    if self.model.rotation:
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


    def newAlignEdgePred(self,targetBoxes,adj,gtGroups,gtGroupAdj,outputBoxes,edgePred,edgeIndexes,predGroups,rel_prop_pred):
        if edgePred is None:
            if targetBoxes is None:
                prec=1
                ap=1
                recall=1
                targIndex = -torch.ones(outputBoxes.size(0)).int()
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
                'FmError' : Fm,
                }
            
            #return torch.tensor([]),torch.tensor([]), targIndex, torch.ones(outputBoxes.size(0)), None, log
            predsGTYes = torch.tensor([])
            predsGTNo = torch.tensor([])
            fullHit = None
            matches=0
            predTypes = None
        else:

            #decide which predicted boxes belong to which target boxes
            #should this be the same as AP_?
            numClasses = 2
            
            if targetBoxes is not None:
                targetBoxes = targetBoxes.cpu()
                if self.model.rotation:
                    targIndex, fullHit, overSegmented = newGetTargIndexForPreds_dist(targetBoxes[0],outputBoxes,1.1,numClasses,hard_thresh=False)
                else:
                    targIndex, fullHit, overSegmented = newGetTargIndexForPreds_iou(targetBoxes[0],outputBoxes,0.4,numClasses,hard_thresh=False,fixed=self.fixedAlign)
            else:
                targIndex=torch.LongTensor(num_pred_bbs).fill_(-1)
                fullHit=torch.BoolTensor(num_pred_bbs).false_()
                overSegmented=torch.BoolTensor(num_pred_bbs).false_()



            #Create gt vector to match edgePred.values()
            num_internal_iters = edgePred.size(-2)
            predsRel = edgePred[...,0] 
            predsOverSeg = edgePred[...,1] 
            predsGroup = edgePred[...,2] 
            predsError = edgePred[...,3] 
            sigPredsAll = torch.sigmoid(predsRel[:,-1])
            predsGTRel = []
            predsGTNoRel = []
            predsGTOverSeg = []
            predsGTNotOverSeg = []
            predsGTGroup = []
            predsGTNoGroup = []
            predsGTNoError = []
            predsGTError = []

            scores = []
            matches=0
            badPred=0
            goodEdge=0
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
                predGroupsT[node] = [targIndex[bb].item() for bb in predGroups[node] if targIndex[bb].item()>=0 and (fullHit[bb] or overSegmented[bb])]
                predGroupsTNear[node] = [targIndex[bb].item() for bb in predGroups[node] if targIndex[bb].item()>=0 and not (fullHit[bb] or overSegmented[bb])]

            for i,(n0,n1) in enumerate(edgeIndexes):
                wasRel=None
                wasOverSeg=None
                wasError=None
                wasGroup=None
                ts0=predGroupsT[n0]
                ts1=predGroupsT[n1]
                ts0near=predGroupsT[n0]
                ts1near=predGroupsT[n1]
                #t1 = targIndexes[n1].item()
                gtGroup0 = getGTGroup(ts0,gtGroups)
                gtGroup1 = getGTGroup(ts1,gtGroups)
                if gtGroup1==-1:
                    gtGroup1=-2 #so gtGroup0!=gtGroup1
                if len(ts0+ts0near)>0 and len(ts1+ts1near)>0:
                    #if len(ts0)==1 and overSegmented[ts0[0]] and len(ts1)==1 and overSegmented[ts1[0]]:S
                    if len(predGroups[n0])==1 and overSegmented[predGroups[n0][0]] and len(predGroups[n1])==1 and overSegmented[predGroups[n1][0]] and ts0[0]==ts1[0]:
                        wasOverSeg=True
                        wasError=False
                    else:
                        wasOverSeg=False
                        gtLinked=False
                        gtLinkedNear=False
                    
                        #if not( (len(ts0)==1 and overSegmented[ts0[0]]) or (len(ts1)==1 and overSegmented[ts1[0]]) ):
                        if not (len(predGroups[n0])==1 and overSegmented[predGroups[n0][0]]) and not (len(predGroups[n1])==1 and overSegmented[predGroups[n1][0]]):
                            for t0 in ts0:
                                for t1 in ts1:
                                    if (min(t0,t1),max(t0,t1)) in adj:
                                        gtLinked=True
                                        break
                                if gtLinked:
                                    break
                        #else we default to a "near" link, don't suppress and don't support

                        gtLinkedNear=gtLinked
                        for t0 in ts0+ts0near:
                            if gtLinkedNear:
                                break
                            for t1 in ts1+ts1near:
                                if (min(t0,t1),max(t0,t1)) in adj:
                                    gtLinkedNear=True
                                    break
                
                        if gtLinked:
                            if pure(ts0,gtGroups) and pure(ts1,gtGroups):
                                matches+=1
                                wasRel=True
                                wasError=False
                                matches+=1
                                scores.append( (sigPredsAll[i],True) )

                                if gtGroup0==gtGroup1:
                                    wasGroup=True
                                else:
                                    wasGroup=False
                            else:
                                wasError=True
                                scores.append( (sigPredsAll[i],False) ) #for the sake of scoring, this is a bad relationship
                                if gtGroup0!=gtGroup1:
                                    wasGroup=False

                        elif gtLinkedNear:
                            scores.append( (sigPredsAll[i],False) ) #for the sake of scoring, this is a bad relationship
                            if gtGroup0==gtGroup1 and pure(ts0,gtGroups) and pure(ts1,gtGroups):
                                wasGroup=True
                        else:
                            if ((min(gtGroup0,gtGroup1),max(gtGroup0,gtGroup1)) in gtGroupAdj) or gtGroup0==gtGroup1:
                                if gtGroup0==gtGroup1 and pure(ts0,gtGroups) and pure(ts1,gtGroups):
                                    wasGroup=True
                                    wasError=False
                                elif gtGroup0!=gtGroup1:
                                    wasGroup=False
                                    wasError=False
                            else:
                                wasRel=False
                                wasError=False
                                wasGroup=False
                                scores.append( (sigPredsAll[i],False) )

                else:
                    #if self.useBadBBPredForRelLoss=='fixed' or (self.useBadBBPredForRelLoss and (predsWithNoIntersection[n0] or predsWithNoIntersection[n1])):
                    if self.useBadBBPredForRelLoss=='full' or np.random.rand()<self.useBadBBPredForRelLoss:
                        wasRel=False
                        #predsGTNoRel.append(predsRel[i])
                    elif sigPredsAll[i]>self.thresh_rel:
                        badPred+=1
                    wasError=True
                    scores.append( (sigPredsAll[i],False) )



                if wasRel is not None:
                    if wasRel:
                        predsGTRel.append(predsRel[i])
                    else:
                        predsGTNoRel.append(predsRel[i])
                    if torch.sigmoid(predsRel[i])>self.thresh_rel:
                        if wasRel:
                            truePosRel+=1
                            saveRelPred[i]='TP'
                        else:
                            falsePosRel+=1
                            saveRelPred[i]='FP'
                    else:
                        if wasRel:
                            falseNegRel+=1
                            saveRelPred[i]='FN'
                        else:
                            trueNegRel+=1
                            saveRelPred[i]='TN'
                else:
                    if torch.sigmoid(predsRel[i])>self.thresh_rel:
                        saveRelPred[i]='UP'
                    else:
                        saveRelPred[i]='UN'
                if wasOverSeg is not None:
                    if wasOverSeg:
                        predsGTOverSeg.append(predsOverSeg[i])
                    else:
                        predsGTNotOverSeg.append(predsOverSeg[i])
                    if torch.sigmoid(predsOverSeg[i])>self.thresh_overSeg:
                        if wasOverSeg:
                            truePosOverSeg+=1
                            saveOverSegPred[i]='TP'
                        else:
                            falsePosOverSeg+=1
                            saveOverSegPred[i]='FP'
                    else:
                        if wasOverSeg:
                            falseNegOverSeg+=1
                            saveOverSegPred[i]='FN'
                        else:
                            trueNegOverSeg+=1
                            saveOverSegPred[i]='TN'
                else:
                    if torch.sigmoid(predsOverSeg[i])>self.thresh_overSeg:
                        saveOverSegPred[i]='UP'
                    else:
                        saveOverSegPred[i]='UN'
                if wasGroup is not None:
                    if wasGroup:
                        predsGTGroup.append(predsGroup[i])
                    else:
                        predsGTNoGroup.append(predsGroup[i])
                    if torch.sigmoid(predsGroup[i])>self.thresh_group:
                        if wasGroup:
                            truePosGroup+=1
                            saveGroupPred[i]='TP'
                            successfulEdge=True
                        else:
                            falsePosGroup+=1
                            saveGroupPred[i]='FP'
                    else:
                        if wasGroup:
                            falseNegGroup+=1
                            saveGroupPred[i]='FN'
                        else:
                            trueNegGroup+=1
                            saveGroupPred[i]='TN'
                else:
                    if torch.sigmoid(predsGroup[i])>self.thresh_group:
                        saveGroupPred[i]='UP'
                    else:
                        saveGroupPred[i]='UN'
                if wasError is not None:
                    if wasError:
                        predsGTError.append(predsError[i])
                    else:
                        predsGTNoError.append(predsError[i])
                    if torch.sigmoid(predsError[i])>self.thresh_error:
                        if wasError:
                            truePosError+=1
                            saveErrorPred[i]='TP'
                        else:
                            falsePosError+=1
                            saveErrorPred[i]='FP'
                    else:
                        if wasError:
                            falseNegError+=1
                            saveErrorPred[i]='FN'
                        else:
                            trueNegError+=1
                            saveErrorPred[i]='TN'
                else:
                    if torch.sigmoid(predsError[i])>self.thresh_error:
                        saveErrorPred[i]='UP'
                    else:
                        saveErrorPred[i]='UN'



            for i in range(len(adj)-matches):
                scores.append( (float('nan'),True) )
        
            #stack all label divisions into tensors
            if len(predsGTRel)>0:
                predsGTRel = torch.stack(predsGTRel,dim=1).to(edgePred.device)
            else:
                predsGTRel = torch.FloatTensor(num_internal_iters,0).to(edgePred.device)
            if len(predsGTNoRel)>0:
                predsGTNoRel = torch.stack(predsGTNoRel,dim=1).to(edgePred.device)
            else:
                predsGTNoRel = torch.FloatTensor(num_internal_iters,0).to(edgePred.device)
            if len(predsGTOverSeg)>0:
                predsGTOverSeg = torch.stack(predsGTOverSeg,dim=1).to(edgePred.device)
            else:
                predsGTOverSeg = torch.FloatTensor(num_internal_iters,0).to(edgePred.device)
            if len(predsGTNotOverSeg)>0:
                predsGTNotOverSeg = torch.stack(predsGTNotOverSeg,dim=1).to(edgePred.device)
            else:
                predsGTNotOverSeg = torch.FloatTensor(num_internal_iters,0).to(edgePred.device)
            if len(predsGTGroup)>0:
                predsGTGroup = torch.stack(predsGTGroup,dim=1).to(edgePred.device)
            else:
                predsGTGroup = torch.FloatTensor(num_internal_iters,0).to(edgePred.device)
            if len(predsGTNoGroup)>0:
                predsGTNoGroup = torch.stack(predsGTNoGroup,dim=1).to(edgePred.device)
            else:
                predsGTNoGroup = torch.FloatTensor(num_internal_iters,0).to(edgePred.device)
            if len(predsGTError)>0:
                predsGTError = torch.stack(predsGTError,dim=1).to(edgePred.device)
            else:
                predsGTError = torch.FloatTensor(num_internal_iters,0).to(edgePred.device)
            if len(predsGTNoError)>0:
                predsGTNoError = torch.stack(predsGTNoError,dim=1).to(edgePred.device)
            else:
                predsGTNoError = torch.FloatTensor(num_internal_iters,0).to(edgePred.device)

            predsGTYes = torch.cat((predsGTRel,predsGTOverSeg,predsGTGroup,predsGTError),dim=1)
            predsGTNo = torch.cat((predsGTNoRel,predsGTNotOverSeg,predsGTNoGroup,predsGTNoError),dim=1)

            recallRel = truePosRel/(truePosRel+falseNegRel) if truePosRel+falseNegRel>0 else 1
            precRel = truePosRel/(truePosRel+falsePosRel) if truePosRel+falsePosRel>0 else 1
            recallOverSeg = truePosOverSeg/(truePosOverSeg+falseNegOverSeg) if truePosOverSeg+falseNegOverSeg>0 else 1
            precOverSeg = truePosOverSeg/(truePosOverSeg+falsePosOverSeg) if truePosOverSeg+falsePosOverSeg>0 else 1
            recallGroup = truePosGroup/(truePosGroup+falseNegGroup) if truePosGroup+falseNegGroup>0 else 1
            precGroup = truePosGroup/(truePosGroup+falsePosGroup) if truePosGroup+falsePosGroup>0 else 1
            recallError = truePosError/(truePosError+falseNegError) if truePosError+falseNegError>0 else 1
            precError = truePosError/(truePosError+falsePosError) if truePosError+falsePosError>0 else 1
            log = {
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
            predTypes = [saveRelPred,saveOverSegPred,saveGroupPred,saveErrorPred]

        if rel_prop_pred is not None:
            if len(adj)>0:
                final_prop_rel_recall = matches/len(adj)
            else:
                final_prop_rel_recall = 1
            if len(edgeIndexes)>0:
                final_prop_rel_prec = matches/len(edgeIndexes)
            else:
                final_prop_rel_prec = 1
            log['final_prop_rel_recall']=final_prop_rel_recall
            log['final_prop_rel_prec']=final_prop_rel_prec

            relPropScores,relPropIds, threshPropRel = rel_prop_pred
            truePropPred=falsePropPred=badPropPred=0
            propPredsPos=[]
            propPredsNeg=[]
            for i,(n0,n1) in enumerate(relPropIds):
                t0 = targIndex[n0].item()
                t1 = targIndex[n1].item()
                ts0=predGroupsT[n0]
                ts1=predGroupsT[n1]
                gtGroup0 = getGTGroup(ts0,gtGroups)
                gtGroup1 = getGTGroup(ts1,gtGroups)
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
                    elif gtGroup0!=gtGroup1:
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
                propPredsPos = torch.stack(propPredsPos).to(relPropScores.device)
            else:
                propPredsPos = None
            if len(propPredsNeg)>0:
                propPredsNeg = torch.stack(propPredsNeg).to(relPropScores.device)
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
            log['propRecall']=propRecall
            log['propFullPrec']=propFullPrec

            proposedInfo = (propPredsPos,propPredsNeg, propRecall, propFullPrec)
        else:
            proposedInfo = None

        return predsGTYes, predsGTNo, targIndex, fullHit, proposedInfo, log, predTypes


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


    def run(self,instance,useGT,threshIntur=None,get=[]):
        numClasses = self.model.numBBTypes
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
                if self.model.predNN or self.model.predClass:
                    if target_num_neighbors is not None:
                        alignedNN_use = target_num_neighbors[0]
                    bbPredNN_use = bbPred[:,:,0]
                    start=1
                else:
                    start=0
                if self.model.predClass:
                    if targetBoxes is not None:
                        alignedClass_use =  targetBoxes[0,:,13:13+self.model.numBBTypes]
                    bbPredClass_use = bbPred[:,:,start:start+self.model.numBBTypes]
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
                if self.model.predNN:
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
                if self.model.predClass:
                    #We really don't care about the class of non-overlapping instances
                    if targetBoxes is not None:
                        toKeep = bbFullHit==1
                        if toKeep.any():
                            bbPredClass_use = bbPred[toKeep][:,:,start:start+self.model.numBBTypes]
                            bbAlignment_use = bbAlignment[toKeep]
                            alignedClass_use =  targetBoxes[0][bbAlignment_use.long()][:,13:13+self.model.numBBTypes] #There should be no -1 indexes in hereS
                        else:
                            alignedClass_use = None
                            bbPredClass_use = None
                    else:
                        alignedClass_use = None
                        bbPredClass_use = None
            else:
                bbPredNN_use = None
                bbPredClass_use = None
        if relPred is not None:
            numEdgePred = relPred.size(0)
            if predPairingShouldBeTrue is not None:
                lenTrue = predPairingShouldBeTrue.size(0)
            else:
                lenTrue = 0
            if predPairingShouldBeFalse is not None:
                lenFalse = predPairingShouldBeFalse.size(0)
            else:
                lenFalse = 0
        else:
            numEdgePred = lenTrue = lenFalse = 0
        numBoxPred = outputBoxes.size(0)
        #if iteration>25:
        #    import pdb;pdb.set_trace()
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



        if not self.model.detector_frozen:
            if targetBoxes is not None:
                targSize = targetBoxes.size(1)
            else:
                targSize =0 
            #import pdb;pdb.set_trace()
            boxLoss, position_loss, conf_loss, class_loss, nn_loss, recall, precision = self.loss['box'](outputOffsets,targetBoxes,[targSize],target_num_neighbors)
            losses['boxLoss'] = boxLoss
            #boxLoss *= self.lossWeights['box']
            #if relLoss is not None:
            #    loss = relLoss + boxLoss
            #else:
            #    loss = boxLoss
        #else:
        #    loss = relLoss


        if self.model.predNN and bbPredNN_use is not None and bbPredNN_use.size(0)>0:
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

        if self.model.predClass and bbPredClass_use is not None and bbPredClass_use.size(0)>0:
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
        if not self.model.detector_frozen:
            if 'nnFinalLoss' in losses:
                log['nn loss improvement (neg is good)'] = losses['nnFinalLoss'].item()-nn_loss
            if 'classFinalLoss' in losses:
                log['class loss improvement (neg is good)'] = losses['classFinalLoss'].item()-class_loss

        if 'bb_stats' in get:
            outputBoxes=torch.cat((outputBoxes[:,0:6],outputBoxes[:,7:]),dim=1) #throw away NN pred
            if targetBoxes is not None:
                targetBoxes = targetBoxes.cpu()
            if targetBoxes is not None:
                target_for_b = targetBoxes[0]
            else:
                target_for_b = torch.empty(0)
            if self.model.rotation:
                ap_5, prec_5, recall_5 =AP_dist(target_for_b,outputBoxes,0.9,numClasses)
            else:
                ap_5, prec_5, recall_5 =AP_iou(target_for_b,outputBoxes,0.5,numClasses)
            prec_5 = np.array(prec_5)
            recall_5 = np.array(recall_5)
            log['bb_AP']=ap_5
            log['bb_prec']=prec_5
            log['bb_recall']=recall_5
            log['bb_Fm_avg']=(2*(prec_5*recall_5)/(prec_5+recall_5)).mean()

        if 'nn_acc' in get:
            if self.model.predNN and bbPred is not None:
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
        numClasses = len(self.data_loader.dataset.classMap)
        image, targetBoxes, adj, target_num_neighbors = self._to_tensor(instance)
        gtGroups = instance['gt_groups']
        gtGroupAdj = instance['gt_groups_adj']
        if useGT:
            allOutputBoxes, outputOffsets, allEdgePred, allEdgeIndexes, allNodePred, allPredGroups, rel_prop_pred = self.model(
                                    image,
                                    targetBoxes,
                                    target_num_neighbors,
                                    True,
                                    otherThresh=self.conf_thresh_init, 
                                    otherThreshIntur=threshIntur, 
                                    hard_detect_limit=self.train_hard_detect_limit)
            #TODO
            #predPairingShouldBeTrue,predPairingShouldBeFalse, eRecall,ePrec,fullPrec,ap,proposedInfo = self.prealignedEdgePred(adj,relPred,relIndexes,rel_prop_pred)
            #if bbPred is not None:
            #    if self.model.predNN or self.model.predClass:
            #        if target_num_neighbors is not None:
            #            alignedNN_use = target_num_neighbors[0]
            #        bbPredNN_use = bbPred[:,:,0]
            #        start=1
            #    else:
            #        start=0
            #    if self.model.predClass:
            #        if targetBoxes is not None:
            #            alignedClass_use =  targetBoxes[0,:,13:13+self.model.numBBTypes]
            #        bbPredClass_use = bbPred[:,:,start:start+self.model.numBBTypes]
            #else:
            #    bbPredNN_use=None
            #    bbPredClass_use=None
            #final_prop_rel_recall = final_prop_rel_prec = None
        else:
            #outputBoxes, outputOffsets: one, predicted at the begining
            #relPred, relIndexes, bbPred, predGroups: multiple, for each step in graph prediction. relIndexes indexes into predGroups, which indexes to outputBoxes
            #rel_prop_pred: if we use prop, one for begining
            allOutputBoxes, outputOffsets, allEdgePred, allEdgeIndexes, allNodePred, allPredGroups, rel_prop_pred = self.model(image,
                    otherThresh=self.conf_thresh_init, otherThreshIntur=threshIntur, hard_detect_limit=self.train_hard_detect_limit)
            #gtPairing,predPairing = self.alignEdgePred(targetBoxes,adj,outputBoxes,relPred)
        ### TODO code prealigned
        losses=defaultdict(lambda:0)
        log={}
        #for graphIteration in range(len(allEdgePred)):
        allEdgePredTypes=[]
        for graphIteration,(outputBoxes,edgePred,nodePred,edgeIndexes,predGroups) in enumerate(zip(allOutputBoxes,allEdgePred,allNodePred,allEdgeIndexes,allPredGroups)):
            #edgePred=allEdgePred[graphIteration]
            #nodePred=allNodePred[graphIteration]
            #edgeIndexes=allEdgeIndexes[graphIteration]
            #predGroups=allPredGroups[graphIteration]

            predEdgeShouldBeTrue,predEdgeShouldBeFalse, bbAlignment, bbFullHit, proposedInfo, logIter, edgePredTypes = self.newAlignEdgePred(targetBoxes,adj,gtGroups,gtGroupAdj,outputBoxes,edgePred,edgeIndexes,predGroups, rel_prop_pred if graphIteration==0 else None)
            allEdgePredTypes.append(edgePredTypes)
            #create aligned GT
            #this was wrong...
                #first, remove unmatched predicitons that didn't overlap (weren't close) to any targets
                #toKeep = 1-((bbNoIntersections==1) * (bbAlignment==-1))
            #remove predictions that overlapped with GT, but not enough
            #toKeep = 1-((bbFullHit==0) * (bbAlignment!=-1)) #toKeep = not (incomplete_overlap and did_overlap)
            #bbAlignment_use = bbAlignment[toKeep]
            assert(not self.model.predNN)
            if self.model.predClass:
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

                node_pred_use_index += node_pred_use_index_sp

                if len(node_pred_use_index)>0:
                    nodePredClass_use = nodePred[node_pred_use_index][:,:,self.model.nodeIdxClass:self.model.nodeIdxClassEnd]
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
                    nodePredConf_use = nodePred[node_conf_use_index][:,:,self.model.nodeIdxConf]
                    nodeGTConf_use = torch.FloatTensor(node_conf_gt).to(nodePred.device)

            ####


            if edgePred is not None:
                numEdgePred = edgePred.size(0)
                if predEdgeShouldBeTrue is not None:
                    lenTrue = predEdgeShouldBeTrue.size(0)
                else:
                    lenTrue = 0
                if predEdgeShouldBeFalse is not None:
                    lenFalse = predEdgeShouldBeFalse.size(0)
                else:
                    lenFalse = 0
            else:
                numEdgePred = lenTrue = lenFalse = 0

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
                    losses['propRelLoss']+=propRelLoss



            
            #Fine tuning detector. Should only happed once
            if not self.model.detector_frozen and graphIteration==0:
                if targetBoxes is not None:
                    targSize = targetBoxes.size(1)
                else:
                    targSize =0 
                #import pdb;pdb.set_trace()
                boxLoss, position_loss, conf_loss, class_loss, nn_loss, recall, precision = self.loss['box'](outputOffsets,targetBoxes,[targSize],target_num_neighbors)
                losses['boxLoss'] += boxLoss
                logIter['bb_position_loss'] = position_loss
                logIter['bb_conf_loss'] = conf_loss
                logIter['bb_class_loss'] = class_loss
                logIter['bb_nn_loss'] = nn_loss

                #boxLoss *= self.lossWeights['box']
                #if relLoss is not None:
                #    loss = relLoss + boxLoss
                #else:
                #    loss = boxLoss
            #else:
            #    loss = relLoss


            if self.model.predNN and nodePredNN_use is not None and nodePredNN_use.size(0)>0:
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

            if self.model.predClass and nodePredClass_use is not None and nodePredClass_use.size(0)>0:
                alignedClass_use = alignedClass_use[:,None] #introduce "time" dimension to broadcast
                class_loss_final = self.loss['classFinal'](nodePredClass_use,alignedClass_use)
                losses['classFinalLoss'] += class_loss_final

            if nodePredConf_use is not None and nodePredConf_use.size(0)>0:
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
                draw_graph(outputBoxes,self.model.used_threshConf,torch.sigmoid(nodePred).cpu().detach(),torch.sigmoid(edgePred).cpu().detach(),edgeIndexes,predGroups,image,edgePredTypes,targetBoxes,self.model,path)
                print('saved {}'.format(path))

            if 'bb_stats' in get:

                if self.model.detector.predNumNeighbors:
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
                if self.model.rotation:
                    ap_5, prec_5, recall_5 =AP_dist(target_for_b,outputBoxes,0.9,numClasses, beforeCls=beforeCls)
                else:
                    ap_5, prec_5, recall_5 =AP_iou(target_for_b,outputBoxes,0.5,numClasses, beforeCls=beforeCls)
                prec_5 = np.array(prec_5)
                recall_5 = np.array(recall_5)
                log['bb_AP_{}'.format(graphIteration)]=ap_5
                log['bb_prec_{}'.format(graphIteration)]=prec_5
                log['bb_recall_{}'.format(graphIteration)]=recall_5
                log['bb_Fm_avg_{}'.format(graphIteration)]=(2*(prec_5*recall_5)/(prec_5+recall_5)).mean()
        
        #log['rel_prec']= fullPrec
        #log['rel_recall']= eRecall
        #log['rel_Fm']= 2*(fullPrec*eRecall)/(eRecall+fullPrec) if eRecall+fullPrec>0 else 0

        if not self.model.detector_frozen:
            if 'nnFinalLoss' in losses:
                log['nn loss improvement (neg is good)'] = losses['nnFinalLoss'].item()-nn_loss
            if 'classFinalLoss' in losses:
                log['class loss improvement (neg is good)'] = losses['classFinalLoss'].item()-class_loss


        if 'nn_acc' in get:
            if self.model.predNN and bbPred is not None:
                predNN_p=bbPred[:,-1,0]
                diffs=torch.abs(predNN_p-target_num_neighbors[0][bbAlignment].float())
                nn_acc = (diffs<0.5).float().mean().item()
                log['nn_acc']=nn_acc

        if proposedInfo is not None:
            propRecall,propPrec = proposedInfo[2:4]
            log['prop_rel_recall'] = propRecall
            log['prop_rel_prec'] = propPrec
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
                 got[name] = [n.detach().cpu() for n in allNodePred]
            elif name=='allEdgePred':
                 got[name] = [n.detach().cpu() for n in allEdgePred]
            elif name=='allEdgeIndexes':
                 got[name] = allEdgeIndexes
            elif name=='allPredGroups':
                 got[name] = allPredGroups
            elif name=='allOutputBoxes':
                 got[name] = allOutputBoxes
            elif name=='allEdgePredTypes':
                 got[name] = allEdgePredTypes
            elif name != 'bb_stats' and name != 'nn_acc':
                raise NotImplementedError('Cannot get [{}], unknown'.format(name))
        return losses, log, got
