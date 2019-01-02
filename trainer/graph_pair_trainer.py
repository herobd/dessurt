import numpy as np
import torch
from base import BaseTrainer
import timeit
from utils import util
from collections import defaultdict
from evaluators import FormsBoxDetect_printer
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist, getTargIndexForPreds_iou, getTargIndexForPreds_dist
from datasets.testforms_graph_pair import display
import random


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
        self.lossWeights = config['loss_weights'] if 'loss_weights' in config else {"box": 1, "edge":1}
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
        warmup_steps = config['trainer']['warmup_steps'] if 'warmup_steps' in config['trainer'] else 1000
        lr_lambda = lambda step_num: min((step_num+1)**-0.3, (step_num+1)*warmup_steps**-1.3)
        self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)

        #default is unfrozen, can be frozen by setting 'start_froze' in the PairingGraph models params
        self.unfreeze_detector = config['trainer']['unfreeze_detector'] if 'unfreeze_detector' in config['trainer'] else None

        self.thresh_conf = config['trainer']['thresh_conf'] if 'thresh_conf' in config['trainer'] else 0.92
        self.thresh_intersect = config['trainer']['thresh_intersect'] if 'thresh_intersect' in config['trainer'] else 0.4

        #we iniailly train the pairing using GT BBs, but eventually need to fine-tune the pairing using the networks performance
        self.stop_from_gt = config['trainer']['stop_from_gt'] if 'stop_from_gt' in config['trainer'] else None
        self.partial_from_gt = config['trainer']['partial_from_gt'] if 'partial_from_gt' in config['trainer'] else None

        self.conf_thresh_init = config['trainer']['conf_thresh_init'] if 'conf_thresh_init' in config['trainer'] else 0.9
        self.conf_thresh_change_iters = config['trainer']['conf_thresh_change_iters'] if 'conf_thresh_change_iters' in config['trainer'] else 5000

        self.train_hard_detect_limit = config['train_hard_detect_limit'] if 'train_hard_detect_limit' in config else 100
        self.val_hard_detect_limit = config['val_hard_detect_limit'] if 'val_hard_detect_limit' in config else 300

    def _to_tensor(self, instance):
        image = instance['img']
        bbs = instance['bb_gt']
        adjaceny = instance['adj']

        if self.with_cuda:
            image = image.to(self.gpu)
            if bbs is not None:
                bbs = bbs.to(self.gpu)
            #adjacenyMatrix = adjacenyMatrix.to(self.gpu)
        return image, bbs, adjaceny

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
            return False
        elif self.partial_from_gt is not None and iteration>=self.partial_from_gt:
            return random.random()>0.5
        else:
            return True

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
        #self.lr_schedule.step()

        ##tic=timeit.default_timer()
        batch_idx = (iteration-1) % len(self.data_loader)
        try:
            thisInstance = self.data_loader_iter.next()
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            thisInstance = self.data_loader_iter.next()
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
        image, targetBoxes, adj = self._to_tensor(thisInstance)
        if self.useGT(iteration):
            outputBoxes, outputOffsets, edgePred = self.model(image,targetBoxes, 
                    otherThresh=self.conf_thresh_init, otherThreshIntur=threshIntur, hard_detect_limit=self.train_hard_detect_limit)
            #_=None
            gtPairing,predPairing = self.prealignedEdgePred(adj,edgePred)
        else:
            outputBoxes, outputOffsets, edgePred = self.model(image,
                    otherThresh=self.conf_thresh_init, otherThreshIntur=threshIntur, hard_detect_limit=self.train_hard_detect_limit)
            gtPairing,predPairing = self.alignEdgePred(targetBoxes,adj,outputBoxes,edgePred)
        if len(predPairing.size())>0 and predPairing.size(0)>0:
            edgeLoss = self.loss['edge'](predPairing,gtPairing)
        else:
            edgeLoss = torch.tensor(0.0,requires_grad=True).to(image.device)
        if not self.model.detector_frozen:
            if targetBoxes is not None:
                targSize = targetBoxes.size(1)
            else:
                targSize =0 
            #import pdb;pdb.set_trace()
            boxLoss, position_loss, conf_loss, class_loss, recall, precision = self.loss['box'](outputOffsets,targetBoxes,[targSize])
            loss = edgeLoss*self.lossWeights['edge'] + boxLoss*self.lossWeights['box']
        else:
            loss = edgeLoss


        ##toc=timeit.default_timer()
        ##print('loss: '+str(toc-tic))
        ##tic=timeit.default_timer()
        gtPairing=predPairing=outputBoxes=outputOffsets=edgePred=image=targetBoxes=thisInstance=None
        edgeLoss = edgeLoss.item()
        if not self.model.detector_frozen:
            boxLoss = boxLoss.item()
        else:
            boxLoss = 0
        loss.backward()
        #what is grads?
        #minGrad=9999999999
        #maxGrad=-9999999999
        #for p in filter(lambda p: p.grad is not None, self.model.parameters()):
        #    minGrad = min(minGrad,p.min())
        #    maxGrad = max(maxGrad,p.max())
        #import pdb; pdb.set_trace()
        torch.nn.utils.clip_grad_value_(self.model.parameters(),1)
        self.optimizer.step()

        loss = loss.item()

        ##toc=timeit.default_timer()
        ##print('bac: '+str(toc-tic))

        #tic=timeit.default_timer()
        metrics={}
        #index=0
        #for name, target in targetBoxes.items():
        #    metrics = {**metrics, **self._eval_metrics('box',name,output, target)}
        #for name, target in targetPoints.items():
        #    metrics = {**metrics, **self._eval_metrics('point',name,output, target)}
        #    metrics = self._eval_metrics(name,output, target)
        #toc=timeit.default_timer()
        #print('metric: '+str(toc-tic))

        #perAnchor={}
        #for i in range(avg_conf_per_anchor.size(0)):
        #    perAnchor['anchor{}'.format(i)]=avg_conf_per_anchor[i]

        log = {
            'loss': loss,
            'boxLoss': boxLoss,
            'edgeLoss': edgeLoss,

            **metrics,
        }

        #if iteration%10==0:
        #image=None
        #queryMask=None
        #targetBoxes=None
        #outputBoxes=None
        #outputOffsets=None
        #loss=None
        #torch.cuda.empty_cache()


        return log#
    def _minor_log(self, log):
        ls=''
        for key,val in log.items():
            ls += key
            if type(val) is float:
                ls +=': {:.6f},\t'.format(val)
            else:
                ls +=': {},\t'.format(val)
        self.logger.info('Train '+ls)

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_box_loss =0
        total_edge_loss =0
        total_val_metrics = np.zeros(len(self.metrics))

        mAP = np.zeros(self.model.numBBTypes)
        mRecall = np.zeros(self.model.numBBTypes)
        mPrecision = np.zeros(self.model.numBBTypes)
        with torch.no_grad():
            losses = defaultdict(lambda: 0)
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.logged:
                    print('iter:{} valid batch: {}/{}'.format(self.iteration,batch_idx,len(self.valid_data_loader)), end='\r')

                image, targetBoxes, adjM = self._to_tensor(instance)

                outputBoxes, outputOffsets, edgePred = self.model(image, hard_detect_limit=self.val_hard_detect_limit)
                #loss = self.loss(output, target)
                loss = 0
                index=0
                
                gtPairing,predPairing = self.alignEdgePred(targetBoxes,adjM,outputBoxes,edgePred)
                if len(predPairing.size())>0 and predPairing.size(0)>0:
                    edgeLoss = self.loss['edge'](predPairing,gtPairing)
                else:
                    edgeLoss = torch.tensor(0.0).to(image.device)
                if not self.model.detector_frozen:
                    boxLoss, position_loss, conf_loss, class_loss, recall, precision = self.loss['box'](outputOffsets,targetBoxes,[targetBoxes.size(1)])
                    loss = edgeLoss*self.lossWeights['edge'] + boxLoss*self.lossWeights['box']
                else:
                    boxLoss=torch.tensor(0.0)
                    loss = edgeLoss
                total_box_loss+=boxLoss.item()
                total_edge_loss+=edgeLoss.item()
                
                if targetBoxes is not None:
                    targetBoxes = targetBoxes.cpu()
                if targetBoxes is not None:
                    target_for_b = targetBoxes[0]
                else:
                    target_for_b = torch.empty(0)
                if self.model.rotation:
                    ap_5, prec_5, recall_5 =AP_dist(target_for_b,outputBoxes,0.9,self.model.numBBTypes)
                else:
                    ap_5, prec_5, recall_5 =AP_iou(target_for_b,outputBoxes,0.5,self.model.numBBTypes)
                mAP += np.array(ap_5)
                mRecall += np.array(recall_5)
                mPrecision += np.array(prec_5)

                total_val_loss += loss.item()
                loss=None
                image=None
                queryMask=None
                targetBoxes=None
                outputBoxes=None
                outputOffsets=None
                #total_val_metrics += self._eval_metrics(output, target)
        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_box_loss': total_box_loss / len(self.valid_data_loader),
            'val_edge_loss': total_edge_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist(),
            'val_recall':(mRecall/len(self.valid_data_loader)).tolist(),
            'val_precision':(mPrecision/len(self.valid_data_loader)).tolist(),
            'val_mAP':(mAP/len(self.valid_data_loader)).tolist(),
            #'val_position_loss':total_position_loss / len(self.valid_data_loader),
            #'val_conf_loss':total_conf_loss / len(self.valid_data_loader),
            #'val_class_loss':tota_class_loss / len(self.valid_data_loader),
        }


    def alignEdgePred(self,targetBoxes,adj,outputBoxes,edgePred):
        if edgePred is None or targetBoxes is None:
            return torch.tensor([]),torch.tensor([])
        targetBoxes = targetBoxes.cpu()
        #decide which predicted boxes belong to which target boxes
        #should this be the same as AP_?
        numClasses = 2
        if self.model.rotation:
            targIndex = getTargIndexForPreds_dist(targetBoxes[0],outputBoxes,0.9,numClasses)
        else:
            targIndex = getTargIndexForPreds_iou(targetBoxes[0],outputBoxes,0.5,numClasses)

        #Create gt vector to match edgePred.values()

        edges = edgePred[0] #edgePred._indices().cpu()
        predsAll = edgePred[1] #edgePred._values()
        newGT = []#torch.tensor((edges.size(0)))
        preds = []
        #for i in range(edges.size(0)):
        i=0
        for n0,n1 in edges:
            #n0 = edges[i,0]
            #n1 = edges[i,1]
            t0 = targIndex[n0]
            t1 = targIndex[n1]
            if t0>=0 and t1>=0:
                newGT.append( int((t0,t1) in adj) )#adjM[ min(t0,t1), max(t0,t1) ])
                preds.append(predsAll[i])
            #else skip this
            i+=1
        if len(preds)==0:
            return torch.tensor([]),torch.tensor([])
        newGT = torch.tensor(newGT).float().to(edgePred[1].device)
        preds = torch.cat(preds).to(edgePred[1].device)
        #assert(preds.requires_grad)
    
        return newGT, preds

    def prealignedEdgePred(self,adj,edgePred):
        if edgePred is None:
            assert(adj is None or len(adj)==0)
            return torch.tensor([]),torch.tensor([])
        edges = edgePred[0] #edgePred._indices().cpu().t()

        gt = torch.empty(len(edges))#edges.size(0))
        #for i in range(edges.size(0)):
        i=0
        for n0,n1 in edges:
            #n0 = edges[i,0]
            #n1 = edges[i,1]
            gt[i] = int((n0,n1) in adj) #(adjM[ n0, n1 ])
            i+=1
    
        #return gt.to(edgePred.device), edgePred._values().view(-1).view(-1)
        return gt.to(edgePred[1].device), edgePred[1].view(-1)
