import numpy as np
import torch
from base import BaseTrainer
import timeit
from utils import util
from collections import defaultdict
from evaluators import FormsBoxDetect_printer
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist
from datasets.testforms_box import display


class BoxDetectTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(BoxDetectTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        if 'loss_params' in config:
            self.loss_params=config['loss_params']
        else:
            self.loss_params={}
        if 'box' in self.loss:
            self.loss['box'] = self.loss['box'](**self.loss_params['box'], 
                    num_classes=model.numBBTypes, 
                    rotation=model.rotation, 
                    scale=model.scale,
                    anchors=model.anchors)
        elif 'overseg' in self.loss:
            self.loss['overseg'] = self.loss['overseg'](**self.loss_params['overseg'], 
                    num_classes=model.numBBTypes, 
                    rotation=model.rotation, 
                    scale=model.scale,
                    anchors=model.anchors)
        if 'line' in self.loss and self.loss['line'] is not None:
            if 'line' in self.loss_params:
                params = self.loss_params['line']
            else:
                params = {}
            self.loss['line'] = self.loss['line'](**params, 
                    num_classes=model.numBBTypes, 
                    scale=model.scale,
                    anchor_h=model.meanH)
        if 'loss_weights' in config:
            self.lossWeights=config['loss_weights']
        else:
            self.lossWeights={'box':0.6, 'line': 0.4, 'point':0.4, 'pixel':8}
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        #for i in range(self.start_iteration,
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        #self.log_step = int(np.sqrt(self.batch_size))
        #lr schedule from "Attention is all you need"
        #base_lr=config['optimizer']['lr']

        self.thresh_conf = config['thresh_conf'] if 'thresh_conf' in config else 0.92
        self.thresh_intersect = config['thresh_intersect'] if 'thresh_intersect' in config else 0.4

    def _to_tensor(self, instance):
        data = instance['img']
        if 'bb_gt' in instance:
            targetBoxes = instance['bb_gt']
            targetBoxes_sizes = instance['bb_sizes']
        else:
            targetBoxes = None
            targetBoxes_sizes = []
        if 'num_neighbors' in instance:
            target_num_neighbors = instance['num_neighbors']
        else:
            target_num_neighbors = None
        if 'line_gt' in instance:
            targetLines = instance['line_gt']
            targetLines_sizes = instance['line_label_sizes']
        else:
            targetLines = {}
            targetLines_sizes = {}
        if 'point_gt' in instance:
            targetPoints = instance['point_gt']
            targetPoints_sizes = instance['point_label_sizes']
        else:
            targetPoints = {}
            targetPoints_sizes = {}
        if 'pixel_gt' in instance:
            targetPixels = instance['pixel_gt']
        else:
            targetPixels = None
        if type(data) is np.ndarray:
            data = torch.FloatTensor(data.astype(np.float32))
        elif type(data) is torch.Tensor:
            data = data.type(torch.FloatTensor)

        def sendToGPU(targets):
            new_targets={}
            for name, target in targets.items():
                if target is not None:
                    new_targets[name] = target.to(self.gpu)
                else:
                    new_targets[name] = None
            return new_targets

        if self.with_cuda:
            data = data.to(self.gpu)
            if targetBoxes is not None:
                targetBoxes=targetBoxes.to(self.gpu)
            targetLines=sendToGPU(targetLines)
            targetPoints=sendToGPU(targetPoints)
            if targetPixels is not None:
                targetPixels=targetPixels.to(self.gpu)
            if target_num_neighbors is not None:
                target_num_neighbors=target_num_neighbors.to(self.gpu)
        return data, targetBoxes, targetBoxes_sizes, targetLines, targetLines_sizes, targetPoints, targetPoints_sizes, targetPixels, target_num_neighbors

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
        self.model.train()
        #self.model.eval()
        #print('WARNING EVAL')

        ##tic=timeit.default_timer()
        batch_idx = (iteration-1) % len(self.data_loader)
        try:
            thisInstance = self.data_loader_iter.next()
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            thisInstance = self.data_loader_iter.next()
        if not self.model.predNumNeighbors:
            del thisInstance['num_neighbors']
        ##toc=timeit.default_timer()
        ##print('data: '+str(toc-tic))
        
        ##tic=timeit.default_timer()

        self.optimizer.zero_grad()

        losses, run_log, out = self.run(thisInstance)

        loss=0
        for name in losses.keys():
            losses[name] *= self.lossWeights[name[:-4]]
            loss += losses[name]
            losses[name] = losses[name].item()
        if len(losses)>0:
            loss.backward()
        ##toc=timeit.default_timer()
        ##print('for: '+str(toc-tic))
        #loss = self.loss(output, target)
        #what is grads?
        #minGrad=9999999999
        #maxGrad=-9999999999
        #for p in filter(lambda p: p.grad is not None, self.model.parameters()):
        #    minGrad = min(minGrad,p.min())
        #    maxGrad = max(maxGrad,p.max())
        torch.nn.utils.clip_grad_value_(self.model.parameters(),1)
        self.optimizer.step()
        
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

        ##tic=timeit.default_timer()
        loss = loss.item()
        ##toc=timeit.default_timer()
        ##print('item: '+str(toc-tic))
        #perAnchor={}
        #for i in range(avg_conf_per_anchor.size(0)):
        #    perAnchor['anchor{}'.format(i)]=avg_conf_per_anchor[i]

        log = {
            'loss': loss,

            **metrics,
            **losses,
            **run_log
        }


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
        val_metrics = {}#defaultdict(lambda: 0.0)
        val_count = defaultdict(lambda: 1)

        with torch.no_grad():
            #losses = defaultdict(lambda: 0)
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.model.predNumNeighbors:
                    del instance['num_neighbors']
                data, targetBoxes, targetBoxes_sizes, targetLines, targetLines_sizes, targetPoints, targetPoints_sizes, targetPixels,target_num_neighbors = self._to_tensor(instance)
                losses,log_run,got = self.run(instance,get=['bbs'],val=True)
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
        
        for val_name in val_metrics:
            if val_count[val_name]>0:
                val_metrics[val_name] /= val_count[val_name]
        return val_metrics





    def run(self,instance,get=[],val=False):
        #print('==forms: {}'.format(instance['imgName']))
        index=0
        losses={}
        log={}
        ##tic=timeit.default_timer()
        #predictions = util.pt_xyrs_2_xyxy(outputBoxes)
        #if self.iteration % self.save_step == 0:
        #    targetPoints={}
        #    targetPixels=None
        #    _,lossC=FormsBoxDetect_printer(None,instance,self.model,self.gpu,self._eval_metrics,self.checkpoint_dir,self.iteration,self.loss['box'])
        #    this_loss, position_loss, conf_loss, class_loss, recall, precision = lossC
        #else:
        data, targetBoxes, targetBoxes_sizes, targetLines, targetLines_sizes, targetPoints, targetPoints_sizes, targetPixels,target_num_neighbors = self._to_tensor(instance)
        if not self.model.predNumNeighbors:
            target_num_neighbors=None
        outputBoxes, outputOffsets, outputLines, outputOffsetLines, outputPoints, outputPixels = self.model(data)

        if 'box' in self.loss:
            this_loss, position_loss, conf_loss, class_loss, nn_loss, recall, precision, recall_noclass, precision_noclass = self.loss['box'](outputOffsets,targetBoxes,targetBoxes_sizes,target_num_neighbors)

            losses['boxLoss']=this_loss#.item()
            log['position_loss']=position_loss
            log['conf_loss']=conf_loss
            log['class_loss']=class_loss
            log['recall']=recall
            log['precision']=precision
            if recall+precision>0:
                log['F1']=2*recall*precision/(recall+precision)
            else:
                log['F1']=0
            log['recall_noclass']=recall_noclass
            log['precision_noclass']=precision_noclass
            if recall_noclass+precision_noclass>0:
                log['F1_noclass']=2*recall_noclass*precision_noclass/(recall_noclass+precision_noclass)
            else:
                log['F1_noclass']=0
            #print('boxLoss:{}'.format(this_loss))
#display(instance)
        elif 'overseg' in self.loss:
            if val:
                this_loss, position_loss, conf_loss, class_loss, rot_loss, recall, precision, gt_covered, pred_covered, recall_noclass, precision_noclass, gt_covered_noclass, pred_covered_noclass = self.loss['overseg'](outputOffsets,targetBoxes,targetBoxes_sizes,calc_stats=True)

                log['recall']=recall
                log['precision']=precision
                log['gt_covered']=gt_covered
                log['pred_covered']=pred_covered
                log['recall_noclass']=recall_noclass
                log['precision_noclass']=precision_noclass
                log['gt_covered_noclass']=gt_covered_noclass
                log['pred_covered_noclass']=pred_covered_noclass
                log['F1_covered_noclass']=2*gt_covered_noclass*pred_covered_noclass/(gt_covered_noclass+pred_covered_noclass)
            else:
                this_loss, position_loss, conf_loss, class_loss, rot_loss, _,_,_,_,_,_,_,_ = self.loss['overseg'](outputOffsets,targetBoxes,targetBoxes_sizes)
            losses['oversegLoss']=this_loss#.item()
            log['position_loss']=position_loss
            log['conf_loss']=conf_loss
            log['class_loss']=class_loss
            log['rot_loss']=rot_loss
        else:
            position_loss=0
            conf_loss=0
            class_loss=0

        index=0
        for name, target in targetLines.items():
            #print('line')
            predictions = outputOffsetLines[index]
            this_loss, line_pos_loss, line_conf_loss, line_class_loss = self.loss['line'](predictions,target,targetLines_sizes[name])
            losses[name+'Loss']=this_loss.item()
            losses[name+'_posLoss']=line_pos_loss
            losses[name+'_confLoss']=line_conf_loss
            losses[name+'_classLoss']=line_class_loss
            index+=1
        index=0
        for name, target in targetPoints.items():
            #print('point')
            predictions = outputPoints[index]
            this_loss, this_position_loss, this_conf_loss, this_class_loss = self.loss['point'](predictions,target,targetPoints_sizes[name], **self.loss_params['point'])
            losses[name+'Loss']=this_loss.item()
            index+=1
        if targetPixels is not None:
            #print('pixel')
            this_loss = self.loss['pixel'](outputPixels,targetPixels, **self.loss_params['pixel'])
            losses['pixelLoss']=this_loss.item()
        ##toc=timeit.default_timer()
        ##print('loss: '+str(toc-tic))
        ##tic=timeit.default_timer()

        got={}
        for name in get:
            if name=='bbs':
                got[name] = outputBoxes

        return losses, log, got
