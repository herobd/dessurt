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



def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

class QATrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(QATrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        if 'loss_params' in config:
            self.loss_params=config['loss_params']
        else:
            self.loss_params={}
        self.lossWeights = config['loss_weights'] if 'loss_weights' in config else {"box": 1, "rel":1}
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        self.valid_data_loader = valid_data_loader

        self.ocr_word_bbs = config['trainer']['word_bbs']

        self.debug = 'DEBUG' in  config['trainer']


        self.amp = config['trainer']['AMP'] if 'AMP' in config['trainer'] else False
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.accum_grad_steps = config['trainer']['accum_grad_steps'] if 'accum_grad_steps' in config['trainer'] else 1


        self.print_pred_every = config['trainer']['print_pred_every'] if  'print_pred_every' in config['trainer'] else 200




    def _to_tensor(self, instance):
        image = instance['img']
        bbs = instance['bb_gt']

        if self.with_cuda:
            image = image.to(self.gpu)
            if bbs is not None:
                bbs = bbs.to(self.gpu)
        return image, bbs

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
        #print("WARNING EVAL")

        #t#ticAll=timeit.default_timer()#t##t#
        batch_idx = (iteration-1) % len(self.data_loader)
        try:
            thisInstance = self.data_loader_iter.next()
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            thisInstance = self.data_loader_iter.next()
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


        #print('\t\t\t\t{} {}'.format(iteration,thisInstance['imgName']))
        if self.amp:
            with torch.cuda.amp.autocast():
                losses, run_log, out = self.run(thisInstance)
        else:
            losses, run_log, out = self.run(thisInstance)
        #t#self.opt_history['full run'].append(timeit.default_timer()-tic)#t#

        #t#tic=timeit.default_timer()#t##t#
        loss=0
        for name in losses.keys():
            losses[name] *= self.lossWeights[name[:-4]]
            loss += losses[name]
            losses[name] = losses[name].item()
        if len(losses)>0:
            assert not torch.isnan(loss)
            if self.accum_grad_steps>1:
                loss /= self.accum_grad_steps
            if self.amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        meangrad=0
        count=0
        for n,m in self.model.named_parameters():
            #if 'answer_decode' in n:
            #    import pdb;pdb.set_trace()
            if m.grad is None:
                continue
            count+=1
            meangrad+=abs(m.grad.data.mean().cpu().item())
            assert not torch.isnan(m.grad.data).any()
        if count!=0:
            meangrad/=count
        if self.accum_grad_steps<2 or iteration%self.accum_grad_steps==0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(),1)
            if self.amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        #t#self.opt_history['backprop'].append(timeit.default_timer()-tic)#t#
        if len(losses)>0:
            loss = loss.item()
        #print('loss:{}, mean grad:{}'.format(loss,meangrad))
        log = {
            'mean abs grad': meangrad,
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

        prefix = 'val_'


        with torch.no_grad():
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.logged:
                    print('iter:{} valid batch: {}/{}'.format(self.iteration,batch_idx,len(self.valid_data_loader)), end='\r')
                losses,log_run, out = self.run(instance)

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
                val_metrics[val_name] =  val_count[val_name]/val_count[val_name]
        return val_metrics






    def run(self,instance,get=[],forward_only=False):#
        image, targetBoxes  = self._to_tensor(instance)
        questions = instance['questions']
        answers = instance['answers']

        #OCR possibilities
        #-All correct
        #-partail corrupt, partail missing
        #-all missing (none)
        
        if self.ocr_word_bbs:
            gtTrans = instance['form_metadata']['word_trans']
        else:
            gtTrans = instance['transcription']
        #t#tic=timeit.default_timer()#t##t#
        if self.ocr_word_bbs: #useOnlyGTSpace and self.use_word_bbs_gt:
            word_boxes = instance['form_metadata']['word_boxes'][None,:,:,].to(image.device) #I can change this as it isn't used later
            targetBoxes_changed=word_boxes
            if self.model.training:
                targetBoxes_changed[:,:,0] += torch.randn_like(targetBoxes_changed[:,:,0])
                targetBoxes_changed[:,:,1] += torch.randn_like(targetBoxes_changed[:,:,1])
                #if self.model_ref.rotation:
                #    targetBoxes_changed[:,:,2] += torch.randn_like(targetBoxes_changed[:,:,2])*0.01
                targetBoxes_changed[:,:,3] += torch.randn_like(targetBoxes_changed[:,:,3])
                targetBoxes_changed[:,:,4] += torch.randn_like(targetBoxes_changed[:,:,4])
                targetBoxes_changed[:,:,3][targetBoxes_changed[:,:,3]<1]=1
                targetBoxes_changed[:,:,4][targetBoxes_changed[:,:,4]<1]=1

        elif targetBoxes is not None:
            targetBoxes_changed=targetBoxes.clone()
            if self.model.training:
                targetBoxes_changed[:,:,0] += torch.randn_like(targetBoxes_changed[:,:,0])
                targetBoxes_changed[:,:,1] += torch.randn_like(targetBoxes_changed[:,:,1])
                #if self.model_ref.rotation:
                #    targetBoxes_changed[:,:,2] += torch.randn_like(targetBoxes_changed[:,:,2])*0.01
                targetBoxes_changed[:,:,3] += torch.randn_like(targetBoxes_changed[:,:,3])
                targetBoxes_changed[:,:,4] += torch.randn_like(targetBoxes_changed[:,:,4])
                targetBoxes_changed[:,:,3][targetBoxes_changed[:,:,3]<1]=1
                targetBoxes_changed[:,:,4][targetBoxes_changed[:,:,4]<1]=1
                #we tweak the classes in the model
        else:
            targetBoxes_changed = None

        if targetBoxes_changed is not None:
            targetBoxes_changed[:,:,5:]=0 #zero out other information to ensure results aren't contaminated
            #TODO corrupt the OCR
            ocr_changed = gtTrans
        else:
            ocr_changed = None

        pred_a, target_a, string_a = self.model(image,targetBoxes_changed,ocr_changed,questions,answers)


        if forward_only:
            return
        #t#self.opt_history['run model'].append(timeit.default_timer()-tic)#t#
        #t#tic=timeit.default_timer()#t##t#
        losses=defaultdict(lambda:0)
        log={}

        losses['answerLoss'] = self.loss['answer'](pred_a,target_a,**self.loss_params['answer'])

        cor_present=0
        for answer,pred in zip(answers,string_a):
            if len(pred)>0 and answer[0]==pred[0]:
                cor_present+=1
        log['present_acc']=cor_present/len(answers)

        if self.print_pred_every>0 and self.iteration%self.print_pred_every==0:
            print('iteration {}'.format(self.iteration))
            for question,answer,pred in zip(questions,answers,string_a):
                print('[Q]:{} [A]:{} [P]:{}'.format(question,answer,pred))





        got={}
        for name in get:
            #else
            raise NotImplementedError('Cannot get [{}], unknown'.format(name))
        return losses, log, got

    def bn_update(self):
        r"""Updates BatchNorm running_mean, running_var buffers in the model.
        It performs one pass over data in `loader` to estimate the activation
        statistics for BatchNorm layers in the model.
        Args:
            loader (torch.utils.data.DataLoader): dataset loader to compute the
                activation statistics on. Each data batch should be either a
                tensor, or a list/tuple whose first element is a tensor
                containing data.
            model (torch.nn.Module): model for which we seek to update BatchNorm
                statistics.
            device (torch.device, optional): If set, data will be trasferred to
                :attr:`device` before being passed into :attr:`model`.
        """
        model=self.model
        loader=self.data_loader
        if not _check_bn(model):
            return
        print('updating bn')
        was_training = model.training
        model.train()
        momenta = {}
        #model.apply(_reset_bn)
        #model.apply(lambda module: _get_momenta(module, momenta))
        n = 0
        with torch.no_grad():
            for instance in loader:

                b = 1#input.size(0)

                momentum = b / float(n + b)
                #for module in momenta.keys():
                #    module.momentum = momentum

                self.newRun(instance,self.useGT(self.iteration),forward_only=True)
                n += b

        #model.apply(lambda module: _set_momenta(module, momenta))
        #model.train(was_training)

