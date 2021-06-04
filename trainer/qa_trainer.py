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
import editdistance

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

        self.ocr_word_bbs = config['trainer']['word_bbs'] if 'word_bbs' in config['trainer'] else False

        self.debug = 'DEBUG' in  config['trainer']


        self.amp = config['trainer']['AMP'] if 'AMP' in config['trainer'] else False
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.accum_grad_steps = config['trainer']['accum_grad_steps'] if 'accum_grad_steps' in config['trainer'] else 1


        self.print_pred_every = config['trainer']['print_pred_every'] if  'print_pred_every' in config['trainer'] else 200
        
        #t#self.opt_history = defaultdict(list)#t#



    def _to_tensor(self, instance):
        image = instance['img']

        if self.with_cuda:
            image = image.to(self.gpu)
        return image

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
        #t#ticAll=timeit.default_timer()#t#

        self.model.train()
        #self.model.eval()
        #print("WARNING EVAL")

        #t##t#ticAll=timeit.default_timer()#t##t#
        batch_idx = (iteration-1) % len(self.data_loader)
        try:
            thisInstance = self.data_loader_iter.next()
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            thisInstance = self.data_loader_iter.next()
        #t#self.opt_history['get data'].append(timeit.default_timer()-ticAll)#t#


        
        if self.accum_grad_steps<2 or iteration%self.accum_grad_steps==1:
            self.optimizer.zero_grad()

        ##toc=timeit.default_timer()
        ##print('for: '+str(toc-tic))
        #loss = self.loss(output, target)
        index=0
        losses={}


        #t#tic=timeit.default_timer()#t#
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
        if self.side_process:
            with open('test/tmp_{}.txt'.format(self.side_process),'a') as f:
                f.write('[{}] {}'.format(self.iteration,self.model_ref.decoder.layers[0].linear1.weight.data[0]))
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
                val_metrics[val_name] =  val_metrics[val_name]/val_count[val_name]
        return val_metrics






    def run(self,instance,get=[],forward_only=False):#
        image = self._to_tensor(instance)
        ocrBoxes = instance['bb_gt']
        questions = instance['questions']
        answers = instance['answers']

        #OCR possibilities
        #-All correct
        #-partail corrupt, partail missing
        #-all missing (none)
        
        if self.ocr_word_bbs:
            gtTrans = [form_metadata['word_trans'] for form_metadata in instance['form_metadata']]
        else:
            gtTrans = instance['transcription']
        #t##t#tic=timeit.default_timer()#t##t#
        if self.ocr_word_bbs: #useOnlyGTSpace and self.use_word_bbs_gt:
            word_boxes = torch.stack([form_metadata['word_boxes'] for form_metadata in instance['form_metadata']],dim=0)
            word_boxes = word_boxes.to(image.device) #I can change this as it isn't used later
            ocrBoxes=word_boxes

        if ocrBoxes is not None:
            
            #ocrBoxes[:,:,5:]=0 #zero out other information to ensure results aren't contaminated
            ocr = gtTrans
        else:
            ocr = None

        #import pdb;pdb.set_trace()
        pred_a, target_a, string_a = self.model(image,ocrBoxes,ocr,questions,answers)


        if forward_only:
            return
        #t##t#self.opt_history['run model'].append(timeit.default_timer()-tic)#t#
        #t##t#tic=timeit.default_timer()#t##t#
        losses=defaultdict(lambda:0)
        log={}
        

        losses['answerLoss'] = self.loss['answer'](pred_a,target_a,**self.loss_params['answer'])


        #t#tic=timeit.default_timer()#t#
        cor_present=0
        total_present=0
        cor_pair=0
        total_pair=0
        score_ed = 0
        total_score = 0
        for b_answers,b_pred in zip(answers,string_a):
            for answer,pred in zip(b_answers,b_pred):
                if len(pred)>0 and len(answer)>0 and answer[0]==pred[0]:
                    cor_present+=1
                total_present+=1
                if len(answer)>2:
                    if answer[2:]==pred[2:]:
                        cor_pair+=1
                    total_pair+=1
                if len(answer)>0 or len(pred)>0:
                    score_ed += editdistance.eval(answer,pred)/((len(answer)+len(pred))/2)
                else:
                    score_ed += 0
                total_score +=1
                
        log['present_acc']=cor_present/total_present
        if total_pair>0:
            log['pair_acc']=cor_pair/total_pair
        log['score_ed'] = score_ed/total_score
        #t#self.opt_history['score'].append(timeit.default_timer()-tic)#t#

        if self.print_pred_every>0 and self.iteration%self.print_pred_every==0:
            print('iteration {}'.format(self.iteration))
            for b,(b_question,b_answer,b_pred) in enumerate(zip(questions,answers,string_a)):
                if ocr is not None:
                    print('{} OCR: {}'.format(b,ocr[b]))
                for question,answer,pred in zip(b_question,b_answer,b_pred):
                    print('{} [Q]:{}\t[A]:{}\t[P]:{}'.format(b,question,answer,pred))





        got={}
        if get is not None:
            for name in get:
                if 'strings'==name:
                    ret=[]
                    for b,(b_question,b_answer,b_pred) in enumerate(zip(questions,answers,string_a)):
                        for question,answer,pred in zip(b_question,b_answer,b_pred):
                            ret.append('{} [Q]:{}\t[A]:{}\t[P]:{}'.format(b,question,answer,pred))
                    got[name]=ret
                else:
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

