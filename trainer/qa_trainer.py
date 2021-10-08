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
from data_sets.testforms_graph_pair import display
import random, os, math
import editdistance

try:
    import easyocr
except:
    pass

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
        self.do_ocr = config['trainer']['do_ocr'] if 'do_ocr' in config['trainer'] else False
        if self.do_ocr and self.do_ocr!='no' and self.do_ocr!='json':
            self.ocr_reader = easyocr.Reader(['en'],gpu=config['cuda'])



    def _to_tensor(self, t):
        if self.with_cuda:
            t = t.to(self.gpu)
        return t

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

        val_metrics = {}#not a default dict since it can have different kinds of data in it
        val_count = defaultdict(int)

        prefix = 'val_'


        with torch.no_grad():
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.logged:
                    print('iter:{} valid batch: {}/{}'.format(self.iteration,batch_idx,len(self.valid_data_loader)), end='\r')
                losses,log_run, out = self.run(instance,valid=True)

                for name,value in log_run.items():
                    if value is not None:
                        val_name = prefix+name
                        if isinstance(value,(list,tuple)):
                            if val_name in val_metrics:
                                val_metrics[val_name]+=sum(value)
                            else:
                                val_metrics[val_name]=sum(value)
                            val_count[val_name]+=len(value)
                        else:   
                            if val_name in val_metrics:
                                val_metrics[val_name]+=value
                            else:
                                val_metrics[val_name]=value
                            val_count[val_name]+=1

                for name,value in losses.items():
                    if value is not None:
                        value = value.item()
                        val_name = prefix+name
                        if val_name in val_metrics:
                            val_metrics[val_name]+=value
                        else:
                            val_metrics[val_name]=value
                        val_count[val_name]+=1



                #total_val_metrics += self._eval_metrics(output, target)

        for val_name in val_metrics:
            if val_count[val_name]>0:
                val_metrics[val_name] =  val_metrics[val_name]/val_count[val_name]
        return val_metrics






    def run(self,instance,get=[],forward_only=False,valid=False):#
        image = self._to_tensor(instance['img'])
        device = image.device
        ocrBoxes = instance['bb_gt']
        questions = instance['questions']
        answers = instance['answers']
        gt_mask = instance['mask_label']
        if gt_mask is not None:
            gt_mask = gt_mask.to(device)

        #OCR possibilities
        #-All correct
        #-partail corrupt, partail missing
        #-all missing (none)

        if self.do_ocr:
            if self.do_ocr == 'no':
                ocr_res=[[]]*image.size(0)
            elif self.do_ocr == 'json':
                ocr_res = instance['pre-recognition']
            else:
                ocr_res=[]
                normal_img = (128*(image[:,0]+1)).cpu().numpy().astype(np.uint8)
                for img in normal_img:
                    ocr_res.append( self.ocr_reader.readtext(img,decoder='greedy+softmax') )
        else:
            if self.ocr_word_bbs:
                gtTrans = [form_metadata['word_trans'] for form_metadata in instance['form_metadata']]
            else:
                
                gtTrans = instance['transcription']
            #t##t#tic=timeit.default_timer()#t##t#
            if self.ocr_word_bbs: #useOnlyGTSpace and self.use_word_bbs_gt:
                word_boxes = torch.stack([form_metadata['word_boxes'] for form_metadata in instance['form_metadata']],dim=0)
                word_boxes = word_boxes.to(device) #I can change this as it isn't used later
                ocrBoxes=word_boxes

            if ocrBoxes is not None:
                
                #ocrBoxes[:,:,5:]=0 #zero out other information to ensure results aren't contaminated
                ocr = gtTrans
            else:
                ocr = None
            ocr_res = (ocrBoxes,ocr)

        #import pdb;pdb.set_trace()
        pred_a, target_a, string_a, pred_mask = self.model(image,ocr_res,questions,answers)

        #pred_a[:,0].sum().backward()
        #print(self.model.start_token.grad)
        #pred_a.sum().backward()
        #print(self.model.image.grad)
        #import pdb;pdb.set_trace()


        if forward_only:
            return
        #t##t#self.opt_history['run model'].append(timeit.default_timer()-tic)#t#
        #t##t#tic=timeit.default_timer()#t##t#
        losses=defaultdict(lambda:0)
        log=defaultdict(list)
        
        losses['answerLoss'] = self.loss['answer'](pred_a,target_a,**self.loss_params['answer'])
        #losses['answerLoss'] = pred_a.sum()
        if 'mask' in self.loss and gt_mask is not None: #we allow gt_mask to be none to not supervise
            mask_labels_batch_mask = instance['mask_labels_batch_mask'].to(device)
            losses['maskLoss'] = self.loss['mask'](pred_mask*mask_labels_batch_mask[:,None,None,None],gt_mask)


        #t#tic=timeit.default_timer()#t#
        score_ed = []
        for b_answers,b_pred in zip(answers,string_a):
            for answer,pred in zip(b_answers,b_pred):
                if len(answer)>0 or len(pred)>0:
                    score_ed.append( editdistance.eval(answer,pred)/((len(answer)+len(pred))/2) )
                else:
                    score_ed.append( 0 )
                
        log['score_ed'] = np.mean(score_ed)

        if valid:
            if gt_mask is not None:
                #compute pixel IoU
                pred_binary_mask = pred_mask>0
                intersection = (pred_binary_mask*gt_mask).sum(dim=3).sum(dim=2)
                union = (pred_binary_mask+gt_mask).sum(dim=3).sum(dim=2)
                iou = (intersection/union).cpu()
            else:
                iou = None
            for b,(b_answers,b_pred,b_questions) in enumerate(zip(answers,string_a,questions)):
                assert len(b_questions)==1
                answer = b_answers[0]
                pred = b_pred[0]
                question = b_questions[0]
            
                #print(question)
                #print(' answ:'+answer)
                #print(' pred:'+pred)
                if question.startswith('al~'):
                    cls = question[3:]
                    try:
                        count_pred = int(pred)
                    except ValueError:
                        count_pred = 0
                    count_gt = int(answer)
                    log['E_all_{}_IoU'.format(cls)].append(iou[b].item())
                    log['E_all_{}_count_err'.format(cls)].append(abs(count_pred-count_gt))
                elif question.startswith('z0') or question.startswith('g0'):
                    cls_pred = pred[:3]
                    cls_gt = answer[:3]
                    log['E_class_acc'].append(cls_pred==cls_gt)
                    
                    if question.startswith('z0'):
                        try:
                            count_pred = int(pred[3:]) if pred[3:]!='ø' else 0
                        except ValueError:
                            count_pred = 0
                        count_gt = int(answer[3:]) if answer[3:]!='ø' else 0
                        log['E_link_count_err'].append(abs(count_pred-count_gt))
                        log['E_link_all_IoU'].append(iou[b].item())
                    else:
                        if 'g0~' in question:
                            if answer[3:] != 'ø':
                                start_gt = answer.find('>')
                                count_gt = int(answer[3:start_gt])
                            else:
                                start_gt = 2
                                count_gt = 0
                            if pred[3:] != 'ø':
                                start_pred = pred.find('>')
                                if start_pred>-1:
                                    try:
                                        count_pred = int(pred[3:start_pred])
                                    except ValueError:
                                        count_pred = 0
                                else:
                                    start_pred=2
                                    count_pred=0
                            else:
                                start_pred=2
                                count_pred=0

                            log['E_link_count_err'].append(abs(count_pred-count_gt))
                            pred_s = pred[start_pred+1:]
                            answer_s = answer[start_gt+1:]
                        else:
                            pred_s = pred[3:]
                            answer_s = answer[3:]

                        log['E_link_step_IoU'].append(iou[b].item())
                        ed = editdistance.eval(answer_s,pred_s)
                        hit = ed/((len(answer_s)+len(pred_s))/2) < 0.1
                        log['E_link_step_acc'].append(int(hit))
                        log['E_link_step_ed'].append(ed)
                elif question.startswith('f1~') or question.startswith('p1~'):
                    ed = editdistance.eval(answer,pred)
                    hit = ed/((len(answer)+len(pred))/2) < 0.1
                    log['E_read_acc'].append(int(hit))
                    log['E_read_ed'].append(ed)
                    if answer[0]=='\\':
                        log['E_read_1stNewline_acc'].append(1 if pred[0]=='\\' else 0)
                elif question.startswith('fli:'):
                    typ = question[4:question.find('~')]
                    ed = editdistance.eval(answer,pred)
                    hit = ed/((len(answer)+len(pred))/2) < 0.1
                    log['E_{}_acc'.format(typ)].append(int(hit))
                    log['E_{}_ed'.format(typ)].append(ed)
                else:
                    print('ERROR: missed question -- {}'.format(question))
                
                #for name,values in log.items():
                #    if name.startswith('E_'):
                #        print('{}: {}'.format(name,values[-1]))



        #t#self.opt_history['score'].append(timeit.default_timer()-tic)#t#

        if self.print_pred_every>0 and self.iteration%self.print_pred_every==0:
            self.logger.info('iteration {}'.format(self.iteration))
            for b,(b_question,b_answer,b_pred) in enumerate(zip(questions,answers,string_a)):
                if self.do_ocr:
                    self.logger.info('{} OCR: ')
                    for res in ocr_res[b]:
                        self.logger.info(res[1][0])

                elif ocr is not None and not self.model_ref.blank_ocr:
                    self.logger.info('{} OCR: {}'.format(b,ocr[b]))
                for question,answer,pred in zip(b_question,b_answer,b_pred):
                    self.logger.info('{} [Q]:{}\t[A]:{}\t[P]:{}'.format(b,question,answer,pred))





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
        if not valid:
            for name in log:
                if isinstance(log[name],list):
                    log[name] = np.mean(log[name])
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

