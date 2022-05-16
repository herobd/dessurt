import numpy as np
import torch
import torch.nn.functional as F
from base import BaseTrainer
import timeit
from utils import util
from collections import defaultdict
import random, os, math
import editdistance
from utils import img_f



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
        self.lossWeights = config['loss_weights']
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        self.valid_data_loader = valid_data_loader


        self.ocr_word_bbs = config['trainer']['word_bbs'] if 'word_bbs' in config['trainer'] else False

        self.debug = False#'DEBUG' in  config['trainer']


        self.amp = config['trainer']['AMP'] if 'AMP' in config['trainer'] else False
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.accum_grad_steps = config['trainer']['accum_grad_steps'] if 'accum_grad_steps' in config['trainer'] else 1


        self.print_pred_every = config['trainer']['print_pred_every'] if  'print_pred_every' in config['trainer'] else 200
        


        self.distillation_temperature = config['trainer'].get('distillation_temperature',2.0)

        self.DEBUG_max_ocr_len=0

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
        self.logger.info('start valid loop')
        start_time = timeit.default_timer()
        self.model.eval()

        val_metrics = {}#not a default dict since it can have different kinds of data in it
        val_count = defaultdict(int)

        prefix = 'val_'


        with torch.no_grad():
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.logged:
                    print('iter:{} valid batch: {}/{}'.format(self.iteration,batch_idx,len(self.valid_data_loader)), end='\r')
                #elif batch_idx%100==0:
                #    self.logger.info('iter:{} valid batch: {}/{}'.format(self.iteration,batch_idx,len(self.valid_data_loader)))
                losses,log_run, out = self.run(instance,valid=True)
                #if batch_idx%100==0:
                #    self.logger.info('    finished valid batch: {}/{}'.format(batch_idx,len(self.valid_data_loader)))

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
        F_measure_prec={}
        F_measure_recall={}
        for val_name in val_metrics:
            if val_count[val_name]>0:
                val_metrics[val_name] =  val_metrics[val_name]/val_count[val_name]
                if 'F_prec' in val_name:
                    last_underscore = val_name.rfind('_')
                    var_name = val_name[last_underscore+1:]
                    F_measure_prec[var_name]=val_metrics[val_name]
                if 'F_recall' in val_name:
                    last_underscore = val_name.rfind('_')
                    var_name = val_name[last_underscore+1:]
                    F_measure_recall[var_name]=val_metrics[val_name]

        names = set(F_measure_prec.keys())
        names.update(F_measure_recall.keys())
        if len(names)>0:
            total_Fms=0
            for name in names:
                p = F_measure_prec[name] if name in F_measure_prec else 1
                r = F_measure_recall[name] if name in F_measure_recall else 1
                f = 2*(p*r)/(p+r) if (p+r) > 0 else 0
                total_Fms+=f
                val_metrics['val_F_Measure_{}'.format(name)]=f
            val_metrics['val_F_Measure_MACRO']=total_Fms/len(names)

        end_time = timeit.default_timer()
        self.logger.info('Time to complete validation loop: {}'.format(end_time-start_time))
        return val_metrics






    def run(self,instance,get=[],forward_only=False,valid=False,run=False):#

        image = self._to_tensor(instance['img'])
        device = image.device
        questions = instance['questions']
        answers = instance['answers']
        noise_token_mask = instance['noise_token_mask']

        gt_mask = instance['mask_label']
        if gt_mask is not None:
            gt_mask = gt_mask.to(device)


        distill= 'bart_logits' in instance and instance['bart_logits'] is not None


        if run:
            string_a,pred_mask = self.model(image,questions,RUN=run)
            string_a = [[string_a]]
        elif distill:
            pred_a, target_a, string_a, pred_logits, pred_last_hidden, batch_mask = self.model(image,questions,answers,distill=True)
        elif 'unlikelihood' in self.loss:
            pred_a, target_a, string_a, pred_mask, pred_logits = self.model(image,questions,answers,get_logits=True)
        else:
            pred_a, target_a, string_a, pred_mask = self.model(image,questions,answers)



        if forward_only:
            return
        losses=defaultdict(lambda:0)
        log=defaultdict(list)
        
        if not run:
            if 'answer' in self.loss:
                if noise_token_mask is not None:
                    noise_token_mask = noise_token_mask[:,1:] #we don't compute loss on start token
                    #This is to mix up tokens (done in dataset) for more robust training
                    if noise_token_mask.size(1)<pred_a.size(1):
                        noise_token_mask = F.pad(noise_token_mask,(0,pred_a.size(1)-noise_token_mask.size(1)))
                    elif noise_token_mask.size(1)>pred_a.size(1):
                        noise_token_mask = noise_token_mask[:,:pred_a.size(1)]

                    pred_a = pred_a * noise_token_mask[:,:,None].to(device) #This will raise the Loss, but prevents model from learning bad (switched) tokens

                losses['answerLoss'] = self.loss['answer'](pred_a,target_a,**self.loss_params['answer'])
                if self.debug:
                    print('answer size: {}'.format(pred_a.size()))

            if 'unlikelihood' in self.loss:
                assert noise_token_mask is None
                losses['unlikelihoodLoss'] = self.loss['unlikelihood'](pred_logits,pred_a,target_a,**self.loss_params['unlikelihood'])

            if 'mask' in self.loss and gt_mask is not None and pred_mask is not None: #we allow gt_mask to be none to not supervise
                mask_labels_batch_mask = instance['mask_labels_batch_mask'].to(device)
                losses['maskLoss'] = self.loss['mask'](pred_mask*mask_labels_batch_mask[:,None,None,None],gt_mask)

            if distill:
                #pred_len = batch_mask.size(1)
                teacher_logits = instance['bart_logits'].to(device)
                teacher_len = teacher_logits.size(1)
                batch_mask = batch_mask[:,:teacher_len,None] #add channel dim for broadcast
                teacher_batch_mask = batch_mask[:,:teacher_len]

                logits_dim = teacher_logits.size(-1)

                pred_logits = pred_logits[:,:teacher_len,:logits_dim]
                

                #cosine loss
                if self.lossWeights['cosine']>0:
                    teacher_last_hidden = instance['bart_last_hidden'].to(device)
                    hidden_dim = teacher_last_hidden.size(-1)

                    pred_last_hidden = torch.masked_select(pred_last_hidden,batch_mask)
                    pred_last_hidden = pred_last_hidden.view(-1,hidden_dim)
                    teacher_last_hidden = torch.masked_select(teacher_last_hidden,teacher_batch_mask)
                    teacher_last_hidden = teacher_last_hidden.view(-1,hidden_dim)

                    target = pred_last_hidden.new(pred_last_hidden.size(0)).fill_(1)
                    losses['cosineLoss'] = F.cosine_embedding_loss(pred_last_hidden, teacher_last_hidden, target,reduction="mean")

                if 'distill_loss_mask' in instance and instance['distill_loss_mask'] is not None:
                    distill_loss_mask = instance['distill_loss_mask'][...,None].to(batch_mask.device)
                    batch_mask *= distill_loss_mask
                    teacher_batch_mask *= distill_loss_mask
                
                if self.debug:
                    print('batch_mask sum: {}'.format(batch_mask.sum()))
                    print('teacher batch_mask sum: {}'.format(teacher_batch_mask.sum()))
                pred_logits = torch.masked_select(pred_logits,batch_mask)
                pred_logits = pred_logits.view(-1,logits_dim)
                teacher_logits = torch.masked_select(teacher_logits,teacher_batch_mask)
                teacher_logits = teacher_logits.view(-1,logits_dim)

                losses['distillationLoss'] = F.kl_div(
                        F.log_softmax(pred_logits / self.distillation_temperature, dim=-1),
                        F.softmax(teacher_logits / self.distillation_temperature, dim=-1),
                        reduction='batchmean')* (self.distillation_temperature ** 2)


        #t#tic=timeit.default_timer()#t#
        score_ed = []
        q_type_scores = defaultdict(list)
        pred_index = 0
        for b_answers,b_pred,b_questions in zip(answers,string_a,questions):
            for answer,pred,question in zip(b_answers,b_pred,b_questions):
                if answer is not None:
                    if len(answer)>0 or len(pred)>0:
                        score_ed.append( editdistance.eval(answer.lower(),pred.lower())/((len(answer)+len(pred))/2) )
                    else:
                        score_ed.append( 0 )

                    if len(answer)>0:
                        #get question start
                        q_end = question.find('~')
                        if q_end < 0:
                            q_end = question.find('>')
                        if q_end <0:
                            print('WARNING, logging sees unhandeled question: '+question)
                        else:
                            q_type = question[0:q_end+1]
                            q_type_scores[q_type].append(score_ed[-1])

                    if question.startswith('mk>') and not run:
                        #get topN accuracy for first token prediction.
                        #this is for internal evaluation of model LM performance
                        right_token = target_a[pred_index,0]
                        scores,preds = torch.sort(pred_a[pred_index,0],descending=True)
                        for N in [10,50,100]:
                            hitN = (preds[:N]==right_token).any()#.float().mean()
                            log['mk_firsttoken_top{}'.format(N)].append(hitN.int().item())
                pred_index += 1

                
        log['score_ed'] = np.mean(score_ed)
        for q_type,scores in q_type_scores.items():
            log['{}_ED'.format(q_type)] = np.mean(scores)

        if valid:
            if gt_mask is not None and pred_mask is not None:
                #compute pixel IoU
                pred_binary_mask = pred_mask>0
                intersection = (pred_binary_mask*gt_mask).sum(dim=3).sum(dim=2)
                union = (pred_binary_mask+gt_mask).sum(dim=3).sum(dim=2)
                iou = (intersection/union).cpu()
            else:
                iou = None
            for b,(b_answers,b_pred,b_questions,b_metadata) in enumerate(zip(answers,string_a,questions,instance['metadata'])):
                assert len(b_questions)==1
                #print('pred '+b_pred[0])
                #print('true '+b_answers[0])
                answer = b_answers[0].lower() if b_answers[0] is not None else None
                pred = b_pred[0].lower()
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
                    log['E_{}_CER'.format(typ)].append(ed/len(answer) if len(answer)>0 else ed)
                elif question == 'read_block0>':
                    ed = editdistance.eval(answer,pred)
                    log['E_line_based_CER'].append(ed/len(answer) if len(answer)>0 else ed)
                elif question == 'read_block>':
                    pass #let it be handled by basic ED
                elif question == 'w0>':
                    if '§' not in answer and '¿' not in answer:
                        ed = editdistance.eval(answer,pred)
                        log['E_line_based_CER'].append(ed/len(answer) if len(answer)>0 else ed)

                        a_words=answer.split(' ')
                        p_words=pred.split(' ')
                        w_ed = editdistance.eval(a_words,p_words)
                        log['E_line_based_WER'].append(w_ed/len(a_words) if len(a_words)>0 else w_ed)
                        #short_ed = editdistance.eval(answer[:len(pred)+2],pred)
                        #log['E_short_CER'].append(short_ed/len(pred) if len(pred)>0 else short_ed)

                        #if log['E_short_CER'][-1]>1 or True:
                        #    print('GT  : '+answer)
                        #    print('pred: '+pred)
                        #    img = (128*(1+image[0])).cpu()
                        #    img = torch.cat((img,img[0:1]),dim=0).permute(1,2,0).numpy()
                        #    img_f.imshow('',img.astype(np.uint8))
                        #    img_f.show()

                            #import pdb;pdb.set_trace()
                elif question.startswith('ne>'):
                    pred_type,pred_word = processNER(pred)
                    gt_type,gt_word = processNER(answer)
                    log['E_NER_acc'].append(1 if pred_type==gt_type else 0)
                    if gt_type!='o':
                        log['F_recall_{}'.format(gt_type)].append(1 if pred_type==gt_type else 0)
                    if pred_type!='o':
                        log['F_prec_{}'.format(pred_type)].append(1 if pred_type==gt_type else   0)
                    log['E_CER'] = editdistance.eval(gt_word,pred_word)/len(gt_word)
                elif question.startswith('ne~'):
                    pred_type = pred[1]
                    gt_type = answer[1]
                    log['E_NER_acc'].append(1 if pred_type==gt_type else 0)
                    if gt_type!='o':
                        log['F_recall_{}'.format(gt_type)].append(1 if pred_type==gt_type else 0)
                    if pred_type!='o':
                        log['F_prec_{}'.format(pred_type)].append(1 if pred_type==gt_type else   0)
                elif question.startswith('ner_'):
                    pred_words = processNERLine(pred)#.split(' ')
                    gt_words = processNERLine(answer)
                    pred_full = ' '.join(w[0] for w in pred_words)
                    pred_full = pred_full.replace(' .','.').replace(' ,',',').replace(' ?','?').replace(' !','!').replace(' ;',';').replace(' :',':').replace(' (','(').replace(' )',')')
                    gt_full = ' '.join(w[0] for w in gt_words)
                    gt_full = gt_full.replace(' .','.').replace(' ,',',').replace(' ?','?').replace(' !','!').replace(' ;',';').replace(' :',':').replace(' (','(').replace(' )',')')

                    #we now step through at be sure we mactch the words up
                    p=0 #pred index
                    g=0 #gt index
                    eds = [None]*len(pred_words)
                    recalls = [None]*len(pred_words)
                    precs = [None]*len(pred_words)
                    aligned = [None]*len(pred_words)
                    total_gt_len=0
                    for p in range(len(pred_words)):
                        eds[p] = [None]*len(gt_words)
                        recalls[p] = [defaultdict(list) for i in range(len(gt_words))]#[None]*len(gt_words)
                        precs[p] = [defaultdict(list) for i in range(len(gt_words))]#[None]*len(gt_words)
                        aligned[p] = [None for i in range(len(gt_words))]
                        for g in range(len(gt_words)):
                            ed = editdistance.eval(pred_words[p][0],gt_words[g][0])
                            if p==0 and g==0:
                                prev_ed=0
                                prev_recall={}
                                prev_prec={}
                                prev_aligned=[]
                            else:
                                step = []
                                if p>0 and g>0:
                                    step.append(((-1,-1),eds[p-1][g-1]))
                                if p>0:
                                    step.append(((-1,0),eds[p-1][g]))
                                if g>0:
                                    step.append(((0,-1),eds[p][g-1]))
                                step.sort(key=lambda s:s[1])
                                sp,sg = step[0][0]
                                prev_ed = eds[p+sp][g+sg]
                                prev_recall = recalls[p+sp][g+sg]
                                prev_prec = precs[p+sp][g+sg]
                                prev_aligned = aligned[p+sp][g+sg]

                            eds[p][g]=ed+prev_ed
                            total_gt_len+=len(gt_words[g][0])

                            aligned[p][g]=prev_aligned+[(gt_words[g],pred_words[p])]

                            #hit = gt_words[g][1] == pred_words[p][1]
                            gt_cls = gt_words[g][1]
                            pred_cls = pred_words[p][1]
                            recalls[p][g].update(prev_recall)
                            if gt_cls!='o':
                                #recalls[p][g]=defaultdict(list)
                                recalls[p][g][gt_cls]=recalls[p][g][gt_cls]+[1 if pred_cls==gt_cls else 0]
                            precs[p][g].update(prev_prec)
                            if pred_cls!='o':
                                #precs[p][g]=defaultdict(list)
                                precs[p][g][pred_cls]=precs[p][g][pred_cls]+[1 if pred_cls==gt_cls else 0]
                    if len(recalls)>0 and len(recalls[-1])>0:
                        for cls,recall in recalls[-1][-1].items():
                            log['F_recall_{}'.format(cls)]+=recall
                            #print('recall {} added: {}'.format(cls,recall))
                    else:
                        for word,cls in gt_words:
                            log['F_recall_{}'.format(cls)].append(0)

                    if len(precs)>0 and len(precs[-1])>0:
                        for cls,prec in precs[-1][-1].items():
                            log['F_prec_{}'.format(cls)]+=prec
                            #print('prec {} added: {}'.format(cls,prec))
                    else:
                        for word,cls in gt_words:
                            log['F_prec_{}'.format(cls)].append(1)
                    if len(pred_words)>0:
                        log['E_approx_CER'].append(eds[-1][-1]/total_gt_len)
                    elif len(gt_words)>0:
                        log['E_approx_CER'].append(1)

                    log['E_true_CER'].append(editdistance.eval(pred_full,gt_full)/len(gt_full))

                    #for gt,pred in aligned[-1][-1]:
                    #    if gt[1]==pred[1]:
                    #        print('{}\t{}\t\t{}\t{}'.format(gt[1],gt[0],pred[1],pred[0]))
                    #    else:
                    #        print('{}\t{}\t\t{}\t{} <<< error'.format(gt[1],gt[0],pred[1],pred[0]))


                elif question.startswith('mk>'):
                    pass #handled earlier
                elif question.startswith('natural_q~'):
                    #Compute Average Normalized Levenshtein Similarity (ANLS).
                    scores = []
                    assert len(b_metadata['all_answers'])==1
                    if b_metadata['all_answers'][0] is not None:
                        for ans in b_metadata['all_answers'][0]:
                            ed = editdistance.eval(ans.lower(),pred)
                            NL = ed/max(len(ans),len(pred))
                            scores.append(1-NL if NL<0.5 else 0)
        
                            #for HW-QA datasets
                            if len(b_metadata['all_answers'][0])==1:
                                log['E_acc@NL0.5'].append(1 if NL<0.5 else 0)

                        log['E_ANLS'].append(max(scores))


                #elif question=='json>':
                #    pred_data = fixLoadJSON(pred)
                # 
                #    #get predicted and gt entities
                elif question.startswith('classify>'):
                    if answer==pred:
                        log['E_class_acc'].append(1)
                    else:
                        log['E_class_acc'].append(0)
                elif question.startswith('record~'):
                    ed = editdistance.eval(answer,pred)
                    log['E_doc_CE'].append(ed/len(answer))
                elif question.startswith('json'):
                    ed = editdistance.eval(answer,pred)
                    log['E_json_CE'].append(ed/len(answer))
                else:
                    print('ERROR: missed question -- {}'.format(question))
                
                #for name,values in log.items():
                #    if name.startswith('E_'):
                #        print('{}: {}'.format(name,values[-1]))



        #t#self.opt_history['score'].append(timeit.default_timer()-tic)#t#

        if self.print_pred_every>0 and self.iteration%self.print_pred_every==0 and not valid:
            self.logger.info('iter {}'.format(self.iteration))
            for b,(b_question,b_answer,b_pred) in enumerate(zip(questions,answers,string_a)):
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
                elif 'pred'==name:
                    assert len(string_a)==1 and len(string_a[0])==1
                    got[name]=string_a[0][0]
                elif 'gt'==name:
                    assert len(answers)==1 and len(answers[0])==1
                    got[name]=answers[0][0]
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

ner_classes=set(cls.lower() for cls in ['N', 'C', 'L', 'T', 'O', 'P', 'G','NORP', 'LAW', 'PER', 'QUANTITY', 'MONEY', 'CARDINAL', 'LOCATION', 'LANGUAGE', 'ORG', 'DATE', 'FAC', 'ORDINAL', 'TIME', 'WORK_OF_ART', 'PERCENT', 'GPE', 'EVENT', 'PRODUCT'])
def processNERLine(line):
    ret = []
    words = line.split(' ')
    words2 = line.split(']')
    if len(words2)>len(words):
        words = words2
        spaced=False
    else:
        spaced=True
    for w in words:
        if len(w)==0:
            continue
        start_b = w.rfind('[')
        if spaced:
            end_b = w.rfind(']')
        else:
            end_b = len(w)
        if start_b!=-1 and end_b!=-1:
            if 'ne:'==w[start_b+1:start_b+4]:
                cls = w[start_b+4:end_b]
            else:
                colon = w.rfind(':')
                if colon!=-1:
                    cls = w[colon+1:end_b]
                else:
                    cls = w[start_b+1:end_b]

            word = w[:start_b]
        elif start_b!=-1:
            cls='o'
            word = w[:start_b]
        else:
            cls='o'
            word = line
        #print('see class: '+cls)
        if cls not in ner_classes:
            cls = 'o'
        ret.append((word,cls))
    return ret

def processNER(pred):
    start_b = pred.find('[')
    end_b = pred.find(']')
    if start_b!=-1 and end_b!=-1:
        cls = pred[start_b+1:end_b]
        word = pred[end_b+1:]
    
    return cls,word

