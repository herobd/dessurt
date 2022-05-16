import os
import math
import json, copy
import timeit
import logging
import torch
import torch.optim as optim
import time
from utils.util import ensure_dir
from collections import defaultdict
from model import *
import re
import numpy as np
try:
    from torch.optim.swa_utils import AveragedModel
except ModuleNotFoundError:
    pass
#from ..model import PairingGraph
from torch.nn.parallel import DistributedDataParallel

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, resume, config, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        #if type(model) is tuple:
        #    self.model,self.model_ref
        self.model = model
        self.model_ref = model

        self.loss = loss
        self.metrics = metrics
        self.name = config['name']
        self.logged = config['super_computer'] if 'super_computer' in config else False
        self.iterations = config['trainer']['iterations']
        self.val_step = config['trainer']['val_step']
        self.save_step = config['trainer']['save_step']
        self.save_step_minor = config['trainer']['save_step_minor'] if 'save_step_minor' in config['trainer'] else None
        self.log_step = config['trainer']['log_step']
        self.verbosity = config['trainer'].get('verbosity',1)
        self.with_cuda = config['cuda'] and torch.cuda.is_available()
        if config['cuda'] and not torch.cuda.is_available():
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        elif config['cuda']:
            self.gpu = torch.device('cuda:' + str(config['gpu']))
            self.model = self.model.to(self.gpu)
        else:
            self.gpu=None

        self.train_logger = train_logger
        if config['optimizer_type']!="none":
            main_params=[]
            slow_params=[]
            slower_params=[]
            not_as_slow_params=[]
            slow_param_names = config['trainer']['slow_param_names'] if 'slow_param_names' in config['trainer'] else []
            slower_param_names = config['trainer']['slower_param_names'] if 'slower_param_names' in config['trainer'] else []
            not_as_slow_param_names = config['trainer']['not_as_slow_param_names'] if 'not_as_slow_param_names' in config['trainer'] else []
            freeze_param_names = config['trainer']['freeze_param_names'] if 'freeze_param_names' in config['trainer'] else []
            only_params = config['trainer']['only_params'] if 'only_params' in config['trainer'] else None
            for name,param in model.named_parameters():
                if only_params is None or any([p in name for p in only_params]):
                    goSlow=False
                    goSlower=False
                    goNotAsSlow=False
                    freeze=False
                    for sp in slower_param_names:
                        if sp in name:
                            goSlower=True
                            break
                    for sp in not_as_slow_param_names:
                        if sp in name:
                            goNotAsSlow=True
                            break
                    for sp in slow_param_names:
                        if sp in name:
                            goSlow=True
                            break
                    for fp in freeze_param_names:
                        if fp in name:
                            freeze=True
                            break
                    if freeze:
                        pass
                    elif goNotAsSlow:
                        not_as_slow_params.append(param)
                    elif goSlower:
                        slower_params.append(param)
                    elif goSlow:
                        slow_params.append(param)
                    elif ('hwr' in name and self.hwr_frozen) or ('style_extractor' in name and self.style_frozen):
                        pass
                    elif 'style_extractor' in name and self.curriculum.need_style_in_disc:
                        discriminator_params.append(param)
                    else:
                        main_params.append(param)
            to_opt = [
                    {'params': main_params}, 
                    {'params': slow_params, 'lr': config['optimizer']['lr']*0.1}]
            if len(not_as_slow_params)>0:
                to_opt.append({'params': not_as_slow_params, 'lr': config['optimizer']['lr']*0.5})
            if len(slower_params)>0:
                to_opt.append({'params': slower_params, 'lr': config['optimizer']['lr']*0.01})
            should_be_in_to_opt = 2 + (1 if len(slower_param_names)>0 else 0) + (1 if len(not_as_slow_param_names)>0 else 0)
            assert should_be_in_to_opt + len(to_opt) #help catch errors in param names
            self.optimizer = getattr(optim, config['optimizer_type'])(to_opt,
                                                                      **config['optimizer'])
                    #self.optimizer = getattr(optim, config['optimizer_type'])(model.parameters(),



        self.swa = config['trainer']['swa'] if 'swa' in config['trainer'] else (config['trainer']['weight_averaging'] if 'weight_averaging' in config['trainer'] else False)
        if self.swa:
            self.swa_model = AveragedModel(self.model)#type(self.model)(config['model'])
            #if config['cuda']:
            #    self.swa_model = self.swa_model.to(self.gpu)
            self.swa_start = config['trainer']['swa_start'] if 'swa_start' in config['trainer'] else config['trainer']['weight_averaging_start']
            #self.swa_c_iters = config['trainer']['swa_c_iters'] if 'swa_c_iters' in config['trainer'] else config['trainer']['weight_averaging_c_iters']
            self.swa_avg_every = config['trainer']['swa_avg_every'] if 'swa_avg_every' in config['trainer'] else 0
            assert(self.val_step>=self.swa_avg_every) #otherwise we'll start evaluating more than the (swa)model is updated



        self.useLearningSchedule = config['trainer']['use_learning_schedule'] if 'use_learning_schedule' in config['trainer'] else False
        if self.useLearningSchedule=='LR_test':
            start_lr=0.000001
            slope = (1-start_lr)/self.iterations
            lr_lambda = lambda step_num: start_lr + slope*step_num
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)
        elif self.useLearningSchedule=='cyclic': #only decreasing
            min_lr_mul = config['trainer']['min_lr_mul'] if 'min_lr_mul' in config['trainer'] else 0.001
            cycle_size = config['trainer']['cycle_size'] if 'cycle_size' in config['trainer'] else 500
            lr_lambda = lambda step_num: (1-(1-min_lr_mul)*((step_num-1)%cycle_size)/(cycle_size-1))
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)
        elif self.useLearningSchedule=='cyclic-full':
            min_lr_mul = config['trainer']['min_lr_mul'] if 'min_lr_mul' in config['trainer'] else 0.25
            cycle_size = config['trainer']['cycle_size'] if 'cycle_size' in config['trainer'] else 500
            def trueCycle (step_num):
                cycle_num = step_num//cycle_size
                if cycle_num%2==0: #even, rising
                    return ((1-min_lr_mul)*((step_num)%cycle_size)/(cycle_size-1)) + min_lr_mul
                else: #odd
                    return (1-(1-min_lr_mul)*((step_num)%cycle_size)/(cycle_size-1))
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,trueCycle)
        elif self.useLearningSchedule=='cyclic-decay':
            min_lr_mul = config['trainer']['min_lr_mul'] if 'min_lr_mul' in config['trainer'] else 0.25
            cycle_size = config['trainer']['cycle_size'] if 'cycle_size' in config['trainer'] else 500
            decay_rate = config['trainer']['decay_rate'] if 'decay_rate' in config['trainer'] else 0.99994 #saturates at about 50000 iterations
            def decayCycle (step_num):
                cycle_num = step_num//cycle_size
                decay = decay_rate**step_num
                if cycle_num%2==0: #even, rising
                    return decay*((1-min_lr_mul)*((step_num)%cycle_size)/(cycle_size-1)) + min_lr_mul
                else: #odd
                    return -decay*(1-min_lr_mul)*((step_num)%cycle_size)/(cycle_size-1) + 1-(1-min_lr_mul)*(1-decay)
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,decayCycle)
        elif self.useLearningSchedule=='1cycle':
            low_lr_mul = config['trainer']['low_lr_mul'] if 'low_lr_mul' in config['trainer'] else 0.25
            min_lr_mul = config['trainer']['min_lr_mul'] if 'min_lr_mul' in config['trainer'] else 0.0001
            cycle_size = config['trainer']['cycle_size'] if 'cycle_size' in config['trainer'] else 1000
            iters_in_trailoff = self.iterations-(2*cycle_size)
            def oneCycle (step_num):
                cycle_num = step_num//cycle_size
                if step_num<cycle_size: #rising
                    return ((1-low_lr_mul)*((step_num)%cycle_size)/(cycle_size-1)) + low_lr_mul
                elif step_num<cycle_size*2: #falling
                    return (1-(1-low_lr_mul)*((step_num)%cycle_size)/(cycle_size-1))
                else: #trail off
                    t_step_num = step_num-(2*cycle_size)
                    return low_lr_mul*(iters_in_trailoff-t_step_num)/iters_in_trailoff + min_lr_mul*t_step_num/iters_in_trailoff

            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,oneCycle)
        elif self.useLearningSchedule=='detector':
            warmup_steps = config['trainer']['warmup_steps'] if 'warmup_steps' in config['trainer'] else 1000
            lr_lambda = lambda step_num: min((step_num+1)**-0.3, (step_num+1)*warmup_steps**-1.3)
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)
        elif self.useLearningSchedule=='step':
            steps = config['trainer']['lr_steps']
            assert(type(steps) is list)
            def stepLR(step_num):
                mul=1
                for step in steps:
                    if step_num>=step:
                        mul*=0.1
                return mul
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,stepLR)
        elif self.useLearningSchedule=='multi_rise':
            steps = config['trainer']['warmup_steps']
            assert(type(steps) is list)
            steps=[0]+steps
            def riseLR(step_num):
                for i,step in enumerate(steps[1:]):
                    if step_num<step:
                        return (step_num-steps[i])*(0.99/(step-steps[i]))+.01
                return 1.0
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,riseLR)
        elif self.useLearningSchedule=='multi_rise then swa':
            steps = config['trainer']['warmup_steps']
            warmup_cap = 1.0
            swa_lr_mul = config['trainer']['swa_lr_mul'] if 'swa_lr_mul' in config['trainer'] else 0.001
            assert(type(steps) is list)
            steps=[0]+steps
            def riseLR(step_num):
                if step_num<self.swa_start:
                    for i,step in enumerate(steps[1:]):
                        if step_num<step:
                            return warmup_cap*((step_num-steps[i])*(0.99/(step-steps[i]))+.01)
                    return 1.0
                else:
                    return swa_lr_mul
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,riseLR)
        elif self.useLearningSchedule=='multi_rise then ramp_to_swa':
            steps = config['trainer']['warmup_steps']
            down_steps = config['trainer']['ramp_down_steps']
            warmup_cap = 1.0
            swa_lr_mul = config['trainer']['swa_lr_mul'] if 'swa_lr_mul' in config['trainer'] else 0.001
            assert(type(steps) is list)
            steps=[0]+steps
            def riseLR(step_num):
                if step_num<self.swa_start-down_steps:
                    for i,step in enumerate(steps[1:]):
                        if step_num<step:
                            return warmup_cap*((step_num-steps[i])*(0.99/(step-steps[i]))+.01)
                    return 1.0
                elif step_num<self.swa_start:
                    return 1 - (1-swa_lr_mul)*(down_steps-(self.swa_start-step_num))/down_steps
                else:
                    return swa_lr_mul
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,riseLR)
        elif self.useLearningSchedule=='multi_rise then ramp_to_lower':
            steps = config['trainer']['warmup_steps']
            down_steps = config['trainer']['ramp_down_steps']
            down_start = config['trainer']['lr_down_start']
            warmup_cap = 1.0
            swa_lr_mul = config['trainer']['lr_mul'] if 'lr_mul' in config['trainer'] else 0.1
            assert(type(steps) is list)
            steps=[0]+steps
            def riseLR(step_num):
                if step_num<down_start-down_steps:
                    for i,step in enumerate(steps[1:]):
                        if step_num<step:
                            return warmup_cap*((step_num-steps[i])*(0.99/(step-steps[i]))+.01)
                    return 1.0
                elif step_num<down_start:
                    return 1 - (1-swa_lr_mul)*(down_steps-(down_start-step_num))/down_steps
                else:
                    return swa_lr_mul
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,riseLR)
        elif self.useLearningSchedule=='multi_rise with cyclic_full then swa':
            steps = config['trainer']['warmup_steps']
            warmup_cap = config['trainer']['warmup_cap']
            min_lr_mul = config['trainer']['min_lr_mul'] if 'min_lr_mul' in config['trainer'] else 0.25
            cycle_size = config['trainer']['cycle_size']
            swa_lr_mul = config['trainer']['swa_lr_mul'] if 'swa_lr_mul' in config['trainer'] else 0.001
            def trueCycle (step_num):
                cycle_num = step_num//cycle_size
                if cycle_num%2==0: #even, rising
                    return ((1-min_lr_mul)*((step_num)%cycle_size)/(cycle_size-1)) + min_lr_mul
                else: #odd
                    return (1-(1-min_lr_mul)*((step_num)%cycle_size)/(cycle_size-1))
            assert(type(steps) is list)
            steps=[0]+steps
            def riseLR(step_num):
                if step_num<self.swa_start:
                    for i,step in enumerate(steps[1:]):
                        if step_num<step:
                            return warmup_cap*((step_num-steps[i])*(0.99/(step-steps[i]))+.01)
                    return trueCycle(step_num)
                else:
                    return swa_lr_mul
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,riseLR)
        elif self.useLearningSchedule=='spike then swa':
            warmup_steps = config['trainer']['warmup_steps'] if 'warmup_steps' in config['trainer'] else 1000
            swa_lr_mul = config['trainer']['swa_lr_mul'] if 'swa_lr_mul' in config['trainer'] else 0.1
            def spikeThenSWA(step_num):
                if step_num<self.swa_start:
                    return min((max(0.000001,step_num-(warmup_steps-3))/100)**-0.1, step_num*(1.485/warmup_steps)+.01)
                else:
                    return swa_lr_mul
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,spikeThenSWA)
        elif self.useLearningSchedule is True:
            warmup_steps = config['trainer']['warmup_steps'] if 'warmup_steps' in config['trainer'] else 1000
            #lr_lambda = lambda step_num: min((step_num+1)**-0.3, (step_num+1)*warmup_steps**-1.3)
            lr_lambda = lambda step_num: min((max(0.000001,step_num-(warmup_steps-3))/100)**-0.1, step_num*(1.485/warmup_steps)+.01)
            #y=((x-(2000-3))/100)^-0.1 and y=x*(1.485/2000)+0.01
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)
        elif self.useLearningSchedule:
            self.logger.info('Unrecognized learning schedule: {}'.format(self.useLearningSchedule))
            exit()
        
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']
        #assert self.monitor_mode == 'min' or self.monitor_mode == 'max'
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.retry_count = config['trainer']['retry_count'] if 'retry_count' in config['trainer'] else 0
        self.start_iteration = 1
        self.iteration=self.start_iteration
        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], self.name)
        self.seperate_log = None
        if 'seperate_log_at' in config['trainer']:
            self.seperate_log = os.path.join(config['trainer']['seperate_log_at'],self.name+'.log')
        ensure_dir(self.checkpoint_dir)
        json.dump(config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)
        self.iteration=999999999999999
        self.side_process=False
        self.reset_iteration = config['trainer']['reset_resume_iteration'] if 'reset_resume_iteration' in config['trainer'] else False
        if resume:
            self._resume_checkpoint(resume)
        if 'multiprocess' in config or 'distributed' in config:
            self.model = DistributedDataParallel(
                    self.model,
                    find_unused_parameters=True) #sometimes not used params... not sure why

    def finishSetup(self):
        """
        things that slave processes shouldn't do
        """
        ensure_dir(self.checkpoint_dir)
        json.dump(self.config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)

    def train(self):
        """
        Full training logic
        """
        sumLog=defaultdict(list)
        sumTime=0
        #for metric in self.metrics:
        #    sumLog['avg_'+metric.__name__]=0

        for self.iteration in range(self.start_iteration, self.iterations + 1):
            if not self.logged:
                print('iteration: {}'.format(self.iteration), end='\r')

            t = timeit.default_timer()
            result=None
            lastErr=None
            if self.useLearningSchedule:
                self.lr_schedule.step()
            for attempt in range(self.retry_count):
                try:
                    result = self._train_iteration(self.iteration)
                    break
                except RuntimeError as err:
                    print(err)
                    torch.cuda.empty_cache() #this is primarily to catch rare CUDA out of memory errors
                    lastErr = err

            if result is None:
                result = self._train_iteration(self.iteration)
                #if self.retry_count>1:
                #    print('Failed all {} times!'.format(self.retry_count))
                #raise lastErr

            elapsed_time = timeit.default_timer() - t
            sumLog['sec_per_iter'].append( elapsed_time )
            #print('iter: '+str(elapsed_time))

            #Stochastic Weight Averaging    https://github.com/timgaripov/swa/blob/master/train.py
            if self.swa and self.iteration>=self.swa_start and (self.swa_avg_every==0 or (self.iteration-self.swa_start)%self.swa_avg_every==0):
                #swa_n = (self.iterations-self.swa_start)//self.swa_c_iters
                #moving_average(self.swa_model, self.model, 1.0 / (swa_n + 1))
                #swa_n += 1
                if self.swa_model is None:
                    self.swa_model = AveragedModel(self.model)
                self.swa_model.update_parameters(self.model)

            if self.side_process:
                continue #when multithreading, current log, and validation, is only collected on master


            for key, value in result.items():
                if key == 'metrics':
                    for i, metric in enumerate(self.metrics):
                        sumLog['avg_'+metric.__name__].append(result['metrics'][i])
                else:
                    sumLog['avg_'+key].append(value)
            
            #log prep
            if (    self.iteration%self.log_step==0 or 
                    self.iteration%self.val_step==0 or 
                    self.iteration % self.save_step == 0 or 
                    (self.save_step_minor is not None and self.iteration % self.save_step_minor==0)
                ):
                log = {'iteration': self.iteration}

                for key, value in result.items():
                    if key == 'metrics':
                        for i, metric in enumerate(self.metrics):
                            log[metric.__name__] = result['metrics'][i]
                    else:
                        log[key] = value

            #LOG
            if self.iteration%self.log_step==0:
                #prinpt()#clear inplace text
                print('                   ', end='\r')
                if self.iteration-self.start_iteration>=self.log_step: #skip avg if started in odd spot
                    for key in sumLog:
                        sumLog[key] = np.mean(sumLog[key])
                    #self._minor_log(sumLog)
                    log = {**log, **sumLog}
                self._minor_log(log)
                for key in sumLog:
                    sumLog[key] = []
                if self.iteration%self.val_step!=0: #we'll do it later if we have a validation pass
                    self.train_logger.add_entry(log)

            #VALIDATION
            if self.iteration%self.val_step==0:
                if self.swa and self.iteration>=self.swa_start:
                    temp_model = self.model.cpu()
                    self.model = self.swa_model
                    self.bn_update()
                    val_result = self._valid_epoch()
                    self.model = temp_model.cuda()
                    for key, value in val_result.items():
                        if 'metrics' in key:
                            for i, metric in enumerate(self.metrics):
                                log['swa_val_' + metric.__name__] = val_result[key][i]
                        else:
                            log['swa_'+key] = value
                else:
                    val_result = self._valid_epoch()
                    for key, value in val_result.items():
                        if 'metrics' in key:
                            for i, metric in enumerate(self.metrics):
                                log['val_' + metric.__name__] = val_result[key][i]
                        else:
                            log[key] = value

                if self.train_logger is not None:
                    if self.iteration%self.log_step!=0:
                        print('                   ', end='\r')
                    #    print()#clear inplace text
                    self.train_logger.add_entry(log)
                    if self.verbosity >= 1:
                        for key, value in log.items():
                            if self.verbosity>=2 or 'avg' in key or 'val' in key:
                                self.logger.info('    {:15s}: {}'.format(str(key), value))
                if self.monitor in log and ((self.monitor_mode == 'min'  and log[self.monitor] < self.monitor_best)\
                        or (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best)):
                    self.monitor_best = log[self.monitor]
                    self._save_checkpoint(self.iteration, log, save_best=True)

            #SAVE
            if self.iteration % self.save_step == 0:
                self._save_checkpoint(self.iteration, log)
                if self.iteration%self.log_step!=0:
                    print('                   ', end='\r')
                #    print()#clear inplace text
                self.logger.info('Checkpoint saved for iteration '+str(self.iteration))
            elif self.iteration % self.save_step_minor == 0:
                self._save_checkpoint(self.iteration, log, minor=True)
                if self.iteration%self.log_step!=0:
                    print('                   ', end='\r')
                #    print()#clear inplace text
                #self.logger.info('Minor checkpoint saved for iteration '+str(self.iteration))

            

    def _train_iteration(self, iteration):
        """
        Training logic for a single iteration

        :param iteration: Current iteration number
        """
        raise NotImplementedError

    def save(self):
        self._save_checkpoint(self.iteration, None)

    def _save_checkpoint(self, iteration, log, save_best=False, minor=False):
        """
        Saving checkpoints

        :param iteration: current iteration number
        :param log: logging information of the ipoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'iteration': iteration,
            'logger': self.train_logger,
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        if 'save_mode' not in self.config or self.config['save_mode']=='state_dict':
            state_dict = self.model.state_dict()
            for k,v in state_dict.items():
                state_dict[k]=v.cpu()
            state['state_dict']= state_dict
            if self.swa and self.swa_model is not None:
                swa_state_dict = self.swa_model.state_dict()
                for k,v in swa_state_dict.items():
                    swa_state_dict[k]=v.cpu()
                state['swa_state_dict']= swa_state_dict
        else:
            state['model'] = self.model.cpu()
            if self.swa:
                state['swa_model'] = self.swa_model.cpu()
        if self.useLearningSchedule:
            state['lr_schedule'] = self.lr_schedule.state_dict()

        torch.cuda.empty_cache() #weird gpu memory issue when calling torch.save()
        if not minor:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint-iteration{}.pth'
                                    .format(iteration))
        else:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint-latest.pth')
            if os.path.exists(filename):
                os.rename(filename,os.path.join(self.checkpoint_dir, 'checkpoint-prev.pth'))
                                
        torch.save(state, filename)
        if not minor:
            #remove minor as this is the latest
            filename_late = os.path.join(self.checkpoint_dir, 'checkpoint-latest.pth')
            try:
                os.remove(filename_late)
            except FileNotFoundError:
                pass
            #os.link(filename,filename_late) #this way checkpoint-latest always does have the latest
            torch.save(state, filename_late) #something is wrong with the linking

        if save_best:
            os.rename(filename, os.path.join(self.checkpoint_dir, 'model_best.pth'))
            self.logger.info("Saved current best: {} ...".format('model_best.pth'))
        else:
            self.logger.info("Saved checkpoint: {} ...".format(filename))

        if self.seperate_log is not None:
            state = {
                'iteration': iteration,
                'logger': self.train_logger,
                'config': self.config
            }
            torch.save(state, self.seperate_log) #something is wrong with thel inkgin




    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        try:
            checkpoint = torch.load(resume_path, map_location=lambda storage, location: storage)
        except:
            resume_path = resume_path.replace('checkpoint-latest','checkpoint-prev')
            checkpoint = torch.load(resume_path, map_location=lambda storage, location: storage)
        if 'override' not in self.config or not self.config['override']:
            self.config = checkpoint['config']
        if not self.reset_iteration:
            self.start_iteration = checkpoint['iteration'] + 1
            self.iteration=self.start_iteration
        if not math.isinf(checkpoint['monitor_best']):
            self.monitor_best = checkpoint['monitor_best']
        else:
            if checkpoint['logger'] is not None:
                for entry in checkpoint['logger'].entries.values():
                    if self.monitor in entry and ((self.monitor_mode=='min' and entry[self.monitor]<self.monitor_best) or (self.monitor_mode=='max' and entry[self.monitor]>self.monitor_best)):
                        self.monitor_best = entry[self.monitor]
        #print(checkpoint['state_dict'].keys())

        if ('save_mode' not in self.config or self.config['save_mode']=='state_dict') and 'state_dict' in checkpoint:

            #Brain surgery, allow restarting with modified model

            did_brain_surgery=False
            remove_keys=[]
            checkpoint['state_dict'] = {(k if not k.startswith('module') else k[7:]):v for k,v in checkpoint['state_dict'].items() if 'relative_position_index' not in k}



            keys=checkpoint['state_dict'].keys()
            init_state_dict = self.model.state_dict()
            for key in keys:

                orig_size = checkpoint['state_dict'][key].size()
                if key in init_state_dict:
                    init_size = init_state_dict[key].size()
                    dims=-1
                    for dim in range(len(orig_size)):
                        if init_size[dim]!=orig_size[dim]:
                            dims=dim
                    if dims>-1:
                        if dims==0:
                            init_state_dict[key][:orig_size[0]] = checkpoint['state_dict'][key][:init_size[0]]
                        elif dims==1:
                            init_state_dict[key][:orig_size[0],:orig_size[1]] = checkpoint['state_dict'][key][:init_size[0],:init_size[1]]
                        elif dims==2:
                            init_state_dict[key][:orig_size[0],:orig_size[1],:orig_size[2]] = checkpoint['state_dict'][key][:init_size[0],:init_size[1],:init_size[2]]
                        elif dims==3:
                            init_state_dict[key][:orig_size[0],:orig_size[1],:orig_size[2],:orig_size[3]] = checkpoint['state_dict'][key][:init_size[0],:init_size[1],:init_size[2],:init_size[3]]
                        else:
                            raise NotImplementedError('no Brain Surgery above 4 dims')
                        checkpoint['state_dict'][key] = init_state_dict[key]
                        self.logger.info('BRAIN SURGERY PERFORMED on {}: {} -> {}'.format(key,orig_size,init_size))
                        did_brain_surgery=True
                else:
                    remove_keys.append(key)
                    did_brain_surgery=True
                    self.logger.info('BRAIN SURGERY PERFORMED removed {}'.format(key))
            for key in init_state_dict.keys():
                if key not in checkpoint['state_dict']:
                    if 'relative_position_index' not in key:
                        self.logger.info('BRAIN SURGERY PERFORMED added {}'.format(key))
                        did_brain_surgery=True
                    checkpoint['state_dict'][key] = init_state_dict[key]

            #specail check for Swin Transformer
            for init_key,value in init_state_dict.items():
                if 'attn_mask' in init_key or 'relative_position_index' in key: # and init_key not in keys:
                    checkpoint['state_dict'][init_key]=value
            for key in remove_keys:
                del checkpoint['state_dict'][key]

            self.model.load_state_dict(checkpoint['state_dict'])
            if self.swa and 'swa_state_dict' in checkpoint:
                self.swa_model = AveragedModel(self.model)
                keys=checkpoint['swa_state_dict'].keys()
                init_state_dict = self.swa_model.state_dict()
                for key in keys:
                    if torch.is_tensor(init_state_dict[key]) and len(init_state_dict[key].size())>0 and init_state_dict[key].size(0)>checkpoint['swa_state_dict'][key].size(0):
                        orig_size = checkpoint['swa_state_dict'][key].size(0)
                        init_state_dict[key][:orig_size] = checkpoint['swa_state_dict'][key]
                        checkpoint['swa_state_dict'][key] = init_state_dict[key]
                        self.logger.info('BRAIN SURGERY PERFORMED on {}'.format(key))
                self.swa_model.load_state_dict(checkpoint['swa_state_dict'])
        else:
            self.model = checkpoint['model']
            if self.swa:
                self.swa_model = checkpoint['swa_model']
        #if self.swa:
        #    self.swa_n = checkpoint['swa_n']
        dont_load_optimizer = self.config['dont_load_optimizer'] if 'dont_load_optimizer' in self.config else False
        if not did_brain_surgery and not dont_load_optimizer and 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if self.with_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda(self.gpu)
            except ValueError as e:
                self.logger.info('WARNING did not load optimizer state_dict. {}'.format(e))
        else:
            self.logger.info('Did not load optimizer')
        if self.useLearningSchedule:
            self.lr_schedule.load_state_dict(checkpoint['lr_schedule'])
        if checkpoint['logger'] is not None:
            self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (iteration {}) loaded".format(resume_path, self.start_iteration))

    def update_swa_batch_norm(self):
        #update_bn(self.data_loader,self.swa_model)
        tmp=self.model.cpu()
        self.model=self.swa_model.train()
        for instance in self.data_loader:
            self.run(instance)
        self.model=tmp

    def validate(self):
        log={}
        if self.swa and self.iteration>=self.swa_start:
            temp_model = self.model.cpu()
            self.model = self.swa_model
            self.bn_update()
            val_result = self._valid_epoch()
            self.model = temp_model.cuda()
            for key, value in val_result.items():
                if 'metrics' in key:
                    for i, metric in enumerate(self.metrics):
                        log['swa_val_' + metric.__name__] = val_result[key][i]
                else:
                    log['swa_'+key] = value
        else:
            val_result = self._valid_epoch()
            for key, value in val_result.items():
                if 'metrics' in key:
                    for i, metric in enumerate(self.metrics):
                        log['val_' + metric.__name__] = val_result[key][i]
                else:
                    log[key] = value
                    #sumLog['avg_'+key] += value

        self.train_logger.add_entry(log)
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha
