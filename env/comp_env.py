import time
import torch
import numpy
import math
import logging
from models.vit_pytorch.distill_comp import Deit_Comp
from .utils import exit_order, get_channel_list, get_model, adjust_learning_rate
from .train import forward
from .inference import inference
from .pruner import pruning
from .data import T_conf, ORIGINAL_ACC
import utils
from timm.scheduler import PlateauLRScheduler

class check_queue():
    def __init__(self, len, default):
        self.values = [default for _ in range(len)]
        
    def append(self, value):
        self.values.append(value)
        self.values.pop(0)
    
    def check_all_below(self, threthold):
        return sum([i>threthold for i in self.values]) == 0
    
class Comp_env():
    def __init__(self,model='resnet_exit_quant', dataset='cifar100', suffix='', test = 0, pt_finetune=False, dist=None, t_conf=None):
        ## train configuration
        logging.info("creating env...")
        logging.info("train/eval/model configuration:")
        self.t_conf = t_conf if t_conf else T_conf(model, dataset, suffix, pt_finetune)
        if dist is None:
            utils.init_distributed_mode(self.t_conf)
        if not t_conf:
            self.t_conf.add_dataloader()
        self.test = test
        if test == 1:
            logging.info('optimizor test for P+100epoch finetune')
        logging.info(vars(self.t_conf))
        
        # train&eval
        self.epoch = 0 # epoch after last compression
        self.epoch_each_finetune = 3
        self.epoch_after_prune = 0 # 5
        self.epoch_after_quant = 0
        self.epoch_after_truncated = 0 # 5
        # self.epoch_after_addexit = 0
        ## lr decay
        self.loss_past = numpy.inf
        self.loss_drop_queue = check_queue(3, 1.0)
        self.current_decay_level = len(self.t_conf.decay_level) - 1 # initially trained model, should have lowest lr
        # model
        self.model = get_model(self.t_conf).to(self.t_conf.device)
        self.model.beta = self.t_conf.beta
        # self.model.BNnoquant(True)
        # self.model.exit_num = 0
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.t_conf.lr, weight_decay=self.t_conf.weight_decay)
        self.main_optimizer = torch.optim.SGD([{"params":i if type(i) is torch.nn.Parameter else i.parameters()} for i in self.model.main_list], lr=self.t_conf.sgdlr*self.t_conf.decay_level[-1],
                                         momentum=self.t_conf.momentum, weight_decay=self.t_conf.sgd_wd)
        self.exit_optimizer = torch.optim.Adam([{"params":i.parameters()} for i in self.model.exit_list], 
                                               lr=self.t_conf.adamlr*self.t_conf.decay_level[-1], weight_decay=self.t_conf.adam_wd)
        self.optimizers = [self.main_optimizer, self.exit_optimizer]
        # exit
        self.exit_to_place = exit_order(self.model)
        self.model.place_layer = []
        self.num_each_add = 2
        
        # quantize
        self.quant_level = 0
        self.w, self.a = 32, 32
        
        self.quant_bcr = 1.0
        self.quant_mcr = 1.0
        
        # prune
        self.prune_ratio = self.t_conf.pruning_ratio
        self.example_inputs = self.t_conf.example_inputs.to(int(self.t_conf.gpus[0])) # traindata[0][0].unsqueeze(0).to(self.device)
        
        self.prune_bcr = 1.0
        self.prune_mcr = 1.0
        
        # state
        self.base_acc, _ = inference(self.model, self.t_conf.test_loader, self.t_conf, beta=self.t_conf.beta)
        logging.info(f'### Base acc: {self.base_acc}')
        self.time = time.time()
        self.action_past = -1
        self.acc_past = self.base_acc
        self.bcr_past = 1.0
        self.mcr_past = 1.0
        self.nomalized_cost_past = 1.0
        self.exit_bcr_past = 1.0
        self.acc = self.base_acc
        self.bcr = 1.0 # bitops compression ratio
        self.mcr = 1.0 # memory compression ratio
        self.exit_bcr = 1.0 # bcr from early exit
        self.acc_drop_r_past = 0
        self.step_num = 0
        self.state_names = ['w bit', 'a bit', 'prune bcr', 'prune mcr', 'exit layer num', 'prune num','finetune num', 'quant num', 'total bcr', 'mcr', 'acc drop', 'acc change']
        self.state_dim = len(self.state_names)
        ## truncated check
        self.acc_drop_queue = check_queue(3, 0.0)
        self.tr_step_num = 66
        self.tr_comp_num = 12
        
        # action
        self.actions = [self.finetune, self.prune, self.add_exit, self.quantize] #, self.delete_exit] # For on-policy, currently, no undo operations that should not occur in practice are added.
        self.action_names = ['F', 'P', 'E', 'Q']
        self.action_dim = len(self.actions)
        self.trajectory = '' # Merge the consecutive F's
        self.full_trajectory = '' # show all F
        self.comp_num = {'F': 0, 'P': 0, 'E': 0, 'Q': 0}
        self.comp_full = {'F':True, 'P': False, 'E': False, 'Q':False}
        self.finetune_action = (0)
        self.comp_action = (1,2)
        self.recover_action = (3)
        
        # reward weight
        self.weight = 0.5
        # setting 1: instantaneous accuracy reduction dominates
        # self.drw = 0.05 * self.weight # default accuracy loss reward weight, 100 if drop 2%
        # self.arw = 0.5 * self.weight # accuracy change reward weight, 1000 if change 1%, 9000 if change 3%
        # setting 2: 
        self.drw = 0.5 * self.weight # default accuracy loss reward weight, 100 if drop 2%
        self.arw = 22.5 * self.weight # 22.5 for Deit-tiny, 45 for others
        self.trw = 0 # 0.015 * self.weight # time reward weight, 60 if 60s
        self.brw = 2.8 * self.weight # bit cr reward weight, 10000 if 2x->4x
        self.mrw = 1.4 * self.weight # memory cr reward weight
        self.r_weight = [self.drw, self.arw, self.trw, self.brw, self.mrw]
        self.tredarw = 0.1 * self.weight
        logging.info('env set up finish.')
        logging.info('env configuration:')
        logging.debug(vars(self))
    
    def state(self):
        return numpy.array([self.w, self.a, self.prune_bcr, self.prune_mcr, len(self.model.place_layer), self.comp_num['P'], self.comp_num['F'], self.comp_num['Q'], self.bcr_total(), self.mcr, self.acc_drop(), self.acc_change()]) # get_channel_list(self.model)
    
    def quantize(self): 
        if self.comp_full['Q']:
            return False
        self.quant_level += 1
        self.comp_num['Q'] += 1
        current_wa = self.t_conf.wa_list[self.quant_level]
        self.w, self.a = current_wa
        self.model.change_quant(self.w, self.a)
        # self.acc, _ = inference(self.model, self.t_conf.test_loader, self.t_conf)
        if not self.t_conf.F_once:
            self.finetune(self.epoch_after_quant)
        self.bcr *= self.t_conf.bcr_mul[self.quant_level]
        self.mcr *= self.t_conf.mcr_mul[self.quant_level]
        return True
    
    def random_action(self):
        return numpy.random.randint(0,len(self.actions))
    
    def reset(self, seed=0):
        past_t_conf = self.t_conf
        self.__init__(model=self.t_conf.model, dataset=self.t_conf.dataset,
                      suffix=self.t_conf.suffix,test=self.test, pt_finetune=self.t_conf.pt_finetune, dist='already', t_conf=past_t_conf)
        return self.state(), None
        
    def reward_func(self):
        dr = 0
        if self.acc_drop() < -0.8:
            acc_drop_r = math.log(0.2,2)
        else: 
            acc_drop_r = math.log(self.acc_drop()+1,2) # 1 when drop = 1, 3 when drop = 7, 5 when drop = 31, 6 when drop = 63
        if self.acc_drop() < 0:
            acc_drop_r = acc_drop_r/4  # special case, limit a minus acc_drop_r
        # acc_drop_r = self.acc_drop()
        ar = (self.acc_drop_r_past-acc_drop_r) 
        self.acc_drop_r_past = acc_drop_r
        tr = 0
        br = (self.acc/self.base_acc*self.bcr*self.exit_bcr-self.acc_past/self.base_acc*self.bcr_past*self.exit_bcr_past)
        mr = (self.acc/self.base_acc*self.mcr-self.acc_past/self.base_acc*self.mcr_past) # self.acc/self.base_acc 
        self.r_list = [dr, ar, tr, br, mr]
        return self.r_list
        
    def weighted_r(self):
        r = []
        for i, j in zip(self.r_list, self.r_weight):
            r.append(i*j)
        return r

    def update_time(self):
        # update time and return time interval
        past_time = self.time
        self.time = time.time()
        return self.time - past_time
    
    def acc_change(self):
        return self.acc-self.acc_past
    
    def acc_drop(self):
        return self.base_acc - self.acc
    
    def bcr_total(self):
        return self.bcr*self.exit_bcr
    
    def step(self, action):
        ### update loss queue and lr
        def change_lr():
            for optimizer, lr in zip(self.optimizers, self.t_conf.lr):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * self.t_conf.decay_level[self.current_decay_level]
        def truncated_condition():
            if self.step_num >= self.tr_step_num:
                logging.info('TRED: step limitation.')
                return True
            elif (self.comp_full['E'] and self.comp_full['Q'] and self.comp_full['P'] and self.comp_full['F']):
                logging.info('TRED: runout compression.')
                return True
            elif (self.action_names[action] == 'F' and self.acc_drop() > self.t_conf.broken_acc_loss/2): 
                logging.info('TRED: broken accuracy.') 
                return True
            else:
                return False
                # sum(self.comp_num.values()) > self.tr_comp_num or\
        def done_condition():
            return False # for optimzation scene, we don't let it done
        self.do = self.actions[action]() 
        if self.do:
            self.full_trajectory += self.action_names[action]
            if self.t_conf.F_merge and self.action_past != -1:
                if not (self.action_names[self.action_past] == 'F' and self.action_names[action] == 'F'):
                    self.trajectory += self.action_names[action] # Merge the consecutive F's
            else:
                self.trajectory += self.action_names[action]
            self.acc, nomalized_cost = inference(self.model, self.t_conf.test_loader, self.t_conf, beta=self.t_conf.beta)
            self.step_num += 1
            self.action_past = action
        else:
            logging.info("invalid action.")
            self.acc, nomalized_cost = self.acc_past, self.nomalized_cost_past
            # self.action_past = -1
        logging.info('acc ' + str(self.acc) + ' nomalized ' + str(nomalized_cost))
        self.exit_bcr = 1.0/nomalized_cost 
        r_list = self.reward_func()
        r = sum(r_list)
        s_next = self.state()
        
        # update
        self.acc_past = self.acc
        self.nomalized_cost_past = nomalized_cost
        self.bcr_past = self.bcr
        self.mcr_past = self.mcr
        self.exit_bcr_past = self.exit_bcr
        
        # update comp_full
        self.comp_full['Q'] = True if self.quant_level >= len(self.t_conf.wa_list) - 1 else False
        self.comp_full['E'] = True if len(self.exit_to_place) <= 0 else False
        self.comp_full['P'] = True if self.comp_num['P'] >= 20 else False # rep pruner is dangerous for repeat pruning on a not completely finetuned model.
        
        # adjust lr
        if self.t_conf.F_once:
            if self.action_names[action] == 'F':
                self.comp_full['F'] = True
            else:
                self.comp_full['F'] = False
        else:
            # multi-F setting with modified Plateau type lr scheduler
            if self.action_names[action] == 'F' and self.loss_drop_queue.check_all_below(0.01): # n steps' loss change <= 1%
                logging.debug("lr decay")
                if self.current_decay_level == len(self.t_conf.decay_level) - 1:
                    self.comp_full['F'] = True
                else:
                    self.comp_full['F'] = False
                    self.current_decay_level += 2
                    if self.current_decay_level >= len(self.t_conf.decay_level):
                        self.current_decay_level = len(self.t_conf.decay_level) - 1
                    change_lr()
                    self.loss_drop_queue.__init__(3, 1.0)
            if self.acc_change() < -0.5 and self.acc_drop() > self.t_conf.bad_acc_loss and self.current_decay_level > 0:
                logging.debug("lr preserve")
                # acc loss (often from compression) + bad acc + lr already decayed
                level_decay = len(self.t_conf.decay_level) - 1
                for i in self.t_conf.acc_change_limitation:
                    if self.acc_change() < i:
                        self.current_decay_level -= level_decay
                        break
                    level_decay -= 1
                if self.current_decay_level < 0:
                    self.current_decay_level = 0
                change_lr()
                self.loss_drop_queue.__init__(3, 1.0)
        dw = True if done_condition() else False
        bonus = 0
        if truncated_condition():
            tred = True   
        else:
            tred = False
        r += bonus
        r_list.append(bonus)
        self.comp_full['F'] = True if self.acc_drop() < 1 * self.t_conf.broken_acc_loss else False 
        if self.acc_drop() > 1.5*self.t_conf.broken_acc_loss:
            self.comp_full['P'] = True # force finetune
            self.comp_full['Q'] = True
            self.comp_full['E'] = True
        return s_next, None, dw, tred, None, self.comp_full, self.do # r and r_list calculated outside.

    def full_inference(self, model, dataset, beta=0.95):
        self.t_conf.dist_eval = True
        self.t_conf.distributed = False
        self.t_conf.dist_url = 'env://'
        utils.init_distributed_mode(self.t_conf)
        self.t_conf.add_dataloader()
        
        self.acc, nomalized_cost = inference(self.model, self.t_conf.test_loader, self.t_conf, full_dataset=True, beta=beta)
        original_acc = ORIGINAL_ACC[model][dataset]
        logging.info('original acc ' + str(original_acc) + ' beta ' + str(beta) + ' acc ' + str(self.acc) + '({:.2f})'.format(original_acc-self.acc) + ' nomalized ' + str(nomalized_cost) + ' bcr ' + str(self.bcr/nomalized_cost) + ' mcr ' + str(self.mcr))
        self.model.beta = self.t_conf.beta
        
    def pt_finetune(self, lrdecay_b=4, lrdecay_e=4, batch_size=None):
        self.t_conf.dist_eval = True
        self.t_conf.distributed = False
        self.t_conf.dist_url = 'env://'
        utils.init_distributed_mode(self.t_conf)
        self.t_conf.add_dataloader()
        if batch_size:
            self.t_conf.reset_batch_size(batch_size)
        print('Entering post train finetuning...')
        F_once = self.t_conf.F_once
        self.t_conf.F_once = True
        self.comp_full['F'] = False
        self.finetune(epochs=100, patience_es=30, patience_t=5, decay_rate=0.2 ,cooldown_t=0, threshold=1e-4, full_place=False, amp=False, lrdecay_b=lrdecay_b, lrdecay_e=lrdecay_e)
        self.comp_full['F'] = True
        self.t_conf.F_once = F_once

    def finetune(self, epochs = 10, patience_es=1,patience_t=1,decay_rate=0.5,cooldown_t=0,threshold=1e-2, full_place=True, amp=True, lrdecay_b=4, lrdecay_e=0):
        if self.comp_full['F']:
            return False
        # if not epochs:
        #     epochs = self.epoch_each_finetune # default finetune epoch
        current_place = self.model.place_layer # backup place_layer
        if full_place:
            self.model.place_layer = self.model.init_place_layer
        place = True if len(self.model.place_layer) > 0 else False
        if self.t_conf.F_once:
            if type(self.model) is Deit_Comp:
                param_groups_main_exit = [[{"params":i if type(i) is torch.nn.Parameter else i} for i in self.model.main_list],
                                          [{"params":i} for i in self.model.exit_list]]
                for param_groups in param_groups_main_exit:
                    newgroups = []
                    for group in param_groups:
                        no_decay, decay = [], []
                        if type(group["params"]) is torch.nn.Parameter:
                            decay.append(group["params"])
                        else:
                            for name, param in group["params"].named_parameters():
                                if param.ndim <= 1 or name.endswith(".bias"):
                                    no_decay.append(param)
                                else:
                                    decay.append(param)
                        group["params"] = decay
                        newgroup = group.copy()
                        newgroup["weight_decay"] = 0
                        newgroup["params"] = no_decay
                        newgroups.append(newgroup)
                    param_groups += newgroups
                    for group in param_groups:
                        if not group["params"]:
                            param_groups.remove(group)
                self.main_optimizer = torch.optim.AdamW(param_groups_main_exit[0], 
                                    lr=self.t_conf.adamlr*self.t_conf.decay_level[self.t_conf.adamwlr_level], weight_decay=self.t_conf.adamw_wd)
                self.exit_optimizer = torch.optim.AdamW(param_groups_main_exit[1], 
                                    lr=self.t_conf.adamlr*self.t_conf.decay_level[self.t_conf.adamwlr_level], weight_decay=self.t_conf.adamw_wd)
            else:
                self.main_optimizer = torch.optim.SGD([{"params":i if type(i) is torch.nn.Parameter else i.parameters()} for i in self.model.main_list], lr=self.t_conf.sgdlr*self.t_conf.decay_level[lrdecay_b],
                                            momentum=self.t_conf.momentum, weight_decay=self.t_conf.sgd_wd)
                self.exit_optimizer = torch.optim.Adam([{"params":i.parameters()} for i in self.model.exit_list], 
                                                lr=self.t_conf.adamlr*self.t_conf.decay_level[lrdecay_e], weight_decay=self.t_conf.adam_wd)
            self.optimizers = [self.main_optimizer, self.exit_optimizer]
            schedulers = []
            for optimizer in self.optimizers:
                schedulers.append(PlateauLRScheduler(
                            optimizer,
                            decay_rate=decay_rate,   
                            patience_t=patience_t,   
                            mode='min',   
                            cooldown_t=cooldown_t,      
                            threshold=threshold,  
                            lr_min=1e-6,  
                            verbose=True
                ))
            early_stop_patience = patience_es  
            acc1, losses = forward(self.t_conf.val_loader, self.model, self.t_conf.criterion, -1, 
                    training=False, optimizer=None, args=self.t_conf, place=True, onlyexit=False, amp=amp)
            best_model_state = self.model.state_dict().copy()
            best_val_acc1 = acc1
            best_epoch = 0
            no_improve_counter = 0 
        self.comp_num['F'] += 1
        for e in range(epochs):
               # same with pruner.pruning, do operation on all exit layers
            logging.debug(self.optimizers[0])
            _, _ = forward(self.t_conf.train_loader, self.model, self.t_conf.criterion, e, 
                    training=True, optimizer=self.optimizers[0], args=self.t_conf, place=False, onlyexit=False, amp=amp)
            if place:
                logging.debug(self.optimizers[1])
                _, _ = forward(self.t_conf.train_loader, self.model, self.t_conf.criterion, e, 
                        training=True, optimizer=self.optimizers[1], args=self.t_conf, place=True, onlyexit=True, amp=amp)
            acc1, losses = forward(self.t_conf.val_loader, self.model, self.t_conf.criterion, e, 
                    training=False, optimizer=None, args=self.t_conf, place=True, onlyexit=False, amp=amp)
            if self.t_conf.F_once:
                if place:
                    schedulers[0].step(e, losses[0])
                    schedulers[1].step(e, losses[1])
                else:
                    schedulers[0].step(e, losses)
                if acc1 > best_val_acc1:
                    best_val_acc1 = acc1
                    best_model_state = self.model.state_dict().copy()
                    no_improve_counter = 0 
                    best_epoch = e
                else:
                    no_improve_counter += 1
                if no_improve_counter >= early_stop_patience:
                    logging.debug(f"No improvement for {early_stop_patience} epochs. Early stopping!")
                    break 
            else:
                loss = sum(losses)
                self.loss_drop_queue.append((self.loss_past-loss)/(loss+1e-9))
                self.loss_past = loss
                self.epoch += 1
        if self.t_conf.F_once:
            self.model.load_state_dict(best_model_state)
            logging.info(f'finetune epoch num: {e}, best occured {best_epoch}')
        self.model.place_layer = current_place
        return True

    def delete_exit(self):
        if len(self.model.place_layer) == 0:
            return False# no place to delete exit
        self.exit_to_place.insert(0, self.model.place_layer.pop())
        # self.model.exit_num = len(self.model.place_layer)
        self.comp_num['E'] = len(self.model.place_layer)
        return True

    def add_exit(self):
        if self.comp_full['E']:
            return False
        for _ in range(self.num_each_add):
            if len(self.exit_to_place) <= 0:
                self.comp_full['E'] = True
                return False# no place to insert exit
            self.model.place_layer.append(self.exit_to_place[0])
            # self.model.exit_num = len(self.model.place_layer)
            self.comp_num['E'] = len(self.model.place_layer)
            self.exit_to_place = self.exit_to_place[1:]
        return True
        
    def prune(self):
        # self.epoch = int(self.epoch/2)
        # if self.comp_num['P'] >= 1 and self.acc_drop() > self.t_conf.bad_acc_loss:
        if self.comp_full['P']:
            return False
        self.comp_num['P'] += 1
        bcr_inc, mcr_inc = pruning(self.model, self.example_inputs, self.prune_ratio)
        # self.acc, _ = inference(self.model, self.t_conf.test_loader, self.t_conf)
        logging.info('prune inc')
        logging.info('bcr inc: '+str(bcr_inc)+', mcr inc: '+str(mcr_inc))
        if not self.t_conf.F_once:
            self.finetune(self.epoch_after_prune)
        self.bcr *= bcr_inc
        self.prune_bcr *= bcr_inc
        self.mcr *= mcr_inc
        self.prune_mcr *= mcr_inc
        return True
