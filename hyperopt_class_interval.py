from hyperopt import hp,tpe,Trials,fmin,space_eval,STATUS_OK,STATUS_FAIL,rand
from nets.nets_interval import *
from utils.dataloaders_interval import get_dataloader_interval
import torch
import os
import pickle
import numpy as np
from pycox.evaluation import EvalSurv
import pandas as pd
import shutil
from torch.utils.tensorboard import SummaryWriter
from RAdam.radam import RAdam
from tqdm import tqdm
from pycox.models import CoxCC,CoxPH,CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
import torchtuples as tt
import time
def square(x):
    return x**2

def linear(x):
    return x

def ibs_calc(S,mask_1,mask_2,gst,gt):

    weights_2 = 1./gt
    weights_2[weights_2==float("Inf")]=0.0
    weights_1 = 1./gst
    weights_1[weights_1==float("Inf")]=0.0
    return ((S**2)*mask_1)*weights_1 + ((1.-S)**2 * mask_2)*weights_2

def ibll_calc(S,mask_1,mask_2,gst,gt):
    weights_2 = 1./gt
    weights_2[weights_2==float("Inf")]=0.0
    weights_1 = 1./gst
    weights_1[weights_1==float("Inf")]=0.0
    return ((torch.log(1-S+1e-6))*mask_1)*weights_1 + ((torch.log(S+1e-6)) * mask_2)*weights_2

def simpsons_composite(S,step_size,n):
    idx_odd = torch.arange(1,n-1,2)
    idx_even = torch.arange(2,n,2)
    S[idx_odd]=S[idx_odd]*4
    S[idx_even]=S[idx_even]*2
    return torch.sum(S)*step_size/3.

class fifo_list():
    def __init__(self,n):
        self.fifo_list = [0]*n
        self.n=n
    def insert(self,el):
        self.fifo_list.pop(0)
        self.fifo_list.append(el)
    def __len__(self):
        return len(self.fifo_list)

    def get_sum(self):
        return sum(self.fifo_list)

class hyperopt_training_interval():
    def __init__(self,job_param,hyper_param_space):
        self.d_out = job_param['d_out']
        self.dataset_string = job_param['dataset_string']
        self.seed = job_param['seed']
        self.total_epochs = job_param['total_epochs']
        self.device = job_param['device']
        self.patience = job_param['patience']
        self.hyperits = job_param['hyperits']
        self.selection_criteria = job_param['selection_criteria']
        self.grid_size  = job_param['grid_size']
        self.validation_interval = job_param['validation_interval']
        self.test_grid_size  = job_param['test_grid_size']
        self.objective = job_param['objective']
        self.net_type = job_param['net_type']
        self.fold_idx = job_param['fold_idx']
        self.savedir = job_param['savedir']
        self.reg_mode = job_param['reg_mode']
        self.ibs_est_deltas = job_param['ibs_est_deltas']
        self.use_sotle = job_param['use_sotle']
        self.global_hyperit = 0
        self.debug = False
        torch.cuda.set_device(self.device)
        self.save_path = f'./{self.savedir}/{self.dataset_string}_seed={self.seed}_fold_idx={self.fold_idx}_objective={self.objective}_{self.net_type}/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            shutil.rmtree(self.save_path)
            os.makedirs(self.save_path)
        self.hyperopt_params = ['bounding_op', 'transformation', 'depth_x', 'width_x','depth_t', 'width_t', 'depth', 'width', 'bs', 'lr','direct_dif','dropout','eps','weight_decay','reg_lambda','T_losses']
        self.get_hyperparameterspace(hyper_param_space)
        if self.reg_mode=='ibs':
            self.reg_func = ibs_calc
        elif self.reg_mode=='ibll':
            self.reg_func = ibll_calc

    def calc_eval_objective(self,S_left,S_right):
        val_likelihood = self.train_objective(S_left,S_right)
        return val_likelihood

    def get_hyperparameterspace(self,hyper_param_space):
        self.hyperparameter_space = {}
        for string in self.hyperopt_params:
            self.hyperparameter_space[string] = hp.choice(string, hyper_param_space[string])

    def __call__(self,parameters_in):
        print(f"----------------new hyperopt iteration {self.global_hyperit}------------------")
        print(parameters_in)
        self.dataloader = get_dataloader_interval(self.dataset_string,parameters_in['bs'],self.seed,self.fold_idx)
        self.cycle_length = self.dataloader.__len__()//self.validation_interval+1
        print('cycle_length',self.cycle_length)
        self.T_losses = parameters_in['T_losses']
        net_init_params = {
            'd_in_x' : 0 if isinstance(self.dataloader.dataset.X,list) else self.dataloader.dataset.X.shape[1],
            'cat_size_list': self.dataloader.dataset.unique_cat_cols,
            'd_in_y' : self.dataloader.dataset.y_left.shape[1],
            'd_out' : self.d_out,
            'bounding_op':parameters_in['bounding_op'],
            'transformation':parameters_in['transformation'],
            'layers_x': [parameters_in['width_x']]*parameters_in['depth_x'],
            'layers_t': [parameters_in['width_t']]*parameters_in['depth_t'],
            'layers': [parameters_in['width']]*parameters_in['depth'],
            'direct_dif':parameters_in['direct_dif'],
            'objective':self.objective,
            'dropout':parameters_in['dropout'],
            'eps':parameters_in['eps']
        }
        self.reg_lambda = parameters_in['reg_lambda']
        self.train_objective = get_objective(self.objective)
        if self.net_type=='survival_net_basic':
            self.model = survival_net_basic_interval(**net_init_params).to(self.device)
        # elif self.net_type=='benchmark':
        #     self.model = MLPVanillaCoxTime(in_features=net_init_params['d_in_x'],
        #                             num_nodes=net_init_params['layers'],
        #                             batch_norm=False,
        #                             dropout=net_init_params['dropout'],
        #                             activation=torch.nn.Tanh) #Actual net to be used

        self.optimizer = RAdam(self.model.parameters(),lr=parameters_in['lr'],weight_decay=parameters_in['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',patience=self.patience//4,min_lr=1e-3,factor=0.9)
        results = self.full_loop()
        del self.optimizer
        self.global_hyperit+=1
        results['net_init_params'] = net_init_params
        torch.cuda.empty_cache()
        return results

    def do_metrics(self,training_loss,likelihood,i):
        val_likelihood= self.validation_score()
        if self.debug:
            test_likelihood = self.test_score()
            self.writer.add_scalar('Loss/train', training_loss, i)
            self.writer.add_scalar('Loss/val', val_likelihood, i)
            self.writer.add_scalar('Loss/test', test_likelihood, i)
            print(
                f'test_likelihood: {test_likelihood} ')
            # self.debug_list.append(test_ibs)
        if self.selection_criteria == 'train':
            criteria = val_likelihood  # minimize #
        print(f'total_loss: {training_loss} likelihood: {likelihood}')
        print(f'criteria score: {criteria} val likelihood: {val_likelihood}')
        return criteria


    def eval_func(self,i,training_loss,likelihood):
        if i % self.cycle_length == 0:
            criteria = self.do_metrics(training_loss,likelihood,i)
            self.scheduler.step(criteria)
            if criteria < self.best:
                self.best = criteria
                print('new best val score: ', self.best)
                print('Dumping model')
                self.dump_model()
                self.counter = 0
            else:
                self.counter += 1
            if self.counter>self.patience:
                return True

    def training_loop(self,epoch):
        self.dataloader.dataset.set(mode='train')
        total_loss_train=0.
        tot_likelihood=0.
        self.model = self.model.train()
        for i,(X,x_cat,y_left,y_right,inf_indicator) in enumerate(tqdm(self.dataloader)):
            y_left = y_left.to(self.device)
            y_right = y_right.to(self.device)
            if not isinstance(x_cat,list): #F
                x_cat = x_cat.to(self.device)
            if not isinstance(X,list):
                X = X.to(self.device)
            S_left,S_right = self.model.forward_cum(X,y_left,y_right,inf_indicator,x_cat)
            loss =self.train_objective(S_left,S_right)
            total_loss =loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            total_loss_train+=total_loss.detach()
            tot_likelihood+=loss.detach()
            if not self.use_sotle:
                if self.eval_func(i,total_loss_train/(i+1),tot_likelihood/(i+1)):
                    return True
        return False

    def eval_loop(self,grid_size):
        # S_series_container = []
        S_left_list = []
        S_right_list = []
        y_left_list = []
        y_right_list = []
        self.model = self.model.eval()
        # durations  = self.dataloader.dataset.invert_duration(self.dataloader.dataset.y.numpy()).squeeze()
        # events  = self.dataloader.dataset.delta.numpy()
        # chunks = self.dataloader.batch_size//50+1
        # t_grid_np = np.linspace(self.dataloader.dataset.min_duration, self.dataloader.dataset.max_duration,
        #                         grid_size)
        # time_grid = torch.from_numpy(t_grid_np).float().unsqueeze(-1)
        for i,(X,x_cat,y_left,y_right,inf_indicator) in enumerate(tqdm(self.dataloader)):
            y_left = y_left.to(self.device)
            y_right = y_right.to(self.device)
            if not isinstance(X,list):
                X = X.to(self.device)
            if not isinstance(x_cat, list):  # F
                x_cat = x_cat.to(self.device)
            S_left, S_right = self.model.forward_cum(X, y_left, y_right, inf_indicator, x_cat)
            # if not isinstance(x_cat, list):
            #     for chk,chk_cat in zip(torch.chunk(X, chunks),torch.chunk(x_cat, chunks)):
            #         input_time = time_grid.repeat((chk.shape[0], 1)).to(self.device)
            #         X_repeat = chk.repeat_interleave(grid_size, 0)
            #         x_cat_repeat = chk_cat.repeat_interleave(grid_size, 0)
            #         S_serie = self.model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
            #         S_serie = S_serie.detach()
            #         S_series_container.append(S_serie.view(-1, grid_size).t().cpu())
            # else:
            #     x_cat_repeat = []
            #     for chk in torch.chunk(X, chunks):
            #         input_time = time_grid.repeat((chk.shape[0], 1)).to(self.device)
            #         X_repeat = chk.repeat_interleave(grid_size, 0)
            #         S_serie = self.model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
            #         S_serie = S_serie.detach()
            #         S_series_container.append(S_serie.view(-1, grid_size).t().cpu())
            S_left_list.append(S_left)
            S_right_list.append(S_right)
            y_left_list.append(y_left.cpu().numpy())
            y_right_list.append(y_right.cpu().numpy())
        S_left_cat = torch.cat(S_left_list)
        S_right_cat = torch.cat(S_right_list)
        # S_series_container = pd.DataFrame(torch.cat(S_series_container,1).numpy())
        # t_grid_np = self.dataloader.dataset.invert_duration(t_grid_np.reshape(-1, 1)).squeeze()
        val_likelihood = self.calc_eval_objective(S_left_cat, S_right_cat,)
        return val_likelihood.item()

    def train_score(self):
        self.dataloader.dataset.set(mode='train')
        return self.eval_loop(self.grid_size)
    def validation_score(self):
        self.dataloader.dataset.set(mode='val')
        return self.eval_loop(self.grid_size)

    def test_score(self):
        self.dataloader.dataset.set(mode='test')
        return self.eval_loop(self.grid_size)
    def dump_model(self):
        torch.save(self.model.state_dict(), self.save_path + f'best_model_{self.global_hyperit}.pt')

    def load_model(self):
        self.model.load_state_dict(torch.load(self.save_path + f'best_model_{self.global_hyperit}.pt'))

    def full_loop(self):
        self.counter = 0
        self.best = np.inf
        self.sotl_e_list = fifo_list(n=self.T_losses)
        if self.debug:
            self.writer =SummaryWriter()
            self.debug_list = []
        for i in range(self.total_epochs):
            if self.training_loop(i):
                break
        if self.debug:
            if self.debug_list:
                print(f'best test ibs {min(self.debug_list)}')
        self.load_model()
        val_likelihood = self.validation_score()
        test_likelihood = self.test_score()

        return self.parse_results(val_likelihood,
                                  test_likelihood)

    def parse_results(self, val_likelihood,
                      test_likelihood,):

        if self.selection_criteria == 'train':
            criteria = val_likelihood
            criteria_test = test_likelihood

        return {'loss': criteria,
                'status': STATUS_OK,
                'test_loss': criteria_test,
                'test_loglikelihood':test_likelihood,
                }


    def run(self):
        trials = Trials()
        best = fmin(fn=self,
                    space=self.hyperparameter_space,
                    algo=tpe.suggest,
                    max_evals=self.hyperits,
                    trials=trials,
                    verbose=True)
        print(space_eval(self.hyperparameter_space, best))
        pickle.dump(trials,
                    open(self.save_path + 'hyperopt_database.p',
                         "wb"))


    def post_process(self):
        trials = pickle.load(open(self.save_path + 'hyperopt_database.p',
                         "rb"))

        if self.selection_criteria == 'train':
            reverse = False
        elif self.selection_criteria == 'concordance':
            reverse = True
        elif self.selection_criteria in ['ibs','ibs_likelihood']:
            reverse = False
        elif self.selection_criteria == 'inll':
            reverse = False

        best_trial = sorted(trials.results, key=lambda x: x['test_loss'], reverse=reverse)[0] #low to high
        data = [best_trial['test_loglikelihood']]
        df = pd.DataFrame([data],columns=['test_loglikelihood'])
        print(df)
        df.to_csv(self.save_path+'best_results.csv',index_label=False)



















