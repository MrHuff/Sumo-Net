from hyperopt import hp,tpe,Trials,fmin,space_eval,STATUS_OK,STATUS_FAIL,rand
from nets.nets import *
from utils.dataloaders import get_dataloader
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
def square(x):
    return x**2

class hyperopt_training():
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
        self.global_hyperit = 0
        self.debug = False
        torch.cuda.set_device(self.device)
        self.save_path = f'./{self.savedir}/{self.dataset_string}_seed={self.seed}_fold_idx={self.fold_idx}_objective={self.objective}_{self.net_type}/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            shutil.rmtree(self.save_path)
            os.makedirs(self.save_path)
        self.hyperopt_params = ['bounding_op', 'transformation', 'depth_x', 'width_x','depth_t', 'width_t', 'depth', 'width', 'bs', 'lr','direct_dif','dropout','eps','weight_decay']
        self.get_hyperparameterspace(hyper_param_space)

    def calc_eval_objective(self,S,f,S_extended,durations,events,time_grid):
        val_likelihood = self.train_objective(S,f)
        eval_obj = EvalSurv(surv=S_extended,durations=durations,events=events,censor_surv='km') #Add index and pass as DF
        conc = eval_obj.concordance_td('antolini')
        ibs = eval_obj.integrated_brier_score(time_grid)
        inll = eval_obj.integrated_nbll(time_grid)
        return val_likelihood,conc,ibs,inll

    def get_hyperparameterspace(self,hyper_param_space):
        self.hyperparameter_space = {}
        for string in self.hyperopt_params:
            self.hyperparameter_space[string] = hp.choice(string, hyper_param_space[string])

    def __call__(self,parameters_in):
        print(f"----------------new hyperopt iteration {self.global_hyperit}------------------")
        print(parameters_in)
        self.dataloader = get_dataloader(self.dataset_string,parameters_in['bs'],self.seed,self.fold_idx)
        self.cycle_length = self.dataloader.__len__()//self.validation_interval
        net_init_params = {
            'd_in_x' : self.dataloader.dataset.X.shape[1],
            'cat_size_list': self.dataloader.dataset.unique_cat_cols,
            'd_in_y' : self.dataloader.dataset.y.shape[1],
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
        self.train_objective = get_objective(self.objective)
        if self.net_type=='survival_net':
            self.model = survival_net(**net_init_params).to(self.device)
        elif self.net_type=='survival_net_variant':
            self.model = survival_net_variant(**net_init_params).to(self.device)
        elif self.net_type=='survival_net_basic':
            self.model = survival_net(**net_init_params).to(self.device)
        elif self.net_type=='survival_net_nocov':
            self.model = survival_net_nocov(**net_init_params).to(self.device)
        elif self.net_type=='ocean_net':
            self.model = ocean_net(**net_init_params).to(self.device)
        elif self.net_type=='cox_net':
            self.model = cox_net(**net_init_params).to(self.device)

        self.optimizer = RAdam(self.model.parameters(),lr=parameters_in['lr'],weight_decay=parameters_in['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',patience=self.patience//4,min_lr=1e-3,factor=0.9)
        results = self.full_loop()
        self.global_hyperit+=1
        results['net_init_params'] = net_init_params
        del self.model
        del self.optimizer
        del self.dataloader
        torch.cuda.empty_cache()
        return results

    def eval_func(self,i,training_loss):
        if i % self.cycle_length == 0:
            val_likelihood, conc, ibs, inll = self.validation_score()
            if self.debug:
                test_likelihood, test_conc, test_ibs, test_inll = self.test_score()
                self.writer.add_scalar('Loss/train', training_loss, i)
                self.writer.add_scalar('Loss/val', val_likelihood, i)
                self.writer.add_scalar('Loss/test', test_likelihood, i)
                self.writer.add_scalar('conc/val', conc, i)
                self.writer.add_scalar('conc/test', test_conc, i)
                self.writer.add_scalar('ibs/val', ibs, i)
                self.writer.add_scalar('ibs/test', test_ibs, i)
                self.writer.add_scalar('inll/val', inll, i)
                self.writer.add_scalar('inll/test', test_inll, i)
                print('test:', test_likelihood, test_conc, test_ibs, test_inll)
            if self.selection_criteria == 'train':
                criteria = val_likelihood  # minimize #
            elif self.selection_criteria == 'concordance':
                criteria = -conc  # maximize
            elif self.selection_criteria == 'ibs':
                criteria = ibs  # minimize
            elif self.selection_criteria == 'inll':
                criteria = inll  # maximize
            print('criteria score: ', criteria)
            print('val conc', conc)
            print('val ibs', ibs)
            print('val inll', inll)
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

    def training_loop(self):
        self.dataloader.dataset.set(mode='train')
        total_loss_train=0
        self.model = self.model.train()
        for i,(X,x_cat,y,delta) in enumerate(tqdm(self.dataloader)):
            X = X.to(self.device)
            y = y.to(self.device)
            delta = delta.to(self.device)
            mask = delta==1
            X_f = X[mask, :]
            y_f = y[mask, :]
            if not isinstance(x_cat,list): #F
                x_cat = x_cat.to(self.device)
                x_cat_f = x_cat[mask,:]
            else:
                x_cat_f = []
            S = self.model.forward_cum(X,y,mask,x_cat)
            f = self.model(X_f,y_f,x_cat_f)
            loss = self.train_objective(S,f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss_train+=loss.detach()
            if self.eval_func(i,total_loss_train/(i+1)):
                return True
        return False

    def eval_loop_kkbox(self,grid_size):
        S_series_container = []
        S_log = []
        f_log = []
        durations = []
        events = []
        self.model = self.model.eval()
        # durations  = self.dataloader.dataset.invert_duration(self.dataloader.dataset.y.numpy()).squeeze()
        # events  = self.dataloader.dataset.delta.numpy()
        chunks = self.dataloader.batch_size // 50 + 1
        with torch.no_grad():
            t_grid_np = np.linspace(self.dataloader.dataset.min_duration, self.dataloader.dataset.max_duration,
                                    grid_size)
            time_grid = torch.from_numpy(t_grid_np).float().unsqueeze(-1)
            for i, (X, x_cat, y, delta) in enumerate(tqdm(self.dataloader)):
                X = X.to(self.device)
                y = y.to(self.device)
                delta = delta.to(self.device)
                mask = delta == 1
                X_f = X[mask, :]
                y_f = y[mask, :]
                if not isinstance(x_cat, list):
                    x_cat = x_cat.to(self.device)
                    x_cat_f = x_cat[mask, :]
                else:
                    x_cat_f = []
                S = self.model.forward_cum(X, y, mask, x_cat)
                f = self.model(X_f, y_f, x_cat_f)
                S_log.append(S)
                f_log.append(f)
                if i*self.dataloader.batch_size<50000:
                    if not isinstance(x_cat, list):
                        for chk, chk_cat in zip(torch.chunk(X, chunks), torch.chunk(x_cat, chunks)):
                            input_time = time_grid.repeat((chk.shape[0], 1)).to(self.device)
                            X_repeat = chk.repeat_interleave(grid_size, 0)
                            x_cat_repeat = chk_cat.repeat_interleave(grid_size, 0)
                            S_serie = self.model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
                            S_series_container.append(S_serie.view(-1, grid_size).t().cpu())
                    else:
                        x_cat_repeat = []
                        for chk in torch.chunk(X, chunks):
                            input_time = time_grid.repeat((chk.shape[0], 1)).to(self.device)
                            X_repeat = chk.repeat_interleave(grid_size, 0)
                            S_serie = self.model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
                            S_series_container.append(S_serie.view(-1, grid_size).t().cpu())
                    durations.append(y.cpu().numpy())
                    events.append(delta.cpu().numpy())
            durations = self.dataloader.dataset.invert_duration(np.concatenate(durations)).squeeze()
            # durations = np.concatenate(durations).squeeze()
            events = np.concatenate(events).squeeze()

            S_log = torch.cat(S_log)
            f_log = torch.cat(f_log)
            # reshape(-1, 1)).squeeze()
            S_series_container = pd.DataFrame(torch.cat(S_series_container, 1).numpy())
            t_grid_np = self.dataloader.dataset.invert_duration(t_grid_np.reshape(-1, 1)).squeeze()
            S_series_container = S_series_container.set_index(t_grid_np)
            # S_series_container=S_series_container.set_index(t_grid_np)
            val_likelihood, conc, ibs, inll = self.calc_eval_objective(S_log, f_log, S_series_container,
                                                                       durations=durations, events=events,
                                                                       time_grid=t_grid_np)
        return val_likelihood.item(), conc, ibs, inll

    def eval_loop(self,grid_size):
        S_series_container = []
        S_log = []
        f_log = []
        durations = []
        events = []
        # self.model = self.model.eval()
        # durations  = self.dataloader.dataset.invert_duration(self.dataloader.dataset.y.numpy()).squeeze()
        # events  = self.dataloader.dataset.delta.numpy()
        chunks = self.dataloader.batch_size//50+1
        with torch.no_grad():
            t_grid_np = np.linspace(self.dataloader.dataset.min_duration, self.dataloader.dataset.max_duration,
                                    grid_size)
            time_grid = torch.from_numpy(t_grid_np).float().unsqueeze(-1)
            for i, (X, x_cat, y, delta) in enumerate(tqdm(self.dataloader)):
                X = X.to(self.device)
                y = y.to(self.device)
                delta = delta.to(self.device)
                mask = delta == 1
                X_f = X[mask, :]
                y_f = y[mask, :]
                if not isinstance(x_cat, list):
                    x_cat = x_cat.to(self.device)
                    x_cat_f = x_cat[mask, :]
                else:
                    x_cat_f = []
                S = self.model.forward_cum(X, y,mask,x_cat)
                f = self.model(X_f, y_f,x_cat_f)
                if not isinstance(x_cat, list):
                    for chk,chk_cat in zip(torch.chunk(X, chunks),torch.chunk(x_cat, chunks)):
                        input_time = time_grid.repeat((chk.shape[0], 1)).to(self.device)
                        X_repeat = chk.repeat_interleave(grid_size, 0)
                        x_cat_repeat = chk_cat.repeat_interleave(grid_size, 0)
                        S_serie = self.model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
                        S_series_container.append(S_serie.view(-1, grid_size).t().cpu())
                else:
                    x_cat_repeat = []
                    for chk in torch.chunk(X, chunks):
                        input_time = time_grid.repeat((chk.shape[0], 1)).to(self.device)
                        X_repeat = chk.repeat_interleave(grid_size, 0)
                        S_serie = self.model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
                        S_series_container.append(S_serie.view(-1, grid_size).t().cpu())
                S_log.append(S)
                f_log.append(f)
                durations.append(y.cpu().numpy())
                events.append(delta.cpu().numpy())
            durations = self.dataloader.dataset.invert_duration(np.concatenate(durations)).squeeze()
            #durations = np.concatenate(durations).squeeze()
            events = np.concatenate(events).squeeze()

            S_log = torch.cat(S_log)
            f_log = torch.cat(f_log)
            # reshape(-1, 1)).squeeze()
            S_series_container = pd.DataFrame(torch.cat(S_series_container,1).numpy())
            t_grid_np = self.dataloader.dataset.invert_duration(t_grid_np.reshape(-1, 1)).squeeze()
            S_series_container=S_series_container.set_index(t_grid_np)
            #S_series_container=S_series_container.set_index(t_grid_np)
            val_likelihood,conc,ibs,inll = self.calc_eval_objective(S_log, f_log,S_series_container,durations=durations,events=events,time_grid=t_grid_np)
        return val_likelihood.item(),conc,ibs,inll

    def validation_score(self):
        self.dataloader.dataset.set(mode='val')
        if self.dataset_string=='kkbox':
            return self.eval_loop_kkbox(self.grid_size)
        else:
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
        if self.debug:
            self.writer =SummaryWriter()
        for i in range(self.total_epochs):
            if self.training_loop():
                break
        self.load_model()
        val_likelihood,val_conc,val_ibs,val_inll = self.validation_score()
        test_likelihood,test_conc,test_ibs,test_inll = self.test_score()

        if self.selection_criteria == 'train':
            criteria = val_likelihood
            criteria_test = test_likelihood
        elif self.selection_criteria == 'concordance':
            criteria = -val_conc
            criteria_test = test_conc
        elif self.selection_criteria == 'ibs':
            criteria = val_ibs
            criteria_test = test_ibs
        elif self.selection_criteria == 'inll':
            criteria = val_inll
            criteria_test = test_inll

        return {'loss': criteria,
                'status': STATUS_OK,
                'test_loss': criteria_test,
                'test_loglikelihood':test_likelihood,
                 'test_conc':test_conc,
                 'test_ibs':test_ibs,
                 'test_inll':test_inll,
                'val_loglikelihood': val_likelihood,
                'val_conc': val_conc,
                'val_ibs': val_ibs,
                'val_inll': val_inll,
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
        elif self.selection_criteria == 'ibs':
            reverse = False
        elif self.selection_criteria == 'inll':
            reverse = False
        best_trial = sorted(trials.results, key=lambda x: x['test_loss'], reverse=reverse)[0] #low to high
        data = [best_trial['test_loglikelihood'],best_trial['test_conc'],best_trial['test_ibs'],best_trial['test_inll'],
                best_trial['val_loglikelihood'],best_trial['val_conc'],best_trial['val_ibs'],best_trial['val_inll']]
        df = pd.DataFrame([data],columns=['test_loglikelihood','test_conc','test_ibs','test_inll',
                                          'val_loglikelihood','val_conc','val_ibs','val_inll'])
        print(df)
        df.to_csv(self.save_path+'best_results.csv',index_label=False)



















