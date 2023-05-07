import copy

from hyperopt import hp, tpe, Trials, fmin, space_eval, STATUS_OK, STATUS_FAIL, rand
from nets.nets import *
from utils.dataloaders import get_dataloader
import torch
import os
import pickle
import numpy as np
from utils.dataloaders import custom_dataloader
from pycox_local.pycox.evaluation import EvalSurv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.deephit_transformation_fix import *
import gpytorch
from nets.GWI import *
kernel_obj = Kernel()

def square(x):
    return x ** 2

def linear(x):
    return x

class GWISurvivalKernel(torch.nn.Module):
    def __init__(self,X,T):
        super(GWISurvivalKernel, self).__init__()
        x_median = kernel_obj.get_median_ls(X,X)
        t_median = kernel_obj.get_median_ls(T,T)
        self.cov_kernel = gpytorch.kernels.RBFKernel()
        self.time_kernel = gpytorch.kernels.RBFKernel()
        self.cov_kernel._set_lengthscale(x_median)
        self.time_kernel._set_lengthscale(t_median)

    def unsqueezeXT(self,x1):
        if x1.dim()==3:
            x = x1[:,:,:-1] #should just be diagonal style setup
            t = x1[:,:,-1].unsqueeze(1) #usually just 1-D hence needs unsqueeze()
        else:
            x = x1[:,:-1]
            t = x1[:,-1] #should be 2d?!
        return x,t
    def forward(self, x1,x2=None):
        if x2 is None:
            x,t = self.unsqueezeXT(x1)
            return self.cov_kernel(x) * self.time_kernel(t)
        else:
            x,t = self.unsqueezeXT(x1)
            x_,t_ = self.unsqueezeXT(x2)
            # if x1.dim()<x2.dim():
            #     x,t = self.unsqueezeXT(x1)
            #
            # elif x1.dim()==x2.dim():
            #     x,t = self.unsqueezeXT(x1)
            #     x_,t_ = self.unsqueezeXT(x2)
            # else:
            #     pass
            return self.cov_kernel(x,x_) * self.time_kernel(t,t_)

def ibs_calc(S, mask_1, mask_2, gst, gt):
    weights_2 = 1. / gt
    weights_2[weights_2 == float("Inf")] = 0.0
    weights_1 = 1. / gst
    weights_1[weights_1 == float("Inf")] = 0.0
    return ((S ** 2) * mask_1) * weights_1 + ((1. - S) ** 2 * mask_2) * weights_2


def ibll_calc(S, mask_1, mask_2, gst, gt):
    weights_2 = 1. / gt
    weights_2[weights_2 == float("Inf")] = 0.0
    weights_1 = 1. / gst
    weights_1[weights_1 == float("Inf")] = 0.0
    return ((torch.log(1 - S + 1e-6)) * mask_1) * weights_1 + ((torch.log(S + 1e-6)) * mask_2) * weights_2


def simpsons_composite(S, step_size, n):
    idx_odd = torch.arange(1, n - 1, 2)
    idx_even = torch.arange(2, n, 2)
    S[idx_odd] = S[idx_odd] * 4
    S[idx_even] = S[idx_even] * 2
    return torch.sum(S) * step_size / 3.


class fifo_list():
    def __init__(self, n):
        self.fifo_list = [0] * n
        self.n = n

    def insert(self, el):
        self.fifo_list.pop(0)
        self.fifo_list.append(el)

    def __len__(self):
        return len(self.fifo_list)

    def get_sum(self):
        return sum(self.fifo_list)


class hyperopt_training():
    def __init__(self, job_param, hyper_param_space, custom_dataloader=None):
        self.d_out = job_param['d_out']
        self.dataset_string = job_param['dataset_string']
        self.seed = job_param['seed']
        self.total_epochs = job_param['total_epochs']
        self.device = job_param['device']
        self.patience = job_param['patience']
        self.hyperits = job_param['hyperits']
        self.selection_criteria = job_param['selection_criteria']
        self.grid_size = job_param['grid_size']
        self.validation_interval = job_param['validation_interval']
        self.test_grid_size = job_param['test_grid_size']
        self.objective = job_param['objective']
        self.net_type = job_param['net_type']
        self.fold_idx = job_param['fold_idx']
        self.savedir = job_param['savedir']
        self.chunks = job_param['chunks']
        self.max_series_accumulation = job_param['max_series_accumulation']
        self.validate_train = job_param['validate_train']
        self.global_hyperit = 0
        self.best = np.inf
        self.debug = False
        #torch.cuda.set_device(self.device)
        self.custom_dataloader = custom_dataloader
        self.save_path = f'{self.savedir}/{self.dataset_string}_seed={self.seed}_fold_idx={self.fold_idx}_objective={self.objective}_{self.net_type}/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.hyperopt_params = ['bounding_op', 'transformation', 'depth_x', 'width_x', 'depth_t', 'width_t', 'depth',
                                'width', 'bs', 'lr', 'direct_dif', 'dropout', 'eps', 'weight_decay','m_factor','m_P','reg','x_s']
        self.deephit_params = ['alpha', 'sigma', 'num_dur']
        self.get_hyperparameterspace(hyper_param_space)

    def calc_eval_objective(self, S, f, S_extended, durations, events, time_grid):
        val_likelihood = self.train_objective(S, f)
        eval_obj = EvalSurv(surv=S_extended, durations=durations, events=events,
                            censor_surv='km')  # Add index and pass as DF
        conc = eval_obj.concordance_td()
        ibs = eval_obj.integrated_brier_score(time_grid)
        inll = eval_obj.integrated_nbll(time_grid)
        return val_likelihood, conc, ibs, inll

    def benchmark_eval(self, y, events, X, wrapper):
        surv = wrapper.predict_surv_df(X)
        t_grid_np = np.linspace(y.min(), y.max(), surv.index.shape[0])
        surv = surv.set_index(t_grid_np)
        ev = EvalSurv(surv=surv, durations=y, events=events, censor_surv='km')
        conc = ev.concordance_td()
        ibs = ev.integrated_brier_score(t_grid_np)
        inll = ev.integrated_nbll(t_grid_np)
        return conc, ibs, inll

    def get_hyperparameterspace(self, hyper_param_space):
        self.hyperparameter_space = {}
        for string in self.hyperopt_params:
            self.hyperparameter_space[string] = hp.choice(string, hyper_param_space[string])

        if self.net_type == 'deephit_benchmark':
            for string in self.deephit_params:
                self.hyperparameter_space[string] = hp.choice(string, hyper_param_space[string])

    def __call__(self, parameters_in):
        print(f"----------------new hyperopt iteration {self.global_hyperit}------------------")
        print(parameters_in)
        sumo_net = self.net_type in ['survival_net_basic', 'weibull_net', 'lognormal_net','survival_GWI']
        if self.custom_dataloader is not None:
            self.dataloader = self.custom_dataloader
            self.dataloader.batch_size = parameters_in['bs']
            x_c = self.dataloader.dataset.x_c
        else:
            self.dataloader = get_dataloader(self.dataset_string, parameters_in['bs'], self.seed, self.fold_idx,
                                             sumo_net=sumo_net)
            x_c = self.dataloader.dataset.X.shape[1]
        self.cycle_length = self.dataloader.__len__() // self.validation_interval + 1
        print('cycle_length', self.cycle_length)
        net_init_params = {
            'd_in_x': x_c,
            'cat_size_list': self.dataloader.dataset.unique_cat_cols,
            'd_in_y': self.dataloader.dataset.y.shape[1],
            'd_out': self.d_out,
            'bounding_op': parameters_in['bounding_op'],
            'transformation': parameters_in['transformation'],
            'layers_x': [parameters_in['width_x']] * parameters_in['depth_x'],
            'layers_t': [parameters_in['width_t']] * parameters_in['depth_t'],
            'layers': [parameters_in['width']] * parameters_in['depth'],
            'direct_dif': parameters_in['direct_dif'],
            'objective': self.objective,
            'dropout': parameters_in['dropout'],
            'eps': parameters_in['eps']
        }
        self.train_objective = get_objective(self.objective)
        if self.net_type == 'weibull_net':
            self.m_q = weibull_net(**net_init_params).to(self.device)
        elif self.net_type == 'lognormal_net':
            self.m_q = lognormal_net(**net_init_params).to(self.device)
        elif self.net_type == 'survival_net_basic':
            self.m_q = survival_net_basic(**net_init_params).to(self.device)
        elif self.net_type == 'survival_GWI':
            self.m_q = survival_GWI(**net_init_params).to(self.device)
        self.event = self.dataloader.dataset.train_delta

        self.X = self.dataloader.dataset.train_X[self.event==1]
        self.Y = self.dataloader.dataset.train_y[self.event==1]
        self.N = self.X.shape[0]
        self.m =int(self.N**0.5 * parameters_in['m_factor'])

        self.cov_dat = torch.cat([self.X,self.Y],dim=1)

        if torch.sum(self.event)>self.m:
            z_mask=torch.randperm(self.N)[:self.m]
            self.Z =  self.X[z_mask, :]
            self.Y_Z= self.Y[z_mask,:]
            p = torch.randperm(self.N)[:self.m]
            self.X_hat =  self.X[p,:]
            self.Y_hat = self.Y[p]
        else:
            self.Z= copy.deepcopy(self.X)
            self.Y_Z= copy.deepcopy(self.Y)
            self.X_hat=copy.deepcopy(self.X)
            self.Y_hat=copy.deepcopy(self.Y)
        z_dat = torch.cat([self.Z,self.Y_Z],dim=1)
        x_dat = torch.cat([self.X_hat,self.Y_hat],dim=1)
        self.p_kernel = GWISurvivalKernel(self.X, self.Y)
        for parameters in self.p_kernel.parameters():
            parameters.requires_grad=False
        self.r = r_param_cholesky_scaling(k=self.p_kernel, Z=z_dat, X=x_dat, sigma=1.0,
                                         parametrize_Z=False)
        self.r.init_L()
        self.r.to(self.device)
        self.model = GWI(
            N=self.dataloader.dataset.train_X.shape[0],
            m_q=self.m_q,
            m_p=parameters_in['m_P'],
            r=self.r,
            sigma=1.0,
            reg=parameters_in['reg'],
            x_s=parameters_in['x_s']
        ).to(self.device)

        self.dump_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=parameters_in['lr'],
                                          weight_decay=parameters_in['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience // 4,
                                                                    min_lr=1e-3, factor=0.9)
        results = self.full_loop()
        del self.optimizer
        self.global_hyperit += 1
        results['net_init_params'] = net_init_params
        torch.cuda.empty_cache()
        return results

    def do_metrics(self, training_loss, likelihood, reg_loss, i):
        val_likelihood, conc, ibs, ibll = self.validation_score()
        if self.debug:
            test_likelihood, test_conc, test_ibs, test_inll = self.test_score()
            self.writer.add_scalar('Loss/train', training_loss, i)
            self.writer.add_scalar('Loss/val', val_likelihood[0], i)
            self.writer.add_scalar('Loss/test', test_likelihood[0], i)
            self.writer.add_scalar('conc/val', conc, i)
            self.writer.add_scalar('conc/test', test_conc, i)
            self.writer.add_scalar('ibs/val', ibs, i)
            self.writer.add_scalar('ibs/test', test_ibs, i)
            self.writer.add_scalar('inll/val', ibll, i)
            self.writer.add_scalar('inll/test', test_inll, i)
            print(
                f'test_likelihood: {test_likelihood[0]} test_likelihood: {test_likelihood[1]} test_conc: {test_conc} test_ibs: {test_ibs}  test_inll: {test_inll}')
            self.debug_list.append(test_ibs)
        if self.selection_criteria in ['train', 'likelihood']:
            criteria = val_likelihood[0]  # minimize #
        elif self.selection_criteria == 'concordance':
            criteria = -conc  # maximize
        elif self.selection_criteria == 'ibs':
            criteria = ibs  # minimize
        elif self.selection_criteria == 'inll':
            criteria = ibll  # "minimize"
        print(f'total_loss: {training_loss} likelihood: {likelihood} reg_loss: {reg_loss}')
        if self.validate_train:
            tr_likelihood, tr_conc, tr_ibs, tr_ibll = self.train_score()
            print(
                f'tr_likelihood: {tr_likelihood[0]} tr_likelihood: {tr_likelihood[1]} tr_conc: {tr_conc} tr_ibs: {tr_ibs}  tr_ibll: {tr_ibll}')
        print(
            f'criteria score: {criteria} val likelihood: {val_likelihood[0]} val likelihood: {val_likelihood[1]} val conc:{conc} val ibs: {ibs} val inll {ibll}')
        return criteria

    def eval_func(self, i, training_loss, likelihood, reg_loss):
        if i % self.cycle_length == 0:
            criteria = self.do_metrics(training_loss, likelihood, reg_loss, i)
            self.scheduler.step(criteria)
            if criteria < self.best:
                self.best = criteria
                print('new best val score: ', self.best)
                print('Dumping model')
                self.dump_model()
                self.counter = 0
            else:
                self.counter += 1
            if self.counter > self.patience:
                return True

    def training_loop(self, epoch): #TODO rewrite training loop logic
        self.dataloader.dataset.set(mode='train')
        total_loss_train = 0.
        tot_likelihood = 0.
        tot_reg_loss = 0.
        self.model = self.model.train()
        for i, (X, x_cat, y, delta) in enumerate(tqdm.tqdm(self.dataloader)):
            X = X.to(self.device)
            y = y.to(self.device)
            delta = delta.to(self.device)
            z_mask=torch.randperm(self.N)[:self.model.x_s]
            Z_prime_cov = self.cov_dat[z_mask, :].to(self.device)
            self.optimizer.zero_grad()
            total_loss = self.model.get_loss(y,X,x_cat,delta,Z_prime_cov)
            total_loss.backward()
            self.optimizer.step()
            total_loss_train += total_loss.detach()
            tot_likelihood += total_loss.detach()
            if self.eval_func(i, total_loss_train / (i + 1), tot_likelihood / (i + 1), tot_reg_loss / (i + 1)):
                return True
        return False

    #rewrite validation loss
    def eval_loop(self, grid_size, chunks=50, max_series_accumulation=50000):
        self.dataloader.dataset.set(mode='val')
        total_loss_val = 0.
        self.model = self.model.eval()
        for i, (X, x_cat, y, delta) in enumerate(tqdm.tqdm(self.dataloader)):
            X = X.to(self.device)
            y = y.to(self.device)
            delta = delta.to(self.device)
            z_mask=torch.randperm(self.N)[:self.model.x_s]
            Z_prime_cov = self.cov_dat[z_mask, :].to(self.device)
            total_loss = self.model.get_loss(y,X,x_cat,delta,Z_prime_cov)
            total_loss_val+=total_loss.item()
        val_loss = total_loss_val/(self.dataloader.dataset.X.shape[0])
        self.model.train()
        return [val_loss, val_loss], 0.0, 0.0, 0.0

    def train_score(self):
        self.dataloader.dataset.set(mode='train')
        return self.eval_loop(self.grid_size, chunks=self.chunks, max_series_accumulation=self.max_series_accumulation)

    def validation_score(self):
        self.dataloader.dataset.set(mode='val')
        return self.eval_loop(self.grid_size, chunks=self.chunks, max_series_accumulation=self.max_series_accumulation)

    def test_score(self):
        self.dataloader.dataset.set(mode='test')
        return self.eval_loop(self.grid_size)

    def dump_model(self):
        torch.save(self.model.state_dict(), self.save_path + f'best_model_{self.global_hyperit}.pt')
        torch.save(self.model, self.save_path + f'best_model_full_{self.global_hyperit}.pt')


    def load_model(self):
        self.model.load_state_dict(torch.load(self.save_path + f'best_model_{self.global_hyperit}.pt'))

    def full_loop(self):
        self.counter = 0
        if self.debug:
            self.writer = SummaryWriter()
            self.debug_list = []
        for i in range(self.total_epochs):
            if self.training_loop(i):
                break
        if self.debug:
            print(f'best test ibs {min(self.debug_list)}')
        self.load_model()
        val_likelihood, val_conc, val_ibs, val_inll = self.validation_score()
        test_likelihood, test_conc, test_ibs, test_inll = self.test_score()

        return self.parse_results(val_likelihood, val_conc, val_ibs, val_inll,
                                  test_likelihood, test_conc, test_ibs, test_inll)

    def parse_results(self, val_likelihood, val_conc, val_ibs, val_inll,
                      test_likelihood, test_conc, test_ibs, test_inll, val_loss_cox=None):
        if self.selection_criteria == 'train':
            if val_loss_cox is None:
                criteria = val_likelihood[0]
                criteria_test = test_likelihood[0]
            else:
                criteria = val_loss_cox
                criteria_test = val_loss_cox
        if self.selection_criteria == 'likelihood':
            criteria = val_likelihood[0]
            criteria_test = test_likelihood[0]
        elif self.selection_criteria == 'concordance':
            criteria = -val_conc
            criteria_test = -test_conc
        elif self.selection_criteria == 'ibs':
            criteria = val_ibs
            criteria_test = test_ibs
        elif self.selection_criteria == 'inll':
            criteria = val_inll
            criteria_test = test_inll

        return {'loss': criteria,
                'status': STATUS_OK,
                'test_loss': criteria_test,
                'test_loglikelihood_1': test_likelihood[0],
                'test_loglikelihood_2': test_likelihood[1],
                'test_conc': test_conc,
                'test_ibs': test_ibs,
                'test_inll': test_inll,
                'val_loglikelihood_1': val_likelihood[0],
                'val_loglikelihood_2': val_likelihood[1],
                'val_conc': val_conc,
                'val_ibs': val_ibs,
                'val_inll': val_inll,
                }

    def run(self):
        if os.path.exists(self.save_path + 'hyperopt_database.p'):
            return
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

        if self.selection_criteria in ['train', 'likelihood']:
            reverse = False
        elif self.selection_criteria == 'concordance':
            reverse = False
        elif self.selection_criteria in ['ibs', 'ibs_likelihood']:
            reverse = False
        elif self.selection_criteria == 'inll':
            reverse = False

        best_trial = sorted(trials.results, key=lambda x: x['test_loss'], reverse=reverse)[0]  # low to high
        data = [best_trial['test_loglikelihood_1'], best_trial['test_loglikelihood_2'], best_trial['test_conc'],
                best_trial['test_ibs'], best_trial['test_inll'],
                best_trial['val_loglikelihood_1'], best_trial['val_loglikelihood_2'], best_trial['val_conc'],
                best_trial['val_ibs'], best_trial['val_inll']]
        df = pd.DataFrame([data],
                          columns=['test_loglikelihood_1', 'test_loglikelihood_2', 'test_conc', 'test_ibs', 'test_inll',
                                   'val_loglikelihood_1', 'val_loglikelihood_2', 'val_conc', 'val_ibs', 'val_inll'])
        print(df)
        df.to_csv(self.save_path + 'best_results.csv', index_label=False)
