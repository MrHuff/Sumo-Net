from hyperopt import hp,tpe,Trials,fmin,space_eval,STATUS_OK,STATUS_FAIL
from nets.nets import survival_net,log_objective
from utils.dataloaders import get_dataloader
import torch
import os

class hyperopt_training():
    def __init__(self,job_param,hyper_param_space):
        self.d_out = job_param['d_out']
        self.dataset_string = job_param['dataset_string']
        self.seed = job_param['seed']
        self.eval_metric = job_param['eval_metric']
        self.hyperopt_params = ['bounding_op','transformation','depth_x','width_x','depth','width','bs','lr']
        self.total_epochs = job_param['total_epochs']
        self.device = job_param['device']
        self.eval_objective = self.get_eval_objective(job_param['eval_metric'])
        self.global_loss_init = job_param['global_loss_init']
        self.patience = job_param['patience']
        self.validation_interval = self.total_epochs//20
        self.global_hyperit = 0
        torch.cuda.set_device(self.device)
        self.job_path = f'./{self.dataset_string}_{self.seed}/'
        if not os.path.exists(self.job_path):
            os.makedirs(self.job_path)

    def get_eval_objective(self,str):
        if str=='train':
            return log_objective
        elif str=='c_score':
            pass

    def get_hyperparameterspace(self,hyper_param_space):
        self.hyperparameter_space = {}
        for string in self.hyperopt_params:
            self.hyperparameter_space[string] = hp.choice(string, hyper_param_space[string])

    def __call__(self,parameters_in):
        self.dataloader = get_dataloader(self.dataset_string,parameters_in['bs'],self.seed)
        net_init_params = {
            'd_in_x' : self.dataloader.dataset.X.shape[1],
            'd_in_y' : self.dataloader.dataset.y.shape[1],
            'd_out' : self.d_out,
            'bounding_op':parameters_in['bounding_op'],
            'transformation':parameters_in['transformation'],
            'layers_x': [parameters_in['width_x']]*parameters_in['depth_x'],
            'layers': [parameters_in['width']]*parameters_in['depth'],
        }
        self.model = survival_net(**net_init_params).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=parameters_in['lr'])
        results = self.full_loop()
        self.global_hyperit+=1
        return results

    def training_loop(self):
        self.dataloader.dataset.set(mode='train')
        for i,(X,y,delta) in enumerate(self.dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            delta = delta.to(self.device)
            f,S = self.model(X,y)
            loss = log_objective(S,f,delta)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval_loop(self):
        total = 0
        with torch.no_grad():
            for i,(X,y,delta) in enumerate(self.dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                delta = delta.to(self.device)
                f, S = self.model(X, y)
                total+= self.eval_metric(f,S,delta)
        return total

    def validation_score(self):
        self.dataloader.dataset.set(mode='val')
        return self.eval_loop()

    def test_score(self):
        self.dataloader.dataset.set(mode='test')
        return self.eval_loop()

    def dump_model(self):
        torch.save(self.model.state_dict(),self.job_path+f'best_model_{self.global_hyperit}.pt')

    def full_loop(self):
        counter = 0
        best = self.global_loss_init
        for i in self.total_epochs:
            self.training_loop()
            if i%self.validation_interval==0:
                val_loss = self.validation_score()
                if val_loss<self.best:
                    best = val_loss
            if counter > self.patience:
                break

        val_loss = self.validation_score()
        test_loss = self.test_score()

        return {'loss': -val_loss, 'status': STATUS_OK, 'test_loss': -test_loss}


























