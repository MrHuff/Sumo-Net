from hyperopt_class import *
import numpy as np
import torch

datasets = ['support',
            'metabric',
            'gbsg',
            'flchain',
            'kkbox',
            'weibull',
            'checkboard',
            'normal'
            ]
job_params = {
    'd_out': 1,
    'dataset_string': datasets[6],
    'seed': 123,
    'eval_metric': 'train',
    'total_epochs': 50,
    'device': 0,
    'global_loss_init': np.inf,
    'patience': 5,
    'hyperits': 3,

}
hyper_param_space = {
    'bounding_op': [square],  # torch.sigmoid, torch.relu, torch.exp,
    'transformation': [torch.nn.functional.tanh],
    'depth_x': [2],
    'width_x': [16,32],
    'depth': [2],
    'width': [16,32],
    'bs': [32,64],
    'lr': [1e-4],
    'direct_dif':[False]

}
if __name__ == '__main__':
    #Evaluate other toy examples to draw further conclusions...
    training_obj = hyperopt_training(job_param=job_params,hyper_param_space=hyper_param_space)
    training_obj.run()