from hyperopt_class import *
import numpy as np
import torch
import GPUtil
datasets = ['support',
            'metabric',
            'gbsg',
            'flchain',
            'kkbox',
            'weibull',
            'checkboard',
            'normal'
            ]





#Write serious job script, figure out post processing pipeline...
if __name__ == '__main__':
    #Evaluate other toy examples to draw further conclusions...
    # Time component might need to be normalized...

    hyper_param_space = {
        # torch.nn.functional.elu,torch.nn.functional.relu,
        'bounding_op': [square],  # torch.sigmoid, torch.relu, torch.exp,
        'transformation': [torch.nn.functional.tanh],
        'depth_x': [2],
        'width_x': [64],
        'depth': [2],
        'width': [64],
        'bs': [250],
        'lr': [1e-3],
        'direct_dif': [False],
        'dropout': [0.5],
        'objective': ['S_mean']  # S_mean

    }
    for i in [0]:
        devices = GPUtil.getAvailable(order='memory', limit=8)
        device = devices[0]
        job_params = {
            'd_out': 1,
            'dataset_string': datasets[i],
            'seed': 123,
            'eval_metric': 'train',
            'total_epochs': 250,
            'device': device,
            'patience': 50,
            'hyperits': 2,
            'selection_criteria':'concordance',
            'grid_size':250,
            'validation_interval':2,
        }
        training_obj = hyperopt_training(job_param=job_params,hyper_param_space=hyper_param_space)
        training_obj.run()