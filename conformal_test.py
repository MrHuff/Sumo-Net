from sumo_net.hyperopt_class import *
import numpy as np

import torch
import GPUtil
import warnings
warnings.simplefilter("ignore")
datasets = ['support',
            'metabric',
            'gbsg',
            'flchain',
            'kkbox',
            'weibull',
            'checkboard',
            'normal'
            ]
#Uppgrade dataloader rip, probably uses some retarded permutation which is really slow.
#Write serious job script, figure out post processing pipeline...
if __name__ == '__main__':
    #Evaluate other toy examples to draw further conclusions...
    # Time component might need to be normalized...
    # eval more frequently...
    hyper_param_space = {
        # torch.nn.functional.elu,torch.nn.functional.relu,
        'bounding_op': [torch.relu],  # torch.sigmoid, torch.relu, torch.exp,
        'transformation': [torch.nn.Tanh()],
        'depth_x': [2],
        'width_x': [32], #adapt for smaller time net
        'depth_t': [1],
        'width_t': [1], #ads
        'depth': [2],
        'width': [16],
        'bs': [500],
        'lr': [1e-2],
        'direct_dif': ['autograd'],
        'dropout': [0.1],
        'eps':[1e-4],
        'weight_decay':[0],
        'T_losses':[90],
        'alpha': [0.2],
        'sigma': [0.1],
        'num_dur': [20],

    }
    for i in [0]:
        device = "cpu:0"
        job_params = {
            'd_out': 1,
            'dataset_string': datasets[i],
            'seed': 1,#,np.random.randint(0,9999),
            'total_epochs': 50,
            'device': device,
            'patience': 50,
            'hyperits': 1,
            'selection_criteria':'train',
            'grid_size':100,
            'test_grid_size':100,
            'validation_interval':2,
            'net_type':'survival_net_basic',
            'objective': 'S_mean',
            'chunks':50,
            'max_series_accumulation':50000,
            'validate_train':False,
            'fold_idx':1 ,
            'savedir':'test',
            'use_sotle':False,
            'conformal':True,
        }
        training_obj = hyperopt_training(job_param=job_params,hyper_param_space=hyper_param_space)
        # training_obj.debug=True
        training_obj.run()
        training_obj.post_process()
        training_obj.fit_conformal()