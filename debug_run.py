from hyperopt_class import *
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
        'bounding_op': [square],  # torch.sigmoid, torch.relu, torch.exp,
        'transformation': [torch.nn.Tanh()],
        'depth_x': [2],
        'width_x': [32], #adapt for smaller time net
        'depth_t': [3],
        'width_t': [1], #ads
        'depth': [1],
        'width': [32],
        'bs': [100],
        'lr': [1e-2],
        'direct_dif': ['autograd'],
        'dropout': [0.0],
        'eps':[1e-4],
        'weight_decay':[0.],
        'reg_lambda':[1.0],
        'T_losses':[90]

    }
    for i in [1]:
        devices = GPUtil.getAvailable(order='memory', limit=8)
        print(devices)
        print(torch.cuda.device_count())
        device = devices[0]
        job_params = {
            'd_out': 1,
            'dataset_string': datasets[i],
            'seed': 9001,#,np.random.randint(0,9999),
            'total_epochs': 5000,
            'device': device,
            'patience': 200,
            'hyperits': 1,
            'selection_criteria':'ibs',
            'grid_size':100,
            'test_grid_size':100,
            'validation_interval':10,
            'net_type':'survival_net',
            'objective': 'S_mean',
            'fold_idx':0 ,
            'savedir':'test',
            'reg_mode':'ibs',
            'ibs_est_deltas':100,
            'use_sotle':False
        }
        training_obj = hyperopt_training(job_param=job_params,hyper_param_space=hyper_param_space)
        training_obj.debug=True
        training_obj.run()
        training_obj.post_process()