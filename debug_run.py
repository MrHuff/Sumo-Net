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
        'transformation': [torch.nn.functional.tanh],
        'depth_x': [2],
        'width_x': [32], #adapt for smaller time net
        'depth_t': [2],
        'width_t': [32], #ads
        'depth': [1],
        'width': [1],
        'bs': [25],
        'lr': [1e-2],
        'direct_dif': [False],
        'dropout': [0.1],
        'eps':[1e-4],
        'weight_decay':[0]

    }
    for i in [-1]:
        devices = GPUtil.getAvailable(order='memory', limit=8)
        print(devices)
        print(torch.cuda.device_count())
        device = devices[0]
        job_params = {
            'd_out': 1,
            'dataset_string': datasets[i],
            'seed': 1337,#,np.random.randint(0,9999),
            'eval_metric': 'train',
            'total_epochs': 5000,
            'device': device,
            'patience': 50,
            'hyperits': 1,
            'selection_criteria':'ibs',
            'grid_size':100,
            'test_grid_size':100,
            'validation_interval':10,
            'net_type':'benchmark',
            'objective': 'S_mean',
            'fold_idx':3,
            'savedir':'test'

        }
        training_obj = hyperopt_training(job_param=job_params,hyper_param_space=hyper_param_space)
        training_obj.debug=True
        training_obj.run()
        training_obj.post_process()