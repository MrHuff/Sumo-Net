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
    hyper_param_space = {
        # torch.nn.functional.elu,torch.nn.functional.relu,
        'bounding_op': [square],  # torch.sigmoid, torch.relu, torch.exp,
        'transformation': [torch.nn.functional.tanh],
        'depth_x': [1],
        'width_x': [32],
        'depth': [1],
        'width': [32],
        'bs': [500],
        'lr': [1e-1],
        'direct_dif': [False],
        'dropout': [0.2],

    }
    for i in [2]:
        devices = GPUtil.getAvailable(order='memory', limit=8)
        print(devices)
        print(torch.cuda.device_count())
        device = devices[0]
        job_params = {
            'd_out': 1,
            'dataset_string': datasets[i],
            'seed': 123,#,np.random.randint(0,9999),
            'eval_metric': 'train',
            'total_epochs': 500,
            'device': device,
            'patience': 50,
            'hyperits': 1,
            'selection_criteria':'train',
            'grid_size':250,
            'test_grid_size':10000,
            'validation_interval':1,
            'net_type':'ocean_net',
        'objective': 'S_mean'  # S_mean

        }
        training_obj = hyperopt_training(job_param=job_params,hyper_param_space=hyper_param_space)
        training_obj.debug=True
        training_obj.run()
        training_obj.post_process()