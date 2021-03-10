from hyperopt_class_interval import *
import numpy as np

import torch
import GPUtil
import warnings
warnings.simplefilter("ignore")
datasets = [
            'essIncData'

            ] +['interval_checkboard','interval_weibull','interval_normal']
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
        'depth_x': [1],
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
    for i in [-1]:
        devices = GPUtil.getAvailable(order='memory', limit=8)
        print(devices)
        print(torch.cuda.device_count())
        device = devices[0]
        job_params = {
            'd_out': 1,
            'dataset_string': datasets[i],
            'seed': 1337,#,np.random.randint(0,9999),
            'total_epochs': 100,
            'device': device,
            'patience': 100,
            'hyperits': 1,
            'selection_criteria':'train',
            'grid_size':100,
            'test_grid_size':100,
            'validation_interval':2,
            'net_type':'survival_net_basic',
            'objective': 'S_mean',
            'fold_idx':0 ,
            'savedir':f'{datasets[i]}_test',
            'reg_mode':'conc',
            'ibs_est_deltas':100,
            'use_sotle':False
        }
        training_obj = hyperopt_training_interval(job_param=job_params, hyper_param_space=hyper_param_space)
        training_obj.debug=True
        training_obj.run()
        training_obj.post_process()