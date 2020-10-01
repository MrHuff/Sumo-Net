from hyperopt_class import *
import numpy as np
import torch
if __name__ == '__main__':
    datasets = ['support',
                'metabric',
                'gbsg',
                'flchain',
                'kkbox'
                ]
    job_params = {
        'd_out': 1,
        'dataset_string': datasets[1],
        'seed': 123,
        'eval_metric': 'train',
        'total_epochs':25,
        'device':0,
        'global_loss_init':np.inf,
        'patience':5,
        'hyperits':5,

    }
    hyper_param_space = {
        'bounding_op':[lambda x:x**2], #torch.sigmoid, torch.relu
        'transformation':[torch.tanh],
        'depth_x':[2,3,4],
        'width_x':[8,16,32],
        'depth':[3,4,5],
        'width':[8,16,32],
        'bs':[32],
        'lr':[1e-3,1e-4]

    }
    training_obj = hyperopt_training(job_param=job_params,hyper_param_space=hyper_param_space)