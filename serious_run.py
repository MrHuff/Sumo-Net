from hyperopt_class import *
import torch
import GPUtil
import argparse
import warnings
warnings.simplefilter("ignore")
from generate_job_parameters import load_obj

def job_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_path', type=str, nargs='?', default='', help='which dataset to run')
    parser.add_argument('--idx', type=int, nargs='?', default=-1, help='which dataset to run')
    return parser

datasets = ['support',
            'metabric',
            'gbsg',
            'flchain',
            'kkbox',
            'weibull',
            'checkboard',
            'normal'
            ]
eval_metrics = [
'train',
'concordance',
'ibs',
'inll'
]
dataset_bs =[[25,50,100,250],[5,10,25,50,100],[5,10,25,50,100],[5,10,25,50,100],[1000,2500,5000],[64,128,256,512,1024],[64,128,256,512,1024],[64,128,256,512,1024]]
dataset_d_x =[[1, 2], [1, 2], [1, 2], [1, 2], [4, 6, 8], [1, 2], [1, 2], [1, 2]]
dataset_d =[[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2,4], [1, 2,4], [1, 2,4]]
dataset_w_x =[[16, 32], [16, 32], [16, 32], [16, 32], [128, 256, 512], [8, 16, 32,64], [8, 16, 32,64], [8, 16, 32,64]]
dataset_w =[[16, 32], [8, 16, 32], [8, 16, 32], [8, 16, 32], [8,16,32], [8, 16, 32,64], [8, 16, 32,64], [8, 16, 32,64]]
dataset_d_t =[[1,2,3],[1,2,3],[1,2,3],[1,2,3],[4,6,8],[1,2],[1,2],[1,2]]
loss_type = ['S_mean','hazard_mean']
if __name__ == '__main__':
    input_args = vars(job_parser().parse_args())
    fold = input_args['job_path']
    idx = input_args['idx']
    jobs = os.listdir(fold)
    jobs.sort()
    args = load_obj(jobs[idx],folder=f'{fold}/')
    d_x = dataset_d_x[args['dataset']]
    w_x = dataset_w_x[args['dataset']]
    d = dataset_d[args['dataset']]
    w = dataset_w[args['dataset']]
    d_t = dataset_d_t[args['dataset']]
    bs = dataset_bs[args['dataset']]
    hyper_param_space = {
        'bounding_op': [square],
        'transformation': [torch.nn.functional.tanh],
        'depth_x': d_x,
        'width_x': w_x,
        'depth': d,
        'width': w,
        'depth_t': d_t,
        'width_t': [8,16,32],  # ads
        'bs': bs,
        'lr': [1e-2],
        'direct_dif': [args['direct_dif']],
        'dropout': [0.0,0.1,0.2,0.3,0.4,0.5], #[0.0,0.7]
        'eps': [1e-3,1e-4,1e-5],
        'weight_decay': [1e-3,1e-4,1e-2,0.1,0],
        'reg_lambda': [2.0,1.0,0.5,0.1],
        'T_losses': [90]

    }

    devices = GPUtil.getAvailable(order='memory', limit=1)
    device = devices[0]
    job_params = {
        'd_out': 1,
        'dataset_string': datasets[args['dataset']],
        'seed': args['seed'],
        'total_epochs': args['total_epochs'],
        'device': device,
        'patience': args['patience'],
        'hyperits':  args['hyperits'],
        'selection_criteria': 'ibs',
        'grid_size':args['grid_size'],
        'test_grid_size': args['test_grid_size'],
        'validation_interval': args['validation_interval'],
        'net_type': args['net_type'],
        'objective': loss_type[args['loss_type']],
        'fold_idx': args['fold_idx'],
        'savedir':args['savedir'],
        'reg_mode':args['reg_mode'],
        'ibs_est_deltas':args['ibs_est_deltas'],
        'use_sotle': False

    }
    training_obj = hyperopt_training(job_param=job_params,hyper_param_space=hyper_param_space)
    training_obj.run()
    training_obj.post_process()