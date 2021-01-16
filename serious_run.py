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
    # parser.add_argument('--dataset', type=int, nargs='?', default=-1, help='which dataset to run')
    # parser.add_argument('--seed', type=int, nargs='?', help='selects the seed to split the data on')
    # parser.add_argument('--eval_metric', type=int, nargs='?', default=0, help='which evaluation metric to use')
    # parser.add_argument('--total_epochs', type=int, nargs='?', default=50, help='total number of epochs to train')
    # parser.add_argument('--grid_size', type=int, nargs='?', default=1000, help='grid_size for evaluation')
    # parser.add_argument('--test_grid_size', type=int, nargs='?', default=10000, help='grid_size for evaluation')
    # parser.add_argument('--hyperits', type=int, nargs='?', default=25, help='nr of hyperits to conduct')
    # parser.add_argument('--patience', type=int, nargs='?', default=50, help='# of epochs before terminating in validation error improvement')
    # parser.add_argument('--validation_interval', type=int, nargs='?', default=5, help='nr of epochs between validations')
    # parser.add_argument('--loss_type', type=int, nargs='?', default=0, help='nr of epochs between validations')
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
dataset_bs =[[25,50,100,250],[5,10,25,50,100],[5,10,25,50,100],[5,10,25,50,100],[1000,2500,5000],[50,100,250,500,1000,2500],[50,100,250,500,1000,2500],[50,100,250,500,1000,2500]]
# dataset_bs =[[25,50,100,250],[5,10,25,50,100],[5,10,25,50,100],[5,10,25,50,100],[1000,2500,5000],[64,128,256,512,1024],[64,128,256,512,1024],[64,128,256,512,1024]]
dataset_d_x =[[1, 2], [1, 2], [1, 2], [1, 2], [4, 6, 8], [1, 2], [1, 2], [1, 2]]
dataset_d =[[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2,4], [1, 2,4], [1, 2,4]]
# dataset_w_x =[[16, 32], [8, 16, 32], [8, 16, 32], [8, 16, 32], [128, 256, 512], [8, 16, 32,64], [8, 16, 32,64], [8, 16, 32,64]]
dataset_w_x =[[16, 32], [8, 16, 32], [8, 16, 32], [8, 16, 32], [128, 256, 512], [64,128,256,512], [64,128,256,512], [64,128,256,512]]
# dataset_w =[[16, 32], [8, 16, 32], [8, 16, 32], [8, 16, 32], [128, 256, 512], [64,128,256,512], [64,128,256,512], [64,128,256,512]]
dataset_w =[[16, 32], [8, 16, 32], [8, 16, 32], [8, 16, 32], [8,16,32], [8, 16, 32,64], [8, 16, 32,64], [8, 16, 32,64]]
dataset_d_t =[[1,2,3],[1,2,3],[1,2,3],[1,2,3],[4,6,8],[1,2],[1,2],[1,2]]

loss_type = ['S_mean','hazard_mean']
#Write serious job script, figure out post processing pipeline...
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
        # torch.nn.functional.elu,torch.nn.functional.relu,
        'bounding_op': [square,torch.nn.functional.relu],  # torch.sigmoid, torch.relu, torch.exp,
        'transformation': [torch.nn.functional.tanh],
        'depth_x': d_x,
        'width_x': w_x,
        'depth': d,
        'width': w,
        'depth_t': d_t,
        'width_t': [8,16,32],  # ads
        'bs': bs,
        'lr': [1e-3,1e-2,1e-1],
        'direct_dif': [args['direct_dif']],
        'dropout': [0.0,0.7],#[0.0,0.1,0.2,0.3,0.4,0.5],
        'eps': [1e-3,1e-4,1e-5],
        'weight_decay': [0,0.01,0.02,0.05,0.1,0.2,0.4],#[1e-3,1e-4,1e-2,0.1,0]

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
        'selection_criteria': eval_metrics[args['eval_metric']],
        'grid_size':args['grid_size'],
        'test_grid_size': args['test_grid_size'],
        'validation_interval': args['validation_interval'],
        'net_type': args['net_type'],
        'objective': loss_type[args['loss_type']],
        'fold_idx': args['fold_idx'],
        'savedir':args['savedir']

    }
    training_obj = hyperopt_training(job_param=job_params,hyper_param_space=hyper_param_space)
    training_obj.run()
    training_obj.post_process()