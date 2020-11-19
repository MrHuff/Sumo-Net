from hyperopt_class import *
import torch
import GPUtil
import argparse
import warnings
warnings.simplefilter("ignore")


def job_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, nargs='?', default=-1, help='which dataset to run')
    parser.add_argument('--seed', type=int, nargs='?', help='selects the seed to split the data on')
    parser.add_argument('--eval_metric', type=int, nargs='?', default=0, help='which evaluation metric to use')
    parser.add_argument('--total_epochs', type=int, nargs='?', default=50, help='total number of epochs to train')
    parser.add_argument('--grid_size', type=int, nargs='?', default=1000, help='grid_size for evaluation')
    parser.add_argument('--test_grid_size', type=int, nargs='?', default=10000, help='grid_size for evaluation')
    parser.add_argument('--hyperits', type=int, nargs='?', default=25, help='nr of hyperits to conduct')
    parser.add_argument('--patience', type=int, nargs='?', default=50, help='# of epochs before terminating in validation error improvement')
    parser.add_argument('--validation_interval', type=int, nargs='?', default=5, help='nr of epochs between validations')
    parser.add_argument('--total_nr_gpu', type=int, nargs='?', default=4, help='nr of epochs between validations')
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
dataset_bs =[[100,250,500,1000],[100,250,500,1000],[100,250,500,1000],[100,250,500,1000],[500,1000,2500,5000],[250],[250],[250]]

#Write serious job script, figure out post processing pipeline...
if __name__ == '__main__':
    args = vars(job_parser().parse_args())
    hyper_param_space = {
        # torch.nn.functional.elu,torch.nn.functional.relu,
        'bounding_op': [square,torch.nn.functional.relu],  # torch.sigmoid, torch.relu, torch.exp,
        'transformation': [torch.nn.functional.tanh],
        'depth_x': [1,2,3,4,5],
        'width_x': [16,32,64,128,256,512],
        'depth': [1,2,3,4,5],
        'width': [16,32,64,128,256,512],
        'bs': dataset_bs[args['dataset']],
        'lr': [1e-3,1e-4,1e-2],
        'direct_dif': [False],
        'dropout': [0.0,0.1,0.2,0.3,0.4,0.5],
        'objective': ['S_mean']  # S_mean
    }

    devices = GPUtil.getAvailable(order='memory', limit=args['total_nr_gpu'])
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
        'validation_interval': args['validation_interval']
    }
    training_obj = hyperopt_training(job_param=job_params,hyper_param_space=hyper_param_space)
    training_obj.run()
    training_obj.post_process()