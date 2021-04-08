import os
import shutil
import pickle

def save_obj(obj, name ,folder):
    with open(f'{folder}'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)

def generate_job_params(directory='job_dir'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    base_dict={
        'dataset': 0,
        'seed':1337,
        'total_epochs':1,
        'patience':21,
        'hyperits':20,
        'grid_size':100,
        'test_grid_size': 100,
        'validation_interval':3,
        'loss_type':0,
        'net_type':'ocean_net',
        'fold_idx': 0,
        'savedir':f'{directory}_results',
        'direct_dif': ['autograd'],
        'use_sotle': False,

    }
    counter = 0
    for fold_idx in [0,1,2,3,4]:
        for dataset in [2]:
            for l_type in [0]:
                # for net_t in ['survival_net_basic','cox_time_benchmark','deepsurv_benchmark','cox_CC_benchmark','cox_linear_benchmark','deephit_benchmark']:
                for net_t in ['survival_net_basic']:
                    base_dict['dataset']=dataset
                    base_dict['loss_type']=l_type
                    base_dict['net_type']=net_t
                    base_dict['fold_idx']=fold_idx
                    save_obj(base_dict,f'job_{counter}',directory+'/')
                    counter +=1

if __name__ == '__main__':
        generate_job_params(directory='testing')