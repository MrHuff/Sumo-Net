import os
import shutil
import pickle

def save_obj(obj, name ,folder):
    with open(f'{folder}'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)

def generate_job_params(directory='job_dir/'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    base_dict={
        'dataset': 0,
        'seed':1337,
        'total_epochs':100,
        'patience':10,
        'hyperits':30,
        'eval_metric':2,
        'grid_size':100,
        'test_grid_size': 100,
        'validation_interval':3,
        'loss_type':0,
        'net_type':'ocean_net',
        'fold_idx': 0,
        'savedir':'conc_test',
        'direct_dif': ['autograd'],
        'reg_mode': 'conc',
        'ibs_est_deltas': 100,
        'use_sotle': False,

    }
    counter = 0
    for fold_idx in [0,1,2,3,4]:
        for dataset in [1]:
            for l_type in [0]:
                for net_t in ['survival_net_basic']:
                    base_dict['dataset']=dataset
                    base_dict['loss_type']=l_type
                    base_dict['net_type']=net_t
                    base_dict['fold_idx']=fold_idx
                    save_obj(base_dict,f'job_{counter}',directory)
                    counter +=1

if __name__ == '__main__':
        generate_job_params(directory='job_conc_test/')