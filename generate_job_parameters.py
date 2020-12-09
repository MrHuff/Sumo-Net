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
        'seed':-1,
        'total_epochs':500,
        'patience':50,
        'hyperits':30,
        'eval_metric':0,
        'grid_size':1000,
        'test_grid_size': 10000,
        'validation_interval':1,
        'loss_type':0
    }
    counter = 0
    for s in [1,2,3,4,5]:
        for dataset in [0,1,2,3]:
            for l_type in [0,1]:
                base_dict['dataset']=dataset
                base_dict['seed']=s
                base_dict['loss_type']=l_type
                save_obj(base_dict,f'job_{counter}',directory)
                counter +=1

if __name__ == '__main__':
    generate_job_params(directory='job_dir/')