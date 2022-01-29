from serious_run import datasets
import GPUtil
from sumo_net.hyperopt_class import *
from utils.plot_utils import *

if __name__ == '__main__':
    save_folder = 'sumo_example_plots_benchmarks'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        shutil.rmtree(save_folder)
        os.makedirs(save_folder)
    for idx in [5,6,7]:
        d_str = datasets[idx]
        if d_str=='weibull':
            t_array = np.linspace(0, 2, num=100)
            x_array = [0.0, 0.3, 1.0]
        elif d_str=='checkboard':
            t_array = np.linspace(0, 1, num=100)
            x_array = [0.1, 0.4]
        elif d_str =='normal':
            t_array = np.linspace(80, 120, num=100)
            x_array = [ 0.2, 0.4, 0.6, 0.8, 1.0]

        hyper_param_space = {
            # torch.nn.functional.elu,torch.nn.functional.relu,
            'bounding_op': [square],  # torch.sigmoid, torch.relu, torch.exp,
            'transformation': [torch.nn.functional.relu],
            'depth_x': [1],
            'width_x': [16],  # adapt for smaller time net
            'depth_t': [1],
            'width_t': [1],  # ads
            'depth': [2],
            'width': [128],
            'bs': [256],
            'lr': [1e-2],
            'direct_dif': ['autograd'],
            'dropout': [0.1],
            'eps': [1e-3],
            'weight_decay': [0]

        }
        devices = GPUtil.getAvailable(order='memory', limit=8)
        device = devices[0]
        job_params = {
            'd_out': 1,
            'dataset_string': datasets[idx],
            'seed': 1337,  # ,np.random.randint(0,9999),
            'eval_metric': 'train',
            'total_epochs': 5000,
            'device': device,
            'patience': 50,
            'hyperits': 1,
            'selection_criteria': 'ibs',
            'grid_size': 100,
            'test_grid_size': 100,
            'validation_interval': 10,
            'net_type': 'benchmark',
            'objective': 'S_mean',
            'fold_idx': 3,
            'savedir': 'test'

        }
        training_obj = hyperopt_training(job_param=job_params, hyper_param_space=hyper_param_space)
        training_obj.run()
        surv = training_obj.wrapper.predict_surv_df(torch.tensor(x_array).unsqueeze(-1).float())
        dl = get_dataloader(d_str,250,1337,3)
        surv.index = dl.dataset.invert_duration(surv.index.values.reshape(-1, 1)).squeeze()
        data = surv[(t_array[0].item() <= surv.index) & (surv.index <= t_array[-1].item())]
        for i in range(len(x_array)):
            plt.plot(data.index.values, data.iloc[:,i].values,
                     label=f'x={x_array[i]}',linewidth=4.0)  # fix
        plt.legend(prop={'size': 48},loc=1)
        plt.xlabel('Time')
        plt.ylabel(r'S(t)')
        plt.savefig(f'{save_folder}/{d_str}_3_survival_plot.png',bbox_inches = 'tight',
    pad_inches = 0.1)
        plt.clf()





