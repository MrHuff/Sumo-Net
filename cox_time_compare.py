import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt
import torch
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv
from utils.dataloaders import toy_data_class
import pandas as pd

if __name__ == '__main__':

    for toy_dat in [7]:
        net = MLPVanillaCoxTime(in_features=1,num_nodes=[64,64,64,64],batch_norm=True,dropout=0.1)
        if toy_dat==5:
            test_X = torch.Tensor([[0], [0.3], [1.0]]).cuda()
            str_name = 'weibull'
        if toy_dat==6:
            test_X = torch.Tensor([[0.1],[0.2],[0.4],[0.6],[0.8],[1.0]]).cuda()
            str_name = 'checkboard'
        if toy_dat==7:
            test_X = torch.Tensor([[0.0],[0.2],[0.4],[0.6],[0.8],[1.0]]).cuda()
            str_name = 'normal'
        cols_leave = ['x1']
        leave = [(col, None) for col in cols_leave]

        x_mapper = DataFrameMapper(leave)
        c = toy_data_class(str_name)
        df_train = c.read_df()
        df_test = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_test.index)
        df_val = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_val.index)
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_val = x_mapper.transform(df_val).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')



        labtrans = CoxTime.label_transform()
        get_target = lambda df: (df['duration'].values, df['event'].values)
        y_train = labtrans.fit_transform(*get_target(df_train))
        y_val = labtrans.transform(*get_target(df_val))
        durations_test, events_test = get_target(df_test)
        val = tt.tuplefy(x_val, y_val)

        model = CoxTime(net, tt.optim.Adam)
        batch_size = 250
        model.optimizer.set_lr(0.001)
        epochs = 512
        callbacks = [tt.callbacks.EarlyStopping()]
        verbose = True
        log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                        val_data=val.repeat(10).cat())
        _ = model.compute_baseline_hazards()
        surv = model.predict_surv_df(input = x_test,batch_size=5000) # passing wrong data... pass real test data...
        print(surv)
        # surv.iloc[:, :x_test.shape[0]].plot()
        # plt.ylabel('S(t | x)')
        # _ = plt.xlabel('Time')
        # plt.savefig(f'./cox_time_survival_plot_dataset_{toy_dat}.png')
        # Ok Issue with validation objective. Comparison is too great?!
        eval_obj = EvalSurv(surv=surv,durations=durations_test,events=events_test,censor_surv='km')
        conc = eval_obj.concordance_td()
        time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
        eval_obj.brier_score(time_grid)
        ibs = eval_obj.integrated_brier_score(time_grid)
        eval_obj.nbll(time_grid).plot()
        inll = eval_obj.integrated_nbll(time_grid)
        df = pd.DataFrame([[conc,ibs,inll]],columns=['test_conc','test_ibs','test_inll'])
        df.to_csv(f'./kvamme_test_data={toy_dat}.csv')







