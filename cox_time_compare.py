import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt
import torch
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime

from utils.dataloaders import toy_data_class


if __name__ == '__main__':

    for toy_dat in [5,6,7]:
        net = MLPVanillaCoxTime(in_features=1,num_nodes=[64,64,64,64],batch_norm=True,dropout=0.1)
        if toy_dat==5:
            test_X = torch.Tensor([[0], [0.3], [1.0]]).cuda()
            str_name = 'weibull'
        if toy_dat==6:
            test_X = torch.Tensor([[0.1],[0.4]]).cuda()
            str_name = 'checkboard'
        if toy_dat==7:
            test_X = torch.Tensor([[0.0],[0.2],[0.4],[0.6],[0.8],[1.0]]).cuda()
            str_name = 'normal'
        cols_leave = ['x_1']
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

        model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
        batch_size = 250
        model.optimizer.set_lr(0.001)
        epochs = 512
        callbacks = [tt.callbacks.EarlyStopping()]
        verbose = True
        log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                        val_data=val.repeat(10).cat())
        _ = model.compute_baseline_hazards()
        surv = model.predict_surv_df(test_X)
        print(surv)

        surv.iloc[:, :test_X.shape[0]].plot()
        plt.ylabel('S(t | x)')
        _ = plt.xlabel('Time')
        plt.savefig(f'./cox_time_survival_plot_dataset_{toy_dat}.png')








