import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxCC,CoxPH,CoxTime
from pycox.evaluation import EvalSurv
from pycox.models.cox_time import MLPVanillaCoxTime


np.random.seed(1234)
_ = torch.manual_seed(123)

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.25)
df_train = df_train.drop(df_val.index)


cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)


x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')



labtrans = CoxTime.label_transform()
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))
durations_test, events_test = get_target(df_test)
val = tt.tuplefy(x_val, y_val)

in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False

# net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
#                               dropout, output_bias=output_bias)
net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout) #Actual net to be used
model = CoxTime(net, tt.optim.Adam,labtrans=labtrans) #the cox time framework, dont do this..
model.optimizer.set_lr(0.01)
epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

batch_size = 256

log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val.repeat(10).cat())

base_haz = model.compute_baseline_hazards()
print(base_haz)
surv = model.predict_surv_df(x_test)
ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
conc = ev.concordance_td()
time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
ibs = ev.integrated_brier_score(time_grid)
inll = ev.integrated_nbll(time_grid)

print(conc,ibs,inll)


