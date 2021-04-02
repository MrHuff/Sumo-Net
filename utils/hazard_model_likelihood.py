import pycox
import matplotlib.pyplot as plt
import torch
import numpy as np

class HazardLikelihoodCoxTime():
    def __init__(self,pycox_model):
        self.model = pycox_model
        base_haz = self.model.compute_baseline_hazards()
        self.base_haz_time,self.base_haz = torch.from_numpy(base_haz.index.values).float(), torch.from_numpy(base_haz.values).float()
        self.min_base_haz_time = self.base_haz_time.min()
        self.max_base_haz_time = self.base_haz_time.max()

    def check_min(self,t,haz_vec):
        haz_vec[t<self.min_base_haz_time] = self.base_haz[0]
        return haz_vec

    def check_max(self,t,haz_vec):
        haz_vec[t>self.max_base_haz_time] = self.base_haz[-1]
        return haz_vec

    def get_base(self,t):
        bool_mask = t<=self.base_haz_time
        idx = torch.arange(bool_mask.shape[1], 0, -1)
        tmp2 = bool_mask * idx
        indices = torch.argmax(tmp2, 1, keepdim=True)
        base_ind = indices-1
        b = self.base_haz[base_ind]
        dist_t = t-self.base_haz_time[base_ind]
        dist = self.base_haz_time[indices]-self.base_haz_time[base_ind]
        delta = (self.base_haz[indices]-self.base_haz[base_ind])/dist
        haz_vec = b+delta*dist_t
        return haz_vec

    def get_vec_base(self,t):
        bool_mask = t>=self.base_haz_time
        tmp2 = bool_mask * self.base_haz #Might have to refine this step later
        return tmp2

    def get_base_haz_interpolate(self,t):
        haz_vec = self.get_base(t)
        haz_vec = self.check_min(t,haz_vec)
        haz_vec = self.check_max(t,haz_vec)
        return haz_vec

    def calculate_hazard(self,X,T,event):
        event = event.bool().squeeze()
        X = X[event,:]
        T=T[event]
        chks = X.shape[0]//5000 + 1
        haz_list = []
        for x,t in zip(torch.chunk(X,chks,dim=0),torch.chunk(T,chks,dim=0)):
            base_haz = self.get_base_haz_interpolate(t)
            exp_g = self.model.predict((x,t)).exp().cpu()
            haz_list.append(exp_g*base_haz)
        hazard = torch.cat(haz_list,dim=0)
        return hazard

    def calculate_cumulative_hazard(self,X,T):
        chks = X.shape[0]//5000 + 1
        c_haz_list = []
        for x,t in zip(torch.chunk(X,chks,dim=0),torch.chunk(T,chks,dim=0)):
            vec_hazard = self.get_vec_base(t)
            new_x = x.repeat_interleave(self.base_haz_time.shape[0],dim=0)
            new_t = self.base_haz_time.repeat(x.shape[0]).unsqueeze(-1)
            exp_g = self.model.predict((new_x,new_t)).view(x.shape[0],-1).exp().cpu()
            cum_hazard = torch.sum(vec_hazard*exp_g,dim=1)
            c_haz_list.append(cum_hazard)
        cum_hazard = torch.cat(c_haz_list,dim=0)
        return cum_hazard

    def calc_likelihood(self,hazard,cum_hazard):
        n = cum_hazard.shape[0]
        return -((hazard + 1e-6).log().sum() - cum_hazard.sum()) / n

class general_likelihood():
    def __init__(self,pycox_model):
        self.model = pycox_model

    def get_S_and_f(self,X,T,event):
        event = event.bool()
        surv = self.model.predict_surv_df(X)
        times = torch.from_numpy(surv.index.values).float()
        S = torch.from_numpy(surv.values).t().float()
        min_time = times.min().item()

        min_bool = (T<min_time).squeeze()
        max_time = times.max().item()
        max_bool = (T>max_time).squeeze()

        bool_mask = T<=times
        idx = torch.arange(bool_mask.shape[1], 0, -1)
        tmp2 = bool_mask * idx
        indices = torch.argmax(tmp2, 1, keepdim=True)
        base_ind = indices-1
        S_t_1 = torch.gather(S,dim=1,index=indices)
        S_t_0 = torch.gather(S,dim=1,index=base_ind)
        delta  = times[indices]-times[base_ind]
        t_prime = T-times[base_ind]
        surv = (1-t_prime/delta)*S_t_0+t_prime/delta*S_t_1
        f = -(S_t_1-S_t_0)/delta
        f[min_bool]=0.0
        f[max_bool]=0.0
        surv[min_bool]= S[min_bool,0].unsqueeze(-1)
        surv[max_bool]= S[max_bool,-1].unsqueeze(-1)
        return surv[~event],f[event]

    def calc_likelihood(self,S, f):
        n = S.shape[0] + f.shape[0]
        return -((f + 1e-6).log().sum() + S.sum()) / n

if __name__ == '__main__':
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn_pandas import DataFrameMapper
    import torch
    import torchtuples as tt

    from pycox.datasets import metabric
    from pycox.models import CoxCC, CoxPH, CoxTime
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
    net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)  # Actual net to be used
    model = CoxTime(net, tt.optim.Adam)  # the cox time framework, dont do this..
    model.optimizer.set_lr(0.01)
    epochs = 512
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = True
    batch_size = 256
    print(x_train.shape)
    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                    val_data=val.repeat(10).cat())
    base_haz = model.compute_baseline_hazards()
    cum_base_haz = model.compute_baseline_cumulative_hazards()
    t,s = torch.from_numpy(base_haz.index.values),torch.from_numpy(base_haz.values)
    reference_t = torch.tensor([[-100.0],[1.5],[200.0]])
    survL = general_likelihood(model)
    print(x_val.shape)
    S,f = survL.get_S_and_f(X=x_val,T=torch.from_numpy(y_val[0]).unsqueeze(-1),event=torch.from_numpy(y_val[1]).unsqueeze(-1))
    print(f.shape)
    print(S.shape)
    L_S = survL.calc_likelihood(S,f)
    print(L_S)
    coxL = HazardLikelihoodCoxTime(model)
    haz =  coxL.calculate_hazard(torch.from_numpy(x_val),torch.from_numpy(y_val[0]).unsqueeze(-1),event=torch.from_numpy(y_val[1]).unsqueeze(-1))
    cum_haz = coxL.calculate_cumulative_hazard(torch.from_numpy(x_val),torch.from_numpy(y_val[0]).unsqueeze(-1))
    print(haz.shape)
    print(cum_haz.shape)
    L = coxL.calc_likelihood(hazard=haz,cum_hazard=cum_haz)
    print(L)
    # surv_base = np.exp(-model.compute_baseline_cumulative_hazards())
    # surv_base.plot()
    # plt.savefig('test.png')
    # surv = model.predict_surv_df(x_test)
    # ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    # conc = ev.concordance_td()
    # time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    # ibs = ev.integrated_brier_score(time_grid)
    # inll = ev.integrated_nbll(time_grid)

