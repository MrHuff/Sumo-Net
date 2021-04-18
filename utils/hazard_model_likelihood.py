import pycox
import matplotlib.pyplot as plt
import torch
import numpy as np
from pycox.evaluation import EvalSurv


class ApproximateLikelihood:
    '''
    Enter a model, covariates, times and events
    model: a pycox model
    x: 2d numpy array
    t: 1d numpy array
    d: 1d numpy array
    baseline_sample_size: how many samples to compute the the baseline hazard
    half_width: k >=1 and densities are evaluated using T_{i-k+1} and T_{i+k}
    '''

    def __init__(self, model, x, t, d, baseline_sample_size, half_width):

        self.model = model
        self.t = t
        self.d = d
        self.x = x
        self.baseline_sample_size = baseline_sample_size
        self.n = len(self.t)
        self.half_width = int(half_width)
        self.mask_observed = self.d == 1
        self.densities = None
        self.survival = None
        self.log_likelihood = None

    def drop_outliers(self, min_time, max_time):

        # Select the outliers and reset self.x, t, d and n
        outlier_mask = (self.t > max_time) or (self.t < min_time)
        self.t = self.t[~outlier_mask]
        self.d = self.d[~outlier_mask]
        self.x = self.x[~outlier_mask]
        self.n = len(self.t)

        return None

    def get_densities(self,surv_df_raw=None):

        # Get the survival dataframe for x_observed, drop duplicate rows
        if surv_df_raw is None:
            survival_df_observed = self.model.predict_surv_df(self.x[self.mask_observed]).drop_duplicates(keep='first')
        else:
            survival_df_observed = surv_df_raw[self.mask_observed,:].t().drop_duplicates(keep='first')
        assert survival_df_observed.index.is_monotonic
        min_index, max_index = 0, len(survival_df_observed.index.values) - 1

        # Create an Eval object
        eval_observed = EvalSurv(survival_df_observed, self.t[self.mask_observed], self.d[self.mask_observed])

        # Get the indices of the survival_df
        indices = eval_observed.idx_at_times(self.t[self.mask_observed])
        left_index = np.minimum(np.maximum(indices - self.half_width + 1, min_index), max_index - 1)
        right_index = np.minimum(indices + self.half_width, max_index)

        # Get the survival probabilities and times
        left_survival = np.array([survival_df_observed.iloc[left_index[i], i] for i in range(len(left_index))])
        right_survival = np.array([survival_df_observed.iloc[right_index[i], i] for i in range(len(right_index))])
        left_time = np.array(survival_df_observed.index[left_index])
        right_time = np.array(survival_df_observed.index[right_index])

        # Approximate the derivative
        delta_survival = left_survival - right_survival
        delta_t = right_time - left_time
        self.densities = delta_survival / delta_t

        return self.densities

    def get_survival(self,surv_df_raw=None):

        # Create the survival_df and the Eval object
        if surv_df_raw is None:
            survival_df_censored = self.model.predict_surv_df(self.x[~self.mask_observed]).drop_duplicates(keep='first')
        else:
            survival_df_censored = surv_df_raw[~self.mask_observed,:].t().drop_duplicates(keep='first')
        eval_censored = EvalSurv(survival_df_censored, self.t[~self.mask_observed], self.d[~self.mask_observed])

        # Get a list of indices of the censored times
        indices = eval_censored.idx_at_times(self.t[~self.mask_observed])

        # Select the survival probabilities
        self.survival = np.array([survival_df_censored.iloc[indices[i], i] for i in range(len(indices))])

        return self.survival

    def get_approximated_likelihood(self,input_dat,target_dat, surv_df_raw=None):
        # Get the survival probabilities and the densities
        if surv_df_raw is None and self.model.__class__.__name__!= 'DeepHitSingle':
            _ = self.model.compute_baseline_hazards(sample=self.baseline_sample_size,input=input_dat,target=target_dat)
        self.get_survival(surv_df_raw)
        self.get_densities(surv_df_raw)

        # Compute the log-likelihood
        self.log_likelihood = np.mean(np.log(np.concatenate((self.survival, self.densities)) + 1e-7))

        return self.log_likelihood
    
class HazardLikelihoodCoxTime():
    # work with cumhazard instead and interpolate that one instead
    # interpolation linear in cumulative hazard early and after...
    def __init__(self, pycox_model):
        self.model = pycox_model
        base_haz_cum = self.model.compute_baseline_cumulative_hazards().drop_duplicates(keep='first')
        base_haz = self.model.compute_baseline_hazards()
        self.base_haz_time, self.base_haz = torch.from_numpy(base_haz.index.values).float(), torch.from_numpy(
            base_haz.values).float()
        self.base_haz_time_cum, self.base_haz_cum = torch.from_numpy(base_haz_cum.index.values).float(), torch.from_numpy(
            base_haz_cum.values).float()
        self.min_base_haz_time = self.base_haz_time.min()
        self.max_base_haz_time = self.base_haz_time.max()
        self.min_beta = self.base_haz_cum[0] / self.min_base_haz_time
        self.max_beta = (self.base_haz_cum[-1] - self.base_haz_cum[-2]) / (
                    self.base_haz_time[-1] - self.base_haz_time[-2])

    def check_min(self, t, haz_vec):
        haz_vec[t < self.min_base_haz_time] =  self.min_beta
        return haz_vec

    def check_max(self, t, haz_vec):
        haz_vec[t > self.max_base_haz_time] = self.max_beta
        return haz_vec

    def get_base(self, t):
        bool_mask = t <=self.base_haz_time_cum
        idx = torch.arange(bool_mask.shape[1], 0, -1)
        tmp2 = bool_mask * idx
        indices = torch.argmax(tmp2, 1, keepdim=True)
        base_ind = torch.relu(indices - 1)
        S_t_1 = self.base_haz_cum[indices]
        S_t_0 = self.base_haz_cum[base_ind]
        delta = self.base_haz_time_cum[indices] - self.base_haz_time_cum[base_ind]
        haz_vec = (S_t_1 - S_t_0) / delta
        return haz_vec

    def get_vec_base(self, t):
        bool_mask = t >= self.base_haz_time
        tmp2 = bool_mask * self.base_haz  # Might have to refine this step later
        return tmp2

    def get_base_haz_interpolate(self, t):
        haz_vec = self.get_base(t)
        haz_vec = self.check_min(t, haz_vec)
        haz_vec = self.check_max(t, haz_vec)
        return haz_vec

    def calculate_hazard(self, X, T, event):
        event = event.bool().squeeze()
        X = X[event, :]
        T = T[event]
        chks = X.shape[0] // 5000 + 1
        haz_list = []
        for x, t in zip(torch.chunk(X, chks, dim=0), torch.chunk(T, chks, dim=0)):
            base_haz = self.get_base_haz_interpolate(t)
            if self.model.__class__.__name__=='CoxTime':
                exp_g = self.model.predict((x, t)).exp().cpu()
            else:
                exp_g = self.model.predict((x)).exp().cpu()
            haz_list.append(exp_g * base_haz)
        hazard = torch.cat(haz_list, dim=0)
        return hazard

    def calculate_cumulative_hazard(self, X, T):
        chks = X.shape[0] // 5000 + 1
        c_haz_list = []
        for x, t in zip(torch.chunk(X, chks, dim=0), torch.chunk(T, chks, dim=0)):
            vec_hazard = self.get_vec_base(t)
            new_x = x.repeat_interleave(self.base_haz_time.shape[0], dim=0)
            new_t = self.base_haz_time.repeat(x.shape[0]).unsqueeze(-1)
            if self.model.__class__.__name__=='CoxTime':
                exp_g = self.model.predict((new_x, new_t)).view(x.shape[0], -1).exp().cpu()
            else:
                exp_g = self.model.predict((new_x)).view(x.shape[0], -1).exp().cpu()
            cum_hazard = torch.sum(vec_hazard * exp_g, dim=1)
            c_haz_list.append(cum_hazard)
        cum_hazard = torch.cat(c_haz_list, dim=0)
        return cum_hazard

    def estimate_likelihood(self, X, T, event):
        if T.dim() != 2:
            T = T.unsqueeze(-1)
        assert T.dim() == 2
        haz = self.calculate_hazard(X, T, event)
        cum_haz = self.calculate_cumulative_hazard(X, T)
        L = self.calc_likelihood(hazard=haz, cum_hazard=cum_haz)
        return L

    def calc_likelihood(self, hazard, cum_hazard):
        n = cum_hazard.shape[0]
        return -((hazard + 1e-6).log().sum() - cum_hazard.sum()) / n

class general_likelihood():
    def __init__(self, pycox_model):
        self.model = pycox_model

    def get_S_and_f_df(self, T, event,df):
        chks = T.shape[0] // 5000 + 1
        S_cat = []
        f_cat = []
        events = []
        df = df.T
        for t, e,surv_df in zip( torch.chunk(T, chks, dim=0), torch.chunk(event, chks, dim=0),np.array_split(df, chks,axis=0)):
            surv_df = surv_df.T
            tmp_df = surv_df.drop_duplicates(keep='first')
            if tmp_df.shape[0]>2:
                surv_df=tmp_df
            times = torch.from_numpy(surv_df.index.values).float()
            surv_tensor = torch.from_numpy(surv_df.values).t().float()
            min_time = times.min().item()
            max_time = times.max().item()
            min_bool = (t <= min_time).squeeze()
            max_bool = (t >= max_time).squeeze()
            surv_tensor = surv_tensor[torch.logical_and(~min_bool, ~max_bool)]
            t = t[torch.logical_and(~min_bool, ~max_bool)]
            e = e[torch.logical_and(~min_bool, ~max_bool)]
            bool_mask = t <= times
            idx = torch.arange(bool_mask.shape[1], 0, -1)
            tmp2 = bool_mask * idx
            indices = torch.argmax(tmp2, 1, keepdim=True)
            base_ind = torch.relu(indices - 1)
            S_t_1 = torch.gather(surv_tensor, dim=1, index=indices)
            S_t_0 = torch.gather(surv_tensor, dim=1, index=base_ind)
            delta = times[indices] - times[base_ind]
            t_prime = t - times[base_ind]
            S = (1 - t_prime / delta) * S_t_0 + t_prime / delta * S_t_1
            f = -(S_t_1 - S_t_0) / delta
            events.append(e)
            S_cat.append(S)
            f_cat.append(f)
        event = torch.cat(events).bool()
        S_cat = torch.cat(S_cat, dim=0)
        f_cat = torch.cat(f_cat, dim=0)
        assert S_cat.shape[0] == event.shape[0]
        assert f_cat.shape[0] == event.shape[0]
        return S_cat[~event], f_cat[event]

    def get_S_and_f(self, X, T, event):
        chks = X.shape[0] // 5000 + 1
        S_cat = []
        f_cat = []
        events = []
        for x, t, e in zip(torch.chunk(X, chks, dim=0), torch.chunk(T, chks, dim=0), torch.chunk(event, chks, dim=0)):
            surv_df = self.model.predict_surv_df(x)
            surv_df = surv_df.drop_duplicates(keep='first')
            times = torch.from_numpy(surv_df.index.values).float()
            surv_tensor = torch.from_numpy(surv_df.values).t().float()
            min_time = times.min().item()
            min_bool = (t <= min_time).squeeze()
            max_time = times.max().item()
            max_bool = (t >= max_time).squeeze()
            surv_tensor = surv_tensor[torch.logical_and(~min_bool, ~max_bool)]
            t = t[torch.logical_and(~min_bool, ~max_bool)]
            e = e[torch.logical_and(~min_bool, ~max_bool)]
            bool_mask = t <= times
            idx = torch.arange(bool_mask.shape[1], 0, -1)
            tmp2 = bool_mask * idx
            indices = torch.argmax(tmp2, 1, keepdim=True)
            base_ind = torch.relu(indices - 1)
            S_t_1 = torch.gather(surv_tensor, dim=1, index=indices)
            S_t_0 = torch.gather(surv_tensor, dim=1, index=base_ind)
            delta = times[indices] - times[base_ind]
            t_prime = t - times[base_ind]
            S = (1 - t_prime / delta) * S_t_0 + t_prime / delta * S_t_1
            f = -(S_t_1 - S_t_0) / delta
            events.append(e)
            S_cat.append(S)
            f_cat.append(f)
        event = torch.cat(events).bool()
        S_cat = torch.cat(S_cat, dim=0)
        f_cat = torch.cat(f_cat, dim=0)
        assert S_cat.shape[0] == event.shape[0]
        assert f_cat.shape[0] == event.shape[0]
        return S_cat[~event], f_cat[event]

    def estimate_likelihood(self, X, T, event):
        if T.dim() != 2:
            T = T.unsqueeze(-1)
        assert T.dim() == 2
        S, f = self.get_S_and_f(X, T, event)
        L = self.calc_likelihood(S, f)
        return L

    def estimate_likelihood_df(self, T, event,df):
        if T.dim() != 2:
            T = T.unsqueeze(-1)
        assert T.dim() == 2
        S, f = self.get_S_and_f_df( T, event,df)
        L = self.calc_likelihood(S, f)
        return L

    def calc_likelihood(self, S, f):
        n = S.shape[0] + f.shape[0]
        return -((f + 1e-6).log().sum() + (S + 1e-6).log().sum()) / n

