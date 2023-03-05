import torch
import tqdm
import gpytorch
import numpy as np
import gc
from nets.nets import log_objective_mean

CDF_APPROX_COEFF = 1.65451
sqrt_pi = np.pi ** 0.5
log2pi = np.log(np.pi * 2)


class Kernel(torch.nn.Module):
    def __init__(self, ):
        super(Kernel, self).__init__()

    def sq_dist(self, x1, x2):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

        # Compute squared distance matrix using quadratic expansion
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))
        # Zero out negative values
        res.clamp_min_(0)

        # res  = torch.cdist(x1,x2,p=2)
        # Zero out negative values
        # res.clamp_min_(0)
        return res

    def covar_dist(self, x1, x2):
        return self.sq_dist(x1, x2).sqrt()

    def get_median_ls(self, X, Y):
        with torch.no_grad():
            if X.shape[0] > 5000:
                idx = torch.randperm(2500)
                X = X[idx, :]
                Y = Y[idx, :]
            d = self.covar_dist(x1=X, x2=Y)
            ret = torch.sqrt(torch.median(d[d >= 0]))
            return ret


def ensure_pos_diag(L):
    v = torch.diag(L)
    v = torch.clamp(v, min=1e-6)
    mask = torch.diag(torch.ones_like(v))
    L = mask * torch.diag(v) + (1. - mask) * L
    return L


def ensure_pos_diag_svgp(K, cap=1e-1):
    v = torch.diag(K)
    v[v <= 0] = cap
    mask = torch.diag(torch.ones_like(v))
    K = mask * torch.diag(v) + (1. - mask) * K
    return K


class r_param_cholesky_scaling(torch.nn.Module):
    def __init__(self, k, Z, X, sigma, scale_init=1.0, parametrize_Z=False):
        super(r_param_cholesky_scaling, self).__init__()
        self.k = k
        self.scale = torch.nn.Parameter(torch.ones(1) * scale_init)
        self.register_buffer('eye', torch.eye(Z.shape[0]))
        self.parametrize_Z = parametrize_Z
        if parametrize_Z:
            self.Z = torch.nn.Parameter(Z)
        else:
            self.register_buffer('Z', Z)
        self.register_buffer('X', X)
        self.sigma = sigma
        self.cap = 1e-1

    def init_L(self):
        with torch.no_grad():
            kx = self.k(self.Z, self.X).evaluate()
            self.kzz = self.k(self.Z).evaluate()
        L_set = False
        for reg in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:  # ,1e-1,1.0,5.0,10.]:
            try:
                with torch.no_grad():
                    chol_L = torch.linalg.cholesky(self.kzz + kx @ kx.t() / self.sigma + self.eye * reg)
                    L = torch.linalg.cholesky(torch.cholesky_inverse(chol_L))
                    # L = torch.linalg.cholesky(torch.inverse(self.kzz  + self.eye*self.sigma))
                    self.L = torch.nn.Parameter(L)
                    L_set = True
            except Exception as e:
                gc.collect()
                print('cholesky init error: ', reg)
                print(e)
                # torch.cuda.empty_cache()
        if not L_set:
            print("Welp this is awkward, it don't want to factorize")

            gc.collect()
            # torch.cuda.empty_cache()
            L = torch.randn_like(self.kzz) * 0.1
            self.L = torch.nn.Parameter(L)
        inv_set = False
        for reg in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:  # , 1e-1, 1.0,2.5,5.0,10.]:
            self.reg = reg
            try:
                if not self.parametrize_Z:
                    with torch.no_grad():
                        self.register_buffer('inv_L', torch.linalg.cholesky(self.kzz + self.eye * self.reg))
                        self.register_buffer('inv_Z', torch.cholesky_inverse(self.inv_L))
                    inv_set = True
            except Exception as e:
                gc.collect()
                print(e)
                print('cholesky inv error: ', reg)
                pass
        if not inv_set:
            gc.collect()
            self.parametrize_Z = True

    def forward(self, x1, x2=None):
        L = torch.tril(self.L) + self.eye * self.reg
        L = ensure_pos_diag(L)
        Z = self.Z
        if x2 is None:
            kzx = self.k(Z, x1).evaluate()
            t = L.t() @ kzx  # L\cdot k(Z,X)

            if self.parametrize_Z:
                kzz = self.k(Z).evaluate()
                chol_z = torch.linalg.cholesky(kzz + self.eye * self.reg)
                sol = torch.cholesky_solve(kzx, chol_z)
            else:
                # sol = torch.cholesky_solve(kzx, self.inv_L)
                sol = self.inv_Z @ kzx
            if len(t.shape) == 3:  # t=[L^T k(Z,X_i),L^T k(Z,X_{i+1}),]
                T_mat = t.permute(0, 2, 1) @ t  # T_mat ok
                inverse_part = kzx.permute(0, 2, 1) @ sol
                out = self.k(x1).evaluate() - inverse_part + T_mat
                return out
            else:
                T_mat = t.t() @ t
                out = self.k(x1).evaluate() - kzx.t() @ sol + T_mat
                return out
        else:
            kzx_1 = self.k(Z, x1).evaluate()
            kzx_2 = self.k(Z, x2).evaluate()
            t = L.t() @ kzx_2
            t_ = kzx_1.t() @ L
            T_mat = t_ @ t
            if self.parametrize_Z:
                kzz = self.k(Z).evaluate()
                chol_z = torch.linalg.cholesky(kzz + self.eye * self.reg)
                sol = torch.cholesky_solve(kzx_2, chol_z)
            else:
                # sol = torch.cholesky_solve(kzx_2, self.inv_L)
                sol = self.inv_Z @ kzx_2
            out = self.k(x1, x2).evaluate() - kzx_1.t() @ sol + T_mat  # /self.sigma
            return out

    def get_sigma_debug(self):
        with torch.no_grad():
            L = torch.tril(self.L) + self.eye * self.reg
            L = ensure_pos_diag(L)
            return L @ L.t()


class GWI(torch.nn.Module):
    def __init__(self, N, m_q, m_p, r, reg=1e-1, sigma=1.0, x_s=250):
        super(GWI, self).__init__()
        self.r = r
        self.m_q = m_q
        self.sigma = sigma
        self.k = self.r.k
        self.reg = reg
        self.m_p = m_p
        self.m = self.r.Z.shape[0]
        self.x_s = x_s
        self.register_buffer('eye', reg * torch.eye(self.m))
        self.register_buffer('big_eye', 100. * torch.eye(self.x_s))
        self.U_calculated = False
        self.N = N

    def get_MPQ(self, batch_X=None):
        raise NotImplementedError

    def get_APQ(self, batch_X, Z_prime, T=None):
        X = batch_X
        rk_hat = 1 / X.shape[0] * self.r(Z_prime, X) @ self.k(X, Z_prime).evaluate()  # self.r.rk(X)/
        if T is not None:
            rk_hat = T * rk_hat
        eigs = torch.linalg.eigvals(rk_hat + self.big_eye)
        eigs = eigs.abs()
        eigs = eigs - self.big_eye.diag()
        eigs = eigs[eigs > 0]
        res = torch.sum(eigs ** 0.5) / self.x_s ** 0.5
        return res

    def get_APQ_diagnose_xs(self, batch_X, Z_prime):
        X = batch_X
        rk_hat = 1 / X.shape[0] * self.r(Z_prime, X) @ self.k(X, Z_prime).evaluate()  # self.r.rk(X)/
        eigs = torch.linalg.eigvals(rk_hat + self.big_eye)
        eigs = eigs.abs()
        eigs = eigs - self.big_eye.diag()
        eigs = eigs[eigs > 0]

        return eigs.detach().cpu().numpy()

    def calc_hard_tr_term(self, X, Z_prime, T=None):
        mpq = self.get_APQ(X, Z_prime, T)
        p_trace = self.k(X.unsqueeze(1)).evaluate().mean()
        q_trace = self.r(X.unsqueeze(1)).mean()
        if T is not None:
            q_trace = q_trace * T
        mat = p_trace + q_trace - 2 * mpq
        return mat, -2 * mpq, q_trace, p_trace

    def posterior_variance(self, X, T=None):
        with torch.no_grad():
            posterior = self.r(X.unsqueeze(1)).squeeze() + self.sigma
            if T is not None:
                posterior = posterior * T
        return posterior ** 0.5

    def survival_likelihood(self, X, X_f, y, y_f, x_cat, x_cat_f, mask):
        x_concat_S = torch.cat([X[~mask],y[~mask]],dim=1)
        cov_mat_S = self.r(x_concat_S)
        L_S = torch.cholesky(cov_mat_S) @ torch.randn_like(y[~mask])
        S, h_S = self.m_q.forward_S(X, y, mask, x_cat,L_S)
        if S.numel() == 0:
            S = S.detach()
        y_f = torch.autograd.Variable(y_f, requires_grad=True)
        x_concat_f = torch.cat([X_f,y_f],dim=1)
        cov_mat_f = self.r(x_concat_f)
        L_f = torch.cholesky(cov_mat_f) @ torch.randn_like(y_f)
        f, h_f = self.m_q.forward_f(X_f, y_f, x_cat_f,L_f)
        if f.numel() == 0:
            f = f.detach()
        total_loss = log_objective_mean(S, f)
        tmp = torch.ones_like(y) * self.m_p
        reg = torch.mean((torch.cat([h_f,h_S],dim=0) - tmp) ** 2)
        return total_loss,reg

    def get_loss(self, y, X, x_cat, delta, Z_prime_cov, T=None):

        mask = delta == 1
        X_f = X[mask, :]
        y_f = y[mask, :]
        if not isinstance(x_cat, list):  # F
            x_cat = x_cat.to(X.device)
            x_cat_f = x_cat[mask, :]
        else:
            x_cat_f = []
        ll,reg = self.survival_likelihood(X, X_f, y, y_f, x_cat, x_cat_f, mask)
        X_cov = torch.cat([X,y],dim=1)
        tot_trace, hard_trace, tr_Q, tr_P = self.calc_hard_tr_term(X_cov, Z_prime_cov, T)
        # print('MPQ: ', hard_trace)
        # print('Tr Q: ', tr_Q)
        # print('Tr P: ', tr_P)
        D = torch.relu((tot_trace + reg)) ** 0.5  # this feels a bit broken?! a small trace term should make a small NLL???????
        sigma = self.sigma
        if T is not None:
            sigma = sigma * T
        log_loss = self.N * tr_Q / (2. * sigma) + ll
        return log_loss / X.shape[0] + D

    def mean_forward(self, X):
        return self.m_q(X)

    def mean_pred(self, X):
        with torch.no_grad():
            return self.m_q(X)
