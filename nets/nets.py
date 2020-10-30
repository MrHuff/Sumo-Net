import torch

class Log1PlusExp(torch.autograd.Function):
    """Implementation of x ↦ log(1 + exp(x))."""
    @staticmethod
    def forward(ctx, x):
        exp = x.exp()
        ctx.save_for_backward(x)
        y = exp.log1p()
        return x.where(torch.isinf(exp),y.half() if x.type()=='torch.cuda.HalfTensor' else y )

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (-x).exp().half() if x.type()=='torch.cuda.HalfTensor' else (-x).exp()
        return grad_output / (1 + y)

log1plusexp = Log1PlusExp.apply
class nn_node(torch.nn.Module):
    def __init__(self,d_in,d_out,transformation=torch.tanh):
        super(nn_node, self).__init__()
        self.w = torch.nn.Linear(d_in,d_out)
        self.f = transformation

    def forward(self,X):
        return self.f(self.w(X))

class bounded_nn_layer(torch.nn.Module):
    def __init__(self,d_in,d_out,bounding_op=lambda x: x**2,transformation=torch.tanh):
        super(bounded_nn_layer, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(*(d_in,d_out)),requires_grad=True)
        self.f = transformation
        self.bounding_op = bounding_op
        self.bias = torch.nn.Parameter(torch.randn(d_out),requires_grad=True)

    def forward(self,X):
        return self.f(X@self.bounding_op(self.W)+self.bias)

class mixed_layer(torch.nn.Module):
    def __init__(self,d_in,d_in_bounded,d_out,bounding_op=lambda x: x**2,transformation=torch.tanh):
        super(mixed_layer, self).__init__()
        self.pos_weights = torch.nn.Parameter(torch.randn(*(d_in_bounded, d_out)), requires_grad=True)
        self.f = transformation
        self.bounding_op = bounding_op
        self.bias = torch.nn.Parameter(torch.randn(d_out), requires_grad=True)
        self.w = torch.nn.Linear(d_in,d_out)

    def forward(self,X,x_bounded):
        return self.f(x_bounded @ self.bounding_op(self.pos_weights) + self.bias + self.w(X))

class survival_net(torch.nn.Module):
    def __init__(self,
                 d_in_x,
                 d_in_y,
                 d_out,
                 layers_x,
                 layers,
                 bounding_op=lambda x: x**2,
                 transformation=torch.tanh,
                 direct_dif = True,
                 objective = 'hazard'
                 ):
        super(survival_net, self).__init__()
        self.init_covariate_net(d_in_x,layers_x,transformation)
        self.init_middle_net(dx_in=layers_x[-1],d_in_y=d_in_y,d_out=d_out,layers=layers,transformation=transformation,bounding_op=bounding_op)
        self.eps = 1e-5
        self.direct = direct_dif
        self.objective  = objective
        if self.objective in ['hazard','hazard_mean']:
            self.f = self.forward_hazard
            self.f_cum = self.forward_cum_hazard
        elif self.objective in ['S','S_mean']:
            self.f=self.forward_f
            self.f_cum=self.forward_S

    def init_covariate_net(self,d_in_x,layers_x,transformation):
        module_list = [nn_node(d_in=d_in_x,d_out=layers_x[0],transformation=transformation)]
        for l_i in range(1,len(layers_x)):
            module_list.append(nn_node(d_in=layers_x[l_i-1],d_out=layers_x[l_i],transformation=transformation))
        self.covariate_net = torch.nn.Sequential(*module_list)

    def init_middle_net(self,dx_in,d_in_y,d_out,layers,transformation,bounding_op):
        self.mixed_layer = mixed_layer(d_in=dx_in, d_in_bounded=d_in_y, d_out=layers[0], transformation=transformation, bounding_op=bounding_op)
        module_list = []
        for l_i in range(1,len(layers)):
            module_list.append(bounded_nn_layer(d_in=layers[l_i - 1], d_out=layers[l_i], transformation=transformation, bounding_op=bounding_op))
        module_list.append(bounded_nn_layer(d_in=layers[-1], d_out=d_out, transformation=lambda x:x, bounding_op=bounding_op))
        self.middle_net = torch.nn.Sequential(*module_list)

    def forward(self,x_cov,y):
        return self.f(x_cov,y)

    def forward_cum(self,x_cov,y,mask):
        return self.f_cum(x_cov, y,mask)

    def forward_S(self,x_cov,y,mask):
        x_cov = x_cov[~mask,:]
        y = y[~mask,:]
        x_cov = self.covariate_net(x_cov)
        h = self.middle_net(self.mixed_layer(x_cov, y))
        return -log1plusexp(h)



    def forward_f(self,x_cov,y):
        x_cov = self.covariate_net(x_cov)
        h = self.middle_net(self.mixed_layer(x_cov, y))
        h_forward = self.middle_net(self.mixed_layer(x_cov, y + self.eps))
        F = h.sigmoid()
        if self.direct:
            F_forward = h_forward.sigmoid()
            f = ((F_forward - F) / self.eps)
        else:
            f = ((h_forward - h) / self.eps)*F*(1-F) #(F)*(1-F), F = h.sigmoid() log(sig(h)) + log(1-sig(h)) = h-2*log1plusexp(h)
        return f

    def forward_cum_hazard(self, x_cov, y, mask):
        x_cov = self.covariate_net(x_cov)
        h = self.middle_net(self.mixed_layer(x_cov, y))
        return torch.exp(10*torch.tanh(h/10.)) #log1plusexp(h)

    def forward_hazard(self, x_cov, y):
        x_cov = self.covariate_net(x_cov)
        h = self.middle_net(self.mixed_layer(x_cov, y))
        h_forward = self.middle_net(self.mixed_layer(x_cov, y + self.eps))
        if self.direct:
            # hazard = (log1plusexp(h_forward) - log1plusexp(h))/self.eps
            hazard = (torch.exp(10*torch.tanh(h_forward/10.)) - torch.exp(10*torch.tanh(h/10)))/self.eps
        else:
            hazard = ((h_forward-h)/self.eps) * torch.sigmoid(h) #*torch.exp(h) #
        return hazard

    def forward_S_eval(self,x_cov,y):
        if self.objective in ['hazard','hazard_mean']:
            S = torch.exp(-self.forward_cum_hazard(x_cov, y, []))
            return S
        elif self.objective in ['S','S_mean']:
            x_cov = self.covariate_net(x_cov)
            h = self.middle_net(self.mixed_layer(x_cov, y))
            return 1-h.sigmoid_()

def get_objective(objective):
    if objective == 'hazard':
        return log_objective_hazard
    if objective == 'hazard_mean':
        return log_objective_hazard_mean
    elif objective == 'S':
        return log_objective
    elif objective=='S_mean':
        return log_objective_mean

def log_objective(S,f):
    return -(f+1e-6).log().sum()-S.sum()

def log_objective_mean(S,f):
    n = S.shape[0]+f.shape[0]
    return -((f+1e-6).log().sum()+S.sum())/n

def log_objective_hazard(cum_hazard,hazard): #here cum_hazard should be a vector of
    # length n, and hazard only needs to be computed for all individuals with
    # delta = 1 I'm not sure how to implement that best?
    return -(  (hazard+1e-6).log().sum()-cum_hazard.sum() )

def log_objective_hazard_mean(cum_hazard,hazard): #here cum_hazard should be a vector of
    # length n, and hazard only needs to be computed for all individuals with
    # delta = 1 I'm not sure how to implement that best?
    n = cum_hazard.shape[0]+hazard.shape[0]
    return -(  (hazard+1e-6).log().sum()-cum_hazard.sum() )/n












