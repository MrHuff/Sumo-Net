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

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class CustomSwish(torch.nn.Module):
    def forward(self, input_tensor):
        return Swish.apply(input_tensor)

class multi_input_Sequential(torch.nn.Sequential):
    def forward(self, inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class multi_input_Sequential_res_net(torch.nn.Sequential):
    def forward(self, inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                output = module(inputs)
                if inputs.shape[1]==output.shape[1]:
                    inputs = inputs+output
                else:
                    inputs = output
        return inputs

log1plusexp = Log1PlusExp.apply
class nn_node(torch.nn.Module): #Add dropout layers, Do embedding layer as well!
    def __init__(self,d_in,d_out,cat_size_list,dropout=0.1,transformation=torch.tanh):
        super(nn_node, self).__init__()

        self.has_cat = len(cat_size_list)>0
        self.latent_col_list = []
        for i,el in enumerate(cat_size_list):
            col_size = el//2+2
            setattr(self,f'embedding_{i}',torch.nn.Embedding(el,col_size))
            self.latent_col_list.append(col_size)
        self.w = torch.nn.Linear(d_in+sum(self.latent_col_list),d_out)
        self.f = transformation
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,X,x_cat=[]):
        if not isinstance(x_cat,list):
            seq = torch.unbind(x_cat,1)
            for i,f in enumerate(seq):
                o = getattr(self,f'embedding_{i}')(f)
                X = torch.cat([X,o],dim=1)
        return self.dropout(self.f(self.w(X)))

class bounded_nn_layer(torch.nn.Module): #Add dropout layers
    def __init__(self, d_in, d_out, bounding_op=lambda x: x ** 2, transformation=torch.tanh):
        super(bounded_nn_layer, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(*(d_in,d_out)),requires_grad=True)
        self.f = transformation
        self.bounding_op = bounding_op
        self.bias = torch.nn.Parameter(torch.randn(d_out),requires_grad=True)

    def forward(self,X):
        return self.f(X@self.bounding_op(self.W)+self.bias)

class mixed_layer(torch.nn.Module): #Add dropout layers
    def __init__(self, d_in, d_in_bounded, d_out, bounding_op=lambda x: x ** 2, transformation=torch.tanh):
        super(mixed_layer, self).__init__()
        self.pos_weights = torch.nn.Parameter(torch.randn(*(d_in_bounded, d_out)), requires_grad=True)
        self.f = transformation
        self.bounding_op = bounding_op
        self.bias = torch.nn.Parameter(torch.randn(d_out), requires_grad=True)
        self.w = torch.nn.Linear(d_in,d_out)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self,X,x_bounded):
        return self.f(x_bounded @ self.bounding_op(self.pos_weights) + self.bias + self.w(X))

class survival_net(torch.nn.Module):
    def __init__(self,
                 d_in_x,
                 cat_size_list,
                 d_in_y,
                 d_out,
                 layers_x,
                 layers,
                 dropout=0.9,
                 bounding_op=lambda x: x**2,
                 transformation=torch.tanh,
                 direct_dif = True,
                 objective = 'hazard'
                 ):
        super(survival_net, self).__init__()
        self.init_covariate_net(d_in_x,layers_x,cat_size_list,transformation,dropout)
        self.init_middle_net(dx_in=layers_x[-1], d_in_y=d_in_y, d_out=d_out, layers=layers,
                             transformation=transformation, bounding_op=bounding_op)
        self.eps = 1e-5
        self.direct = direct_dif
        self.objective  = objective
        if self.objective in ['hazard','hazard_mean']:
            self.f = self.forward_hazard
            self.f_cum = self.forward_cum_hazard
        elif self.objective in ['S','S_mean']:
            self.f=self.forward_f
            self.f_cum=self.forward_S

    def init_covariate_net(self,d_in_x,layers_x,cat_size_list,transformation,dropout):
        module_list = [nn_node(d_in=d_in_x,d_out=layers_x[0],cat_size_list=cat_size_list,transformation=transformation,dropout=dropout)]
        for l_i in range(1,len(layers_x)):
            module_list.append(nn_node(d_in=layers_x[l_i-1],d_out=layers_x[l_i],cat_size_list=[],transformation=transformation,dropout=dropout))
        self.covariate_net = multi_input_Sequential_res_net(*module_list)

    def init_middle_net(self, dx_in, d_in_y, d_out, layers, transformation, bounding_op):
        # self.mixed_layer = mixed_layer(d_in=dx_in, d_in_bounded=d_in_y, d_out=layers[0], transformation=transformation, bounding_op=bounding_op,dropout=dropout)
        module_list = [mixed_layer(d_in=dx_in, d_in_bounded=d_in_y, d_out=layers[0], bounding_op=bounding_op,
                                   transformation=transformation)]
        for l_i in range(1,len(layers)):
            module_list.append(bounded_nn_layer(d_in=layers[l_i - 1], d_out=layers[l_i], bounding_op=bounding_op,
                                                transformation=transformation))
        module_list.append(
            bounded_nn_layer(d_in=layers[-1], d_out=d_out, bounding_op=bounding_op, transformation=lambda x: x))
        self.middle_net = multi_input_Sequential_res_net(*module_list)

    def forward(self,x_cov,y,x_cat=[]):
        return self.f(x_cov,y,x_cat)

    def forward_cum(self,x_cov,y,mask,x_cat=[]):
        return self.f_cum(x_cov, y,mask,x_cat)

    def forward_S(self,x_cov,y,mask,x_cat=[]):
        x_cov = x_cov[~mask,:]
        y = y[~mask,:]
        if not isinstance(x_cat,list):
            x_cat=x_cat[~mask,:]
        #Fix categorical business
        x_cov = self.covariate_net((x_cov,x_cat))
        h = self.middle_net((x_cov, y))
        return -log1plusexp(h)

    def forward_f(self,x_cov,y,x_cat=[]):
        x_cov = self.covariate_net((x_cov,x_cat))
        h = self.middle_net((x_cov, y))
        h_forward = self.middle_net((x_cov, y + self.eps))
        F = h.sigmoid()
        if self.direct:
            F_forward = h_forward.sigmoid()
            f = ((F_forward - F) / self.eps)
        else:
            f = ((h_forward - h) / self.eps)*F*(1-F) #(F)*(1-F), F = h.sigmoid() log(sig(h)) + log(1-sig(h)) = h-2*log1plusexp(h)
        return f

    def forward_cum_hazard(self, x_cov, y, mask,x_cat=[]):
        x_cov = self.covariate_net((x_cov,x_cat))
        h = self.middle_net((x_cov, y))
        return log1plusexp(h)

    def forward_hazard(self, x_cov, y,x_cat=[]):
        x_cov = self.covariate_net((x_cov,x_cat))
        h = self.middle_net((x_cov, y))
        h_forward = self.middle_net((x_cov, y + self.eps))
        if self.direct:
            hazard = (log1plusexp(h_forward) - log1plusexp(h)) / self.eps
        else:
            hazard = torch.sigmoid(h) * ((h_forward - h) / self.eps)
        return hazard

    def forward_S_eval(self,x_cov,y,x_cat=[]):
        if self.objective in ['hazard','hazard_mean']:
            S = torch.exp(-self.forward_cum_hazard(x_cov, y, [],x_cat))
            return S
        elif self.objective in ['S','S_mean']:
            x_cov = self.covariate_net((x_cov,x_cat))
            h = self.middle_net((x_cov, y))
            return 1-h.sigmoid_()


class ocean_net(torch.nn.Module):
    def __init__(self,
                 d_in_x,
                 cat_size_list,
                 d_in_y,
                 d_out,
                 layers_x,
                 layers,
                 dropout=0.9,
                 bounding_op=lambda x: x ** 2,
                 transformation=torch.tanh,
                 direct_dif=True,
                 objective='S_mean'
                 ):
        super(ocean_net, self).__init__()

        self.bounding_op = bounding_op
        self.covariate_net = self.init_covariate_net(d_in_x, layers_x, cat_size_list, transformation, dropout)
        self.middle_net = self.init_middle_net(dx_in=layers_x[-1], d_in_y=d_in_y, d_out=d_out, layers=layers,
                             transformation=transformation, bounding_op=bounding_op)
        self.prod_net_t = self.init_bounded_net(d_in_y,d_out,layers,transformation,bounding_op)
        self.prod_net_x = self.init_covariate_net_2(d_in_x, layers_x,d_out, cat_size_list, transformation, dropout)

        self.net_t = self.init_bounded_net(d_in_y,d_out,layers,transformation,bounding_op)
        self.net_x = self.init_covariate_net_2(d_in_x, layers_x,d_out, cat_size_list, transformation, dropout)


        self.eps = 1e-3
        self.direct = direct_dif
        self.objective = objective
        if self.objective in ['hazard', 'hazard_mean']:
            self.f = self.forward_hazard
            self.f_cum = self.forward_cum_hazard
        elif self.objective in ['S', 'S_mean']:
            self.f = self.forward_f
            self.f_cum = self.forward_S

    def init_covariate_net(self, d_in_x, layers_x, cat_size_list, transformation, dropout):
        module_list = [
            nn_node(d_in=d_in_x, d_out=layers_x[0], cat_size_list=cat_size_list, transformation=transformation,
                    dropout=dropout)]
        for l_i in range(1, len(layers_x)):
            module_list.append(
                nn_node(d_in=layers_x[l_i - 1], d_out=layers_x[l_i], cat_size_list=[], transformation=transformation,
                        dropout=dropout))
        return multi_input_Sequential_res_net(*module_list)

    def init_covariate_net_2(self, d_in_x, layers_x,d_out, cat_size_list, transformation, dropout):
        module_list = [
            nn_node(d_in=d_in_x, d_out=layers_x[0], cat_size_list=cat_size_list, transformation=transformation,
                    dropout=dropout)]
        for l_i in range(1, len(layers_x)):
            module_list.append(
                nn_node(d_in=layers_x[l_i - 1], d_out=layers_x[l_i], cat_size_list=[], transformation=transformation,
                        dropout=dropout))
        module_list.append(
            nn_node(d_in=layers_x[ -1], d_out=d_out, cat_size_list=[], transformation=transformation,
                    dropout=dropout))
        return multi_input_Sequential_res_net(*module_list)

    def init_middle_net(self, dx_in, d_in_y, d_out, layers, transformation, bounding_op):
        # self.mixed_layer = mixed_layer(d_in=dx_in, d_in_bounded=d_in_y, d_out=layers[0], transformation=transformation, bounding_op=bounding_op,dropout=dropout)
        module_list = [mixed_layer(d_in=dx_in, d_in_bounded=d_in_y, d_out=layers[0], bounding_op=bounding_op,
                                   transformation=transformation)]
        for l_i in range(1, len(layers)):
            module_list.append(bounded_nn_layer(d_in=layers[l_i - 1], d_out=layers[l_i], bounding_op=bounding_op,
                                                transformation=transformation))
        module_list.append(
            bounded_nn_layer(d_in=layers[-1], d_out=d_out, bounding_op=bounding_op, transformation=lambda x: x))
        return multi_input_Sequential_res_net(*module_list)

    def init_bounded_net(self, dx_in, d_out, layers, transformation, bounding_op):
        # self.mixed_layer = mixed_layer(d_in=dx_in, d_in_bounded=d_in_y, d_out=layers[0], transformation=transformation, bounding_op=bounding_op,dropout=dropout)
        module_list = [bounded_nn_layer(d_in=dx_in, d_out=layers[0], bounding_op=bounding_op,
                                   transformation=transformation)]
        for l_i in range(1, len(layers)):
            module_list.append(bounded_nn_layer(d_in=layers[l_i - 1], d_out=layers[l_i], bounding_op=bounding_op,
                                                transformation=transformation))
        module_list.append(
            bounded_nn_layer(d_in=layers[-1], d_out=d_out, bounding_op=bounding_op, transformation=lambda x: x))
        return multi_input_Sequential_res_net(*module_list)

    def forward(self, x_cov, y, x_cat=[]):
        return self.f(x_cov, y, x_cat)

    def forward_cum(self, x_cov, y, mask, x_cat=[]):
        return self.f_cum(x_cov, y, mask, x_cat)

    def forward_S(self, x_cov_in, y, mask, x_cat=[]):
        x_cov_in = x_cov_in[~mask, :]
        y = y[~mask, :]
        if not isinstance(x_cat, list):
            x_cat = x_cat[~mask, :]
        # Fix categorical business
        x_cov = self.covariate_net((x_cov_in, x_cat))
        h_xt = self.middle_net((x_cov, y))
        h_xh_t = self.prod_net_t((y))* self.bounding_op(self.prod_net_x((x_cov_in, x_cat)))
        h_t = self.net_t((y))
        h_x = self.net_x((x_cov_in, x_cat))
        h = h_xh_t + h_t + h_x +h_xt
        return -log1plusexp(h)

    def forward_f(self, x_cov_in, y, x_cat=[]):
        x_cov = self.covariate_net((x_cov_in, x_cat))
        h_xt = self.middle_net((x_cov, y))
        h_xt_forward = self.middle_net((x_cov, y + self.eps))
        h_t = self.net_t((y))
        h_t_forward = self.net_t((y+ self.eps))
        prod_x = self.bounding_op(self.prod_net_x((x_cov_in, x_cat)))
        h_xh_t = self.prod_net_t((y))
        h_xh_t_forward = self.prod_net_t((y+self.eps))
        h_x = self.net_x((x_cov_in, x_cat))
        h = h_xt + h_xh_t*prod_x + h_t + h_x
        F = h.sigmoid()

        if self.direct:
            F_forward = (h_xt_forward + h_xh_t_forward*prod_x + h_t_forward + h_x).sigmoid()
            f = ((F_forward - F) / self.eps)
        else:
            diff = h_xt_forward+h_t_forward-(h_xt + h_t) + prod_x*(h_xh_t_forward-h_xh_t)
            f = (diff / self.eps) * F * (
                        1 - F)  # (F)*(1-F), F = h.sigmoid() log(sig(h)) + log(1-sig(h)) = h-2*log1plusexp(h)
        return f

    def forward_cum_hazard(self, x_cov_in, y, mask, x_cat=[]):
        x_cov = self.covariate_net((x_cov_in, x_cat))
        h_xt = self.middle_net((x_cov, y))
        h_xh_t = self.prod_net_t((y))*self.bounding_op(self.prod_net_x((x_cov_in, x_cat)))
        h_t = self.net_t((y))
        h_x = self.net_x((x_cov_in, x_cat))
        h = h_xt + h_xh_t + h_t + h_x

        return log1plusexp(h)

    def forward_hazard(self, x_cov_in, y, x_cat=[]):
        x_cov = self.covariate_net((x_cov_in, x_cat))
        h_xt = self.middle_net((x_cov, y))
        h_xt_forward = self.middle_net((x_cov, y + self.eps))
        h_t = self.net_t((y))
        h_t_forward = self.net_t((y+ self.eps))
        prod_x =self.bounding_op(self.prod_net_x((x_cov_in, x_cat)))
        h_xh_t = self.prod_net_t((y))
        h_xh_t_forward = self.prod_net_t((y+self.eps))
        h_x = self.net_x((x_cov_in, x_cat))
        h = h_xt + h_xh_t*prod_x + h_t + h_x

        if self.direct:
            h_forward = h_xt_forward + h_xh_t_forward*prod_x + h_t_forward + h_x
            hazard = (log1plusexp(h_forward) - log1plusexp(h)) / self.eps
        else:
            diff = h_xt_forward+h_t_forward-(h_xt + h_t) + prod_x*(h_xh_t_forward-h_xh_t)
            hazard = torch.sigmoid(h) * (diff / self.eps)
        return hazard

    def forward_S_eval(self, x_cov_in, y, x_cat=[]):

        if self.objective in ['hazard', 'hazard_mean']:
            S = torch.exp(-self.forward_cum_hazard(x_cov_in, y, [], x_cat))
            return S

        elif self.objective in ['S', 'S_mean']:
            x_cov = self.covariate_net((x_cov_in, x_cat))
            h_xt = self.middle_net((x_cov, y))
            h_xh_t = self.prod_net_t((y)) * self.bounding_op(self.prod_net_x((x_cov_in, x_cat)))
            h_t = self.net_t((y))
            h_x = self.net_x((x_cov_in, x_cat))
            h = h_xt + h_xh_t + h_t + h_x
            return 1 - h.sigmoid_()


class cox_net(torch.nn.Module):
    def __init__(self,
                 d_in_x,
                 cat_size_list,
                 d_in_y,
                 d_out,
                 layers_x,
                 layers,
                 dropout=0.9,
                 bounding_op=lambda x: x ** 2,
                 transformation=torch.tanh,
                 direct_dif=True,
                 objective='S_mean'
                 ):
        super(cox_net, self).__init__()

        self.bounding_op = bounding_op

        self.net_t = self.init_bounded_net(d_in_y,d_out,layers,transformation,bounding_op)
        self.net_x = self.init_covariate_net_2(d_in_x, layers_x,d_out, cat_size_list, transformation, dropout)


        self.eps = 1e-5
        self.direct = direct_dif
        self.objective = objective
        if self.objective in ['hazard', 'hazard_mean']:
            self.f = self.forward_hazard
            self.f_cum = self.forward_cum_hazard
        elif self.objective in ['S', 'S_mean']:
            self.f = self.forward_f
            self.f_cum = self.forward_S

    def init_covariate_net_2(self, d_in_x, layers_x,d_out, cat_size_list, transformation, dropout):
        module_list = [
            nn_node(d_in=d_in_x, d_out=layers_x[0], cat_size_list=cat_size_list, transformation=transformation,
                    dropout=dropout)]
        for l_i in range(1, len(layers_x)):
            module_list.append(
                nn_node(d_in=layers_x[l_i - 1], d_out=layers_x[l_i], cat_size_list=[], transformation=transformation,
                        dropout=dropout))
        module_list.append(
            nn_node(d_in=layers_x[ -1], d_out=d_out, cat_size_list=[], transformation=transformation,
                    dropout=dropout))
        return multi_input_Sequential_res_net(*module_list)

    def init_bounded_net(self, dx_in, d_out, layers, transformation, bounding_op):
        # self.mixed_layer = mixed_layer(d_in=dx_in, d_in_bounded=d_in_y, d_out=layers[0], transformation=transformation, bounding_op=bounding_op,dropout=dropout)
        module_list = [bounded_nn_layer(d_in=dx_in, d_out=layers[0], bounding_op=bounding_op,
                                   transformation=transformation)]
        for l_i in range(1, len(layers)):
            module_list.append(bounded_nn_layer(d_in=layers[l_i - 1], d_out=layers[l_i], bounding_op=bounding_op,
                                                transformation=transformation))
        module_list.append(
            bounded_nn_layer(d_in=layers[-1], d_out=d_out, bounding_op=bounding_op, transformation=lambda x: x))
        return multi_input_Sequential_res_net(*module_list)

    def forward(self, x_cov, y, x_cat=[]):
        return self.f(x_cov, y, x_cat)

    def forward_cum(self, x_cov, y, mask, x_cat=[]):
        return self.f_cum(x_cov, y, mask, x_cat)

    def forward_S(self, x_cov_in, y, mask, x_cat=[]):
        x_cov_in = x_cov_in[~mask, :]
        y = y[~mask, :]
        if not isinstance(x_cat, list):
            x_cat = x_cat[~mask, :]
        # Fix categorical business
        h_t = self.net_t((y))
        h_x = self.net_x((x_cov_in, x_cat))
        h = h_t + h_x
        return -log1plusexp(h)

    def forward_f(self, x_cov_in, y, x_cat=[]):
        h_t = self.net_t((y))
        h_t_forward = self.net_t((y+ self.eps))
        h_x = self.net_x((x_cov_in, x_cat))
        h = h_t + h_x
        F = h.sigmoid()

        if self.direct:
            F_forward = ( h_t_forward + h_x).sigmoid()
            f = ((F_forward - F) / self.eps)
        else:
            diff = (h_t_forward - h)
            f = (diff / self.eps) * F * (
                        1 - F)  # (F)*(1-F), F = h.sigmoid() log(sig(h)) + log(1-sig(h)) = h-2*log1plusexp(h)
        return f

    def forward_cum_hazard(self, x_cov_in, y, mask, x_cat=[]):
        h_t = self.net_t((y))
        h_x = self.net_x((x_cov_in, x_cat))
        h =  h_t + h_x

        return log1plusexp(h)

    def forward_hazard(self, x_cov_in, y, x_cat=[]):
        h_t = self.net_t((y))
        h_t_forward = self.net_t((y+ self.eps))
        h_x = self.net_x((x_cov_in, x_cat))
        h =h_x + h_t
        if self.direct:
            h_forward = h_t_forward+h_x
            hazard = (log1plusexp(h_forward) - log1plusexp(h)) / self.eps
        else:
            diff = (h_t_forward-h_t)
            hazard = torch.sigmoid(h) * (diff / self.eps)
        return hazard

    def forward_S_eval(self, x_cov_in, y, x_cat=[]):

        if self.objective in ['hazard', 'hazard_mean']:
            S = torch.exp(-self.forward_cum_hazard(x_cov_in, y, [], x_cat))
            return S

        elif self.objective in ['S', 'S_mean']:
            h_t = self.net_t((y))
            h_x = self.net_x((x_cov_in, x_cat))
            h = h_t*h_x
            return 1 - h.sigmoid_()

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

def log_objective_hazard_mean(cum_hazard,hazard):
    n = cum_hazard.shape[0]
    return -(  (hazard+1e-6).log().sum()-cum_hazard.sum() )/n













