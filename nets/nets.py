import torch

class nn_node(torch.nn.Module):
    def __init__(self,d_in,d_out,transformation=torch.tanh):
        super(nn_node, self).__init__()
        self.w = torch.nn.Linear(d_in,d_out)
        self.f = transformation

    def forward(self,X):
        return self.f(self.w(X))

class bounded_nn_node(torch.nn.Module):
    def __init__(self,d_in,d_out,bounding_op=lambda x: x**2,transformation=torch.tanh):
        super(bounded_nn_node, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(*(d_in,d_out)),requires_grad=True)
        self.f = transformation
        self.bounding_op = bounding_op
        self.bias = torch.nn.Parameter(torch.randn(d_out),requires_grad=True)

    def forward(self,X):
        return self.f(X@self.bounding_op(self.W)+self.bounding_op(self.bias))

class mixed_node(torch.nn.Module):
    def __init__(self,d_in,d_in_bounded,d_out,bounding_op=lambda x: x**2,transformation=torch.tanh):
        super(mixed_node, self).__init__()
        self.pos_weights = torch.nn.Parameter(torch.randn(*(d_in_bounded, d_out)), requires_grad=True)
        self.f = transformation
        self.bounding_op = bounding_op
        self.pos_bias = torch.nn.Parameter(torch.randn(d_out), requires_grad=True)
        self.w = torch.nn.Linear(d_in,d_out)

    def forward(self,X,x_bounded):
        return self.f(x_bounded @ self.bounding_op(self.pos_weights) + self.bounding_op(self.pos_bias) + self.w(X))

class survival_net(torch.nn.Module):
    def __init__(self,
                 d_in_x,
                 d_in_y,
                 d_out,
                 layers_x,
                 layers,
                 bounding_op=lambda x: x**2,
                 transformation=torch.tanh
                 ):
        super(survival_net, self).__init__()
        self.init_covariate_net(d_in_x,layers_x,transformation)
        self.init_middle_net(dx_in=layers_x[-1],d_in_y=d_in_y,d_out=d_out,layers=layers,transformation=transformation,bounding_op=bounding_op)
        self.eps = 1e-6
    def init_covariate_net(self,d_in_x,layers_x,transformation):
        module_list = [nn_node(d_in=d_in_x,d_out=layers_x[0],transformation=transformation)]
        for l_i in range(1,len(layers_x)):
            module_list.append(nn_node(d_in=layers_x[l_i-1],d_out=layers_x[l_i],transformation=transformation))
        self.covariate_net = torch.nn.Sequential(*module_list)

    def init_middle_net(self,dx_in,d_in_y,d_out,layers,transformation,bounding_op):
        self.mixed_node = mixed_node(d_in=dx_in,d_in_bounded=d_in_y, d_out=layers[0], transformation=transformation,bounding_op=bounding_op)
        module_list = []
        for l_i in range(1,len(layers)):
            module_list.append(bounded_nn_node(d_in=layers[l_i-1],d_out=layers[l_i],transformation=transformation,bounding_op=bounding_op))
        module_list.append(bounded_nn_node(d_in=layers[-1],d_out=d_out,transformation=lambda x:x,bounding_op=bounding_op))
        self.middle_net = torch.nn.Sequential(*module_list)

    def forward(self,x_cov,y):
        x_cov = self.covariate_net(x_cov)
        h = self.middle_net(self.mixed_node(x_cov,y))
        h_forward = self.middle_net(self.mixed_node(x_cov,y+self.eps))

        F = h.sigmoid()
        f = ((h_forward-h)/self.eps) *F*(1-F) #this is fucked figure something smart out...
        return f , 1-F

def log_objective(S,f,delta):
    return (delta*f+(1-delta)*S).sum()















