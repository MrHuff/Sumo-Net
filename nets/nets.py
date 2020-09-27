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
        self.W = torch.nn.Parameter(torch.randn(*(d_in_bounded,d_out)),requires_grad=True)
        self.f = transformation
        self.bounding_op = bounding_op
        self.bias = torch.nn.Parameter(torch.randn(d_out),requires_grad=True)
        self.w = torch.nn.Linear(d_in,d_out)

    def forward(self,X,x_bounded):
        return self.f(x_bounded@self.bounding_op(self.W)+self.bounding_op(self.bias)+self.w(X))











