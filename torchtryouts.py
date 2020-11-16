import numpy as np
import torch

a = np.array([1., 2., 3.])
print('1',a >1.4)

a = torch.tensor(a)
print('2',a > 1.3)

b = torch.tensor([1.3], dtype=torch.float64)
c = torch.tensor(np.array([1.3]))
print(b)
print(c)
print('3',type(b))
print('4',type(c))

print('5',torch.max(a, b))

a[ a> 1.4] = 0
print('6',a)

print('7',torch.exp(torch.tensor([3])))