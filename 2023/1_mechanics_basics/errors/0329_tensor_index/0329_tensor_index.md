I have this functions for a network

```
import torch
import torch.nn as nn
# =================================================================

N = 256 # size of the array of Function F

# =================================================================

# Activation function for fitting
class F_function(nn.Module):
    """
    Activation function for fitting, works by saving points in a parametrized array
    """
    def __init__(self):
        super(F_function, self).__init__()
        self.force = nn.Parameter(torch.rand(N+1), requires_grad=True)
        
    def forward(self, X):
        """
        does an interpolation, in theory takes the input
        int i = v;
        and tries to do
        return F[v] 
        to find the corresponding value to that ´v´
        """

        x, v = X[0], X[1]

        floor_v = torch.floor(v)
        ceil_v = (floor_v + 1).clamp(max=N) #adresses the overflow problem
        alpha = v - floor_v

        #return [x, (1 - alpha) * self.force[floor_v] + alpha * self.force[ceil_v]]
        return torch.tensor([x, (1 - alpha) * self.force[floor_v.int()] + alpha * self.force[ceil_v.int()]])
    

import torch.nn.functional as F

class W_matrix(nn.Module):
    """ we need a linear thats not trainable """
    def __init__(self, dt):
        super(W_matrix, self).__init__()
        self.weights = torch.Tensor([[1, dt], [0, 1]])
        self.bias = torch.Tensor([0, 0])
        
    def forward(self, x): 
        return F.linear(x, self.weights, self.bias)
```

# the implementation
```
class diffNet(nn.Module):
    def __init__(self, depth):
        super(diffNet, self).__init__()
        layers = []
        layers.append(nn.Linear(2, 2))
        for i in range(depth):
            layers.append( F_function() )
            layers.append( W_matrix(dt=1e-02) )
        self.layers = nn.Sequential(*layers)
        
    def forward(self, X):
        return self.layers(X)
```

# then using data set
```
import pandas as pd
df = pd.read_csv('.\data\songforce_v100\songforce.csv')
df.head()

data_worked = pd.DataFrame()
inicial = []
final = []

for index in range(10,len(df)-1):
    final.append(df.velocity[index])
    inicial.append(df.velocity[index - 10])
```

# print(inicial)
```
data_worked['inicial'] = inicial
data_worked['final'] = final
```

# separate 
```
Ndata = len(data_worked)
datatraining = data_worked[:202]
datavalidation = data_worked[202:250]
datatesting = data_worked[250:-1]
```

# where the shape of the data X 
```
X = torch.tensor( [ [0 for i in range(len(datatraining))], datatraining.inicial  ] ,  dtype=torch.float32).T
Y = torch.tensor( [ [0 for i in range(len(datatraining))], datatraining.final  ] ,  dtype=torch.float32)

X.size()

#torch.Size([202, 2])

model = diffNet(depth=10)

ypred = model.forward(X)
```

# But I get the error when trying to use a tensor as an index
IndexError                                Traceback (most recent call last)
c:\msys64\home\Code\ML-physics\impl_2_mechanics\2_fitting_a_F_of_v.ipynb Cell 20 in 
      1 model = diffNet(depth=10)
----> 3 ypred = model.forward(X)

c:\msys64\home\Code\ML-physics\impl_2_mechanics\2_fitting_a_F_of_v.ipynb Cell 20 in diffNet.forward(self, X)
     11 def forward(self, X):
---> 12     return self.layers(X)

File c:\Users\fbien.DESKTOP-6FMEAR7\.conda\envs\deepl\lib\site-packages\torch\nn\modules\module.py:1501, in Module._call_impl(self, *args, **kwargs)
   1496 # If we don't have any hooks, we want to skip the rest of the logic in
   1497 # this function, and just call forward.
   1498 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1499         or _global_backward_pre_hooks or _global_backward_hooks
   1500         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1501     return forward_call(*args, **kwargs)
   1502 # Do not call functions when jit is used
   1503 full_backward_hooks, non_full_backward_hooks = [], []

File c:\Users\fbien.DESKTOP-6FMEAR7\.conda\envs\deepl\lib\site-packages\torch\nn\modules\container.py:217, in Sequential.forward(self, input)
    215 def forward(self, input):
    216     for module in self:
--> 217         input = module(input)
    218     return input
...
     31 alpha = v - floor_v
     33 #return [x, (1 - alpha) * self.force[floor_v] + alpha * self.force[ceil_v]]
---> 34 return torch.tensor([x, (1 - alpha) * self.force[floor_v] + alpha * self.force[ceil_v]])

IndexError: tensors used as indices must be long, int, byte or bool tensors


torch.Size([202, 2])

____

# Problem was
Having tensor of tensor, there were problems

# Fixed by using
conversion to ints
```
self.force[floor_v] ->  self.force[floor_v.int()]
```

and applying torch stack for combining two tensor without redefining the new tensor
```
torch.stack([x, (1 - alpha) * self.force[floor_v] + alpha * self.force[ceil_v]])
```
Then I applied the transpose to have the correct shape



Example of code
```
        if len(X) == 1:
            x, v = X[0], X[1]
        else:
            x, v = X[:,0], X[:,1]

        floor_v = torch.floor(v)
        ceil_v = (floor_v + 1).clamp(max=N) #adresses the overflow problem
        alpha = v - floor_v


        return torch.stack([x, (1 - alpha) * self.force[floor_v.int()] + alpha * self.force[ceil_v.int()]]).T
```