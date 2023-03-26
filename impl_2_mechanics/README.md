# Adapting a NN to a Differential Equation

Based on

‌Hashimoto, K., Sugishita, S., Tanaka, A., & Tomiya, A. (2018). Deep learning and holographic QCD. Physical Review D, 98(10). https://doi.org/10.1103/physrevd.98.106014

The steps are divided in 

- Data Generation: where goes `integrate_data.py`
-


# Data Generation
The way is worked on the paper, is the unknown differential equation as a unknown function that's fixed, it separates different cases, where this function only depends on one of the variables

- only dependent on position (q for general coordinate)
$$
\frac{dp}{dt} = f(q)
$$

- only dependent on speed (here for generality, I write momentum)
$$
\frac{dp}{dt} = f(p)
$$

Its important to remember:
- $\Delta t$ is fixed with the depth of the Neural Network, so on the generation of data, only the initial conditions are controlled
- The unknown function is represented by an array as big as the range of inputs

This function Force is implemmented as a class:

```pytorch
import torch
import torch.nn as nn

# Activation function for fitting
class F(nn.Module):
    """
    Activation function for fitting, works by saving points in a parametrized array
    """
    def __init__(self):
        super(F, self).__init__()
        self.force = nn.Parameter(torch.rand(256), requires_grad=True)
        
    def forward(self, X):
        """
        does an interpolation, in theory takes the input
        int i = v;
        and tries to do
        return F[v] 
        to find the corresponding value to that ´v´
        """

        x, v = X[0], X[1]

        floor_v = v.floor().long()
        ceil_v = (floor_v + 1).clamp(max=255)
        alpha = v - floor_v.float()
        return [x, (1 - alpha) * self.force[floor_v] + alpha * self.force[ceil_v]]
        
```

Optionally a force can be implemented like:

$$
F(\begin{matrix} x \\ v \end{matrix} ) = \left( \begin{matrix} x \\ f(v) \end{matrix}  \right)
$$

if the function was only one dimensional input (not the case, but useful for some people who read this) is possible to separate inside a forward method in pytorch:

```
x0, x1 = torch.unbind(x, dim=1)
x0 = x0 
x1 = self.force(x1)
x = torch.stack((x0, x1), dim=1)
return x
```

## Loss functions and smoothness
- adding physics constrain to the function
- adding continuity condition to make the function more real like





