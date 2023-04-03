import torch
import torch.nn as nn
import torch.nn.functional as F

# =================================================================

N = 256 # size of the array of Function F
dt = 1e-3


# =================================================================

# ================== components of the Net ========================
class F_function(nn.Module):
    """
    Activation function for fitting, works by saving points in a parametrized array
    """
    def __init__(self, dt):
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

        if len(X) == 1: # one data point case
            x, v = X[0], X[1]
        else:
            x, v = X[:,0], X[:,1]

        floor_v = torch.floor(v)
        ceil_v = (floor_v + 1).clamp(max=N) #adresses the overflow problem
        alpha = v - floor_v

        # print(f"max: {max(ceil_v.int())}")
        return torch.stack([x, v + dt* ( (1 - alpha) * self.force[floor_v.int()] + alpha * self.force[ceil_v.int()] ) ] ).T


class W_matrix(nn.Module):
    """ we need a linear thats not trainable """
    def __init__(self, dt):
        super(W_matrix, self).__init__()
        self.weights = torch.Tensor([[1, dt], [0, 1]])
        self.bias = torch.Tensor([0, 0])
        
    def forward(self, x): 
        return F.linear(x, self.weights, self.bias)



# =============== the Net ==================

class diffNet(nn.Module):
    def __init__(self, depth, debug=True):
        super(diffNet, self).__init__()
        
        w_mat = W_matrix(dt= dt) # defined one time, always the same
        f_function = F_function(dt= dt) #defined one time, to only have N parameters

        layers = []
        layers.append(w_mat)
        
        for i in range(depth):
            layers.append( f_function )
            layers.append( w_mat )
        self.layers = nn.Sequential(*layers)
        
    def forward(self, X):
        return self.layers(X)


def plot_f(force_parameters):
    # graphing the array inside of this
    force_values = force_parameters.detach().numpy()
    plt.plot(force_values)
    plt.show()



# ============= Loss function ==============

def smooth_loss(force_params):
    """
    L2 error function for the smoothness of points
    """
    summatory = 0.0
    for i in range(len(force_params)-1):
        summatory += ( force_params[i+1] - force_params[i] )**2
    return summatory

def physics_constrain(force_params):
    """
    The force for friction should be 0 when the input is 0
    This constrain can be changed to other
    """

    return (force_params[0])**2

