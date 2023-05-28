import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# =================================================================


# =================================================================

# ================== components of the Net ========================
class F_function_discrete(nn.Module):
    """
    Activation function for fitting, works by saving points in a parametrized array
    """
    def __init__(self, N=256, dt=5e-2, force_type='velocity'):
        super(F_function_discrete, self).__init__()
        self.N = N
        self.dt = dt
        self.force = nn.Parameter(torch.rand(N+1), requires_grad=True)
        self.force_type = force_type
        
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

        if self.force_type == 'velocity':
            y = v
        elif self.force_type == 'position':
            y = x
        else:
            print("Force type values are 'position' or 'velocity'")
        

        floor_y = torch.floor(y).long()
        ceil_y = (floor_y + 1).clamp(max=self.N).long() #adresses the overflow problem
        alpha = y - floor_y

        # print(f"max: {max(ceil_v.int())}")
        return torch.stack([x, v + self.dt* ( (1 - alpha) * self.force[floor_y] + alpha * self.force[ceil_y] ) ] ).T


class F_percentage_function(nn.Module):
    """
    Activation function for fitting, works by saving points in a parametrized array
    its domain is constructed with lower_limit and upper_limit
    then the ammount of points it contains is decided with N

    this is the advanced version of F_function

    force_type= "position" or "velocity"
    """

    def __init__(self, N, lower_limit, upper_limit, dt, force_type="position"):
        super(F_percentage_function, self).__init__()
        self.force = nn.Parameter(torch.rand(N), requires_grad=True)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.N = N
        self.dt = dt
        self.force_type = force_type
        
    def forward(self, X):
        """
        does an interpolation based on percentage, 
        - whats the percentage of the data point
        - where is the closest force point to this percentage
        - then returns [x, v + dt * selected_force]
        """

        # extracting values
        if len(list(X.shape)) == 1: # one data point case
            x, v = X[0], X[1]
        else:
            x, v = X[:,0], X[:,1]

        # percentage for selecting the force point 
        if self.force_type == 'position':
            y = x
        elif self.force_type == 'velocity':
            y = v
        else:
            print("force_type should be= 'position' or 'velocity'")


        y_percentage = (y - self.lower_limit)/(self.upper_limit - self.lower_limit) * 100
        # Select the component of self.force that is closest to the percentage value
        idx = torch.argmin(torch.abs(
            y_percentage.unsqueeze(1) - torch.arange(0, 100, 100/self.force.shape[0])
            ), dim=1)
        
        # Get the selected force value
        selected_force = self.force[idx]

        # Return the updated input
        return torch.stack([x, v + self.dt * selected_force]).T


    def plot_force(self):
        """
        plots the force points, giving not the index but the real meaning of inputs x
        """
        force_points = np.array(self.force.detach())
        x_points = np.linspace(self.lower_limit, self.upper_limit, self.N)
        
        plt.plot(x_points, force_points)
        if self.force_type == 'position':
            plt.xlabel("x")
        elif self.force_type == 'velocity':
            plt.xlabel("v")
        
        plt.ylabel("force")
        plt.show()

class W_matrix(nn.Module):
    """ we need a linear thats not trainable """
    def __init__(self, dt):
        super(W_matrix, self).__init__()
        self.weights = torch.Tensor([[1, dt], [0, 1]])
        self.bias = torch.Tensor([0, 0])
        
    def forward(self, x): 
        return F.linear(x, self.weights, self.bias)

def narrow_gaussian(x, mu=0, sigma=0.01):
    """ used as an approximation to dirac delta, it reacts close when x is close to mu """
    #pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * torch.exp(-(x - mu)**2 / (2 * sigma**2))
    # is not necessary to include other factors, when x=mu output will be 1
    pdf = torch.exp(-(x - mu)**2 / (2 * sigma**2))
    return pdf
    

class T_postprocessing(nn.Module):
    """
    X takes a tensor and compares its position coordinate
    to the xreal, if its inside a given distance, the function returns 0
    else it will return 1

    if the return is close to 0, it gives out the velocity as well
    """
    

    def __init__(self, xreal = 0, epsilon = 1e-3, debug=False):
        super(T_postprocessing, self).__init__()
        self.xreal = xreal
        self.epsilon = epsilon
        self.debug = debug
        
        
    def forward(self, X):
        if len(list(X.shape)) == 1: # one data point case
            x, v = X[0], X[1]
        else:
            x, v = X[:,0], X[:,1]

        # if x its close to xreal, this will return close to 0
        # else it returns close to 1
        activation_result = 1 - narrow_gaussian(x, mu=self.xreal, sigma=self.epsilon)
        
        if self.debug:
            print("on post processing, activation result was:")
            print(activation_result)

        # if K is not close to 0, its a failed forward
        if (activation_result >= self.epsilon): 
            infinity = 1e20
            v = torch.full((len(v),), float(infinity))

        return torch.stack([activation_result, v]).T


# =============== the Net ==================

class diffNet(nn.Module):
    def __init__(self, depth, lower_limit, upper_limit, dt, N=256,
        ftype='percentage',post_process=False, pp_mu=0, pp_epsilon=1e-3,
        debug=True, force_type="velocity"):
        """
        Parameters:

        Usage:
        """
        super(diffNet, self).__init__()
        

        if ftype == "percentage":
            self.activation_function = F_percentage_function(N, lower_limit, upper_limit, dt, force_type)
        elif ftype == "discrete":
            self.activation_function = F_function_discrete(dt= dt, N=N, force_type=force_type) 
        else:
            print("Use arguments ftype='percentage' or 'discrete'")


        layers = []

        w_mat = W_matrix(dt= dt) # defined one time, always the same

        layers.append(w_mat)
        
        for i in range(depth):
            layers.append( self.activation_function )
            layers.append( w_mat )
        
        if post_process:
            post_process_layer = T_postprocessing(pp_mu, pp_epsilon, debug=debug)
            layers.append(post_process_layer)


        self.layers = nn.Sequential(*layers)
        
    def forward(self, X):
        return self.layers(X)


    def plot_parameters(self):
        self.activation_function.plot_force()

        



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



if __name__ == '__main__':
    from dataset import MyDataset, DataLoader
    import torch
    import torch.nn as nn

    import pandas as pd
    import matplotlib.pyplot as plt

    my_dataset = MyDataset(pd.read_csv("./data/song_combined_df.csv", index_col=0), step_size=50)
    my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)

    # Loading the model
    model = torch.load("./model/040223.pt")

    plt.plot(list(model.parameters())[0].detach())
    plt.title("Loaded parameters of activation function")
    plt.show()

    # Hyperparameters
    smooth_rate = 0.03
    constrain_rate = 1.0 # F(v=0) = 0
    targets_rate = 1.05

    # Define number of epochs
    num_epochs = 50

    # Define loss function and optimizer
    L2_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.4)

    running_loss = 0
    loss_array = []

    #used for graphs
    max_i = len(my_dataloader)

    # Iterate through epochs
    for epoch in range(num_epochs):
        # Iterate through data in the DataLoader
        for i, data in enumerate(my_dataloader, 0):
            # Get inputs and targets from data
            inputs, targets = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model.forward(inputs)

            L2_computed_loss = L2_loss(outputs, targets)
            # Compute loss
            loss = smooth_rate * smooth_loss(list(model.parameters())[0]) +\
                constrain_rate*physics_constrain(list(model.parameters())[0]) + targets_rate * L2_computed_loss


            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print("L2_loss:", L2_computed_loss.detach().item())
                loss_array.append(L2_computed_loss.detach().item())
                print('[Epoch %d, Mini-batch %5d] Loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        
        if epoch % 100 == 10:
            # saving the model
            torch.save(model, f"./model/song_040223_epoch{epoch}.pt")
            
            #ploting
            plt.plot(list(model.parameters())[0].detach(), alpha = epoch/num_epochs)
            plt.show()
            plt.savefig(f"./evolution_force/song_epoch_{epoch}_loss_{running_loss :.2f}.png")


    # finish training
    torch.save(model, f"./model/song_040223_epoch{epoch}.pt")
    print('Finished training')

    plt.plot(list(model.parameters())[0].detach())
    plt.title("After training parameters of activation function")
    plt.show()

    # plot loss
    plt.plot(loss_array)
    plt.title("evolution of L2 loss output target")
    plt.show()