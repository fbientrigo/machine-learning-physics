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

        floor_v = torch.floor(v).long()
        ceil_v = (floor_v + 1).clamp(max=N).long() #adresses the overflow problem
        alpha = v - floor_v

        # print(f"max: {max(ceil_v.int())}")
        return torch.stack([x, v + dt* ( (1 - alpha) * self.force[floor_v] + alpha * self.force[ceil_v] ) ] ).T


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



if __name__ == '__main__':
    from dataset import MyDataset, DataLoader
    import torch
    import torch.nn as nn

    import pandas as pd
    import glob
    import matplotlib.pyplot as plt

    my_dataset = MyDataset(pd.read_csv("./data/song_combined_df.csv", index_col=0))
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