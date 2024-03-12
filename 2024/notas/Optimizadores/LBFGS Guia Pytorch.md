https://johaupt.github.io/blog/pytorch_lbfgs.html

# Optimizing Neural Networks with LFBGS in PyTorch

_How to use LBFGS instead of stochastic gradient descent for neural network training instead in PyTorch_

## Why?

If you ever trained a zero hidden layer model for testing you may have seen that it typically performs worse than a linear (logistic) regression model. By wait? Aren’t these the same thing? Why would the zero hidden layer network be worse? Exactly. There is, of course, a good explanation and it is model estimation. We typically train neural networks using variants of stochastic gradient descent. We typically train regression models using optimization methods than are not stochastic and make use of second derivates.

If you have ever trained a one-hidden-layer network in scikit-learn, you might have seen that one option for the optimizer there is the same as for logistic regression: the [_Limited memory Broyden Fletcher Goldfarb Shanno_ algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS). Using the second order derivate to guide optimization should make convergence faster, although the time and memory requirement might make it infeasible for very deep networks and mini-batch training is not available in PyTorch out-of-the-box.

I work on tabular datasets where good convergence is a larger concern than deep architectures, so I’d be happy to get more stable convergence and regularize my network more to make sure I’m not overfitting. So let’s check out how to use LBFGS in PyTorch!

## Alright, how?

The PyTorch [documentation](https://pytorch.org/docs/stable/optim.html) says

> Some optimization algorithms such as Conjugate Gradient and LBFGS need to reevaluate the function multiple times, so you have to pass in a closure that allows them to recompute your model. The closure should clear the gradients, compute the loss, and return it.

It also provides an example:
```python
````
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)
    ```

Note how the function `closure()` contains the same steps we typically use before taking a step with SGD or Adam. In other words, if the optimizer needs the gradient once, like SGD or Adam, it's simple to calculate the gradient with `.backward()` and pass it to the optimizer. If the optimizer needs to calculate the gradient itself, like LBFGS, then we pass instead a function that wraps the steps we typically do once for others optimizers. 

Let's see how that looks in a simple feed forward network:


```python
import torch
from torch import nn
from torch.autograd import Variable

from torch.optim import Adam, LBFGS
from torch.utils.data import Dataset, DataLoader
```

```
temp = [1,2,3,4,5]
list(zip(temp, temp[:-1]))
```

```
[(1, 1), (2, 2), (3, 3), (4, 4)]
```

I define a somewhat flexible feed-forward network below. The action happens in method `train()`. We replace the gradient calculation with the `closure` function that does the same thing, plus two checks suggested [here](http://sagecal.sourceforge.net/pytorch/index.html) in case `closure` is called only to calculate the loss.


```python
class NNet(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, loss, sigmoid=False):
        super().__init__()
        

        self.input_dim = input_dim
        self.layer_sizes = hidden_layer_sizes
        self.iter = 0
        # The loss function could be MSE or BCELoss depending on the problem
        self.lossFct = loss

        # We leave the optimizer empty for now to assign flexibly
        self.optim = None

        
        hidden_layer_sizes = [input_dim] + hidden_layer_sizes
        last_layer = nn.Linear(hidden_layer_sizes[-1], 1)
        self.layers =\
            [nn.Sequential(nn.Linear(input_, output_), nn.ReLU())
             for input_, output_ in 
             zip(hidden_layer_sizes, hidden_layer_sizes[1:])] +\
            [last_layer]
        
        # The output activation depends on the problem
        if sigmoid:
            self.layers = self.layers + [nn.Sigmoid()]
            
        self.layers = nn.Sequential(*self.layers)

        
    def forward(self, x):
        x = self.layers(x)
        return x
    
    def train(self, data_loader, epochs, validation_data=None):

        for epoch in range(epochs):
            running_loss = self._train_iteration(data_loader)
            val_loss = None
            if validation_data is not None:
                y_hat = self(validation_data['X'])
                val_loss = self.lossFct(input=y_hat, target=validation_data['y']).detach().numpy()
                print('[%d] loss: %.3f | validation loss: %.3f' %
                  (epoch + 1, running_loss, val_loss))
            else:
                print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss))
            
            
                
    def _train_iteration(self,data_loader):
        running_loss = 0.0
        for i, (X,y) in enumerate(data_loader):
            
            X = X.float()
            y = y.unsqueeze(1).float()
            
            X_ = Variable(X, requires_grad=True)
            y_ = Variable(y)
              
            ### Comment out the typical gradient calculation
#             pred = self(X)
#             loss = self.lossFct(pred, y)
            
#             self.optim.zero_grad()
#             loss.backward()
            
            ### Add the closure function to calculate the gradient.
            def closure():
                if torch.is_grad_enabled():
                    self.optim.zero_grad()
                output = self(X_)
                loss = self.lossFct(output, y_)
                if loss.requires_grad:
                    loss.backward()
                return loss
            
            self.optim.step(closure)
            
            # calculate the loss again for monitoring
            output = self(X_)
            loss = closure()
            running_loss += loss.item()
               
        return running_loss
    
    # I like to include a sklearn like predict method for convenience
    def predict(self, X):
        X = torch.Tensor(X)
```

```python
class ExperimentData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx]
```

### Experiment Setup

Let’s test the whole thing on some simulated data from `sklearn`’s `make_classification`.

```
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
```

```
X,y = make_classification(n_samples=20000, n_features=100, n_informative=80, n_redundant=0, n_clusters_per_class=20, class_sep=1, random_state=123)

X, X_val, y, y_val = train_test_split(X, y, test_size=0.5, random_state=123)

# Don't forget to prepare the data for the DataLoader
data = ExperimentData(X,y)
```

```
INPUT_SIZE = X.shape[1]
EPOCHS=5 # Few epochs to avoid overfitting
```

## First test: Zero Layer Networks

We’ll check first if we can reach the AUC of the logistic regression model trained with sklearn and its (almost deprecated) _liblinear_ optimizer.
```
HIDDEN_LAYER_SIZE = []
```
### Adam

Let’s try Adam as an optimizer first. We would use that with a mini-batch and I use the default parameters.

```python
data_loader = DataLoader(data, batch_size=128)

net = NNet(INPUT_SIZE, HIDDEN_LAYER_SIZE, loss = nn.BCELoss(), sigmoid=True)

net.optim = Adam(net.parameters())


net.train(data_loader, EPOCHS, validation_data = {"X":torch.Tensor(X_val), "y":torch.Tensor(y_val).unsqueeze(1)})

```

```
pred_val["adam"] = net.predict(X_val)
```

### LFBGS

LFBGS is next. The implementation in PyTorch doesn’t work for mini-batches, so we’ll input the full dataset at the same time. Better hope your dataset is reasonably sized!
```
data_loader = DataLoader(data, batch_size=X.shape[0])
```

```
net = NNet(INPUT_SIZE, HIDDEN_LAYER_SIZE, loss = nn.BCELoss(), sigmoid=True)
```

```
net.optim = LBFGS(net.parameters(), history_size=10, max_iter=4)
```

A batch version of LBFGS is available at [https://github.com/nlesc-dirac/pytorch/blob/master/torch/optim/lbfgs.py](https://github.com/nlesc-dirac/pytorch/blob/master/torch/optim/lbfgs.py)

