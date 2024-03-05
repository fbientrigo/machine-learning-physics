# Training Data Generation (subroutines.py)
Using a dataset (H,M) it is generated a polynomial fit `f`:

### def `accept_test_middle(H, M, noise_std=0.01, f=None)`
    checks if a point (H,M) lies close enough to a function, accounting for noise
    if is close enought in a 2 dimensional distance;

    H being the horizontal axis, is asking if its image f(H) is close enough to M
    |f(H) - M| <= |noise|
    
    assuming normal distribution noise, f is commonly given by a polynomial fit in data

### def `generate_training_data(noise_std=0.004, data_size=10000, Hrange=[0, 0.022], Mrange=[0, 0.11], f=None)`
    Generate data of 2 classes, those that are close enough to the function accounting for noise
    and those that aren't. So the training becomes a supervised learning problem of many points in a grid.



### class `Phi_Pi_DataSet(DataSet)`
    This code defines a custom dataset class called Phi_Pi_DataSet which inherits from the PyTorch Dataset class. The purpose of this class is to handle data for a machine learning model.

    The __init__ method initializes the dataset by concatenating positive and negative examples of three different arrays (phi, Pi, and ans) and saving them as class attributes. These arrays contain float values. An optional transform parameter is also accepted, which can be used to apply some transformation to the data.

    The __getitem__ method is called when an item is retrieved from the dataset. It retrieves the corresponding values for phi, Pi, and ans for the given index. If a transform function was provided, it is applied to these values before they are returned.

    The __len__ method returns the length of the dataset.