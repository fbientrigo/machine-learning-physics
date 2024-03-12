# Generative way
The problem is of labeled data way, as we have the infinite eigenvalues possible inside of a function $\Lambda(z)$ where $z\in [0,1]$. This in the forward problems allows to create and populate a bigger data set
$$
V \rightarrow \hat \Lambda
$$
the inverse problem relates to generate the potential starting from the generating function, for this we can be inspired on the generative deep learning models of the last years, where we have a "label", a generating function in our case, and the target.

A diffusion model is that we have an image, we include noise and we train the model to clean the noise step by step,
in our case, having the forward model it that can help us to calculate the error of our cleaned image.

### The diffusion process
Diffusion models a