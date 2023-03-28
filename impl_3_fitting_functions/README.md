# Papers & Ideas


On the use of a Hypergeometric Activation Function:
- Vieira, N. (2023). Bicomplex Neural Networks with Hypergeometric Activation Functions. Advances in Applied Clifford Algebras, 33(2). https://doi.org/10.1007/s00006-023-01268-w

‌
On the discovery of parametric activation functions:
- Bingham, G., & Miikkulainen, R. (2020). Discovering Parametric Activation Functions. ArXiv.org. https://arxiv.org/abs/2006.03179v4


Symtorch on process:
- clayton-r. (2023). clayton-r/symtorch: Initial Commit. GitHub. https://github.com/clayton-r/symtorch
- ‌John. (2021, August 8). Turning SymPy expressions into PyTorch modules. Python Awesome; Python Awesome. https://pythonawesome.com/turning-sympy-expressions-into-pytorch-modules/

‌Hyperparameter Tunning
- Stewart, M. (2019, July 9). Simple Guide to Hyperparameter Tuning in Neural Networks. Medium; Towards Data Science. https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594


Avoid overfitting:
- Brownlee, J. (2018, December 16). How to Avoid Overfitting in Deep Learning Neural Networks - MachineLearningMastery.com. MachineLearningMastery.com. https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization-error/

‌
‌






# Fitting functions

## Fitting by Random Array Points
Previous research have used an array to store the function image, where $x$ is the pre image, then we search throught the array using the integer value of $x$, where it has the need to be inside of the limits of the function's array lenght:

let $\Omega$ be the entire span of values that this approach accepts, being from 0 (index 0 and value of $x=0$) to $x=$`len(f_array)`

where $f_\text{array}(x) = f[x]$, assuming $x \in \Omega$, and for interpolation is defined $\alpha = x - \lfloor x \rfloor$


$$
f(x) = \begin{cases} 
f[x] \; \space ; \forall x \in \mathbb Z  \\

(1 - \alpha ) f[ \lfloor x \rfloor ] + \alpha f[ \lceil x \rceil ]
\; \space ; \forall x \notin \mathbb Z 
\end{cases}
$$

## Using parametrizable functions
A general approach to functions that allows parameters to controll its shape, are hypergeometric, that assume a base shape and have clear conditions of convergence
$$
{}_pF_q ((a_i)_{1:p}; (b_j)_{1:q} ; z ) =  \sum_n\frac{\Pi_{i=1}^p (a_i)_n}{\Pi_{i=1}^q (b_j)_n} \frac{z^n}{n!}
$$

by example some simple hypergeometric functions:

$$
{}_1F_0 (a; ; z) = \sum_{n=0}^\infty (a)_n \frac{z^n}{n!} = \frac{1}{(1-z)^a}
$$

more can be found on: 

DLMF: Chapter 15 Hypergeometric Function. (2023). Nist.gov. https://dlmf.nist.gov/15

‌So the approach on training is using the function data to find its fit to a hypergeometric; I say fitting because we will have no knowledge on the original function to be able to do this process.

There are two approachs:
- fitting by random array points (each neuron contains an activation function of many terms of the shape of a hypergeometric, where its components are trainable)
- a deep neural network that learns to map function inputs to hypergeometric parameters