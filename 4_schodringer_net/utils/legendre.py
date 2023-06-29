import torch

# genera polinomios de legendre usando recurrencia
def Pl(n, x):
    if n == 0:
        return 0*x + 1 # esto hace que devuelva un tensor
    elif n == 1:
        return x
    else:
        return ((2*n + 1) * x * Pl(n-1, x) - n * Pl(n-2,x))/(n+1)