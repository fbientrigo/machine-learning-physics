import torch
from torch.autograd import Function, Variable

# Funciones auxiliares para los polinomios de Legendre
def Pl(n, x):
    if n == 0:
        return 0 * x + 1
    elif n == 1:
        return x
    else:
        n = n - 1
        p1 = (2 * n + 1) * x * Pl(n, x)
        p2 = n * Pl(n - 1, x)
        return (p1 - p2) / (n + 1)

def dxPl(n, x):
    if n == 0:
        return 0 * x + 0
    elif n == 1:
        return 0 * x + 1
    else:
        n = n - 1
        return (n + 1) * Pl(n, x) + x * dxPl(n, x)

class PLegendre(Function):
    """Legendre Polynomial function for one legendre polynomial"""
    @staticmethod
    def forward(ctx, n, input):
        ctx.save_for_backward(input)
        ctx.n = n
        output = Pl(n, input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        n = ctx.n
        grad_output_var = torch.autograd.Variable(grad_output, requires_grad=False)
        input_var = torch.autograd.Variable(input, requires_grad=True)
        dxpl = dxPl(n, input_var)
        return None, grad_output_var * dxpl


class LegendrePolynomial():
    """ Generate a sum of legendre polynomials that work with torch grad
    coefx = torch.tensor([np.random.randint(10) for i in range(9)],dtype=torch.float64, requires_grad=True)
    fx = LegendrePolynomial(coefx)
    """
    def __init__(self, coef, domain=None):
        self.coef = coef
        self.domain = domain

    def __call__(self, x):
        result = 0
        for idx,C in enumerate(self.coef):
            result += C * PLegendre.apply(idx, x)
        return result

        

