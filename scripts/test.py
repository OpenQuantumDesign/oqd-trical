from rich import print as pprint

import torch

########################################################################################

N = 2
x = torch.ones((3 * N))
x.requires_grad_(True)


def f(x):
    x = x.reshape(3, N)
    return (
        0.5 * (2 * torch.pi * torch.tensor([5, 5, 2]))[:, None] ** 2 * x**2 * 2
    ).sum()


G = torch.autograd.grad(f(x), x)
H = torch.autograd.functional.hessian(f, x)

pprint(G)
pprint(H)
