import sys

from functools import cached_property

import torch
from torch import optim

########################################################################################


class Crystal:
    def __init__(self, potential, *, N=10, dimension=3):
        self.potential = potential
        self.dimension = dimension
        self.N = N
        pass

    @cached_property
    def equilibrium_position(self):
        x = torch.nn.Parameter(torch.empty(self.dimension * self.N))
        torch.nn.init.normal_(x)

        max_epochs = 10000
        lr = 1e-3
        optimizer = optim.AdamW([x], lr=lr)

        for epoch in range(max_epochs):
            optimizer.zero_grad()

            loss = self.potential(x)
            grad = torch.norm(self.potential.gradient(x)[0])
            loss.backward()

            optimizer.step()

            print(
                "{:<150}".format(
                    "\r"
                    + "[{:<60}] ".format(
                        "="
                        * (
                            (torch.floor(torch.tensor(epoch + 1) / max_epochs * 60)).to(
                                torch.long
                            )
                            - 1
                        )
                        + ">"
                        if epoch + 1 < max_epochs
                        else "=" * 60
                    )
                    + "{:<40}".format(
                        "Epoch {:d}/{:d}: Energy = {:.3f}, Gradient = {:.3f}".format(
                            epoch + 1, max_epochs, loss, grad
                        )
                    )
                ),
                end="",
            )
            sys.stdout.flush()
        return x.reshape(self.dimension, self.N)
