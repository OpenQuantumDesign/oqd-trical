import sys

from abc import ABC, abstractmethod

import torch
from torch import optim

########################################################################################


class OptimizerBase(ABC):
    @abstractmethod
    def optimize(self, parameters):
        pass


class TorchOptimizer(OptimizerBase):
    def __init__(
        self, optimizer=optim.Adam, max_steps=10000, optimizer_kwargs={"lr": 1e-3}
    ):
        super().__init__()

        self.optimizer = optimizer
        self.max_steps = max_steps
        self.optimizer_kwargs = optimizer_kwargs
        pass

    def optimize(self, loss, parameters):
        parameters = torch.nn.Parameter(parameters)
        torch.nn.init.normal_(parameters)

        optimizer = self.optimizer((parameters,), **self.optimizer_kwargs)

        for step in range(self.max_steps):
            optimizer.zero_grad()

            L = loss(parameters)
            L.backward()

            optimizer.step()

            print(
                "{:<150}".format(
                    "\r"
                    + "[{:<60}] ".format(
                        "="
                        * (
                            (
                                torch.floor(
                                    torch.tensor(step + 1) / self.max_steps * 60
                                )
                            ).to(torch.long)
                            - 1
                        )
                        + ">"
                        if step + 1 < self.max_steps
                        else "=" * 60
                    )
                    + "{:<40}".format(
                        "Steps {:d}/{:d}: Loss = {:.3f}".format(
                            step + 1, self.max_steps, L
                        )
                    )
                ),
                end="",
            )
            sys.stdout.flush()

        return parameters
