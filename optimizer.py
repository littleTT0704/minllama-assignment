from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0["
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0["
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                # State should be stored in this dictionary
                state = self.state[p]

                # Update first and second moments of the gradients
                t = state.get("t", 0) + 1
                m = (
                    group["betas"][0] * state.get("m", torch.zeros_like(p))
                    + (1 - group["betas"][0]) * grad
                )
                v = (
                    group["betas"][1] * state.get("v", torch.zeros_like(p))
                    + (1 - group["betas"][1]) * grad**2
                )
                state["t"] = t
                state["m"] = m
                state["v"] = v

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                if group["correct_bias"]:
                    alpha = (
                        group["lr"]
                        * (1 - group["betas"][1] ** t) ** 0.5
                        / (1 - group["betas"][0] ** t)
                    )
                else:
                    alpha = group["lr"]

                # Update parameters
                p.data -= alpha * m / (v**0.5 + group["eps"])

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                p.data -= group["lr"] * group["weight_decay"] * p.data

        return loss
