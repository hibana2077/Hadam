import torch
import math
from torch.optim.optimizer import Optimizer

class HAdam(Optimizer):
    r"""
    Implements HAdam — Adam with k‑th raw moment of gradients.

    Arguments:
        params (iterable): model parameters.
        lr (float, optional): learning rate (α). Default: 1e-3
        betas (Tuple[float, float], optional): coefficients (β₁, β₂). Default: (0.9, 0.999)
        eps (float, optional): term added to denominator for numerical stability. Default: 1e-8
        order (int, optional): k, must be >=2. Even numbers are recommended. Default: 2 (Adam)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, order=2):
        if order < 2:
            raise ValueError("order must be >= 2")
        defaults = dict(lr=lr, betas=betas, eps=eps, order=order)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            k = group["order"]
            lr = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)   # m_t
                    state["exp_moment"] = torch.zeros_like(p.data)  # V_t

                exp_avg, exp_moment = state["exp_avg"], state["exp_moment"]

                state["step"] += 1
                t = state["step"]

                # Update biased first moment estimate (m_t)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased k‑th raw moment estimate (V_t)
                exp_moment.mul_(beta2).add_(grad.abs().pow(k), alpha=1 - beta2)

                # Bias corrections
                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t

                # Compute update — note k‑th root in denominator
                denom = exp_moment.div(bias_c2).pow(1.0 / k).add_(eps)
                step_size = lr / bias_c1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
