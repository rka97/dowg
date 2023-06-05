import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

from torch.optim import Optimizer

class DoWG(Optimizer):
    """Implements DoWG optimization algorithm.
    
    Args:
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4). Also used as the default squared distance estimate.
    """

    def __init__(self, params, eps=1e-4):
        defaults = dict(eps=eps)
        self.eps = eps
        super(DoWG, self).__init__(params, defaults)

    def step(self):
        """Performs a single optimization step."""
        state = self.state

        with torch.no_grad():
            device = self.param_groups[0]['params'][0].device

            # Initialize state variables if needed
            if 'rt2' not in state:
                state['rt2'] = torch.Tensor([self.eps]).to(device)
            if 'vt' not in state:
                state['vt'] = torch.Tensor([0]).to(device)

            grad_sq_norm = torch.Tensor([0]).to(device)
            curr_d2 = torch.Tensor([0]).to(device)
            
            for idx, group in enumerate(self.param_groups):
                group_state = state[idx]
                if 'x0' not in group_state:
                    group_state['x0'] = [torch.clone(p) for p in group["params"]]
                    
                grad_sq_norm += torch.stack([(p.grad ** 2).sum() for p in group["params"]]).sum()
                curr_d2 += torch.stack([((p - p0) ** 2).sum() for p, p0 in zip(group["params"], group_state['x0'])]).sum()
            
            state['rt2'] = torch.max(state['rt2'], curr_d2)
            state['vt'] += (state['rt2'] * grad_sq_norm)
            rt2, vt = state['rt2'], state['vt']
            
            for group in self.param_groups:
                for p in group['params']:
                    gt_hat = rt2 * p.grad.data
                    denom = torch.sqrt(vt).add_(group['eps'])
                    p.data.addcdiv_(gt_hat, denom, value=-1.0)
        return None

class CDoWG(Optimizer):
    """Implements CDoWG-- a coordinate-wise version of DoWG.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
    """

    def __init__(self, params, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(CDoWG, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            with torch.no_grad():
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    state = self.state[p]

                    # Initialize state variables
                    if 'x0' not in state:
                        state['x0'] = torch.clone(p).detach()
                    if 'rt2' not in state:
                        state['rt2'] = torch.zeros_like(p.data).add_(1e-4)
                    if 'vt' not in state:
                        state['vt'] = torch.zeros_like(p.data)

                    state['rt2'] = torch.max(state['rt2'], (p - state['x0']) ** 2)
                    rt2, vt = state['rt2'], state['vt']
                    vt.add_(rt2 * grad ** 2)
                    gt_hat = rt2 * grad
                    denom = vt.sqrt().add_(group['eps'])
                    p.data.addcdiv_(gt_hat, denom, value=-1.0)
        return loss
