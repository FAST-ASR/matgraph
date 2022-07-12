# SPDX-License-Identifier: MIT

import torch
from .fsm import pdfposteriors

class FSMLogMarginal(torch.autograd.Function):
    """Compute the log-marginal probabily of a sequence given a graph."""

    @staticmethod
    def forward(ctx, input, seqlengths, fsm) -> torch.Tensor:
        posts, logprob = pdfposteriors(fsm, input, seqlengths)
        ctx.save_for_backward(posts)
        return sum(logprob)

    @staticmethod
    def backward(ctx, f_grad):
        input_grad, = ctx.saved_tensors
        return torch.mul(input_grad, f_grad), None, None


class LFMMILoss(torch.nn.Module):
    """Lattice-free MMI loss function."""

    def __init__(self, denfsms, numfsms, den_scale=1.0):
        super().__init__()
        self.denfsms = denfsms
        self.numfsms = numfsms
        self.den_scale = den_scale

    def forward(self, input, seqlengths):
        num_llh = FSMLogMarginal.apply(input, seqlengths, self.numfsms)
        den_llh = FSMLogMarginal.apply(input, seqlengths, self.denfsms)
        return -(num_llh - self.den_scale * den_llh)

