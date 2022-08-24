# SPDX-License-Identifier: MIT

import torch
from .fsm import pdfposteriors, BatchFSM, FSM

class FSMLogMarginal(torch.autograd.Function):
    """Compute the log-marginal probabily of a sequence given a graph."""

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, seqlengths: torch.Tensor, fsm: BatchFSM
    ) -> torch.Tensor:
        """
        Args:
          input: Sequences of PDF log-likelihoods (with shape B x T x C)
          seqlengths: Array with length of each sequence (with shape B)

        Returns:
          logprob: Total probability of the sequence
        """
        posts, logprob = pdfposteriors(fsm, input, seqlengths.detach().numpy())
        ctx.save_for_backward(posts)
        return logprob  # shape B

    @staticmethod
    def backward(ctx, f_grad):
        input_grad, = ctx.saved_tensors
        f_grad = f_grad.unsqueeze(1).unsqueeze(2)  # shape Bx1x1
        return f_grad * input_grad, None, None  # broadcast mulitply


class LFMMILoss(torch.nn.Module):
    """Lattice-free MMI loss function."""

    def __init__(self, denfsm: FSM, den_scale=1.0, do_avg=False):
        super().__init__()
        self.denfsm = denfsm
        self.den_scale = den_scale
        self.do_avg = do_avg

    def forward(self, input: torch.Tensor, seqlengths: torch.Tensor, numfsms: BatchFSM):
        """
        Args:
          input: pdf log-likelihoods of shape B x T x C
          seqlengths: 1D tensor of sequnce lengths of shape B
          numfsm: numerator FSMs

        Returns:
          - LF-MMI loss (scalar)
        """
        denfsms = BatchFSM.from_list([self.denfsm for _ in range(input.size(0))])
        num_llh = FSMLogMarginal.apply(input, seqlengths, numfsms)
        den_llh = FSMLogMarginal.apply(input, seqlengths, denfsms)

        loss = -(num_llh - self.den_scale * den_llh)

        if self.do_avg:
            return loss.sum() / seqlengths.sum()
        return loss.sum()
