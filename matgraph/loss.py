# SPDX-License-Identifier: MIT

import torch

class FSMLogMarginal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, lengths, fsms):
        print("forward pass")
        #ttl, grad = forward_backward(...)
        ctx.save_for_backward(grad)
        return ttl

    @staticmethod
    def backward(ctx, out_grad):
        grad, = ctx.saved_tensors
        return torch.mul(grad, out_grad), None, None

class LFMMILoss(torch.nn.Module):

    def __init__(self, den_fsm):
        super().__init__()
        self.den_fsm = den_fsm

    def forward(self, x, x_lengths, num_fsms):
        #den_graphs =
        pass

