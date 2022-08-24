import torch

from .fsm import BatchFSM, FSM
from .loss import FSMLogMarginal

from itertools import permutations
from typing import List


def multi_loss(f, llhs, seqlengths, numfsms):
    return torch.sum(
        torch.stack(
            [
                f(llh, seq, fsm)  # f returns B size vector
                for llh, seq, fsm in zip(llhs, seqlengths, numfsms)
            ],
            dim=1,
        ),  # B x S
        dim=1,  # sum accross speakers
    )  # B


class PIT_LFMMILoss(torch.nn.Module):
    """Permutation invariant training wrapper for MMI loss."""

    def __init__(self, denfsm: FSM, den_scale=1.0, do_avg=False):
        super().__init__()
        self.denfsm = denfsm
        self.den_scale = den_scale
        self.do_avg = do_avg

    def forward(
        self, est_llhs: torch.Tensor, seqlengths: torch.Tensor, numfsms: List[BatchFSM]
    ):
        """
        Args:
          est_llhs: estimated pdf log-likelihoods for each speaker (shape B x S x T x C)
          seqlengths: sequence lengths of shape for each speaker (shape B x S)
          numfsms: numerator FSMs for each speaker (shape S x B)

        Returns:
          -  PIT LF-MMI loss (scalar)
        """
        n_spkrs = len(numfsms)
        batch_size = est_llhs.size(0)
        assert (
            est_llhs.size(1) == n_spkrs
        ), f"Expected number of sources is {n_spkrs}, got {input.size(1)}"
        llhs = est_llhs.permute(1, 0, 2, 3)  # speakers are now first dimension
        seqlengths = seqlengths.t()  # S x B
        denfsms = BatchFSM.from_list([self.denfsm for _ in range(batch_size)])

        log_marginal = FSMLogMarginal.apply
        den_llh = torch.sum(
            torch.stack(
                [log_marginal(llh, seq, denfsms) for llh, seq in zip(llhs, seqlengths)],
                dim=1,
            ),  # B x S
            dim=1,
        )  # B

        perms = torch.tensor(list(permutations(range(n_spkrs))))
        num_llh = torch.stack(
            [
                multi_loss(log_marginal, llhs[perm], seqlengths[perm], numfsms)
                for perm in perms
            ],
            dim=1,
        )  # B x #perms
        num_llh, _ = torch.min(num_llh, dim=1)  # B
        loss = -(num_llh - self.den_scale * den_llh)

        if self.do_avg:
            return loss.sum() / n_spkrs * seqlengths.sum()

        return loss.sum()
