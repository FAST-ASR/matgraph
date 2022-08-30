import torch

from .fsa import BatchCompiledFSA, CompiledFSA
from .loss import FSMLogMarginal

from itertools import permutations
from typing import List

fsm_log_marginal = FSMLogMarginal.apply


def multi_loss(llhs, seqlengths, fsms):
    res = [
        fsm_log_marginal(llh, seqlengths, fsm) for llh, fsm in zip(llhs, fsms)
    ]  # S x B
    res = torch.stack(res, dim=1)  # B x S
    return res


class PIT_LFMMILoss(torch.nn.Module):
    """Permutation invariant training wrapper for MMI loss."""

    def __init__(self, denfsm: CompiledFSA, den_scale=1.0, do_avg=False):
        super().__init__()
        self.denfsm = denfsm
        self.den_scale = den_scale
        self.do_avg = do_avg

    def forward(
        self,
        est_llhs: torch.Tensor,
        seqlengths: torch.Tensor,
        numfsms: List[BatchCompiledFSA],
    ):
        """
        Args:
          est_llhs: estimated pdf log-likelihoods for each speaker (shape S x B x T x C)
          seqlengths: sequence lengths of shape for each speaker (shape B)
          numfsms: numerator FSMs for each speaker (shape S x B)

        Returns:
          -  PIT LF-MMI loss (scalar)
        """
        n_spkrs = len(numfsms)
        batch_size = est_llhs.size(1)
        assert (
            est_llhs.size(0) == n_spkrs
        ), f"Expected number of sources is {n_spkrs}, got {input.size(0)}"
        assert (
            seqlengths.ndim == 1 and seqlengths.size(0) == batch_size
        ), f"seqlengths.ndim {seqlengths.ndim} has to be 1 and its size {seqlengths.size(0)} == {batch_size}"

        llhs = est_llhs  # speakers are first dimension
        denfsms = BatchCompiledFSA.from_list([self.denfsm for _ in range(batch_size)])
        denfsms = [denfsms for _ in range(n_spkrs)]

        den_llh = multi_loss(llhs, seqlengths, denfsms)  # B x S

        perms = permutations(range(n_spkrs))
        num_llh_perms = [
            multi_loss(llhs, seqlengths, [numfsms[i] for i in perm]) for perm in perms
        ]  # #perms x (B x S)
        num_llh_perms = torch.stack(num_llh_perms, dim=0)  # shape: #perms x B x S

        _, idxs = torch.max(
            num_llh_perms.sum(dim=2), dim=0
        )  # find best permutation for each mixture
        idxs = torch.stack([idxs for _ in range(n_spkrs)], dim=1)
        
        num_llh = torch.gather(
            num_llh_perms, 0, idxs.unsqueeze(0)
        )  # recreate num_llh from best perm
        # num_llh = torch.stack([num_llh_perms[i, n, :] for n,i in enumerate(idxs)])
        num_llh = num_llh.squeeze()

        loss = -(num_llh - self.den_scale * den_llh)
        if self.do_avg:
            return loss.sum() / n_spkrs * seqlengths.sum()

        return loss.sum()
