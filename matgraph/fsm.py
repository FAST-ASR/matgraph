# SPDX-License-Identifier: MIT

import math
import torch
from collections.abc import MutableMapping
from juliacall import Main as jl


# Set the Julia environment.
jl.seval("using Adapt")
jl.seval("using Serialization")
jl.seval("using CUDA")
jl.seval("using CUDA.CUSPARSE")
jl.seval("using DLPack")
jl.seval("using MarkovModels")
jl.seval("using Semirings")
jl.seval("using SparseArrays")
jl.seval("using PythonCall")

jl.seval("""
toarray(T, X) = [pyconvert(T, x) for x in X]
""")

jl.seval("""
transfer(x, fn::PythonCall.Py) =
    DLPack.unsafe_wrap(DLPack.DLManagedTensor(fn(x)), x.py)
""")

jl.seval("""
share(x, fn::PythonCall.Py) =
    DLPack.share(x, fn)
""")

jl.seval("""
function expandbatch(x, seqlengths)
    map(t -> expand(t...),
        zip(eachslice(x, dims = 1), seqlengths))
end
""")


class FSM:

    @classmethod
    def from_files(cls, path_fsm, path_smap):
        return FSM(jl.deserialize(path_fsm), jl.deserialize(path_smap))

    def __init__(self, fsm, smap):
        self.fsm = fsm
        self.smap = smap

    def cuda(self):
        self.fsm = jl.adapt(jl.CuArray, self.fsm)
        self.smap = jl.adapt(jl.CuArray, self.smap)


class BatchFSM:

    @classmethod
    def from_list(cls,fsms):
        bfsm = jl.rawunion(*[f.fsm for f in fsms])
        smaps = [f.smap for f in fsms]
        return BatchFSM(bfsm, smaps)

    def __init__(self, bfsm, smaps):
        self.bfsm = bfsm
        self.smaps = smaps

    def cuda(self):
        bfsm = jl.adapt(jl.CuArray, self.bfsm)
        smaps = [jl.CuSparseMatrixCSR(jl.adapt(jl.CuArray, C))
                 for C in self.smaps]
        return BatchFSM(bfsm, smaps)


def pdfposteriors(bfsm, X, seqlengths):
    # We assume the X to be shaped as B x L x D where B is the batch
    # size, L is the sequence length and D is the features dimension.
    X = jl.permutedims(jl.transfer(X, torch.to_dlpack), (3, 1, 2))
    X = jl.expandbatch(X, jl.toarray(jl.Int, seqlengths))
    X = prepare_data(bfsm, X, seqlengths)
    Cs = jl.toarray(jl.typeof(bfsm.smaps[1]), bfsm.smaps)
    Z, ttl = jl.pdfposteriors(bfsm.bfsm, X, Cs)
    return jl.share(jl.permutedims(Z, (2, 3, 1)), torch.from_dlpack), \
           jl.share(ttl, torch.from_dlpack)

