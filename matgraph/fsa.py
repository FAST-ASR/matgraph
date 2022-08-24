# SPDX-License-Identifier: MIT

import math
from torch.utils.dlpack import from_dlpack, to_dlpack
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
convertbatch(T, X) = copyto!(similar(X, T), X)
""")

jl.seval("""
function expandbatch(x, seqlengths)
    map(t -> expand(t...),
        zip(eachslice(x, dims = 1), seqlengths))
end
""")

jl.seval("""
fsmtype(::FSM{K}) where K = K
""")


class CompiledFSA:

    @classmethod
    def from_files(cls, path_fsa, path_smap):
        return CompiledFSA(jl.deserialize(path_fsa), jl.deserialize(path_smap))

    def __init__(self, fsa, smap):
        self.fsa = fsa
        self.smap = smap

    def cuda(self):
        self.fsa = jl.adapt(jl.CuArray, self.fsa)
        self.smap = jl.CuSparseMatrixCSR(jl.adapt(jl.CuArray, self.smap))
        return self


class BatchCompiledFSA:

    @classmethod
    def from_list(cls, fsas):
        bfsa = jl.rawunion(*[f.fsa for f in fsas])
        smaps = [f.smap for f in fsas]
        return BatchCompiledFSA(bfsa, smaps)

    def __init__(self, bfsa, smaps):
        self.bfsa = bfsa
        self.smaps = smaps

    def cuda(self):
        bfsa = jl.adapt(jl.CuArray, self.bfsa)
        smaps = [jl.CuSparseMatrixCSR(jl.adapt(jl.CuArray, C))
                 for C in self.smaps]
        return BatchCompiledFSA(bfsa, smaps)


def pdfposteriors(bfsa, X, seqlengths):
    # We assume the X to be shaped as B x L x D where B is the batch
    # size, L is the sequence length and D is the features dimension.
    X = jl.permutedims(jl.transfer(X, torch.to_dlpack), (3, 1, 2))
    X = jl.convertbatch(jl.fsmtype(bfsm.bfsm), X)
    X = jl.expandbatch(X, jl.toarray(jl.Int, seqlengths))
    Cs = jl.toarray(jl.typeof(bfsm.smaps[0]), bfsm.smaps)
    Z, ttl = jl.pdfposteriors(bfsm.bfsm, X, Cs)
    return jl.share(jl.permutedims(Z, (2, 3, 1)), torch.from_dlpack), \
           jl.share(ttl, torch.from_dlpack)

