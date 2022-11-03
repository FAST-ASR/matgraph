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
fsmtype(::CompiledFSM{K}) where K = K
""")

jl.seval("""
function loadbatch(fsmfiles)
	batch([deserialize(pyconvert(String, f)) for f in fsmfiles]...)
end
""")

jl.seval("""
take_(x) = take!(x)
""")

jl.seval("""
pdfposteriors(fsa, X) = MarkovModels.pdfposteriors2(fsa, X)
""")


class CompiledFSA:

    @classmethod
    def from_file(cls, path_fsa):
        return CompiledFSA(jl.deserialize(path_fsa))

    def __init__(self, fsa):
        self.fsa = fsa

    def cuda(self):
        self.fsa = jl.adapt(jl.CuArray, self.fsa)
        return self


class BatchCompiledFSA:

    @classmethod
    def from_list(cls, fsas):
        bfsa = jl.batch(*[f.fsa for f in fsas])
        return BatchCompiledFSA(bfsa)

    @classmethod
    def from_files(cls, files):
        #bfsa = jl.batch(*[f.fsa for f in fsas])
        bfsa = jl.loadbatch(files)
        return BatchCompiledFSA(bfsa)

    @classmethod
    def from_bin(cls, fsmbytes):
        bfsa = jl.batch(*[
            jl.deserialize(jl.IOBuffer(jl.Array(f)))
            for f in fsmbytes
        ])
        #bfsa = jl.deserialize(jl.IOBuffer(jl.Array(fsmbytes)))
        return BatchCompiledFSA(bfsa)

    #def to_bin(self):
    #    buf = jl.IOBuffer()
    #    jl.serialize(buf, self.bfsa)
    #    jl.flush(buf)
    #    b = bytes(jl.take_(buf))
    #    jl.close(buf)
    #    return b

    def __init__(self, bfsa):
        self.bfsa = bfsa

    def cuda(self):
        bfsa = jl.adapt(jl.CuArray, self.bfsa)
        return BatchCompiledFSA(bfsa)


def pdfposteriors(bfsa, X, seqlengths):
    # We assume the X to be shaped as B x L x D where B is the batch
    # size, L is the sequence length and D is the features dimension.
    X = jl.permutedims(jl.transfer(X, to_dlpack), (3, 1, 2))
    X = jl.convertbatch(jl.fsmtype(bfsa.bfsa), X)
    X = jl.expandbatch(X, jl.toarray(jl.Int, seqlengths))
    #Z, ttl = jl.MarkovModels.pdfposteriors2(bfsa.bfsa, X)
    Z, ttl = jl.pdfposteriors(bfsa.bfsa, X)
    Z, ttl = jl.share(jl.permutedims(Z, (2, 3, 1)), from_dlpack), \
           jl.share(ttl, from_dlpack)
    #Z = jl.share(jl.permutedims(Z, (2, 3, 1)), from_dlpack)
    return Z, ttl

