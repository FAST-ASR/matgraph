
import torch
from juliacall import Main as jl

# Set the Julia environment.
jl.seval("using Serialization")
jl.seval("using CUDA")
jl.seval("using DLPack")
jl.seval("using MarkovModels")
jl.seval("using Semirings")
jl.seval("using SparseArrays")
jl.seval("using PythonCall")
jl.seval("""
transfer(x, fn::PythonCall.Py) =
    DLPack.unsafe_wrap(DLPack.DLManagedTensor(fn(x)), x.py)
""")

def transfer(x):
    return jl.transfer(x, torch.to_dlpack)

from .fsm import *

