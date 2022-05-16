# SPDX-License-Identifier: MIT

import json
from juliacall import Main as jl

def fsm_from_json(jsonstr):
    return jl.FSM(jsonstr)

def union(fsm1, fsm2):
    return jl.union(fsm1, fsm2)

def cat(fsm1, fsm2):
    return jl.cat(fsm1, fsm2)

jl.seval("""
MarkovModels.compose(fsm1, fsms, sep) =
    compose(fsm1, pyconvert(Array, fsms), sep)
""")
def compose(fsm1, fsms, sep=""):
    if not sep:
        sep = jl.one(jl.UnionConcatSemiring)
    else:
        sep = jl.Label(sep)
    return jl.compose(fsm1, fsms, sep)

def determinize(fsm):
    return jl.determinize(fsm)

def minimize(fsm):
    return jl.minimize(fsm)

def propagate(fsm):
    return jl.propagate(fsm)

def renorm(fsm):
    return jl.renorm(fsm)
