# SPDX-License-Identifier: MIT

import math
from collections.abc import MutableMapping
from juliacall import Main as jl

def fsm_from_json(jsonstr):
    return jl.FSM(jsonstr)

def Label(x):
    return jl.Label(x)

def union(fsm1, fsm2):
    return jl.union(fsm1, fsm2)

def cat(fsm1, fsm2):
    return jl.cat(fsm1, fsm2)

def compose(fsm1, fsms):
    return jl.compose(fsm1, fsms)

def determinize(fsm):
    return jl.determinize(fsm)

def minimize(fsm):
    return jl.minimize(fsm)

def propagate(fsm):
    return jl.propagate(fsm)

def renorm(fsm):
    return jl.renorm(fsm)

def fsmdict():
    return jl.Dict()

def linear_fsm(seq, precision="Float32", init_sil_prob=0, sil_prob=0,
               final_sil_prob=0, sil="sil"):
    """Create a linear FSM from `seq`."""
    if init_sil_prob > 0:
        init = [[1, math.log(init_sil_prob)], [2, math.log(1 - init_sil_prob)]]
        labels = [sil, seq[0]]
        arcs = [[1, 2, 0]]
        nstates = 2
    else:
        init = [[1, 0]]
        labels = [seq[0]]
        nstates = 1
        arcs = []

    for i in range(1, len(seq)):
        if sil_prob > 0:
            arcs.append([nstates, nstates+1, math.log(sil_prob)])
            arcs.append([nstates, nstates+2, math.log(1 - sil_prob)])
            arcs.append([nstates+1, nstates+2, 0])
            labels.append(sil)
            labels.append(seq[i])
            nstates += 2
        else:
            arcs.append([nstates, nstates+1, 0])
            labels.append(seq[i])
            nstates += 1

    if final_sil_prob > 0:
        final = [[nstates, math.log(1-final_sil_prob)], [nstates+1, 0]]
        arcs.append([nstates, nstates+1, math.log(final_sil_prob)])
        labels.append(sil)
    else:
        final = [[nstates, 0]]


    topo = f"""
    {{
        "semiring": "LogSemiring{{{precision}}}",
        "initstates": {init},
        "arcs": {arcs},
        "finalstates": {final},
        "labels": ["{'","'.join(labels)}"]
    }}
    """
    return renorm(fsm_from_json(topo))

def save(path, obj):
    jl.serialize(path, obj)

def load(path):
    return jl.deserialize(path)

jl.seval("mergengrams(ng1, ng2) = mergewith((x, y) -> x .+ y, ng1, ng2) ")
def merge_ngrams(ng1, ng2):
    return jl.mergengrams(ng1, ng2)

def ngrams(fsm, order=2):
    return jl.totalngramsum(fsm, order = order)

