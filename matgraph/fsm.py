# SPDX-License-Identifier: MIT

import json
from juliacall import Main as jl

jl.seval("""
function make_fsm(init, arcs, final, labels)
    K = LogSemiring{Float32}
    i = [ a => K(b) for (a, b) in pyconvert(Array{Tuple}, init)]
    T = [ (a, b) => K(c) for (a, b, c) in pyconvert(Array{Tuple}, arcs)]
    f = [ a => K(b) for (a, b) in pyconvert(Array{Tuple}, final)]
    l = [Label(a) for a in pyconvert(Array, labels)]
    FSM(i, T, f, l)
end
""")

def fsm_from_json(jsonstr):
    data = json.loads(jsonstr)
    return jl.make_fsm(data["initstates"], data["arcs"], data["finalstates"],
                       data["labels"])

def union(fsm1, fsm2):
    return jl.union(fsm1, fsm2)

def cat(fsm1, fsm2):
    return jl.cat(fsm1, fsm2)

jl.seval("""
MarkovModels.compose(fsm1, fsms) =
    compose(fsm1, pyconvert(Array, fsms))
""")
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
