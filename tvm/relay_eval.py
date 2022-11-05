
import os
import numpy as np

from tvm  import relay
from typing import (
    Dict,
    Union,
    Tuple,
    List,
)
import tvm
from tvm import relay
from tvm.contrib import relay_viz
from tvm.contrib.relay_viz.interface import (
    VizEdge,
    VizNode,
    VizParser,
)
from tvm.contrib.relay_viz.terminal import (
    TermGraph,
    TermPlotter,
    TermVizParser,
)


model_path = "/home/wanger/code/test_code/model.pb"


x = relay.var("x", shape=[1, 512])
y = relay.var("y", shape=[1, 512])

tmp = relay.multiply(x, y)

weight1 = relay.var("w1", shape=[512, 2048])
bias1 = relay.var("b1", shape=[2048])
m1 = relay.nn.matmul(tmp, weight1)
m1 = relay.nn.bias_add(m1, bias1)

weight2 = relay.var("w2", shape=[512, 2048])
bias2 = relay.var("b2", shape=[2048])
m2 = relay.nn.matmul(tmp, weight2)
m2 = relay.nn.bias_add(m2, bias2)

res = m1 + m2

w1 = tvm.nd.array(np.random.random([512, 2048]).astype(np.float32))
w2 = tvm.nd.array(np.random.random([512, 2048]).astype(np.float32))
b1 = tvm.nd.array(np.random.random([2048]).astype(np.float32))
b2 = tvm.nd.array(np.random.random([2048]).astype(np.float32))
params = {"w1":w1, "w2":w2, "b1":b1, "b2":b2}

func = relay.Function([x, y, weight1, weight2, bias1, bias2], res)

mod = tvm.IRModule.from_expr(func)

mod = relay.transform.ToMixedPrecision("float16")(mod)
    
relay_pass = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ToBasicBlockNormalForm(),
            relay.transform.Legalize(),
            relay.transform.SimplifyInference(),
            relay.transform.CombineParallelDense(),
            relay.transform.FoldConstant(),
            relay.transform.FoldScaleAxis(),
            relay.transform.SimplifyExpr(),
            relay.transform.CanonicalizeCast(),
            relay.transform.CanonicalizeOps(),
            relay.transform.FastMath(),
            relay.transform.FoldConstant(),
        ]
    )
mod = relay_pass(mod)
graph_attr = {"color": "red"}
node_attr = {"color": "blue"}
edge_attr = {"color": "black"}
get_node_attr = {"color": "green"}
dot_plotter = relay_viz.DotPlotter(
            graph_attr=graph_attr,
            node_attr=node_attr,
            edge_attr=edge_attr)

viz = relay_viz.RelayVisualizer(
    mod,
    relay_param=params,
    plotter=dot_plotter,
    parser=relay_viz.DotVizParser())
viz.render("eval")

lib = relay.build(mod, target="llvm", params=params)
with open("graph.json", "w") as f:
    f.write(lib.get_graph_json())
