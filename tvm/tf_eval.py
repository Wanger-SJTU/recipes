
import tensorflow.compat.v1 as tf
import os
import numpy as np
from tvm import relay
from tvm.contrib import graph_executor
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

with tf.gfile.GFile(model_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

mod, params = relay.frontend.from_tensorflow(graph_def, shape={"x":[1,512], "y":[1,512]}, outputs=["out"])

pass_seq = tvm.transform.Sequential(
    [
        relay.transform.RemoveUnusedFunctions(),
        relay.transform.ForwardFoldScaleAxis()
    ]
)
with tvm.transform.PassContext(opt_level=3):
    mod = pass_seq(mod)

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



x = np.random.random([1,512]).astype(np.float32)
y = np.random.random([1,512]).astype(np.float32)
# 获取预测tensor
pred = tf.get_default_graph().get_tensor_by_name("out:0")  # mobilenet_v2

sess = tf.InteractiveSession(graph =tf.get_default_graph())
output = sess.run(pred, feed_dict = {"x:0":x, "y:0":y})

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="llvm", params=params)

dev = tvm.cpu(0)
dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))
m.set_input("x", tvm.nd.array(x))
m.set_input("y", tvm.nd.array(y))
m.run()
out = m.get_output(0)

# np.testing.assert_allclose(output, out.numpy(), rtol=1e-5)