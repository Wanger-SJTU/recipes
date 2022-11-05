import tvm
from tvm import te
from tvm import relay
from tvm.relay.ty import TupleType, TensorType
from tvm.relay.expr_functor import ExprVisitor


def infer_type(node):
    """A method to infer the type of a relay expression."""
    mod = tvm.IRModule.from_expr(node)
    mod = relay.transform.InferType()(mod)
    return mod["main"].ret_type


class RelayVisualizer(ExprVisitor):
    def __init__(self, name="relay_ir"):
        super().__init__()
        self._name = name

    def load_json(self, json_str):
        self._node_dict = {}
        self._node_names = []
        self._node_ids = {}
        self._ignore_nodes = set()

    def visualize(self, json_str, path=None):
        #infer type before visit the entry function
        path = path or self._name+'.prototxt'
        self.load_json(json_str)
        
        #write graph to prototxt
        with open(path, 'w') as f:
            f.write('name : "{}"\n'.format(self._name))
            for k in self._node_names:
                if k in self._ignore_nodes:
                    continue
                node_des = self._node_dict[k]
                topo = ['top:"{}"'.format(k)]+['bottom:"{}"'.format(p)
                                               for p in node_des.get('parents', [])]
                layer_param = ['idx:'+str(node_des["idx"])] + \
                    ['in_{} {}'.format(idx, _tensor_des(i)) for idx, i in enumerate(node_des.get("inputs", []))] + \
                    ['out_{} {}'.format(idx, _tensor_des(o)) for idx, o in enumerate(
                        node_des.get("outputs", []))]
                if "attrs" in node_des:
                    layer_param += ['attrs '+str(node_des["attrs"])]
                f.write('layer {{{0}name:"{1}"{0}type:"{2}"{0}{3}{0}layer_param {{{0}  {4}\n  }}\n}}\n'.format(
                    '\n  ', k, node_des["op"], '\n  '.join(topo), '\n    '.join(layer_param)))
