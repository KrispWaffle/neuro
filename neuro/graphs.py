import numpy as np
from graphviz import Digraph
import os

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for parent in v._parents:
                edges.add((parent, v))
                build(parent)
    build(root)
    return nodes, edges

def draw_dot(root, format='jpg', rankdir='LR'):
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    for n in nodes:
        data_str = np.array2string(n.data, precision=4, separator=',', suppress_small=True)
        grad_str = np.array2string(n.grad, precision=4, separator=',', suppress_small=True) if n.grad is not None else "None"
        shape_str = str(n.data.shape)
        label = f"{{ data {data_str} | grad {grad_str} | shape {shape_str} }}"
        dot.node(name=str(id(n)), label=label, shape='record')
        
        if n._op:
            op_node_id = str(id(n)) + n._op
            dot.node(name=op_node_id, label=n._op)
            dot.edge(op_node_id, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot
def print_graph(root):
  
    print(_format_graph_recursive(root))

def _format_graph_recursive(root, indent_level=0):
  
    indent = "  " * indent_level

    if not hasattr(root, '_ctx') or root._ctx is None:
        op_name = "NONE" 
        dtype_str = str(root.data.dtype)
        if(os.getenv("VIZ") == "1"):
            return f"{indent}Op(Ops.{op_name}, val={root.data}, dtypes.{dtype_str}, src=())"
        else:
            return f"{indent}Op(Ops.{op_name}, dtypes.{dtype_str}, src=())"

    op_name = root._ctx.__class__.__name__.upper()
    dtype_str = str(root.data.dtype)
    
    src_strings = []
    for parent in root._ctx.parents:
        src_strings.append(_format_graph_recursive(parent, indent_level + 1))
    
    src_block = ",\n".join(src_strings)
    if(os.getenv("VIZ") == "1"):
        return (
        f"{indent}Op(Ops.{op_name}, val={root.data}, dtypes.{dtype_str}, src=(\n"
        f"{src_block}\n"
        f"{indent}))"
        )
    else:
        return (
        f"{indent}Op(Ops.{op_name},  dtypes.{dtype_str}, src=(\n"
        f"{src_block}\n"
        f"{indent}))"
        )
           
