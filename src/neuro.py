from typing import Optional
import numpy as np
from graphviz import Digraph
import math
import random

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

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    for n in nodes:
        data_str = np.array2string(n.data, precision=4, separator=',', suppress_small=True)
        grad_str = np.array2string(n.grad, precision=4, separator=',', suppress_small=True) if n.grad is not None else "None"

        dot.node(name=str(id(n)), label=f"{{ data {data_str} | grad {grad_str} }}", shape='record')
        
        if n._op:
            op_node_id = str(id(n)) + n._op
            dot.node(name=op_node_id, label=n._op)
            dot.edge(op_node_id, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

class Tensor:
    def __init__(self, data, _op=None,requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None  
        self._parents = set()  
        self._op = _op
      

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, grad={self.grad}) "

    def backward(self):
        if self.data.shape == ():
            self.grad = np.ones_like(self.data)
        elif self.data.ndim==1: 
            self.grad = np.eye(self.data.shape[0])
        else:
            self.grad  = np.ones_like(self.data)
        
        print(f"Initialized gradient of {self} to {self.grad}")

       
        visited = set()
        topo = []
        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._parents:
                    build_topo(parent)
                topo.append(tensor)
        build_topo(self)

      
        for tensor in reversed(topo):
            print(f"Backpropagating through {tensor}")
            tensor._backward()

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data + other.data,'+', requires_grad=self.requires_grad or other.requires_grad,)


        out._parents.update({self, other})

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
                print(f"Updated gradient of {self} to {self.grad}")
            if other.requires_grad:
                other.grad += out.grad
                print(f"Updated gradient of {other} to {other.grad}")
        out._backward = _backward

        return out
    def __mul__(self,other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data * other.data,'*', requires_grad=self.requires_grad or other.requires_grad,)
        out._parents.update({self, other})
        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
                print(f"Updated gradient of {self} to {self.grad}")
            if other.requires_grad:
                other.grad += self.data * out.grad
                print(f"Updated gradient of {other} to {other.grad}")
        out._backward = _backward
        return out  


    def matmul(self, other ):
        assert self.data.shape[-1] == other.data.shape[-2]
        
        out = Tensor(np.matmul(self.data,other.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = out.grad @ other.data.T
            if other.requires_grad:
                other.grad = self.data.T @ out.grad
        out._backward = _backward
        return out 

    
    def sigmoid(self):
        x = self.data
        s = 1/(1+math.pow(math.e, (-1*x)))
        out = Tensor(s,'sigmoid',requires_grad=self.requires_grad )
        out._parents = {self}
        def _backward():
            if self.requires_grad:
               self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Tensor(t,'tanh', requires_grad=self.requires_grad)
        out._parents = {self}
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
 
    def ones(shape: tuple[int,int]):
        return Tensor(np.full((shape[0], shape[1]), 1, dtype=np.float32))
    def rand(shape: tuple[int, int], min, max):
        data = [[random.randint(min, max)] * shape[1] for _ in range(shape[0])]
        return Tensor(data)
    def sum(self):
        out = Tensor(np.sum(self.data), requires_grad=self.requires_grad)


        out._parents = {self}

        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad
                print(f"Updated gradient of {self} to {self.grad}")
        out._backward = _backward

        return out
def dot(vec, vec2):
        if(len(vec) != len(vec2)):
            raise ValueError("Vectors must have same length")
        result = 0
        for i in range(len(vec)):
            result+=vec[i] * vec2[i]
        return result
