from typing import Optional
import numpy as np
from graphviz import Digraph
import math
import random
import os
import logging

 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.getenv("DEBUG") == "1" else logging.INFO)
logging.basicConfig(
    level=logger.level,
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
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
def log_operation(op_name):
   
    def decorator(func):
        def wrapper(*args, **kwargs):
            
            logger.debug(f"Entering {op_name} with args={args}, kwargs={kwargs}")
            
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {op_name} with result={result}")
 
            
            return result
        
        return wrapper
   
    return decorator

class Node:
    def __init__(self, data,dtype, _op=None,requires_grad=False, ):
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None  
        self._parents = set()  
        self._op = _op
    def backward(self):
        if self.data.shape == ():
            self.grad = np.ones_like(self.data)
        elif self.data.ndim==1: 
            self.grad = np.eye(self.data.shape[0])
        else:
            self.grad  = np.ones_like(self.data)
        
        logging.debug(f"Initialized gradient of {self} to {self.grad}")

       
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
            logging.debug(f"Backpropagating through {tensor}")
            tensor._backward()
        
class Tensor:
    
    def __init__(self, data,dtype=np.float32, _op=None,requires_grad=False, _parents=None):
       self.node  = Node(np.array(data,dtype=dtype,),dtype, _op,_parents)
       self.requires_grad = requires_grad     

    @property
    def data(self):
        return self.node.data
    @property
    def grad(self):
        return self.node.grad
    @grad.setter
    def grad(self, value):
        self.node.grad = value
    
    def backward(self):
        if self.requires_grad:
            self.node.backward()



    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, grad={self.grad}) "

    
    @log_operation("addition")
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data + other.data,'+', requires_grad=self.requires_grad or other.requires_grad,)


        out._parents.update({self, other})

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
                logging.debug(f"Updated gradient of {self} to {self.grad}")
            if other.requires_grad:
                other.grad += out.grad
                logging.debug(f"Updated gradient of {other} to {other.grad}")
        out._backward = _backward

        return out
    @log_operation("subtraction")
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data - other.data,'-', requires_grad=self.requires_grad or other.requires_grad,)


        out._parents.update({self, other})

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
                logging.debug(f"Updated gradient of {self} to {self.grad}")
            if other.requires_grad:
                other.grad += out.grad
                logging.debug(f"Updated gradient of {other} to {other.grad}")
        out._backward = _backward

        return out
    
    @log_operation("multiply")
    def __mul__(self,other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data * other.data,'*', requires_grad=self.requires_grad or other.requires_grad,)
        out._parents.update({self, other})
        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
                logging.debug(f"Updated gradient of {self} to {self.grad}")
            if other.requires_grad:
                other.grad += self.data * out.grad
                logging.debug(f"Updated gradient of {other} to {other.grad}")
        out._backward = _backward
        return out  
    @log_operation("truediv")
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data/other.data, "/",  requires_grad=self.requires_grad or other.requires_grad)
        out._parents.update({self,other})
        def _backward():
            if self.requires_grad:
                self.grad += (1/other.grad) * out.grad
                logging.debug(f"Updated gradient of {self} to {self.grad}")
            if other.requires_grad:
                other.grad += (-self.grad/(other.grad**2))*out.grad
                logging.debug(f"Updated gradient of {self} to {self.grad}")
        out._backward=_backward
        return out
    @log_operation("pow")
    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        
        out = Tensor(self.data ** other.data, '**', requires_grad=self.requires_grad or other.requires_grad)
        
        out._parents.update({self,other})

        def _backward():
            if self.requires_grad:
                self.grad += (other*self.data**(other-1)) *out.grad
                logging.debug(f"Updated gradient of {self} to {self.grad}")

            if other.requires_grad:
                other.grad+=((self.data**other)*np.log(math.e))*out.grad
                logging.debug(f"Updated gradient of {self} to {self.grad}")
        out._backward = _backward
        return out
    @log_operation("matmul")    
    def matmul(self, other ):
        assert self.data.shape[-1] == other.data.shape[-2]
        
        out = Tensor(np.matmul(self.data,other.data), requires_grad=self.requires_grad)
        out._parents.update({self,other})
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out 
    @log_operation("relu")
    def relu(self):
        if hasattr(self.data, "shape") or len(self.data.shape[2]) > 2:
            y = np.where(self.data >0,self.data,0)
        else:
            y = self.data if self.data>0 else 0


        out = Tensor(y, 'relu', requires_grad=self.requires_grad)
        
        out._parents.update({self})
        def _backward():
            if self.requires_grad:
                if hasattr(self.data, "shape") or len(self.data.shape[2]) >= 2:
                   
                   grad_mask = np.array(np.where(self.data> 0,1,0))   
                else:
                    print("Applied2")
                    grad_mask = 1.0 if self.data > 0 else 0.0
                
                self.grad=self.grad + grad_mask 
                
        out._backward = _backward
        return out
    @log_operation("sigmoid")    
    def sigmoid(self):
        x = self.data
        s = 1/(1+math.pow(math.e, (-1*x)))
        out = Tensor(s,'sigmoid',requires_grad=self.requires_grad )
        out._parents.update({self})
        def _backward():
            if self.requires_grad:
               
               self.grad += s * (1 - s) * out.grad
               logging.debug(f"Updated gradient of {self} to {self.grad}")
        out._backward = _backward
        return out
    @log_operation("tanh")
    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1)/(np.exp(2*x) + 1)
        out = Tensor(t,'tanh', requires_grad=self.requires_grad)

        out._parents = {self}
        def _backward():
            self.grad += (1 - t**2) * out.grad
            logging.debug(f"Updated gradient of {self} to {self.grad}")
        out._backward = _backward

        return out
    
    
    def zero_grad(self):
        self.grad = 0
        return self
    
    def ones(shape: tuple[int,int]):
        return Tensor(np.full((shape[0], shape[1]), 1, dtype=np.float32))
    def rand(shape: tuple[int, int], min, max):
        
        return Tensor(np.random.randint(min,max, size=shape))

    def sum(self):
        out = Tensor(np.sum(self.data), requires_grad=self.requires_grad)
        logging.debug(f"output tensor for sum created {out}")


        out._parents = {self}
        logging.debug(f"parent tensors saved {out._parents}")
        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad
                logging.debug(f"Updated gradient of {self} to {self.grad}")
        out._backward = _backward

        return out
def dot(vec, vec2):
        if(len(vec) != len(vec2)):
            raise ValueError("Vectors must have same length")
        result = 0
        for i in range(len(vec)):
            result+=vec[i] * vec2[i]
        return result


@log_operation("softmax")
def softmax(tensor, beta=1):
        
        scaled = tensor.data * beta
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        
        out = Tensor(probs, 'softmax', requires_grad=tensor.requires_grad)

        out._parents.update({tensor})
        def _backward():
            if tensor.requires_grad:
                jacobian =np.diagflat(probs) - np.outer(probs,probs)
                tensor.grad += jacobian @ out.grad
                logging.debug(f"Updated gradient of {tensor} to {tensor.grad}")
        out._backward = _backward
        return out
        
@log_operation("Cross-entropy loss")    
def crossELoss(logits,y_true):
        probs = softmax(logits)
        eps = 1e-12
        probs_clipped = np.clip(probs.data, eps, 1 - eps)
        loss = -np.sum(y_true * np.log(probs_clipped)) / y_true.shape[0]
        loss_tensor = Tensor(loss, "cross_entropy", requires_grad=logits.requires_grad)
        loss_tensor._parents = {logits}
        def _backward():
            if logits.requires_grad:
            # Gradient: (probs - y_true)
                grad = (probs.data - y_true) / y_true.shape[0]
                logits.grad += grad
                logging.debug(f"Updated gradient of {logits} to {logits.grad}")
        loss_tensor._backward = _backward
        return loss_tensor

def MSE(y,x ):
    error = 0 
    n  = range(len(y.data))
    for i in n:
        error += math.pow(y.data[i] - x.data[i],2)
    return error/len(y.data)