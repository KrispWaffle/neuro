from typing import Optional, Type
import numpy as np
import os
import logging
from neuro.instr import *
#from codegen import example
 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.getenv("DEBUG") == "1" else logging.INFO)
logging.basicConfig(
    level=logger.level,
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log_operation(op_name):
   
    def decorator(func):
        def wrapper(*args, **kwargs):
            
            logger.debug(f"Entering {op_name} with args={args}, kwargs={kwargs}")
            
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {op_name} with result={result}")
 
            
            return result
        
        return wrapper
   
   
    return decorator





comp = Program()
class Op:
    def __init__(self, *parents):
        self.parents = parents         
        self.saved_tensors = ()
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
    def forward(self, *xs, **kwargs):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError
    @classmethod
    def compute(tOp, *input_tensors, **kwargs):
        ctx = tOp(*input_tensors) 
        out_data = ctx.forward(*[t.data for t in input_tensors], **kwargs)

        requires_grad = any(t.requires_grad for t in input_tensors)
        out = Tensor(out_data, requires_grad=requires_grad)
        out._ctx = ctx
            
        
        return out
from neuro.t_ops import *

class Tensor(Op):
    def __init__(self, data, device="cpu", node: Instr | None = None,requires_grad=False):
        self.dtype = np.float32
        self.data = np.array(data, dtype=self.dtype)
        self.requires_grad = requires_grad
        self.node = node or comp.append_and_return(Instr(OpType.CONST,self.dtype, arg=self.data))
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._ctx: Optional[Op] = None
        self.shape = np.shape(self.data)                       
        self.device = device
    def backward(self):
                        
        visited, topo = set(), []
        def build(t: "Tensor"):
            if t in visited:
                return
            visited.add(t)
            if t._ctx:
                for p in t._ctx.parents:
                    build(p)
            topo.append(t)
        build(self)
  
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            if t._ctx:
                grads = t._ctx.backward(t.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                for p, g in zip(t._ctx.parents, grads):
                    if g is None or p.grad is None:
                        continue
                    p.grad += g
    @log_operation("addition")
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        n = comp.append_and_return(
            Instr(OpType.ADD, self.dtype, (self.node, other.node)))
        self.node = n
        requires_grad= self.requires_grad or other.requires_grad
        out = Tensor(data=None, requires_grad=requires_grad, node=n)

        if requires_grad:
            out._ctx = Sub(self, other)
        return out
                    

    @log_operation("subtraction")
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        n = comp.append_and_return(
            Instr(OpType.SUB, self.dtype, (self.node, other.node)))
        self.node = n    
        return Sub.compute(self, other)


    @log_operation("multiply")
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        n = comp.append_and_return(
            Instr(OpType.MUL, self.dtype, (self.node, other.node)))
        self.node = n    
        return Mul.compute(self, other)


    @log_operation("truediv")
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        n = comp.append_and_return(
            Instr(OpType, self.dtype, (self.node, other.node)))
        self.node = n    
        return TrueDiv.compute(self, other)

    @log_operation("pow")
    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Pow.compute(self, other)

    @log_operation("matmul")
    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        n = comp.append_and_return(
            Instr(OpType.MATMUL, self.dtype, (self.node, other.node)))
        self.node = n
        return MatMul.compute(self, other)
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        n = comp.append_and_return(
            Instr(OpType.MATMUL, self.dtype, (self.node, other.node)))
        self.node = n
        return MatMul.compute(self, other)
    
    @log_operation("relu")
    def relu(self):
        n = comp.append_and_return(
            Instr(OpType.RELU, self.dtype, self.node))
        self.node = n
        return Relu.compute(self)
    def zero_grad(self):
        self.grad = np.zeros_like(self.data) if self.requires_grad else None

    @staticmethod
    def ones(shape):
        return Tensor(np.ones(shape, dtype=np.float32))

    @staticmethod
    def rand(shape, low, high):
        return Tensor(np.random.randint(low, high, size=shape, dtype=np.int32))

    def sum(self):
        return Sum.compute(self)

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, grad={self.grad})"
    
    def realize(self,program: Program = comp):
        sorted =  topo_sort(program)
    
        buffer:dict[float, np.array] = {}
        for i in sorted:
            if i.op == OpType.CONST:
                buffer[i.id] = np.array(i.arg, dtype=i.dtype)
            else:
           
                operation = OP_MAP[i.op]
           
                input_data = [buffer[s.id] for s in i.src]
                buffer[i.id] = operation(input_data)
        self.data = buffer[sorted[-1].id]
        return buffer[sorted[-1].id]