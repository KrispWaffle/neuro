import numpy as np
from neuro.core import Op

def sum_to_shape(grad, shape):
    # Match og shape that produced it 
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] > 1 :
            grad = grad.sum(axis=i, keepdims=True)
    return grad
class Add(Op):
    def forward(self, a, b):
        
        self.save_for_backward(a,b)
        return a+b
    def backward(self, grad_out):
        a, b = self.saved_tensors
        return sum_to_shape(grad_out, a.shape), sum_to_shape(grad_out, b.shape)
class Sub(Op):
    def forward(self, a, b):
        self.save_for_backward(a,b)
        return a-b
    def backward(self, grad_out):
        a, b = self.saved_tensors
        return sum_to_shape(grad_out, a.shape), sum_to_shape(-grad_out, b.shape)
class Mul(Op):
    def forward(self, a, b):
        self.save_for_backward(a,b)
        return a*b
    def backward(self, grad_out):
        a,b=self.saved_tensors
        return (sum_to_shape(b*grad_out , a.shape), sum_to_shape(a*grad_out, b.shape))
class MatMul(Op):
    def forward(self, a,b):
        self.save_for_backward(a,b)
        return a @ b 
    def backward(self, grad_out):
        a,b = self.saved_tensors
        return (sum_to_shape(grad_out@b.T,a.shape),sum_to_shape(a.T@grad_out, b.shape))
class Sum(Op):
    def forward(self ,a ):
        self.save_for_backward(a)
        return np.sum(a)
    def backward(self, grad_out):
        a, = self.saved_tensors
        return np.ones_like(a) * grad_out
class TrueDiv(Op):
    def forward(self,a,b):
        self.save_for_backward(a,b)
        return a/b
    def backward(self, grad_out):
        a,b = self.saved_tensors
        grad_a = grad_out / b
        grad_b = grad_out * (-a / (b ** 2))
        return sum_to_shape(grad_a, a.shape), sum_to_shape(grad_b, b.shape)
class Pow(Op):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a ** b
    def backward(self, grad_out):
        a,b = self.saved_tensors
        grad_a = grad_out * (b * (a ** (b - 1)))
        grad_b = grad_out * ((a ** b) * np.log(a))
        return sum_to_shape(grad_a, a.shape), sum_to_shape(grad_b, b.shape)
    
class Relu(Op):
    def forward(self,a ):
        self.save_for_backward(a)
        return np.maximum(0, a)
    def backward(self, grad_out):
        a = self.saved_tensors
        grad = grad_out * (a > 0)
        return grad