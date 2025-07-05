#  Neuro — A Lightweight Autograd Tensor Library

**Neuro** is a minimalist NumPy-based tensor library built for learning and experimentation. It features basic tensor operations with automatic differentiation (autograd), inspired by deep learning frameworks like PyTorch and Tinygrad, but written from scratch for clarity and extensibility.

---

##  Features

- ✅ NumPy-backed `Tensor` class  
- ✅ Automatic differentiation (reverse-mode autodiff)  
- ✅ Core tensor ops: `+`, `-`, `*`, `/`, `**`, `matmul`  
- ✅ Activation functions: `relu`, `sigmoid`, `tanh`  
- ✅ Simple graph visualization using `Graphviz`  
- ✅ Logging and operation tracing for debugging  
- ✅ `backward()` support for computing gradients  
- ✅ Easy-to-extend with decorators like `@log_operation`

---

##  Example

```python
from neuro.core import Tensor

x = Tensor([2.0], requires_grad=True)
w = Tensor([3.0], requires_grad=True)
b = Tensor([1.0], requires_grad=True)

y = x * w + b
y.backward()

print("y:", y)
print("dy/dx:", x.grad)
print("dy/dw:", w.grad)
print("dy/db:", b.grad)
