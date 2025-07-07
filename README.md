#  Neuro — A Lightweight Autograd Tensor Library

**Neuro** is a minimalist NumPy-based tensor library made for learning and just messing around.
Future goal being adding support for generating optimized GPU and CPU kernels.  

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
