from neuro.core import *
from neuro.graphs import print_graph
import torch
x = Tensor([[1., 2., 3.],
    [4., 5., 6.]],requires_grad=True)
y = Tensor([[7., 8.],
    [9., 10.],
    [11., 12.]], requires_grad=True)
A = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.]
], requires_grad=True)
B = torch.tensor([
    [7., 8.],
    [9., 10.],
    [11., 12.]
], requires_grad=True)
C = A @ B
z= x@y
loss = C.sum()
loss2 = z.sum()
loss2.backward()
loss.backward()
print("--- Gradients ---")
print("Gradient with respect to A (A.grad):\n", A.grad)
print("\nGradient with respect to B (B.grad):\n", B.grad)
print("Gradient with respect to x (x.grad):\n", x.grad)
print("\nGradient with respect to y (y.grad):\n", y.grad)
print_graph(z)
#print(x.grad)   # should be something sensible, e.g. [-0.25  1.  ]
#print(y.grad)   # should be [3. 3.]
