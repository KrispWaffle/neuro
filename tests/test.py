import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.neuro import *


# Define Tensors
a = Tensor([[1,2,3],[1,2,3]], requires_grad=True)  # a = 2
b = Tensor([[1,2,3],[1,2,3]], requires_grad=True)  # b = 3
c = Tensor([[1,2,3],[1,2,3]], requires_grad=True)  # c = 4
d = Tensor([[1,2,3],[1,2,3]], requires_grad=True)  # d = 5
e = Tensor([[1,2,3],[1,2,3]], requires_grad=True)  # e = 6

# Compute: L = ((a + b) * c + d) * e
f = a + b   # f = (2 + 3) = 5
g = f * c   # g = 5 * 4 = 20
h = g + d   # h = 20 + 5 = 25
L = h * e   # L = 25 * 6 = 150

L.backward()
graph = draw_dot(L)
graph.render('test', format='png')


