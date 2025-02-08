from neuro import *


# Define Tensors
a = Tensor(2, requires_grad=True)  # a = 2
b = Tensor(3, requires_grad=True)  # b = 3
c = Tensor(4, requires_grad=True)  # c = 4
d = Tensor(5, requires_grad=True)  # d = 5
e = Tensor(6, requires_grad=True)  # e = 6

# Compute: L = ((a + b) * c + d) * e
f = a + b   # f = (2 + 3) = 5
g = f * c   # g = 5 * 4 = 20
h = g + d   # h = 20 + 5 = 25
L = h * e   # L = 25 * 6 = 150

L.backward()
graph = draw_dot(L)
graph.render('test', format='png')