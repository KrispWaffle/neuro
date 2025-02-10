from src.neuro import *

scalar = Tensor(3.0, requires_grad=True)
vector = Tensor([2.0, 4.0, 6.0], requires_grad=True)
matrix = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

print("Testing scalar:")
scalar.backward()

print("\nTesting vector:")
vector.backward()

print("\nTesting matrix:")
matrix.backward()