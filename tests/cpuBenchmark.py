import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from neuro.core import *

A = Tensor.rand((4096,4096), 1,20)
B = Tensor.rand((4096,4096), 1,20)

for i in range(100000):
    z = A.matmul(B)
    print(z)
