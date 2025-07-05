import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from neuro.neuro import *


x = Tensor([1,3,4])
x.grad = 1
x.zero_grad()
print(x.grad)

