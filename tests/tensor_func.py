import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.neuro import *



x = Tensor.rand((3000,4000), 1,10)
y = Tensor.rand((3000,4000), 1,10)

z = x**y

print(z)



