import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.neuro import *


x= Tensor([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0],[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]])
y = x.softmax()
print(y.sum())



