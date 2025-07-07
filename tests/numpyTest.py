import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from neuro.core import Tensor

x = Tensor([0.0, 0.0, 1.0], requires_grad=True)
print(x.data)
grad_mask = Tensor(np.where(x.data> 0,1,0))
print(grad_mask)   