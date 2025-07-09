from neuro.core import Tensor




for i in range(100000):
    A = Tensor.rand((8192,8192), 1,20)
    B = Tensor.rand((8192,8192), 1,20)
    z = A @ B
    
    print(z)