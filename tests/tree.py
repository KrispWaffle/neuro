from neuro.core import  Tensor, program, Instr
from neuro.instr import emit_cuda_kernel


X = Tensor([1], requires_grad=True)
Y= Tensor([1],requires_grad=True)

Z = X+Y
print(Z.data)
H = Z * X
print(H.node.src)
H.realize()
print(Z.data)
print(H.data)

for i in program:
    print(i)