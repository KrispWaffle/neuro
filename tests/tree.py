from neuro.core import  Tensor, comp
from neuro.instr import emit_cuda_kernel


X = Tensor([1], requires_grad=True)
Y= Tensor([1],requires_grad=True)
Z = X+Y
print(Z.realize())
tensors = []
tensors.append(Z)


#print(X.node.src)

#print_comp(comp)
