import math
from neuro.core import *

def dot(vec, vec2):
        if(len(vec) != len(vec2)):
            raise ValueError("Vectors must have same length")
        result = 0
        for i in range(len(vec)):
            result+=vec[i] * vec2[i]
        return result


@log_operation("softmax")
def softmax(tensor, beta=1):
        
        scaled = tensor.data * beta
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        
        out = Tensor(probs, 'softmax', requires_grad=tensor.requires_grad)

        out._parents.update({tensor})
        def _backward():
            if tensor.requires_grad:
                jacobian =np.diagflat(probs) - np.outer(probs,probs)
                tensor.grad += jacobian @ out.grad
                logging.debug(f"Updated gradient of {tensor} to {tensor.grad}")
        out._backward = _backward
        return out
        
@log_operation("Cross-entropy loss")    
def crossELoss(logits,y_true):
        probs = softmax(logits)
        eps = 1e-12
        probs_clipped = np.clip(probs.data, eps, 1 - eps)
        loss = -np.sum(y_true * np.log(probs_clipped)) / y_true.shape[0]
        loss_tensor = Tensor(loss, "cross_entropy", requires_grad=logits.requires_grad)
        loss_tensor._parents = {logits}
        def _backward():
            if logits.requires_grad:
            # Gradient: (probs - y_true)
                grad = (probs.data - y_true) / y_true.shape[0]
                logits.grad += grad
                logging.debug(f"Updated gradient of {logits} to {logits.grad}")
        loss_tensor._backward = _backward
        return loss_tensor
@log_operation("MSE")   
def MSE(y,x ):
    error = 0 
    n  = range(len(y.data))
    for i in n:
        error += math.pow(y.data[i] - x.data[i],2)
    return error/len(y.data)


@log_operation("relu")
def relu(self):
    y = np.where(self.data > 0, self.data, 0.0)
    out = Tensor(y, 'relu', requires_grad=self.requires_grad)
    out._parents.add(self)

    def _backward():
        if self.requires_grad:
            grad_mask = (self.data > 0).astype(self.data.dtype)
            self.grad += grad_mask * out.grad
    out._backward = _backward
    return out
@log_operation("sigmoid")
def sigmoid(self):
    s = 1.0 / (1.0 + np.exp(-self.data))
    out = Tensor(s, 'sigmoid', requires_grad=self.requires_grad)
    out._parents.add(self)

    def _backward():
        if self.requires_grad:
            self.grad += s * (1 - s) * out.grad
    out._backward = _backward
    return out

@log_operation("tanh")
def tanh(self):
    t = np.tanh(self.data)
    out = Tensor(t, 'tanh', requires_grad=self.requires_grad)
    out._parents.add(self)

    def _backward():
        if self.requires_grad:
            self.grad += (1 - t ** 2) * out.grad
    out._backward = _backward
    return out