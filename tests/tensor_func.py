import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.neuro import *


def test_basic_operations():
    """Test basic arithmetic operations"""
    # Addition
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    c = a + b
    assert np.allclose(c.data, 5.0), "Addition failed"
    
    # Multiplication
    d = a * b
    assert np.allclose(d.data, 6.0), "Multiplication failed"
    
    # Subtraction
    e = b - a
    assert np.allclose(e.data, 1.0), "Subtraction failed"
    
    # Division
    f = b / a
    assert np.allclose(f.data, 1.5), "Division failed"
    
    print("Basic operations: ✓")

def test_gradients():
    """Test gradient calculations"""
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = a * b
    c.backward()
    
    assert np.allclose(a.grad, 3.0), "Multiplication grad a failed"
    assert np.allclose(b.grad, 2.0), "Multiplication grad b failed"
    
    # Test sum gradient
    d = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    e = d.sum()
    e.backward()
    assert np.allclose(d.grad, np.ones_like(d.data)), "Sum gradient failed"
    
    print("Gradient calculations: ✓")

def test_relu():
    """Test ReLU activation"""
    a = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
  
    b = a.relu()
    
    # Forward pass
    assert np.allclose(b.data, [0.0, 0.0, 2.0]), "ReLU forward failed"
    
    # Backward pass
    b.backward()
    assert np.allclose(a.grad, [0.0, 0.0, 1.0]), "ReLU backward failed"
    
    print("ReLU: ✓")

def test_sigmoid():
    """Test sigmoid activation"""
    a = Tensor(0.0, requires_grad=True)
    b = a.sigmoid()
    
    # Forward pass
    assert np.allclose(b.data, 0.5), "Sigmoid forward failed"
    
    # Backward pass
    b.grad = 1.0
    b.backward()
    assert np.allclose(a.grad, 0.25), "Sigmoid backward failed"
    
    print("Sigmoid: ✓")

def test_matmul():
    """Test matrix multiplication"""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    c = a.matmul(b)
    
    # Forward pass
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(c.data, expected), "Matmul forward failed"
    
    # Backward pass
    c.backward()
    assert np.allclose(a.grad, np.array([[11.0, 15.0], [11.0, 15.0]])), "Matmul a grad failed"
    assert np.allclose(b.grad, np.array([[4.0, 4.0], [6.0, 6.0]])), "Matmul b grad failed"
    
    print("Matrix multiplication: ✓")

def test_softmax():
    """Test softmax implementation"""
    logits = Tensor([2.0, 1.0, 0.1], requires_grad=True)
    probs = softmax(logits)
    
    # Forward pass
    expected = np.exp([2.0, 1.0, 0.1]) / np.sum(np.exp([2.0, 1.0, 0.1]))
    assert np.allclose(probs.data, expected), "Softmax forward failed"
    
    # Numerical stability test
    stable_logits = Tensor([1000.0, 1001.0, 1002.0], requires_grad=True)
    stable_probs = softmax(stable_logits)
    assert not np.any(np.isnan(stable_probs.data)), "Numerical stability failed"
    
    print("Softmax: ✓")

def test_cross_entropy():
    """Test cross-entropy implementation"""
    logits = Tensor([2.0, 1.0, 0.1], requires_grad=True)
    y_true = np.array([1.0, 0.0, 0.0])
    loss = crossELoss(logits, y_true)
    
    # Forward pass
    probs = softmax(logits).data
    expected_loss = -np.sum(y_true * np.log(probs))
    assert np.allclose(loss.data, expected_loss), "Cross-entropy forward failed"
    
    # Backward pass
    loss.backward()
    expected_grad = (probs - y_true)
    assert np.allclose(logits.grad, expected_grad), "Cross-entropy backward failed"
    
    print("Cross-entropy: ✓")

def test_autograd_graph():
    """Test computational graph construction"""
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = a * b
    d = c + Tensor(1.0)
    d.backward()
    
    assert len(d._parents) == 1, "Graph construction failed"
    assert a.grad == 3.0 and b.grad == 2.0, "Graph backward failed"
    
    print("Computational graph: ✓")

def test_batch_processing():
    """Test batch processing capabilities"""
    # Batch of 2 samples, 3 classes each
    logits = Tensor([
        [2.0, 1.0, 0.1],
        [0.1, 2.0, 1.0]
    ], requires_grad=True)
    
    y_true = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    loss = crossELoss(logits, y_true)
    loss.backward()
    
    probs = softmax(logits).data
    expected_grad = (probs - y_true) / y_true.shape[0]
    
    assert np.allclose(logits.grad, expected_grad), "Batch gradient failed"
    print("Batch processing: ✓")

if __name__ == "__main__":
    test_basic_operations()
    test_gradients()
    test_relu()
    test_sigmoid()
    test_matmul()
    test_softmax()
    test_cross_entropy()
    test_autograd_graph()
    test_batch_processing()
    print("All tests passed")