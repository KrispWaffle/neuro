### To-Do
  
  
  - Add a simple SGD optimizer.
  - Create a Linear layer class.
  - Build a Sequential model abstraction to stack layers.




## Current 
  - Implement MSELoss()



## Done:
  [x] working on finishing matmul need to add back prop 
  [x] grad prop with tensor values (other than scalars)
  [x] Add Tensor.relu()
  [x] Implement Tensor.matmul() to support matrix multiplications.
  [x] Tensor.softmax().
  [x] CrossEntropyLoss().






### High Priority


- **Handle Gradient Accumulation:**  
  - Implement a method (e.g., `zero_grad()`) to reset gradients between training steps.
  
- **Shape Handling and Broadcasting:**  
  - Ensure that operations (addition, multiplication, etc.) handle NumPy broadcasting rules.
  - Add checks for tensor shapes to prevent unexpected behavior during forward and backward passes.

---

### Medium Priority

- **Enhance Error Handling:**  
  - Improve shape and type checking in functions (like ReLU) to catch errors early.
  - Add meaningful error messages for mismatched dimensions or unsupported operations.

- **Refine Type Annotations:**  
  - Add and improve type annotations throughout  code for clarity and to help catch bugs .

- **Vectorization for Batch Operations:**  
  - Optimize backward functions to efficiently handle batched inputs, reducing reliance on for-loops.

---

### Low Priority

- **Memory Management:**  
  - Consider using weak references for parent nodes to avoid circular references and potential memory leaks in deep graphs.

- **Modularize for Neural Network Extensions:**  
  - Separate autodiff engine from neural network layer definitions.
  - Plan for additional modules (like layers, optimizers, and regularization methods) to support larger networks.

- **Extend Operations:**  
  - Add more neural network–specific operations (e.g., convolution, pooling, batch normalization) as your framework grows.

