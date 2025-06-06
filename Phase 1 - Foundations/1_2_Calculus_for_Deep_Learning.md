## 📘 Topic 1.2: **Calculus for Deep Learning**

### 🔑 Key Concepts

1. **Derivatives**

   * Measures how a function changes as its input changes.
   * Notation:

     * $f'(x)$, $\frac{df}{dx}$, or $\nabla f(x)$

2. **Chain Rule**

   * Used when functions are composed:

     * $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$
   * Critical in **backpropagation**.

3. **Partial Derivatives**

   * Derivatives w\.r.t. one variable while keeping others constant.
   * Notation:

     * $\frac{\partial f}{\partial x}$

4. **Gradient**

   * Vector of partial derivatives for all input variables.
   * Points in direction of steepest ascent.

     * Used in **gradient descent** to minimize loss.

5. **Jacobian**

   * Matrix of all partial derivatives of vector-valued functions.
   * Important in multivariable networks and transformations.

6. **Hessian**

   * Square matrix of second-order partial derivatives.
   * Used in advanced optimization (Newton’s Method).

---

### 🧠 Intuition for Deep Learning

* **Training** = Adjust weights to minimize loss.
* Derivatives & gradients = Tell how much to tweak weights.
* Chain rule = Allows calculating how changes flow across many layers (backpropagation).

---

### 🧪 Exercises

#### ✅ Conceptual

1. What is the role of the chain rule in backpropagation?
2. How does the gradient help minimize a loss function?
3. What is the difference between partial derivatives and full derivatives?

#### ✅ Math Practice

1. Find the derivative of $f(x) = x^2 + 3x + 5$
2. Use chain rule to find:
   If $f(x) = \sin(2x^2)$, find $f'(x)$

#### ✅ Coding (Symbolic & Numerical Derivatives)

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, sin

# Symbolic differentiation
x = symbols('x')
f = x**2 + 3*x + 5
df_dx = diff(f, x)
print("Symbolic derivative of f(x):", df_dx)

# Chain rule example
f2 = sin(2*x**2)
df2_dx = diff(f2, x)
print("df/dx for sin(2x^2):", df2_dx)

# Numerical Gradient
def f(x):
    return x**2 + 3*x + 5

def numerical_gradient(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

x_val = 2.0
grad = numerical_gradient(f, x_val)
print(f"Numerical gradient at x = {x_val}: {grad}")
```
