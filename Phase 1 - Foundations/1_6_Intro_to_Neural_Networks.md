## 📘 Topic 1.6: **Intro to Neural Networks**

### 🧠 What is a Neural Network?

A **neural network** is a series of mathematical functions that map input data to output predictions. It’s inspired (loosely) by how neurons work in the brain.

---

### 🧱 1. **Perceptron (Single-Layer)**

The **perceptron** is the simplest neural unit:

$$
y = f(w_1x_1 + w_2x_2 + \dots + w_nx_n + b)
$$

* $w$: weights
* $x$: inputs
* $b$: bias
* $f$: activation function (e.g. step, sigmoid, ReLU)

---

### 🔁 2. **Feedforward Neural Network (Multi-Layer Perceptron)**

* **Input layer**: raw data
* **Hidden layers**: learn patterns
* **Output layer**: final prediction

Each layer performs:

$$
z = W \cdot x + b \\
a = f(z)
$$

---

### 🔑 3. **Activation Functions**

These introduce **non-linearity** so the model can learn complex patterns.

#### 🔸 **Sigmoid**

$$
f(x) = \frac{1}{1 + e^{-x}} \quad \text{(output between 0 and 1)}
$$

#### 🔸 **Tanh**

$$
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \quad \text{(output between -1 and 1)}
$$

#### 🔸 **ReLU (Rectified Linear Unit)**

$$
f(x) = \max(0, x) \quad \text{(most commonly used)}
$$

---

### 🧮 4. **Cost (Loss) Functions**

Quantify how bad your model's prediction is.

#### For regression:

* **MSE**: $\frac{1}{n} \sum (y - \hat{y})^2$

#### For classification:

* **Cross-entropy**:

$$
- [y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})]
$$

---

### ⚙️ 5. **Training a Neural Network**

1. **Forward pass** → predict
2. **Loss computation** → calculate error
3. **Backward pass** (backpropagation) → compute gradients
4. **Weight update** → using gradient descent or optimizers like Adam

---

### 🧪 Exercises

#### ✅ Conceptual

1. What happens if you remove the activation function?
2. Why is ReLU preferred in deep networks?
3. What’s the difference between MSE and cross-entropy?

---

#### ✅ Code (Simple NN using PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple dataset: XOR logic
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

# Model definition
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = SimpleNN()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(1000):
    out = model(X)
    loss = loss_fn(out, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test
with torch.no_grad():
    print("Predictions:", model(X).round().squeeze())
```
