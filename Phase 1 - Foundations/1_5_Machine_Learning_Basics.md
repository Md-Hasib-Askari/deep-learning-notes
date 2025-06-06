## 📘 Topic 1.5: **Machine Learning Basics**

### 🔑 Key Concepts

---

### 📌 1. **Types of Learning**

#### ✅ **Supervised Learning**

* Learn from labeled data.
* Examples: Classification (spam vs. ham), Regression (predict price)
* Algorithm sees:
  `X = [features], Y = [target labels]`

#### ✅ **Unsupervised Learning**

* No labels, only features.
* Goal: find hidden patterns or groupings.
* Examples: Clustering (K-means), Dimensionality reduction (PCA)

#### ✅ **Semi-supervised & Reinforcement Learning** *(advanced, but worth knowing)*

---

### 📌 2. **Train/Validation/Test Split**

* **Train Set**: Used to train the model (learn patterns)
* **Validation Set**: Used to tune hyperparameters and avoid overfitting
* **Test Set**: Evaluates final model performance

> Rule of thumb:
> 60% train, 20% validation, 20% test
> (or 80/10/10 depending on dataset size)

---

### 📌 3. **Loss Functions**

#### 🔸 Regression

* **MSE (Mean Squared Error)**:
  $\frac{1}{n} \sum (y - \hat{y})^2$

#### 🔸 Classification

* **Binary Cross-Entropy**:

  $$
  L = -[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})]
  $$

* **Categorical Cross-Entropy**: for multi-class classification

---

### 📌 4. **Optimization**

* **Gradient Descent**:
  Update weights:
  $w = w - \eta \cdot \nabla L(w)$

* Types:

  * **Batch**: entire dataset
  * **Stochastic (SGD)**: one sample
  * **Mini-batch**: group of samples

* Advanced: Adam, RMSprop, Adagrad

---

### 🧠 Why This Matters for DL

> Neural networks = machine learning on steroids
> But the **core principles** are the same:

* Learn patterns from data
* Use loss to quantify performance
* Optimize weights to improve predictions

---

### 🧪 Exercises

#### ✅ Conceptual

1. What’s the difference between supervised and unsupervised learning?
2. Why is the validation set important?
3. Explain how MSE and cross-entropy differ.

---

#### ✅ Coding Task (Simple ML)

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate dummy data
X, y = make_classification(n_samples=1000, n_features=5, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```
