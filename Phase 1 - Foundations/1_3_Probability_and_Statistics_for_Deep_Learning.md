## 📘 Topic 1.3: **Probability and Statistics for Deep Learning**

### 🔑 Key Concepts

---

### 📊 1. **Probability Basics**

* **Random Variable**: A variable whose value is the outcome of a random phenomenon.
* **Probability Distribution**: Describes how probabilities are distributed over values.

  * Discrete: Bernoulli, Binomial
  * Continuous: Gaussian (Normal), Uniform

---

### 📈 2. **Key Distributions**

#### ✅ **Bernoulli Distribution**

* Outcome: 0 or 1
* $P(x=1) = p,\quad P(x=0) = 1 - p$

#### ✅ **Binomial Distribution**

* Number of successes in $n$ independent trials
* $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$

#### ✅ **Normal (Gaussian) Distribution**

* Continuous, bell-shaped
* Defined by:
  $\mu$ (mean), $\sigma$ (std dev)
* PDF:
  $\frac{1}{\sqrt{2\pi\sigma^2}} e^{ -\frac{(x - \mu)^2}{2\sigma^2} }$

---

### 📐 3. **Descriptive Statistics**

* **Mean**: Average
* **Median**: Middle value
* **Mode**: Most frequent value
* **Variance**: Spread of the data
  $\text{Var}(X) = E[(X - \mu)^2]$
* **Standard Deviation**: Square root of variance

---

### 📌 4. **Bayes’ Theorem**

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

* Used in **generative models**, **Naive Bayes**, **Bayesian neural nets**

---

### 🧠 Intuition for Deep Learning

* **Uncertainty**: Models predict probabilities (not certainties)
* **Distributions**: Used in loss functions (cross-entropy, KL divergence)
* **Bayesian Thinking**: Used in probabilistic models and variational inference

---

### 🧪 Exercises

#### ✅ Conceptual

1. What's the difference between a PDF and PMF?
2. Why is the Gaussian distribution important in deep learning?
3. Explain how Bayes' Theorem could be used in classification.

---

#### ✅ Coding (Python + NumPy/Matplotlib)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli, binom

# Gaussian distribution plot
x = np.linspace(-5, 5, 100)
mu, sigma = 0, 1
pdf = norm.pdf(x, mu, sigma)
plt.plot(x, pdf)
plt.title('Standard Normal Distribution')
plt.show()

# Sampling from distributions
ber_samples = bernoulli.rvs(p=0.3, size=1000)
binom_samples = binom.rvs(n=10, p=0.5, size=1000)
normal_samples = np.random.normal(loc=0, scale=1, size=1000)

print("Bernoulli samples mean:", np.mean(ber_samples))
print("Binomial samples std dev:", np.std(binom_samples))
print("Normal samples variance:", np.var(normal_samples))
```
