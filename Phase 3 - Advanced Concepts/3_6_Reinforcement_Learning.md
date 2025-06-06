
## 📘 3.6: **Reinforcement Learning (RL)**

### 🎯 Goal

Train agents to make sequential decisions by **interacting with an environment** to **maximize rewards**.

---

## 🧠 1. Core Concepts

| Concept         | Description                            |
| --------------- | -------------------------------------- |
| **Agent**       | Learner or decision-maker              |
| **Environment** | What the agent interacts with          |
| **State (s)**   | A snapshot of the environment          |
| **Action (a)**  | Decision taken by the agent            |
| **Reward (r)**  | Feedback from the environment          |
| **Policy (π)**  | Strategy mapping states to actions     |
| **Value (V)**   | Expected long-term return from a state |

---

## 🔁 2. The RL Loop

```plaintext
Agent → takes action → Environment → returns reward & new state → Agent …
```

### 🔹 Objective:

Maximize cumulative reward (expected return) over time:

$$
R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

Where $\gamma \in [0,1]$ is the **discount factor**.

---

## 🧮 3. Categories of RL Algorithms

### 🔹 Value-Based Methods

Learn a **value function** to evaluate states or state-action pairs.

#### 🔸 Q-Learning

* Learn Q(s, a): expected reward for taking action `a` in state `s`
* **Update rule**:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
$$

---

### 🔹 Policy-Based Methods

Directly learn the **policy function** π(a|s) without using value functions.

#### 🔸 Policy Gradient

* Updates the policy parameters in the direction that improves performance:

$$
\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot R]
$$

---

### 🔹 Actor-Critic Methods

Combines both:

* **Actor**: Updates the policy π
* **Critic**: Estimates value function V(s) or Q(s, a)

---

## 🧠 4. Deep Q Networks (DQN)

> Use **neural networks** to approximate Q-values.

### 🔸 Key Ideas:

* Replay Buffer: store transitions and sample mini-batches
* Target Network: stabilize training by delaying Q-target updates

### 🔧 Simple DQN Flow:

1. Input: state
2. Output: Q-values for each action
3. Choose action: `argmax(Q)`
4. Update with Bellman equation

---

## 📚 Libraries & Tools

* `Gymnasium` (OpenAI Gym): RL environment simulation
* `Stable-Baselines3`: High-level RL algorithms
* `RLlib`, `CleanRL`: Research & production-ready frameworks

---

## 🧪 Exercises

### ✅ Conceptual

1. What’s the difference between policy-based and value-based methods?
2. Why is experience replay useful in DQN?

### ✅ Practical

* Implement Q-learning on FrozenLake (from `gymnasium`)
* Train a DQN agent on CartPole-v1
* Try PPO (Proximal Policy Optimization) using `Stable-Baselines3`
