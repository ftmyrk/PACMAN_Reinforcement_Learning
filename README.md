# Deep Q-Network Agent for Playing PAC-MAN

This project implements a Deep Q-Network (DQN) Reinforcement Learning agent to play **Ms. Pac-Man** using the Atari environment provided by the OpenAI Gym (Gymnasium) Arcade Learning Environment. The agent learns to navigate the maze, avoid ghosts, and maximize score by interacting with pixel-based game frames.

This project demonstrates:
- Practical application of **Deep Reinforcement Learning**
- Training agents directly from **raw visual input**
- Designing and tuning a **CNN-based Q-network**
- Stabilizing learning using **experience replay** and **target networks**
- Understanding **training dynamics** (reward trends, convergence behavior)

---

## 🎮 Environment

- **Game:** Ms. Pac-Man (Atari 2600 via ALE)
- **Action Space:** 9 discrete actions  
- **Observation:** Raw pixel frames (210 × 160 × 3)
- **Framework:** `gymnasium.make("ALE/MsPacman-v5")`

---

## 🧠 Neural Network Architecture (DQN)

The agent uses a **Convolutional Neural Network** to approximate the Q-function:
- **Input: 210×160×3 RGB frame**
- ↓ Conv2D (32 filters, 8×8)
- ↓ Conv2D (64 filters, 4×4)
- ↓ Conv2D (64 filters, 3×3)
- ↓ Fully Connected (512 units)
- **Output: Q-values for each action**

Training Objective:

$$L(\theta) = \[(r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta))^2$$
\]

Stabilization Methods:
- **Experience Replay Buffer**
- **Target Network Updates**
- **Epsilon-Greedy Exploration** (ε decreases over time)

---

## 🚀 Training Setup

| Property | Value |
|--------|-------|
| Episodes | **10,000** |
| Exploration Strategy | Epsilon-Greedy |
| Replay Buffer | Yes |
| Target Network | Yes |
| Input | Raw Frames (no RAM features) |

### Results Summary
The agent:
- Reached a **max score of ~ 4,120**
- Achieved stable **mean reward convergence around ~500**
- Showed clear improvement over random policy

Reward distribution and performance curves are included in the `report.pdf`.

---

## 📦 Repository Contents

| File | Description |
|------|-------------|
| `PACMAN_DQN.py` | Main training script & network architecture |
| `best_model.pth` | Trained DQN weights |
| `report.pdf` | Full write-up with results, graphs, and analysis |


---

## 🏁 Run the Agent

### Install Dependencies
```bash
pip install gymnasium[atari] torch numpy matplotlib
```

Train the Model
```bash
python PACMAN_DQN.py
```
