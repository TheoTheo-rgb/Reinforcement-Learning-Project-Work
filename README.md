# Frozen Lake from First Principles using Q-Learning

A complete, framework-free Reinforcement Learning solution to the **8×8 Frozen
Lake** problem. The environment, the Q-Learning agent, the training loop and the
evaluation are all implemented from scratch in pure Python / NumPy — **no
Gymnasium, OpenAI Gym, Stable Baselines or RLlib** are used.

> **Course:** DSCD 614 – Reinforcement Learning · University of Ghana,
> Department of Computer Science · Programming Assignment 1.

---

## 1. Introduction

### What is Reinforcement Learning?

Reinforcement Learning (RL) is a branch of machine learning in which an **agent**
learns to make decisions by **interacting** with an **environment**. At each time
step the agent observes a **state**, selects an **action**, and receives a
numerical **reward** plus the **next state**. The agent is never told the correct
action; instead it must discover, through trial and error, a **policy** (a mapping
from states to actions) that maximises the expected sum of (discounted) future
rewards. This trial-and-error loop, balancing the need to **explore** unfamiliar
actions against the desire to **exploit** what already looks good, is what
distinguishes RL from supervised learning.

### What is Frozen Lake?

Frozen Lake is a classic grid-world benchmark. An agent must cross a frozen lake
from a **Start (S)** cell to a **Goal (G)** cell, walking only on **Frozen (F)**
tiles and avoiding **Holes (H)**, which end the episode in failure. This project
uses the standard 8×8 map:

```
S F F F F F F F
F F F F F F F F
F F F H F F F F
F F F H F F F F
F F F H F F F F
F H H F F F H F
F H F F H F H F
F F F H F F F G
```

The agent starts at the top-left and must learn a policy that reaches the
bottom-right goal while avoiding the ten holes.

---

## 2. Environment Design

The environment is implemented as the `FrozenLakeEnv` class in
[`environment.py`](environment.py).

### State representation

States are encoded as a **single integer index** computed from the grid
coordinates:

```
state = row * n_cols + col        # 0 .. 63 for an 8x8 grid
```

Helper methods `to_state(row, col)` and `to_rc(state)` convert between the two
representations.

### Action representation

| Action | Meaning |
|:------:|:--------|
| 0 | Left  |
| 1 | Down  |
| 2 | Right |
| 3 | Up    |

An action that would move the agent off the grid leaves it in place (it bumps
into the wall — movement boundaries are enforced).

### Reward structure

The classic **sparse** reward is used:

| Event | Reward | Episode ends? |
|:------|:------:|:-------------:|
| Reach the Goal (G) | **+1.0** | yes (success) |
| Fall into a Hole (H) | 0.0 | yes (failure) |
| Ordinary Frozen step (F) | 0.0 | no |

Because the discount factor γ < 1, an early +1 is worth more than a late one, so
this sparse reward already drives the agent toward the **shortest safe path**
without any hand-crafted shaping.

The environment also supports an optional **slippery** mode (stochastic
transitions) used for the bonus task — see below.

---

## 3. Q-Learning Algorithm

The agent is implemented as the `QLearningAgent` class in [`agent.py`](agent.py).

### Description

Q-Learning is a **model-free, off-policy, temporal-difference** control
algorithm. It learns an **action-value function** `Q(s, a)`, the expected
discounted return of taking action `a` in state `s` and then acting greedily
thereafter. The values are stored in a table of shape `(n_states, n_actions)`,
initialised to zeros. Once `Q` is accurate, the optimal policy is simply
`π(s) = argmax_a Q(s, a)`.

### The update equation

After every transition `(s, a, r, s')` the agent applies exactly:

```
Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

- **α (learning rate)** — how much each new experience overwrites the old estimate.
- **γ (discount factor)** — how much future rewards count relative to immediate ones.
- **r + γ · max_a' Q(s', a')** — the *TD target*, a one-step bootstrap estimate of the true value.
- The bracketed quantity is the *TD error*; for a terminal `s'` the bootstrap term is dropped (there is no future).

### Exploration strategy

Actions are chosen with an **ε-greedy** policy: with probability ε a random
action is taken (exploration), otherwise the greedy action `argmax_a Q(s, a)` is
taken (exploitation). ε starts high (1.0) and is **decayed exponentially** after
each episode toward a small floor (0.01):

```
ε ← max(ε_min, ε · ε_decay)
```

This explores the lake aggressively early on, then increasingly commits to the
best known route.

---

## 4. Training Procedure

Training is driven by [`train.py`](train.py).

### Hyperparameters used

| Hyperparameter | Value |
|:---------------|:-----:|
| Episodes | 20,000 |
| Learning rate (α) | 0.1 |
| Discount factor (γ) | 0.99 |
| Initial ε | 1.0 |
| Minimum ε | 0.01 |
| ε decay (per episode) | 0.9995 |
| Max steps per episode | 200 |
| Random seed | 42 |

These were chosen after the hyperparameter study in
[`experiments.py`](experiments.py); see the report for details.

---

## 5. Results

### Final success rate

Evaluated greedily (ε = 0) over 200 episodes on the deterministic map:

| Metric | Value |
|:-------|:-----:|
| **Success rate** | **100.0 %** |
| Average reward | 1.000 |
| Successful runs | 200 / 200 |
| Failures | 0 |
| Steps to goal | 14 (optimal) |

The 14-step greedy path equals the Manhattan distance from S(0,0) to G(7,7),
so the agent learned a **provably optimal** route.

### Learned policy

```
↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
→ → → → ↓ ↓ ↓ ↓
↑ ↑ ↑ H → → ↓ ↓
↑ ↑ ↑ H → → ↓ ↓
↑ ↑ ↑ H → → → ↓
↑ H H → → ↑ H ↓
↑ H ← ← H ↓ H ↓
← ← ← H ← → → G
```

The greedy trajectory runs **down** out of the start, **right** across row 1 to
avoid the wall of holes in column 3, then **down the right-hand side** to the
goal. (Arrows in cells far from the optimal path are not meaningful — those
states are rarely visited, so their values stay near zero.)

Visual results saved in [`results/`](results/):

| File | Shows |
|:-----|:------|
| `training_curves.png` | success rate, reward and ε over training |
| `learned_policy.png` | policy arrows + state-value heatmap + optimal path |
| `hyperparameter_study.png` | convergence speed vs α and γ |
| `exploration_comparison.png` | pure vs decaying ε-greedy |
| `slippery_comparison.png` | deterministic vs slippery dynamics |

### Discussion of performance

On the deterministic lake the agent reaches a **perfect 100 % success rate** and
an **optimal 14-step policy**. The learning curve shows success climbing from
near-zero (random walking) to ~100 % as ε decays, typically crossing the 90 %
mark around **3,500 episodes**. Every reasonable hyperparameter setting solved
the task; the differences appear in *convergence speed* rather than final
success. On the **slippery** variant the same algorithm reaches roughly **87 %**
— the remaining failures are unavoidable, because random slips can push even an
optimal policy into a hole.

---

## 6. Bonus Tasks

All three optional tasks are implemented (`experiments.py` / `environment.py`):

- **Option A — Stochastic transitions.** `FrozenLakeEnv(is_slippery=True)` adds
  classic slippery dynamics: the intended move happens with probability 1/3, and
  with 1/3 each the agent slips to a perpendicular direction.
- **Option B — Visualisation.** Training performance is plotted as learning
  curves and comparison charts (see `results/`).
- **Option C — Exploration comparison.** Pure ε-greedy (fixed ε = 0.1) vs
  decaying ε-greedy (1.0 → 0.01) are trained and compared.

---

## 7. Execution Instructions

```bash
# 1. (optional) create a virtual environment
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. train the agent (saves Q-table, stats and learning-curve figure)
python train.py --episodes 20000

# 4. evaluate the trained agent over >= 100 episodes
python evaluate.py --episodes 100

# 5. (optional) render the policy / value heatmap figure
python visualize_policy.py

# 6. (optional) run the full hyperparameter study + bonus experiments
python experiments.py

# train / evaluate the slippery (stochastic) variant instead
python train.py --slippery --episodes 30000
python evaluate.py --slippery --episodes 1000
```

All output artefacts (Q-table, statistics, figures) are written to `results/`.

---

## Repository structure

```
frozen-lake-qlearning/
├── environment.py        # FrozenLakeEnv: grid, dynamics, rewards, rendering
├── agent.py              # QLearningAgent: Q-table, ε-greedy, TD update
├── train.py              # training loop + learning-curve plots
├── evaluate.py           # greedy evaluation + metrics
├── visualize_policy.py   # policy / value heatmap figure
├── experiments.py        # hyperparameter study + bonus comparisons
├── utils.py              # shared helpers (training loop, smoothing, policy grid)
├── requirements.txt
├── README.md
├── report.pdf            # technical report (≤ 5 pages)
└── results/              # saved Q-table, statistics and figures
```
