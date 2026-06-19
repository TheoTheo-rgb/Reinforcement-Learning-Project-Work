"""
agent.py
========

A tabular Q-Learning agent implemented from scratch.

The agent stores a Q-table of shape (n_states, n_actions) and learns through
the standard temporal-difference update:

        Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

Exploration uses an epsilon-greedy policy with exponential epsilon decay so
that the agent explores aggressively early on and increasingly exploits its
learned values as training progresses.
"""

from __future__ import annotations

import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent with an epsilon-greedy behaviour policy.

    Parameters
    ----------
    n_states, n_actions : int
        Size of the state and action spaces.
    alpha : float
        Learning rate (step size) in (0, 1].
    gamma : float
        Discount factor in [0, 1).
    epsilon : float
        Initial exploration probability.
    epsilon_min : float
        Lower bound for epsilon during decay.
    epsilon_decay : float
        Multiplicative decay factor applied to epsilon after each episode
        (epsilon <- max(epsilon_min, epsilon * epsilon_decay)). Set to 1.0 to
        keep epsilon fixed (a pure epsilon-greedy strategy).
    seed : int, optional
        Seed for the agent's random number generator (tie-breaking and
        exploration), kept separate from the environment's RNG.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        seed: int | None = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)

        # Q-table initialised to zeros: optimistic enough for a sparse +1 reward
        # and the natural starting point "the agent knows nothing".
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def select_action(self, state: int, greedy: bool = False) -> int:
        """Choose an action for `state`.

        If `greedy` is True (used at evaluation time) the best known action is
        always returned. Otherwise an epsilon-greedy choice is made: with
        probability epsilon a uniformly-random action is taken (exploration),
        and otherwise the greedy action is taken (exploitation).
        """
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self._greedy_action(state)

    def _greedy_action(self, state: int) -> int:
        """Return argmax_a Q(state, a), breaking ties uniformly at random."""
        q_values = self.q_table[state]
        best = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best))

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Apply one Q-Learning update for the observed transition.

        Implements exactly:

            Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

        For a terminal `next_state` there is no future, so the bootstrap term
        gamma * max_a' Q(s', a') is dropped (treated as zero).
        """
        best_next = 0.0 if done else float(self.q_table[next_state].max())
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Decay epsilon towards its minimum (no effect if decay == 1.0)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    # Policy / value extraction & persistence
    # ------------------------------------------------------------------ #
    def get_policy(self) -> np.ndarray:
        """Return the greedy action for every state as a 1-D array."""
        return np.argmax(self.q_table, axis=1)

    def get_state_values(self) -> np.ndarray:
        """Return V(s) = max_a Q(s, a) for every state."""
        return np.max(self.q_table, axis=1)

    def save(self, path: str) -> None:
        """Persist the Q-table to a .npy file."""
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        """Load a Q-table previously saved with `save`."""
        self.q_table = np.load(path)
