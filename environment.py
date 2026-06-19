"""
environment.py
==============

A from-scratch implementation of the 8x8 Frozen Lake grid-world environment.

No Reinforcement Learning frameworks (Gymnasium, OpenAI Gym, Stable Baselines,
RLlib, ...) are used. The grid, dynamics, reward function and termination logic
are all implemented in plain Python / NumPy.

State representation
--------------------
States are encoded as a single integer index:

        state = row * n_cols + col

so for an 8x8 grid the states are 0 .. 63. Helper methods convert between the
integer index and (row, col) coordinates.

Action representation
---------------------
        0 = Left
        1 = Down
        2 = Right
        3 = Up

Reward structure (default, "classic" sparse reward)
---------------------------------------------------
        +1.0   for reaching the Goal (G)
         0.0   for falling into a Hole (H)   -> episode ends in failure
         0.0   for every ordinary (Frozen) step

With a discount factor gamma < 1 this sparse reward already pushes the agent
towards the *shortest* safe path, because a reward of +1 received sooner is
worth more than the same reward received later.
"""

from __future__ import annotations

import numpy as np


# The standard 8x8 map specified in the assignment.
DEFAULT_MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

# Human-readable action names and the arrow glyphs used when printing a policy.
ACTION_NAMES = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
ACTION_ARROWS = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}  # ← ↓ → ↑

# Row/column displacement produced by each action.
#   Left  -> column - 1
#   Down  -> row    + 1
#   Right -> column + 1
#   Up    -> row    - 1
ACTION_DELTAS = {
    0: (0, -1),   # Left
    1: (1, 0),    # Down
    2: (0, 1),    # Right
    3: (-1, 0),   # Up
}


class FrozenLakeEnv:
    """A deterministic (optionally slippery) Frozen Lake environment.

    Parameters
    ----------
    grid_map : list[str], optional
        The map as a list of equal-length strings using the characters
        S (start), F (frozen), H (hole) and G (goal). Defaults to the
        standard 8x8 assignment map.
    is_slippery : bool, optional
        If True the ice is slippery: an action only moves the agent in the
        intended direction with probability 1/3, and with probability 1/3 each
        in the two perpendicular directions (the classic Frozen Lake stochastic
        dynamics). Defaults to False (deterministic).
    reward_goal, reward_hole, reward_step : float, optional
        The reward returned when reaching the goal, falling into a hole, or
        taking an ordinary step. Defaults to the classic sparse reward
        (+1 / 0 / 0).
    max_steps : int, optional
        Maximum number of steps before an episode is truncated. Prevents the
        agent from wandering forever during early exploration. Defaults to 200.
    seed : int, optional
        Seed for the internal random number generator (used only when slippery).
    """

    def __init__(
        self,
        grid_map: list[str] | None = None,
        is_slippery: bool = False,
        reward_goal: float = 1.0,
        reward_hole: float = 0.0,
        reward_step: float = 0.0,
        max_steps: int = 200,
        seed: int | None = None,
    ) -> None:
        self.grid = [list(row) for row in (grid_map or DEFAULT_MAP)]
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])

        # Validate that every row has the same length.
        if any(len(row) != self.n_cols for row in self.grid):
            raise ValueError("All rows in the map must have the same length.")

        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4

        self.is_slippery = is_slippery
        self.reward_goal = reward_goal
        self.reward_hole = reward_hole
        self.reward_step = reward_step
        self.max_steps = max_steps

        self.rng = np.random.default_rng(seed)

        # Locate the start cell (defaults to (0, 0) if no 'S' is present).
        self.start_state = 0
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r][c] == "S":
                    self.start_state = self.to_state(r, c)

        # Episode bookkeeping, initialised by reset().
        self.state = self.start_state
        self.steps = 0

    # ------------------------------------------------------------------ #
    # Coordinate <-> integer-index helpers
    # ------------------------------------------------------------------ #
    def to_state(self, row: int, col: int) -> int:
        """Convert (row, col) coordinates into a single integer state index."""
        return row * self.n_cols + col

    def to_rc(self, state: int) -> tuple[int, int]:
        """Convert an integer state index back into (row, col) coordinates."""
        return divmod(state, self.n_cols)

    def cell(self, state: int) -> str:
        """Return the map character ('S'/'F'/'H'/'G') at a given state."""
        r, c = self.to_rc(state)
        return self.grid[r][c]

    # ------------------------------------------------------------------ #
    # Core environment API
    # ------------------------------------------------------------------ #
    def reset(self) -> int:
        """Reset the agent to the start state and return that state."""
        self.state = self.start_state
        self.steps = 0
        return self.state

    def get_state(self) -> int:
        """Return the agent's current state index."""
        return self.state

    def is_terminal(self, state: int | None = None) -> bool:
        """Return True if `state` (default: current state) is a hole or goal."""
        s = self.state if state is None else state
        return self.cell(s) in ("H", "G")

    def _move(self, state: int, action: int) -> int:
        """Apply a single deterministic move, clipping at the grid boundary."""
        r, c = self.to_rc(state)
        dr, dc = ACTION_DELTAS[action]
        # Enforce movement boundaries: an attempt to leave the grid keeps the
        # agent in place (it bumps into the wall).
        nr = min(max(r + dr, 0), self.n_rows - 1)
        nc = min(max(c + dc, 0), self.n_cols - 1)
        return self.to_state(nr, nc)

    def _slippery_action(self, action: int) -> int:
        """Sample the actually-executed action under slippery dynamics.

        With probability 1/3 the intended action is taken; with probability
        1/3 each the agent slips to one of the two perpendicular actions.
        """
        if action in (0, 2):           # intended move is horizontal (Left/Right)
            candidates = [action, 1, 3]  # -> can slip Up or Down
        else:                          # intended move is vertical (Up/Down)
            candidates = [action, 0, 2]  # -> can slip Left or Right
        return int(self.rng.choice(candidates))

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        """Take one step in the environment.

        Parameters
        ----------
        action : int
            One of 0 (Left), 1 (Down), 2 (Right), 3 (Up).

        Returns
        -------
        next_state : int
            The resulting state index.
        reward : float
            The reward obtained on this transition.
        done : bool
            True if the episode has ended (goal reached, hole entered, or the
            step limit was hit).
        info : dict
            Diagnostic information, e.g. {'success': True, 'truncated': False}.
        """
        if action not in ACTION_DELTAS:
            raise ValueError(f"Invalid action {action!r}; must be 0, 1, 2 or 3.")

        # If we somehow step from a terminal state, just report it as done.
        if self.is_terminal(self.state):
            return self.state, 0.0, True, {"success": self.cell(self.state) == "G"}

        executed = self._slippery_action(action) if self.is_slippery else action
        self.state = self._move(self.state, executed)
        self.steps += 1

        cell = self.cell(self.state)
        if cell == "G":
            reward, done, success = self.reward_goal, True, True
        elif cell == "H":
            reward, done, success = self.reward_hole, True, False
        else:
            reward, done, success = self.reward_step, False, False

        truncated = False
        if not done and self.steps >= self.max_steps:
            done, truncated = True, True

        info = {"success": success, "truncated": truncated, "executed_action": executed}
        return self.state, reward, done, info

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self, mode: str = "human") -> str:
        """Render the grid with the agent's position marked by 'A'.

        When the agent stands on the start cell it is shown as 'A'. The
        rendered string is both printed (mode='human') and returned.
        """
        ar, ac = self.to_rc(self.state)
        lines = []
        for r in range(self.n_rows):
            row_chars = []
            for c in range(self.n_cols):
                if (r, c) == (ar, ac):
                    row_chars.append("A")          # the agent
                else:
                    row_chars.append(self.grid[r][c])
            lines.append(" ".join(row_chars))
        out = "\n".join(lines)
        if mode == "human":
            print(out)
            print()
        return out


if __name__ == "__main__":
    # Tiny smoke-test: walk the deterministic top row then down the right
    # column, which is the optimal 14-step path from S(0,0) to G(7,7).
    env = FrozenLakeEnv()
    state = env.reset()
    print("Initial grid (A = agent):")
    env.render()

    optimal = [2] * 7 + [1] * 7          # Right x7, Down x7
    total = 0.0
    for a in optimal:
        state, reward, done, info = env.step(a)
        total += reward
        if done:
            break
    print(f"Reached state {state} ({env.cell(state)}), "
          f"total reward = {total}, success = {info['success']}")
