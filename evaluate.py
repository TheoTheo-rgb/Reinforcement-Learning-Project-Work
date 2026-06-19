"""
evaluate.py
===========

Evaluate a trained Q-Learning agent.

Loads the Q-table saved by train.py (results/q_table.npy), runs the greedy
policy for a number of episodes (>= 100 by default) and reports:

    * Success Rate (%)
    * Average Reward
    * Number of Failures
    * Number of Successful Runs

It also prints the extracted policy as an arrow grid.

Example
-------
    python evaluate.py --episodes 100
    python evaluate.py --slippery --episodes 500
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from environment import FrozenLakeEnv
from agent import QLearningAgent
from utils import policy_to_grid, evaluate_policy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Q-Learning agent.")
    p.add_argument("--episodes", type=int, default=100, help="evaluation episodes")
    p.add_argument("--q-table", type=str,
                   default=os.path.join(RESULTS_DIR, "q_table.npy"),
                   help="path to the saved Q-table (.npy)")
    p.add_argument("--max-steps", type=int, default=200, help="max steps per episode")
    p.add_argument("--slippery", action="store_true",
                   help="evaluate on the slippery variant (match training)")
    p.add_argument("--seed", type=int, default=123, help="random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.q_table):
        raise FileNotFoundError(
            f"Q-table not found at {args.q_table}. Run train.py first."
        )

    env = FrozenLakeEnv(is_slippery=args.slippery, max_steps=args.max_steps,
                        seed=args.seed)
    agent = QLearningAgent(env.n_states, env.n_actions, seed=args.seed)
    agent.load(args.q_table)

    print("=" * 64)
    print(f"Evaluating trained agent over {args.episodes} episodes")
    print(f"  (slippery = {args.slippery})")
    print("=" * 64)

    print("\nExtracted policy (\u2190 \u2193 \u2192 \u2191, H = hole, G = goal):\n")
    print(policy_to_grid(env, agent.get_policy()))

    metrics = evaluate_policy(env, agent, n_episodes=args.episodes, render_first=True)

    print("\n" + "-" * 64)
    print("RESULTS")
    print("-" * 64)
    print(f"  Episodes        : {metrics['n_episodes']}")
    print(f"  Success Rate    : {metrics['success_rate']:.1f}%")
    print(f"  Average Reward  : {metrics['avg_reward']:.3f}")
    print(f"  Successful Runs : {metrics['successes']}")
    print(f"  Failures        : {metrics['failures']}")
    if not np.isnan(metrics["avg_steps_on_success"]):
        print(f"  Avg steps to goal (successful runs): "
              f"{metrics['avg_steps_on_success']:.1f}")
    print("-" * 64)


if __name__ == "__main__":
    main()
