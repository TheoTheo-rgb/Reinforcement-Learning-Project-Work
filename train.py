"""
train.py
========

Train the Q-Learning agent on the Frozen Lake environment.

Running this script:
    1. builds the environment and agent (hyperparameters are CLI-configurable),
    2. trains for the requested number of episodes,
    3. saves the learned Q-table to results/q_table.npy,
    4. saves the training statistics to results/training_stats.npz,
    5. writes a learning-curve figure to results/training_curves.png,
    6. prints the learned policy and a short evaluation summary.

Example
-------
    python train.py --episodes 20000 --alpha 0.1 --gamma 0.99
    python train.py --slippery            # train on the stochastic variant
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless backend, safe on any machine/server
import matplotlib.pyplot as plt

from environment import FrozenLakeEnv
from agent import QLearningAgent
from utils import run_training, policy_to_grid, evaluate_policy, moving_average

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Q-Learning on Frozen Lake.")
    p.add_argument("--episodes", type=int, default=20000, help="training episodes")
    p.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    p.add_argument("--epsilon", type=float, default=1.0, help="initial epsilon")
    p.add_argument("--epsilon-min", type=float, default=0.01, help="minimum epsilon")
    p.add_argument("--epsilon-decay", type=float, default=0.9995, help="epsilon decay")
    p.add_argument("--max-steps", type=int, default=200, help="max steps per episode")
    p.add_argument("--slippery", action="store_true", help="use slippery dynamics")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    return p.parse_args()


def plot_training_curves(stats: dict, path: str, window: int = 200) -> None:
    """Save a 3-panel figure: success rate, episode reward and epsilon."""
    rewards = stats["rewards"]
    successes = stats["successes"]
    epsilons = stats["epsilons"]
    episodes = np.arange(1, len(rewards) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    succ_ma = moving_average(successes, window) * 100
    axes[0].plot(np.arange(len(succ_ma)) + window, succ_ma, color="#1f77b4")
    axes[0].set_title(f"Success rate (moving avg, window={window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_ylim(-2, 102)
    axes[0].grid(alpha=0.3)

    rew_ma = moving_average(rewards, window)
    axes[1].plot(np.arange(len(rew_ma)) + window, rew_ma, color="#2ca02c")
    axes[1].set_title(f"Episode reward (moving avg, window={window})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average reward")
    axes[1].grid(alpha=0.3)

    axes[2].plot(episodes, epsilons, color="#d62728")
    axes[2].set_title("Exploration rate (epsilon) over time")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = FrozenLakeEnv(
        is_slippery=args.slippery,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    print("=" * 64)
    print("Training Q-Learning agent on Frozen Lake")
    print("=" * 64)
    print(f"  Map size      : {env.n_rows}x{env.n_cols}  ({env.n_states} states)")
    print(f"  Slippery ice  : {args.slippery}")
    print(f"  Episodes      : {args.episodes}")
    print(f"  alpha (lr)    : {args.alpha}")
    print(f"  gamma         : {args.gamma}")
    print(f"  epsilon       : {args.epsilon} -> {args.epsilon_min} "
          f"(decay {args.epsilon_decay})")
    print("-" * 64)

    stats = run_training(env, agent, args.episodes, log_every=max(1, args.episodes // 10))

    # Persist artefacts.
    q_path = os.path.join(RESULTS_DIR, "q_table.npy")
    stats_path = os.path.join(RESULTS_DIR, "training_stats.npz")
    fig_path = os.path.join(RESULTS_DIR, "training_curves.png")
    agent.save(q_path)
    np.savez(stats_path, **stats)
    plot_training_curves(stats, fig_path)

    print("-" * 64)
    print("Learned policy (\u2190 \u2193 \u2192 \u2191, H = hole, G = goal):\n")
    print(policy_to_grid(env, agent.get_policy()))

    metrics = evaluate_policy(env, agent, n_episodes=100)
    print("\nGreedy evaluation over 100 episodes:")
    print(f"  Success rate : {metrics['success_rate']:.1f}%")
    print(f"  Avg reward   : {metrics['avg_reward']:.3f}")
    print(f"  Successes    : {metrics['successes']}")
    print(f"  Failures     : {metrics['failures']}")
    if not np.isnan(metrics["avg_steps_on_success"]):
        print(f"  Avg steps to goal (successful runs): "
              f"{metrics['avg_steps_on_success']:.1f}")

    print("-" * 64)
    print(f"Saved Q-table   -> {q_path}")
    print(f"Saved stats     -> {stats_path}")
    print(f"Saved curves    -> {fig_path}")


if __name__ == "__main__":
    main()
