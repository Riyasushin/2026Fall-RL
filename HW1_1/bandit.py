"""Plot epsilon-greedy learning curves for four exploration rates."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import matplotlib

# Allow the script to run on a headless machine while still supporting --show.
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


class NormalDistBandit:
    def __init__(self, means, stds):
        assert len(means) == len(stds), "Means and stds must be the same length."
        self.n = len(means)
        self.means = np.array(means)
        self.stds = np.array(stds)
        assert all(self.stds >= 0), "Stds must be positive."

    def pull(self, k):
        assert 0 <= k < self.n, f"Invalid arm {k}."
        return np.random.normal(loc=self.means[k], scale=self.stds[k])


def epsilon_greedy(values, epsilon):
    """Select greedily, with the exploration rule from the starter code."""
    assert len(values) > 1, "There should be 2 or more values."
    # Keep the starter code's exploration-probability correction.
    eps = epsilon * len(values) / (len(values) - 1)
    if random.random() <= eps:
        return random.randint(0, len(values) - 1)
    return int(np.argmax(values))


def run_bandit(epsilon, bandit, iterations):
    """Return the cumulative average reward for one epsilon value."""
    values = np.zeros(bandit.n, dtype=np.float64)
    counts = np.zeros(bandit.n, dtype=np.int64)
    average_reward = np.zeros(iterations, dtype=np.float64)

    for i in range(1, iterations):
        action = epsilon_greedy(values, epsilon)
        counts[action] += 1
        reward = bandit.pull(action)
        values[action] = (
            values[action] * (counts[action] - 1) + reward
        ) / counts[action]
        average_reward[i] = (
            average_reward[i - 1] * (i - 1) + reward
        ) / i
    return average_reward


def write_summary(path, curves, iterations):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["epsilon", "final_average_reward", "last_1000_average"])
        for epsilon, curve in curves.items():
            tail = curve[max(1, iterations - 1000):]
            writer.writerow(
                [epsilon, f"{curve[-1]:.6f}", f"{tail.mean():.6f}"]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the plot window after saving it (not available headlessly).",
    )
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("--iterations must be at least 2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n = 5
    bandit = NormalDistBandit(
        means=np.arange(-n, n + 1), stds=np.ones(11, dtype=np.float64)
    )
    epsilons = [0.01, 0.05, 0.1, 0.2]

    curves = {}
    for epsilon in epsilons:
        curves[epsilon] = run_bandit(epsilon, bandit, args.iterations)
        tail = curves[epsilon][max(1, args.iterations - 1000):]
        print(
            f"epsilon={epsilon:.2f}  final={curves[epsilon][-1]:.4f}  "
            f"last-1000={tail.mean():.4f}"
        )

    figure, axis = plt.subplots(figsize=(8.2, 4.7), dpi=160)
    for epsilon, curve in curves.items():
        axis.plot(
            np.arange(args.iterations),
            curve,
            linewidth=1.35,
            label=fr"$\epsilon={epsilon:.2f}$",
        )
    axis.axhline(5.0, color="0.35", linestyle="--", linewidth=0.9,
                 label="optimal mean = 5")
    axis.set_xlabel("Iterations")
    axis.set_ylabel("Cumulative average reward")
    axis.set_title("Epsilon-greedy on an 11-arm Gaussian bandit")
    axis.set_xlim(0, args.iterations - 1)
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()

    plot_path = args.output_dir / "epsilon_curves.png"
    figure.savefig(plot_path, bbox_inches="tight", facecolor="white")
    summary_path = args.output_dir / "epsilon_summary.csv"
    write_summary(summary_path, curves, args.iterations)
    print(f"Saved plot: {plot_path}")
    print(f"Saved summary: {summary_path}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
