#!/usr/bin/env python3
"""Visualize Jack's Car Rental optimal policy and value (Sutton Fig 4.2 style)."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

ROOT = Path(__file__).resolve().parent


def load_matrix(path: Path) -> np.ndarray:
    rows = []
    with path.open() as f:
        for row in csv.reader(f):
            rows.append([float(x) for x in row])
    return np.array(rows)


def main() -> None:
    policy = load_matrix(ROOT / "policy.csv")
    value = load_matrix(ROOT / "value.csv")

    # Book-style orientation: y-axis = cars at loc1 (A), increasing upward
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # --- Policy heatmap ---
    ax = axes[0]
    vmax = 5
    cmap = plt.get_cmap("RdBu_r")
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(
        policy,
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(r"Optimal policy $\pi_*$")
    ax.set_xlabel("Cars at location 2 (B)")
    ax.set_ylabel("Cars at location 1 (A)")
    ax.set_xticks(range(0, 21, 5))
    ax.set_yticks(range(0, 21, 5))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=range(-5, 6))
    cbar.set_label("cars moved (A→B positive)")

    # Annotate a subset of cells for readability
    for i in range(0, 21, 2):
        for j in range(0, 21, 2):
            a = int(policy[i, j])
            ax.text(j, i, f"{a}", ha="center", va="center", fontsize=6, color="black")

    # --- Value surface as heatmap ---
    ax = axes[1]
    im2 = ax.imshow(
        value,
        origin="lower",
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(r"State-value function $V_*$")
    ax.set_xlabel("Cars at location 2 (B)")
    ax.set_ylabel("Cars at location 1 (A)")
    ax.set_xticks(range(0, 21, 5))
    ax.set_yticks(range(0, 21, 5))
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label("expected return")

    fig.suptitle("HW1.3 Jack's Car Rental — Policy Iteration (γ=0.9)", fontsize=13)
    fig.tight_layout()
    out = ROOT / "policy_value_fig.png"
    fig.savefig(out, dpi=160)
    print(f"Wrote {out}")

    # Terminal-style result image for submission screenshot requirement
    result_text = (ROOT / "policy_result.txt").read_text()
    # Keep policy matrix + header for a compact screenshot
    lines = result_text.splitlines()
    # header through end of policy matrix (before blank + value header)
    cut = 0
    for i, line in enumerate(lines):
        if line.startswith("State-value"):
            cut = i
            break
    snippet = "\n".join(lines[:cut]).rstrip() + "\n"

    fig2, ax2 = plt.subplots(figsize=(11, 9))
    ax2.axis("off")
    ax2.text(
        0.01,
        0.99,
        snippet,
        family="monospace",
        fontsize=7.5,
        va="top",
        ha="left",
        transform=ax2.transAxes,
    )
    fig2.tight_layout()
    out2 = ROOT / "policy_result.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
