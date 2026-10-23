# Run this once in a separate Colab cell before executing the script:
# !pip -q install pandas matplotlib

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files

# ====== Fixed settings ======
EXITS = ["E1", "E6", "E7"]

# Area around each exit (m²)
EXIT_AREAS = {
    "E1": 10.3,
    "E6": 7.0,
    "E7": 8.31,
}


def read_nonnegative_float(prompt: str) -> float:
    """Read a numeric value that cannot be negative."""
    while True:
        try:
            value = float(input(prompt))
            if value < 0:
                print("   ⚠️ The value cannot be negative.")
                continue
            return value
        except ValueError:
            print("   ⚠️ Invalid number. Please enter a numeric value.")


def read_positive_int(prompt: str) -> int:
    """Read an integer greater than zero."""
    while True:
        try:
            value = int(input(prompt))
            if value <= 0:
                print("   ⚠️ The value must be at least 1.")
                continue
            return value
        except ValueError:
            print("   ⚠️ Invalid integer.")


# ====== Scenario name and output path ======
scenario_name = input("\nScenario name: ").strip() or "unnamed_scenario"
safe_scenario_name = re.sub(r"[^A-Za-z0-9_-]+", "_", scenario_name).strip("_")
safe_scenario_name = safe_scenario_name or "unnamed_scenario"

output_path = f"/content/{safe_scenario_name}_exit_density_by_stage.png"

# ====== Initial exit populations: manual input ======
print("\nEnter the initial number of people at each exit:")
people_dynamic = {
    ex: read_nonnegative_float(f"Initial number of people at {ex}: ")
    for ex in EXITS
}


def get_density(exit_id: str) -> float:
    """Current exit density in person/m²."""
    return people_dynamic[exit_id] / EXIT_AREAS[exit_id]


# ====== Groups and travel times ======
num_starts = read_positive_int("\nHow many starting groups will be evacuated? ")

start_nodes = []
people_at_start = {}
time_by_start_exit = {}

for i in range(1, num_starts + 1):
    print("\n" + "=" * 55)
    print(f"Group input {i}")

    while True:
        start_node = input("Start-node name (for example, N27): ").strip()
        if not start_node:
            print("   ⚠️ Start-node name cannot be empty.")
            continue
        if start_node in start_nodes:
            print("   ⚠️ This start node has already been entered.")
            continue
        start_nodes.append(start_node)
        break

    people_at_start[start_node] = read_nonnegative_float(
        f"Number of people at {start_node}: "
    )

    print(f"\nEnter only the travel time from {start_node} to each exit:")
    for exit_id in EXITS:
        time_by_start_exit[(start_node, exit_id)] = read_nonnegative_float(
            f"Travel time {start_node} → {exit_id} (s): "
        )

# Largest groups are assigned first, exactly as in the previous logic.
start_nodes_sorted = sorted(
    start_nodes,
    key=lambda node: people_at_start[node],
    reverse=True,
)

# Stage 0: before assigning any group.
density_history = [
    {
        "stage": 0,
        **{exit_id: get_density(exit_id) for exit_id in EXITS},
    }
]

# ====== Time-only route selection ======
for stage, start_node in enumerate(start_nodes_sorted, start=1):
    chosen_exit = min(
        EXITS,
        key=lambda exit_id: time_by_start_exit[(start_node, exit_id)] / 60.0,
    )

    people_dynamic[chosen_exit] += people_at_start[start_node]

    density_history.append(
        {
            "stage": stage,
            **{exit_id: get_density(exit_id) for exit_id in EXITS},
        }
    )

# ====== The only output: exit-density change chart ======
history_df = pd.DataFrame(density_history)

fig, ax = plt.subplots(figsize=(10.5, 5.8))

line_colors = {
    "E1": "#1f77b4",
    "E6": "#2ca02c",
    "E7": "#9467bd",
}
markers = {
    "E1": "o",
    "E6": "s",
    "E7": "D",
}

for exit_id in EXITS:
    ax.step(
        history_df["stage"],
        history_df[exit_id],
        where="post",
        marker=markers[exit_id],
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linewidth=2.8,
        color=line_colors[exit_id],
        label=rf"$\rho_{{{exit_id}}}$",
        zorder=3,
    )

    final_x = history_df["stage"].iloc[-1]
    final_y = history_df[exit_id].iloc[-1]
    ax.annotate(
        rf"$\rho_{{{exit_id}}}$ = {final_y:.2f}",
        xy=(final_x, final_y),
        xytext=(10, 0),
        textcoords="offset points",
        va="center",
        fontsize=11,
        color=line_colors[exit_id],
    )

ax.set_xlabel("Stage", fontsize=13)
ax.set_ylabel(r"Exit Density (person / m$^2$)", fontsize=13)
ax.set_title("Stage-by-Stage Exit Density Changes", fontsize=14, pad=12)
ax.set_xticks(history_df["stage"])
ax.grid(axis="y", linestyle="--", alpha=0.30)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=11, loc="best")
ax.margins(x=0.08, y=0.12)
fig.tight_layout()

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\n✅ Chart saved: {output_path}")

# Automatically download the only output file.
files.download(output_path)
