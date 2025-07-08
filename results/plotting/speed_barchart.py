import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -----------------------------------------------------------
# 1. Data
# -----------------------------------------------------------
df = pd.DataFrame({
    "Benchmark": [
        "Continual World",
        "CORA (Minihack)",
        "COOX"
    ],
    "Hours per 1e7 timesteps": [
        50,          # Continual World
        1.5,         # CORA
        10/60        # COOX  (10 minutes → 0.1667 h)
    ]
})

# -----------------------------------------------------------
# 2. Plot
# -----------------------------------------------------------
colors = ['#4E79A7', '#F28E2B', '#59A14F']   # pick any palette you like

fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(
    x=df["Benchmark"],
    height=df["Hours per 1e7 timesteps"],
    color=colors,
    width=0.5,
    log=True
)

# axis styling
ax.set_xlabel("Benchmark", fontweight='bold')
ax.set_ylabel(r"Hours per $\mathbf{10^{7}}$ timesteps", fontweight='bold')
# ax.set_title("Average wall-clock time per $10^{7}$ timesteps")
# plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

# # Legend
# legend_handles = [Patch(facecolor=colors[i], label=df["Benchmark"][i])
#                   for i in range(len(df))]
# ax.legend(handles=legend_handles)

plt.tight_layout()

# -----------------------------------------------------------
# 3. Save
# -----------------------------------------------------------
fig.savefig("wall_clock_time_per_1e7_steps_coloured.png",
            dpi=300, bbox_inches="tight")

plt.show()