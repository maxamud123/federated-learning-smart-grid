"""
Plot FL experiment comparison: Baseline vs Optimized (Top-K compression)
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Data from experiment runs ──────────────────────────────────────────────────
rounds = list(range(1, 51))

baseline_rmse = [
    0.5728, 0.5569, 0.5481, 0.5505, 0.5468, 0.5435, 0.5422, 0.5458, 0.5408, 0.5424,
    0.5374, 0.5438, 0.5416, 0.5414, 0.5411, 0.5451, 0.5418, 0.5396, 0.5439, 0.5422,
    0.5477, 0.5468, 0.5442, 0.5475, 0.5565, 0.5523, 0.5551, 0.5541, 0.5586, 0.5604,
    0.5583, 0.5636, 0.5605, 0.5619, 0.5690, 0.5688, 0.5704, 0.5688, 0.5789, 0.5808,
    0.5735, 0.5792, 0.5766, 0.5761, 0.5769, 0.5851, 0.5883, 0.5856, 0.5886, 0.5906
]

baseline_mae = [
    0.4194, 0.3865, 0.3836, 0.3828, 0.3774, 0.3781, 0.3727, 0.3767, 0.3734, 0.3774,
    0.3710, 0.3746, 0.3751, 0.3752, 0.3776, 0.3747, 0.3739, 0.3688, 0.3709, 0.3751,
    0.3766, 0.3817, 0.3764, 0.3773, 0.3824, 0.3804, 0.3786, 0.3777, 0.3842, 0.3845,
    0.3825, 0.3808, 0.3842, 0.3803, 0.3903, 0.3908, 0.3911, 0.3903, 0.3974, 0.3943,
    0.3939, 0.3947, 0.3966, 0.3927, 0.3967, 0.4000, 0.4056, 0.4026, 0.4045, 0.4046
]

optimized_rmse = [
    0.7698, 0.6565, 0.6202, 0.5944, 0.5742, 0.5700, 0.5652, 0.5632, 0.5640, 0.5627,
    0.5638, 0.5630, 0.5624, 0.5623, 0.5605, 0.5612, 0.5647, 0.5638, 0.5649, 0.5651,
    0.5634, 0.5615, 0.5610, 0.5604, 0.5600, 0.5613, 0.5605, 0.5590, 0.5595, 0.5569,
    0.5563, 0.5583, 0.5577, 0.5577, 0.5563, 0.5559, 0.5559, 0.5569, 0.5563, 0.5563,
    0.5571, 0.5570, 0.5562, 0.5584, 0.5562, 0.5560, 0.5555, 0.5567, 0.5565, 0.5588
]

optimized_mae = [
    0.6359, 0.4840, 0.4371, 0.4073, 0.3925, 0.3906, 0.3885, 0.3880, 0.3899, 0.3890,
    0.3897, 0.3892, 0.3884, 0.3881, 0.3869, 0.3875, 0.3901, 0.3893, 0.3898, 0.3896,
    0.3875, 0.3860, 0.3858, 0.3855, 0.3849, 0.3856, 0.3847, 0.3836, 0.3841, 0.3823,
    0.3820, 0.3831, 0.3831, 0.3826, 0.3824, 0.3815, 0.3813, 0.3817, 0.3815, 0.3817,
    0.3824, 0.3821, 0.3813, 0.3822, 0.3820, 0.3817, 0.3822, 0.3823, 0.3819, 0.3826
]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    "Federated Learning: Baseline vs Optimized (Top-K Compression, k=0.1)\n"
    "Thesis: Optimizing FL for Resource-Constrained Edge Devices in Smart Grids",
    fontsize=13, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── Plot 1: RMSE over rounds ───────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(rounds, baseline_rmse,  color="#e74c3c", linewidth=2,   label="Baseline (FedAvg, no compression)")
ax1.plot(rounds, optimized_rmse, color="#2ecc71", linewidth=2,   label="Optimized (Top-K, k=0.1)")
ax1.axvline(x=11, color="#e74c3c", linestyle="--", alpha=0.5, label="Baseline best (Round 11)")
ax1.axvline(x=47, color="#2ecc71", linestyle="--", alpha=0.5, label="Optimized best (Round 47)")
ax1.set_xlabel("Round", fontsize=11)
ax1.set_ylabel("RMSE", fontsize=11)
ax1.set_title("RMSE per FL Round", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, 50)

# ── Plot 2: MAE over rounds ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(rounds, baseline_mae,  color="#e74c3c", linewidth=2, label="Baseline")
ax2.plot(rounds, optimized_mae, color="#2ecc71", linewidth=2, label="Optimized")
ax2.set_xlabel("Round", fontsize=11)
ax2.set_ylabel("MAE", fontsize=11)
ax2.set_title("MAE per FL Round", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 50)

# ── Plot 3: Final comparison bar chart ────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
metrics  = ["Final RMSE", "Final MAE", "Best RMSE"]
baseline_vals  = [0.5906, 0.4046, 0.5374]
optimized_vals = [0.5588, 0.3826, 0.5555]

x     = np.arange(len(metrics))
width = 0.35
bars1 = ax3.bar(x - width/2, baseline_vals,  width, label="Baseline",  color="#e74c3c", alpha=0.85)
bars2 = ax3.bar(x + width/2, optimized_vals, width, label="Optimized", color="#2ecc71", alpha=0.85)

ax3.set_ylabel("Score", fontsize=11)
ax3.set_title("Final Metrics Comparison", fontsize=12, fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(metrics, fontsize=9)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis="y")
ax3.set_ylim(0, 0.7)

for bar in bars1:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7.5)
for bar in bars2:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7.5)

# ── Stats box ─────────────────────────────────────────────────────────────────
stats_text = (
    "Training Time\n"
    f"  Baseline:  4217s (~70 min)\n"
    f"  Optimized: 1686s (~28 min)\n"
    f"  Speedup:   2.5x faster\n\n"
    "Communication (per round)\n"
    f"  Baseline:  100% gradients\n"
    f"  Optimized: 10% gradients\n"
    f"  Reduction: 90%"
)
fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9,
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.savefig("experiments/results/comparison_plot.png", dpi=150, bbox_inches="tight")
print("Plot saved to: experiments/results/comparison_plot.png")
plt.show()
