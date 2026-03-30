import csv
import statistics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# CSV 읽기
data = {}
with open("benchmark_results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["variant"], row["o3"])
        data.setdefault(key, []).append(float(row["time_sec"]))

labels = {
    ("online", "no"):  "Online\n(no O3)",
    ("online", "yes"): "Online\n(O3)",
    ("naive",  "no"):  "Naive\n(no O3)",
    ("naive",  "yes"): "Naive\n(O3)",
}
order = [("online","no"), ("online","yes"), ("naive","no"), ("naive","yes")]
colors = ["#7bafd4", "#2166ac", "#f4a582", "#d6604d"]

avgs = [statistics.mean(data[k]) for k in order]
mins = [min(data[k]) for k in order]
maxs = [max(data[k]) for k in order]
x = np.arange(len(order))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Parallel Softmax Benchmark (NTHREADS=16, N=2^28, 10 runs)", fontsize=13)

# --- 왼쪽: Bar chart with min/max error bars ---
ax = axes[0]
bars = ax.bar(x, avgs, color=colors, width=0.5, zorder=3)
err_low  = [a - mn for a, mn in zip(avgs, mins)]
err_high = [mx - a  for a, mx in zip(avgs, maxs)]
ax.errorbar(x, avgs, yerr=[err_low, err_high],
            fmt="none", color="black", capsize=6, linewidth=1.5, zorder=4)

for bar, avg in zip(bars, avgs):
    ax.text(bar.get_x() + bar.get_width()/2, avg + 0.005,
            f"{avg:.3f}s", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([labels[k] for k in order])
ax.set_ylabel("Time (sec)")
ax.set_title("Average Time with Min/Max Range")
ax.set_ylim(0, max(maxs) * 1.2)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

# --- 오른쪽: Line chart (run별 추이) ---
ax2 = axes[1]
for k, color in zip(order, colors):
    runs = list(range(1, len(data[k]) + 1))
    ax2.plot(runs, data[k], marker="o", label=labels[k].replace("\n", " "), color=color)

ax2.set_xlabel("Run #")
ax2.set_ylabel("Time (sec)")
ax2.set_title("Per-Run Time")
ax2.legend(fontsize=8)
ax2.grid(linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("benchmark_results.png", dpi=150)
print("저장 완료: benchmark_results.png")
plt.show()