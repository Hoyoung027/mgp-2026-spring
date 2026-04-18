import csv
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# Experiment 1: Thread Scaling
# ============================================================
gemv_data = defaultdict(list)
gemm_data = defaultdict(list)

with open("data/results_thread_scaling.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = int(row["threads"])
        v = float(row["avg_sec"])
        if row["func"] == "gemv_sse":
            gemv_data[t].append(v)
        elif row["func"] == "gemm_sse":
            gemm_data[t].append(v)

gemv_threads = sorted(gemv_data)
gemv_times   = [sum(gemv_data[t]) / len(gemv_data[t]) for t in gemv_threads]
gemm_threads = sorted(gemm_data)
gemm_times   = [sum(gemm_data[t]) / len(gemm_data[t]) for t in gemm_threads]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Experiment 1: Thread Scaling (N=2048)", fontsize=14)

ax = axes[0]
ax.plot(gemv_threads, [v * 1e3 for v in gemv_times], marker="o", color="steelblue")
ax.set_title("gemv_sse")
ax.set_xlabel("Threads")
ax.set_ylabel("Avg Time (ms)")
ax.set_xticks(gemv_threads)
ax.grid(True, linestyle="--", alpha=0.5)
for x, y in zip(gemv_threads, [v * 1e3 for v in gemv_times]):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

ax = axes[1]
ax.plot(gemm_threads, gemm_times, marker="o", color="tomato")
ax.set_title("gemm_sse")
ax.set_xlabel("Threads")
ax.set_ylabel("Avg Time (sec)")
ax.set_xticks(gemm_threads)
ax.grid(True, linestyle="--", alpha=0.5)
for x, y in zip(gemm_threads, gemm_times):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("data/exp1_thread_scaling.png", dpi=150)
plt.show()
print("saved: data/exp1_thread_scaling.png")

# ============================================================
# Experiment 2: GEMM vs Freivalds
# ============================================================
method_times = defaultdict(list)

with open("data/results_gemm_vs_freivalds.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        method_times[row["method"]].append(float(row["avg_sec"]))

avg = {m: sum(vs) / len(vs) for m, vs in method_times.items()}
methods = ["gemm", "freivalds"]
times   = [avg[m] for m in methods]
colors  = ["tomato", "steelblue"]

fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle("Experiment 2: GEMM vs Freivalds (N=2048)", fontsize=14)

bars = ax.bar(methods, times, color=colors, width=0.4)
ax.set_ylabel("Avg Time (sec)")
ax.set_yscale("log")
ax.grid(True, axis="y", linestyle="--", alpha=0.5)

for bar, t in zip(bars, times):
    label = f"{t:.4f}s" if t >= 0.01 else f"{t*1e3:.4f}ms"
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.2,
            label, ha="center", va="bottom", fontsize=10)

speedup = avg["gemm"] / avg["freivalds"]
ax.set_title(f"Speedup of Freivalds over GEMM: {speedup:.0f}x")

plt.tight_layout()
plt.savefig("data/exp2_gemm_vs_freivalds.png", dpi=150)
plt.show()
print("saved: data/exp2_gemm_vs_freivalds.png")
