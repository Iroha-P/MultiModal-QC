"""Generate ablation experiment plots."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ABLATION_DIR = Path("outputs/ablation")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_rank():
    path = ABLATION_DIR / "rank_results.json"
    if not path.exists():
        print("rank_results.json not found, skipping.")
        return
    with open(path) as f:
        data = json.load(f)

    ranks = [d["rank"] for d in data]
    acc = [d["accuracy"] for d in data]
    f1 = [d["f1"] for d in data]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("LoRA Rank", fontsize=13)
    ax1.set_ylabel("Accuracy", fontsize=13, color="tab:blue")
    ax1.plot(ranks, acc, "o-", color="tab:blue", linewidth=2, markersize=8, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(ranks)

    ax2 = ax1.twinx()
    ax2.set_ylabel("F1 Score", fontsize=13, color="tab:red")
    ax2.plot(ranks, f1, "s--", color="tab:red", linewidth=2, markersize=8, label="F1")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=11)

    plt.title("LoRA Rank Ablation", fontsize=14)
    fig.tight_layout()
    out = ABLATION_DIR / "ablation_rank.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_data():
    path = ABLATION_DIR / "data_results.json"
    if not path.exists():
        print("data_results.json not found, skipping.")
        return
    with open(path) as f:
        data = json.load(f)

    ratios = [f"{int(d['ratio']*100)}%" for d in data]
    samples = [d["samples"] for d in data]
    acc = [d["accuracy"] for d in data]
    f1 = [d["f1"] for d in data]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Training Data Ratio (samples)", fontsize=13)
    ax1.set_ylabel("Accuracy", fontsize=13, color="tab:blue")
    ax1.plot(range(len(ratios)), acc, "o-", color="tab:blue", linewidth=2, markersize=8, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(range(len(ratios)))
    ax1.set_xticklabels([f"{r}\n({s})" for r, s in zip(ratios, samples)])

    ax2 = ax1.twinx()
    ax2.set_ylabel("F1 Score", fontsize=13, color="tab:red")
    ax2.plot(range(len(ratios)), f1, "s--", color="tab:red", linewidth=2, markersize=8, label="F1")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=11)

    plt.title("Training Data Size Ablation", fontsize=14)
    fig.tight_layout()
    out = ABLATION_DIR / "ablation_data.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_rank()
    plot_data()
