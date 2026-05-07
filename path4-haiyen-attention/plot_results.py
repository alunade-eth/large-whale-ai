#!/usr/bin/env python3

import argparse
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Log parsing

_ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?"
    r"tokens/sec/GPU:\s*([\d.]+)"
)


def parse_log(path: Path, step_lo: int = 1, step_hi: int = 9999):
    """Return list of (step, tps) tuples within [step_lo, step_hi]."""
    results = []
    with open(path) as fh:
        for line in fh:
            m = _ITER_RE.search(line)
            if m:
                step, tps = int(m.group(1)), float(m.group(2))
                if step_lo <= step <= step_hi:
                    results.append((step, tps))
    return results


def window_stats(path: Path, step_lo: int = 200, step_hi: int = 250):
    data = parse_log(path, step_lo, step_hi)
    if not data:
        raise ValueError(f"No data in [{step_lo},{step_hi}] in {path}")
    vals = [tps for _, tps in data]
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


# Data

LOG_DIR = Path(__file__).parent / "logs"

RUNS = {
    # (mean, std, log_path_or_None)
    "TE\n(baseline)": (10533, 0, None),   # course reference run; no log available
    "cuDNN\n(baseline)": (None, None, LOG_DIR / "gipfel-cudnn-train-8b-250s-4n-2048491.log"),
    "FA3\n(no overlap-pg)": (None, None, LOG_DIR / "gipfel-fa3v4-train-8b-250s-4n-nopg-2058726.log"),
    "FA3\n(+ overlap-pg)": (None, None, LOG_DIR / "gipfel-fa3v4-train-8b-250s-4n-pg-2059247.log"),
}

FA3_DETAIL_LOGS = {
    "FA3 (no overlap-pg)": LOG_DIR / "gipfel-fa3v4-train-8b-250s-4n-nopg-2058726.log",
    "FA3 (+ overlap-pg)":  LOG_DIR / "gipfel-fa3v4-train-8b-250s-4n-pg-2059247.log",
}


# Figure 1 - bar chart


def figure_bar(out_path: Path):
    labels, means, stds = [], [], []
    for label, (mean, std, log) in RUNS.items():
        if log is not None:
            mean, std = window_stats(log)
        labels.append(label)
        means.append(mean)
        stds.append(std)

    # colours: grey for baselines, green shades for FA3
    colours = ["#7f8c8d", "#95a5a6", "#2ecc71", "#27ae60"]
    hatches = ["", "", "//", ""]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=colours, hatch=hatches,
                  edgecolor="white", linewidth=0.8,
                  error_kw=dict(elinewidth=1.2, ecolor="#2c3e50"))

    # value labels on top of each bar
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 30,
                f"{mean:,.0f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # reference line at TE baseline
    te_val = means[0]
    ax.axhline(te_val, color="#7f8c8d", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Throughput (tok/s/GPU)", fontsize=11)
    ax.set_title("LLaMA-8B throughput - 4 nodes / 16 GH200 GPUs\n"
                 "seq=4096, steps 200–250", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    ymin = min(means) * 0.97
    ymax = max(m + s for m, s in zip(means, stds)) * 1.03
    ax.set_ylim(ymin, ymax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved {out_path}")



# Figure 2 - throughput over training steps (FA3 runs)


def figure_training_curve(out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colours = ["#2ecc71", "#27ae60"]
    for (label, log), colour in zip(FA3_DETAIL_LOGS.items(), colours):
        data = parse_log(log, step_lo=1, step_hi=250)
        steps = [s for s, _ in data]
        tps   = [t for _, t in data]
        ax.plot(steps, tps, color=colour, linewidth=1.2,
                label=label, alpha=0.85)

    # baselines as horizontal lines
    ax.axhline(10533, color="#7f8c8d", linewidth=1.0,
               linestyle="--", label="TE baseline (10,533)")
    ax.axhline(10555, color="#95a5a6", linewidth=1.0,
               linestyle=":",  label="cuDNN baseline (10,555)")

    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Throughput (tok/s/GPU)", fontsize=11)
    ax.set_title("FA3 throughput over training - 4 nodes / 16 GH200 GPUs\n"
                 "LLaMA-8B, seq=4096", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved {out_path}")



# Main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "figures"))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    figure_bar(out / "fig1_throughput_comparison.pdf")
    figure_training_curve(out / "fig2_training_curve.pdf")


if __name__ == "__main__":
    main()
