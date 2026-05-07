#!/usr/bin/env python3
"""
Parse a Megatron-LM training log and compute throughput statistics.

Usage:
    python3 parse_baseline.py <log_file> --steps-from 200 --steps-to 250

Prints tokens/sec/GPU and TFLOP/s/GPU averaged over the requested step range.
"""

import argparse
import re
import statistics
import sys
from pathlib import Path


_ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+).*?"
    r"tokens/sec/GPU:\s*([\d.]+)"
)


def parse_log(path: Path, step_lo: int, step_hi: int):
    tokens_list = []
    tflops_list = []

    with open(path) as fh:
        for line in fh:
            m = _ITER_RE.search(line)
            if m:
                step = int(m.group(1))
                if step_lo <= step <= step_hi:
                    tflops_list.append(float(m.group(2)))
                    tokens_list.append(float(m.group(3)))

    if not tokens_list:
        sys.exit(
            f"No iterations in [{step_lo}, {step_hi}] found in {path}.\n"
            "Check that the log contains 'tokens/sec/GPU' entries "
            "(requires the 0001-log-tokens-per-sec-to-wandb.patch)."
        )

    return tokens_list, tflops_list


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", help="Path to the Megatron-LM training log")
    ap.add_argument("--steps-from", type=int, default=200, metavar="N", help="First step to include (default: 200)")
    ap.add_argument("--steps-to", type=int, default=250, metavar="N", help="Last step to include (default: 250)")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        sys.exit(f"File not found: {log_path}")

    tokens_list, tflops_list = parse_log(log_path, args.steps_from, args.steps_to)

    n = len(tokens_list)
    tok_mean = statistics.mean(tokens_list)
    tok_std = statistics.stdev(tokens_list) if n > 1 else 0.0
    tfl_mean = statistics.mean(tflops_list)
    tfl_std = statistics.stdev(tflops_list) if n > 1 else 0.0

    print(f"Log file  : {log_path.name}")
    print(f"Steps     : {args.steps_from} – {args.steps_to}  ({n} data points)")
    print(f"tokens/sec/GPU  : {tok_mean:>9.1f}  ±{tok_std:6.1f}")
    print(f"TFLOP/s/GPU     : {tfl_mean:>9.2f}  ±{tfl_std:6.2f}")


if __name__ == "__main__":
    main()
