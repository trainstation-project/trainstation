# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plot MFU and tokens/day against training step from a torchtitan stdout log.

The rank-0 metrics line looks like:
    step: 12  loss: 8.1  ...  tps: 8,299  tflops: 231.67  mfu: 9.27%
where ``tps`` is tokens per second *per GPU*. Cluster tokens/day is computed as
``tps * num_gpus * 86400``.

Usage:
    python scripts/amd/plot_train_metrics.py train.log --num-gpus 8 --out-dir figs \
        --title "Qwen3-30B-A3B, 8x MI355X"
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STEP_RE = re.compile(
    r"step:\s*(\d+).*?tps:\s*([\d,]+).*?tflops:\s*([\d.]+).*?mfu:\s*([\d.]+)%"
)

# Reference data-viz palette (light mode).
_SURFACE = "#fcfcfb"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3df"
_SERIES = "#2a78d6"
_RAW = "#a9c8ee"


def parse_log(path: str) -> dict[int, tuple[float, float, float]]:
    """Return {step: (tps_per_gpu, tflops, mfu_pct)}; later duplicates win."""
    metrics: dict[int, tuple[float, float, float]] = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = _STEP_RE.search(_ANSI_RE.sub("", line))
            if m:
                metrics[int(m.group(1))] = (
                    float(m.group(2).replace(",", "")),
                    float(m.group(3)),
                    float(m.group(4)),
                )
    return dict(sorted(metrics.items()))


def moving_average(values: list[float], window: int) -> list[float]:
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def _style_axes(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.grid(axis="y", color=_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_TEXT_SECONDARY, length=0, labelsize=10)


def plot_metric(
    steps: list[int],
    values: list[float],
    *,
    ylabel: str,
    title: str,
    fmt,
    out_path: str,
    window: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    _style_axes(ax)

    smooth = moving_average(values, window)
    ax.plot(steps, values, color=_RAW, linewidth=1.25, label="per step")
    ax.plot(
        steps,
        smooth,
        color=_SERIES,
        linewidth=2,
        solid_capstyle="round",
        label=f"{window}-step moving avg",
    )
    ax.annotate(
        fmt(smooth[-1], None),
        xy=(steps[-1], smooth[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=_TEXT_PRIMARY,
        fontsize=10,
    )

    ax.set_xlim(steps[0], steps[-1])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.set_xlabel("Training step", color=_TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=_TEXT_SECONDARY)
    ax.set_title(title, loc="left", color=_TEXT_PRIMARY, fontsize=12)
    legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(_TEXT_SECONDARY)

    fig.tight_layout(rect=(0, 0, 0.95, 1))
    fig.savefig(out_path, facecolor=_SURFACE)
    plt.close(fig)


def _fmt_tokens(v: float, _pos) -> str:
    if v >= 1e9:
        return f"{v / 1e9:.1f}B"
    if v >= 1e6:
        return f"{v / 1e6:.0f}M"
    return f"{v:,.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", help="torchtitan stdout log (rank 0)")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--title", default="")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    metrics = parse_log(args.log)
    if not metrics:
        raise ValueError(f"No metrics lines found in {args.log}")
    steps = list(metrics)
    tps = [v[0] for v in metrics.values()]
    mfu = [v[2] for v in metrics.values()]
    tokens_per_day = [t * args.num_gpus * 86400 for t in tps]

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = f"{args.title} -- " if args.title else ""
    plot_metric(
        steps,
        mfu,
        ylabel="MFU (%)",
        title=f"{prefix}MFU",
        fmt=lambda v, _pos: f"{v:.1f}%",
        out_path=os.path.join(args.out_dir, "mfu_vs_step.png"),
        window=args.window,
    )
    plot_metric(
        steps,
        tokens_per_day,
        ylabel=f"Tokens/day ({args.num_gpus} GPUs)",
        title=f"{prefix}Tokens/day",
        fmt=_fmt_tokens,
        out_path=os.path.join(args.out_dir, "tokens_per_day_vs_step.png"),
        window=args.window,
    )

    # Skip the first steps (compile / warmup) for the steady-state summary.
    skip = min(10, len(steps) - 1)
    ss_mfu = sum(mfu[skip:]) / len(mfu[skip:])
    ss_tpd = sum(tokens_per_day[skip:]) / len(tokens_per_day[skip:])
    print(
        f"steps={steps[0]}..{steps[-1]}  steady-state (step>{steps[skip - 1]}): "
        f"mfu={ss_mfu:.2f}%  tokens/day={ss_tpd / 1e9:.2f}B"
    )


if __name__ == "__main__":
    main()
