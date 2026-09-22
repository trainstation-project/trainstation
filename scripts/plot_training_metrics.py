# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plot per-step loss and global tokens/day from TorchTitan TensorBoard events."""

import argparse
import csv
import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    result[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def main() -> None:
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens-per-step", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--rolling-window", type=int, default=50)
    parser.add_argument("--title", default="TorchTitan training")
    parser.add_argument(
        "--run-details",
        default="Measured training metrics; throughput excludes the final checkpoint.",
    )
    args = parser.parse_args()
    if args.tokens_per_step <= 0:
        parser.error("tokens-per-step must be positive")
    if not 0 <= args.warmup_steps < args.expected_steps:
        parser.error("warmup-steps must be in [0, expected-steps)")
    if not 1 <= args.rolling_window <= args.expected_steps - args.warmup_steps:
        parser.error("rolling-window must fit within the measured steps")

    paths = sorted(args.log_dir.rglob("events.out.tfevents.*"))
    if not paths:
        raise ValueError(f"No TensorBoard events found in {args.log_dir}")
    scalars: dict[str, dict[int, float]] = {}
    for path in paths:
        events = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
        for tag in cast(list[str], events.Tags()["scalars"]):
            series = scalars.setdefault(tag, {})
            for event in events.Scalars(tag):
                if event.step in series:
                    raise ValueError(f"Duplicate {tag} at step {event.step}")
                series[event.step] = event.value

    steps = np.arange(1, args.expected_steps + 1)

    def values(tag: str) -> np.ndarray:
        series = scalars[tag]
        if sorted(series) != steps.tolist():
            raise ValueError(f"{tag} does not cover steps 1..{args.expected_steps}")
        result = np.array([series[step] for step in steps], dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError(f"Non-finite values in {tag}")
        return result

    loss = values("loss_metrics/global_avg_loss")
    seconds = values("time_metrics/end_to_end(s)")
    grad_norm = values("grad_norm")
    if (seconds <= 0).any():
        raise ValueError("Step durations must be positive")
    daily_tokens = args.tokens_per_step * 86400 / seconds
    smooth_loss = rolling_mean(loss, args.rolling_window)
    measured = steps > args.warmup_steps
    smooth_daily = np.full(len(steps), np.nan)
    # Pool tokens and elapsed time; averaging instantaneous rates overstates speed.
    smooth_daily[measured] = (
        args.tokens_per_step
        * 86400
        / rolling_mean(seconds[measured], args.rolling_window)
    )
    steady_seconds = float(seconds[measured].sum())
    steady_tps = int(measured.sum()) * args.tokens_per_step / steady_seconds
    window = args.rolling_window
    summary = {
        "completed_steps": args.expected_steps,
        "tokens_per_step": args.tokens_per_step,
        "total_training_tokens": args.expected_steps * args.tokens_per_step,
        "measurement_first_step": args.warmup_steps + 1,
        "measurement_last_step": args.expected_steps,
        "measurement_seconds": steady_seconds,
        "mean_step_seconds": float(seconds[measured].mean()),
        "global_tokens_per_second": steady_tps,
        "global_tokens_per_day": steady_tps * 86400,
        "first_loss": float(loss[0]),
        "last_loss": float(loss[-1]),
        "first_window_mean_loss": float(loss[:window].mean()),
        "last_window_mean_loss": float(loss[-window:].mean()),
        "rolling_window": window,
        "all_losses_and_grad_norms_finite": True,
        "throughput_note": "Global extrapolation, excluding warmup and final checkpoint.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "loss",
                "grad_norm",
                "step_seconds",
                "tokens_per_day",
                "loss_rolling_mean",
                "tokens_per_day_rolling",
                "warmup",
            ]
        )
        for index, step in enumerate(steps):
            writer.writerow(
                [
                    step,
                    loss[index],
                    grad_norm[index],
                    seconds[index],
                    daily_tokens[index],
                    smooth_loss[index],
                    smooth_daily[index],
                    int(not measured[index]),
                ]
            )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    for name, raw, smooth, ylabel, color, subtitle in (
        (
            "loss_vs_step",
            loss,
            smooth_loss,
            "Training cross-entropy loss",
            "#007f86",
            f"First / last {window}-step mean: "
            f"{loss[:window].mean():.3f} / {loss[-window:].mean():.3f}",
        ),
        (
            "tokens_per_day_vs_step",
            daily_tokens / 1e9,
            smooth_daily / 1e9,
            "Global tokens/day (billions, extrapolated)",
            "#6251b5",
            f"Steps {args.warmup_steps + 1}-{args.expected_steps}: "
            f"{steady_tps * 86400 / 1e9:.2f}B tokens/day | "
            f"{steady_tps:,.0f} tokens/s",
        ),
    ):
        fig, ax = plt.subplots(figsize=(10.5, 5.5))
        fig.subplots_adjust(left=0.1, right=0.98, bottom=0.17, top=0.79)
        fig.suptitle(args.title, x=0.1, y=0.96, ha="left", fontsize=16, weight="bold")
        fig.text(0.1, 0.875, subtitle, fontsize=12, color="#374151")
        if args.warmup_steps:
            ax.axvspan(
                0.5,
                args.warmup_steps + 0.5,
                color="#e5e7eb",
                label=f"Warmup (steps 1-{args.warmup_steps})",
            )
        ax.plot(steps, raw, color=color, alpha=0.22, linewidth=0.8, label="Per step")
        ax.plot(
            steps,
            smooth,
            color=color,
            linewidth=2,
            label=f"{window}-step rolling average",
        )
        ax.set(xlabel="Training step", ylabel=ylabel, xlim=(1, args.expected_steps))
        if name == "tokens_per_day_vs_step":
            ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        ax.legend(loc="best", frameon=False, fontsize=9)
        fig.text(
            0.1,
            0.035,
            args.run_details,
            fontsize=9,
            color="#4b5563",
        )
        fig.savefig(args.output_dir / f"{name}.png", dpi=200)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
