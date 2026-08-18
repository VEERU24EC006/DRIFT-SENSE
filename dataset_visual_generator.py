#!/usr/bin/env python
"""Render Slide-6 visuals: one SUCCESS and one HONEST FAILURE case.

For each case the figure shows three panels:
  1) the reference image,
  2) the full 1000x1000 search image with the PREDICTED location (green X)
     and the TRUE location (red open circle),
  3) a zoomed region around the target so both markers are clearly visible.

Also prints the self-evaluation accuracy (frozen 30 benchmark) and runtime.

Usage:
    python make_slide6_visuals.py
    python make_slide6_visuals.py --success sample_005 --failure sample_025
"""

import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import evaluation_script as ev

BENCH = "benchmark.csv"


def case_images(sample_id):
    rows = {r["sample"]: r for r in ev.load_manifest()}
    if sample_id not in rows:
        raise SystemExit(f"sample {sample_id} not in dataset/manifest.csv")
    m = rows[sample_id]
    search = ev.load_gray(ev.dataset_path(m["search"]))
    ref = ev.load_gray(ev.dataset_path(m["reference"]))
    return search, ref, (m["gt_x"], m["gt_y"])


def predict(search, ref):
    t0 = time.perf_counter()
    r = ev.match_pair(search, ref)
    ms = (time.perf_counter() - t0) * 1000.0
    return (r["x"], r["y"]), r, ms


def zoom_window(search, pred, gt, margin=90):
    cx = (pred[0] + gt[0]) / 2.0
    cy = (pred[1] + gt[1]) / 2.0
    span = max(np.hypot(pred[0] - gt[0], pred[1] - gt[1]) / 2.0, 140.0) + margin
    x0 = int(max(0, cx - span))
    x1 = int(min(search.shape[1], cx + span))
    y0 = int(max(0, cy - span))
    y1 = int(min(search.shape[0], cy + span))
    return x0, x1, y0, y1


def draw(search, ref, pred, gt, title, subtext, out_path):
    fig = plt.figure(figsize=(17, 5.4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(ref, cmap="gray")
    ax1.set_title("Reference image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(search, cmap="gray")
    ax2.plot(pred[0], pred[1], marker="X", color="lime", ms=13, mew=3,
             label="Predicted")
    ax2.plot(gt[0], gt[1], marker="o", color="red", ms=13, mfc="none", mew=3,
             label="True")
    ax2.set_title("Search image (1000x1000)")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.axis("off")

    x0, x1, y0, y1 = zoom_window(search, pred, gt)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(search[y0:y1, x0:x1], cmap="gray",
               extent=[x0, x1, y1, y0])
    ax3.plot(pred[0], pred[1], marker="X", color="lime", ms=13, mew=3)
    ax3.plot(gt[0], gt[1], marker="o", color="red", ms=13, mfc="none", mew=3)
    ax3.set_title(f"Zoomed region (x:{x0}-{x1}, y:{y0}-{y1})")
    ax3.axis("off")

    fig.text(0.5, 0.005, subtext, ha="center", fontsize=9, color="#333333")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  saved -> {out_path}")


def benchmark_accuracy():
    n, ok = 0, 0
    with open(BENCH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("error_px"):
                continue
            n += 1
            if float(r["error_px"]) <= 0.65:
                ok += 1
    return (ok / n * 100.0) if n else float("nan"), n


def pick_cases(rng, success, failure):
    """Randomly choose one success (<=0.65 px) and one failure (>5 px) case."""
    rows = {}
    with open(BENCH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("error_px"):
                rows[r["search"].split("/")[-1].replace("_search.png", "")] = float(r["error_px"])
    success_pool = [s for s, e in rows.items() if e <= 0.65]
    failure_pool = [s for s, e in rows.items() if e > 5.0]
    if not success_pool or not failure_pool:
        raise SystemExit("benchmark.csv must contain both success (<=0.65 px) "
                         "and failure (>5 px) cases")
    if success is None:
        success = str(rng.choice(success_pool))
    if failure is None:
        failure = str(rng.choice(failure_pool))
    return success, failure


def grid_lock_reason(sample_id, pred, gt, err):
    """Diagnose a failure: is the error a grid-aligned periodic lock-on?"""
    manifest = {r["sample"]: r for r in ev.load_manifest()}
    m = manifest.get(sample_id)
    pitch = (400.0 / m["scale"]) if m else 44.0   # parent pitch 400 / search scale
    dx, dy = pred[0] - gt[0], pred[1] - gt[1]
    nx, ny = round(dx / pitch), round(dy / pitch)
    residual = float(np.hypot(dx - nx * pitch, dy - ny * pitch))
    if (nx != 0 or ny != 0) and residual < 0.35 * pitch:
        return (f"PERIODIC-LATTICE LOCK-ON: the predicted location is ~({nx},{ny}) "
                f"lattice periods (pitch ~{pitch:.0f} px) from the true one - i.e. a "
                f"visually identical DRAM pattern instance (error {err:.0f} px).")
    return (f"Mis-location on a repetitive layout (error {err:.0f} px): the winner is "
            f"a plausible but wrong lattice instance; regional Census did not "
            f"disambiguate it.")


def census_at(search, ref, gt):
    """Regional Census score at an arbitrary (x, y) using the canonical template."""
    try:
        template = ev.transform_reference_feature(ref, 1.0, 0.10, 0.0)
        c = {"x": gt[0], "y": gt[1], "template": template}
        ev.evaluate_census(search, c)
        return float(c["census"]) if c.get("census_valid") else float("nan")
    except Exception:
        return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--success", default=None, help="explicit success sample id")
    p.add_argument("--failure", default=None, help="explicit failure sample id")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for random case selection (default: random)")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    success, failure = pick_cases(rng, a.success, a.failure)
    print(f"Selected cases -> SUCCESS: {success}, FAILURE: {failure}")

    acc, n = benchmark_accuracy()
    print(f"Frozen benchmark: {acc:.1f}% of {n} cases within 0.65 px")

    print(f"SUCCESS case ({success}):")
    search, ref, gt = case_images(success)
    pred, r, ms = predict(search, ref)
    err = float(np.hypot(pred[0] - gt[0], pred[1] - gt[1]))
    print(f"  pred=({pred[0]:.2f},{pred[1]:.2f}) gt=({gt[0]:.2f},{gt[1]:.2f}) "
          f"err={err:.3f} px decision={r['decision']} runtime={ms:.0f} ms")
    draw(search, ref, pred, gt,
         f"SUCCESS  |  {success}  |  error {err:.2f} px  |  runtime {ms:.0f} ms",
         "Correct localization: predicted within 0.65 px of the true center. "
         "Green X = predicted, red circle = true.",
         "slide_success.png")

    print(f"FAILURE case ({failure}):")
    search, ref, gt = case_images(failure)
    pred, r, ms = predict(search, ref)
    err = float(np.hypot(pred[0] - gt[0], pred[1] - gt[1]))
    print(f"  pred=({pred[0]:.2f},{pred[1]:.2f}) gt=({gt[0]:.2f},{gt[1]:.2f}) "
          f"err={err:.3f} px decision={r['decision']} runtime={ms:.0f} ms")
    reason = grid_lock_reason(failure, pred, gt, err)
    census_true = census_at(search, ref, gt)
    if np.isfinite(census_true):
        winner_census = r.get("census") if np.isfinite(r.get("census", np.nan)) else float("nan")
        if np.isfinite(winner_census) and abs(census_true - winner_census) < 0.05:
            reason += (" The true location scores nearly identically to the winner "
                       f"(regional Census {census_true:.3f} vs {winner_census:.3f}), "
                       "so the ambiguity is structural, not a local-matching error.")
    print(f"  reason: {reason}")
    draw(search, ref, pred, gt,
         f"HONEST FAILURE  |  {failure}  |  error {err:.0f} px",
         f"Why it failed: {reason}",
         "slide_failure.png")


if __name__ == "__main__":
    main()