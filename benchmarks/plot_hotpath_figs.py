"""Figures 1 and 2 of ``benchmarks/RESULTS.md``, drawn from the numbers ``bench_hotpath.py`` measured.

    python benchmarks/plot_hotpath_figs.py      # reads benchmarks/output/hotpath_figdata.json

``bench_hotpath.py`` (stage D) writes that file and calls :func:`fig1` and :func:`fig2`; this script
redraws the two figures from it without re-running the benchmark.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "output", "hotpath_figdata.json")
FIGDIR = os.path.join(HERE, "figures")

C_FWD, C_GRAD, C_OTHER, INK2 = "#2a78d6", "#eb6834", "#a8a7a0", "#52514e"
COMPONENTS = ("oscillation", "spline_product", "norm_product", "segment_sum", "likelihood_priors")


def fig1(rows, path, note=""):
    """Forward and gradient cost vs number of events (left), and their ratio (right)."""
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True,
                                  gridspec_kw=dict(width_ratios=(1.4, 1)))
    for stat, filled in (("poisson", True), ("barlow-beeston", False)):
        rs = sorted((r for r in rows if r["stat"] == stat), key=lambda r: r["n_events"])
        if not rs:
            continue
        n = np.array([r["n_events"] for r in rs], float)
        fwd = np.array([r["fwd_ms"] for r in rs])
        grad = np.array([r["grad_ms"] for r in rs])
        tag = "Poisson" if stat == "poisson" else "Barlow-Beeston-lite"
        for y, c, name in ((fwd, C_FWD, "forward pass"), (grad, C_GRAD, "gradient")):
            ax.plot(n, y, "-" if filled else ":", color=c, linewidth=2 if filled else 1.2,
                    marker="o", markersize=7, markerfacecolor=c if filled else "white",
                    markeredgewidth=1.5, label=f"{name}, {tag}")
        axr.plot(n, grad / fwd, "o", markersize=8, color="#0b0b0b",
                 markerfacecolor="#0b0b0b" if filled else "white", markeredgewidth=1.5, label=tag)
    p = sorted((r for r in rows if r["stat"] == "poisson"), key=lambda r: r["n_events"])
    n0, f0 = p[1]["n_events"], p[1]["fwd_ms"]
    xs = np.array([p[0]["n_events"], p[-1]["n_events"]], float)
    ax.plot(xs, f0 * xs / n0, color=INK2, linewidth=1, linestyle=(0, (1, 3)))
    ax.text(xs[-1], f0 * xs[-1] / n0 * 0.62, "proportional\nto events", ha="right", va="top",
            fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MC events")
    ax.set_ylabel("time per evaluation (ms)")
    ax.set_title("Cost grows linearly with the number of events", fontsize=11)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ratios = [r["grad_ms"] / r["fwd_ms"] for r in p]
    axr.axhspan(min(ratios), max(ratios), color=C_GRAD, alpha=0.12, linewidth=0)
    axr.set_xscale("log")
    axr.set_ylim(0, 9)
    axr.set_xlim(xs[0] / 2, xs[-1] * 2)
    axr.set_xlabel("MC events")
    axr.set_ylabel("gradient time / forward time")
    axr.set_title(f"A gradient costs {min(ratios):.1f}-{max(ratios):.1f}x a forward pass (Poisson)",
                  fontsize=11)
    axr.grid(True, axis="y", alpha=0.25)
    axr.legend(frameon=False, fontsize=9, loc="upper left")
    bb = [r["grad_ms"] / r["fwd_ms"] for r in rows if r["stat"] == "barlow-beeston"]
    if bb:
        axr.text(0.98, 0.04, f"Barlow-Beeston-lite: {min(bb):.1f}-{max(bb):.1f}x\n(its forward pass is cheaper)",
                 transform=axr.transAxes, ha="right", va="bottom", fontsize=9, color=INK2)
    if note:
        fig.supxlabel(note, fontsize=8.5, color=INK2)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig2(item2, path):
    """Share of the forward pass per component, one horizontal bar per event count."""
    ns = sorted(item2, key=float)
    fig, ax = plt.subplots(figsize=(10, 2.9), constrained_layout=True)
    for k, n in enumerate(ns):
        d = item2[n]
        parts = [("oscillation", d["oscillation"], C_FWD), ("spline systematics", d["spline_product"], C_GRAD),
                 ("everything else", sum(d[c] for c in COMPONENTS[2:]), C_OTHER)]
        total = sum(v for _, v, _ in parts)
        left = 0.0
        for name, v, c in parts:
            share = v / total
            ax.barh(k, share, left=left, color=c, edgecolor="white", linewidth=2, height=0.62,
                    label=name if k == 0 else None)
            if share > 0.06:
                ax.text(left + share / 2, k, f"{v:.1f} ms\n{100 * share:.0f}%", ha="center", va="center",
                        fontsize=9.5, color="white" if c != C_OTHER else "#0b0b0b")
            left += share
        ax.text(1.01, k, f"{total:.1f} ms", va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([f"{int(float(n)):,} events" for n in ns])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of the forward pass (components timed separately; everything else = "
                  "normalisations, histogramming, likelihood and priors)", fontsize=9)
    ax.set_title("Where one forward pass spends its time (CPU): oscillation and spline systematics, "
                 "about half each", fontsize=11)
    ax.set_title(ax.get_title(), fontsize=11, pad=30)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(data_path=DATA, figdir=FIGDIR):
    with open(data_path) as fh:
        d = json.load(fh)
    fig1(d["item1_rows"], os.path.join(figdir, "fig1_logdensity_grad_vs_nevents.png"), d.get("fig1_note", ""))
    fig2(d["item2_ms"], os.path.join(figdir, "fig2_component_breakdown.png"))


if __name__ == "__main__":
    main(*sys.argv[1:])
