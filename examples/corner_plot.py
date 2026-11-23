"""Corner plot with several posterior estimates overlaid (68% and 95% regions), used by the README figures.

Each layer is a dict: ``cols`` (list of 1-D arrays, one per parameter, same order as ``labels``),
``label``, ``color``, and ``style``: ``"filled"`` (shaded regions, filled histogram) or ``"line"``
(contour lines and a step histogram; ``linestyle`` optional).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

INK2 = "#52514e"


def _hpd_levels(h, fracs=(0.95, 0.68)):
    flat = np.sort(h.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    return [float(flat[min(np.searchsorted(cum, f), flat.size - 1)]) for f in fracs]


def _lighten(color, f=0.55):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def corner(layers, truth_vals, labels, title, path, stats_text="", smooth=1.2, bins=60):
    n = len(labels)
    lo = [min(np.percentile(L["cols"][i], 0.1) for L in layers) for i in range(n)]
    hi = [max(np.percentile(L["cols"][i], 99.9) for L in layers) for i in range(n)]
    rng = [(a - 0.04 * (b - a), b + 0.04 * (b - a)) for a, b in zip(lo, hi)]
    fig, axes = plt.subplots(n, n, figsize=(2.55 * n, 2.55 * n), constrained_layout=True)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                edges = np.linspace(*rng[i], bins + 1)
                for L in layers:
                    if L["style"] == "filled":
                        ax.hist(L["cols"][i], bins=edges, density=True, color=_lighten(L["color"]),
                                edgecolor="none")
                    ax.hist(L["cols"][i], bins=edges, density=True, histtype="step", color=L["color"],
                            linewidth=1.4, linestyle=L.get("linestyle", "-"))
                ax.axvline(truth_vals[i], color="black", linestyle=":", linewidth=1.2)
                ax.set_yticks([])
                ax.set_xlim(*rng[i])
            else:
                xc = yc = None
                for L in layers:
                    h, xe, ye = np.histogram2d(L["cols"][j], L["cols"][i], bins=bins, range=[rng[j], rng[i]])
                    h = gaussian_filter(h.T, smooth)
                    l95, l68 = _hpd_levels(h)
                    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
                    if L["style"] == "filled":
                        ax.contourf(xc, yc, h, levels=[l95, l68, h.max() * 1.01],
                                    colors=[_lighten(L["color"]), L["color"]])
                    else:
                        ax.contour(xc, yc, h, levels=[l95, l68], colors=L["color"],
                                   linestyles=L.get("linestyle", "-"), linewidths=1.4)
                ax.plot(truth_vals[j], truth_vals[i], marker="+", color="black", markersize=11, markeredgewidth=1.6)
                ax.set_xlim(*rng[j])
                ax.set_ylim(*rng[i])
            if i == n - 1:
                ax.set_xlabel(labels[j])
                ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i])
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            elif j != i:
                ax.tick_params(labelleft=False)
    handles = []
    for L in layers:
        if L["style"] == "filled":
            handles.append(Patch(color=L["color"], label=f"{L['label']}: 68%"))
            handles.append(Patch(color=_lighten(L["color"]), label=f"{L['label']}: 95%"))
        else:
            handles.append(Line2D([], [], color=L["color"], linestyle=L.get("linestyle", "-"), linewidth=1.4,
                                  label=f"{L['label']}: 68%, 95%"))
    handles.append(Line2D([], [], color="black", marker="+", linestyle=":", markersize=10, label="true value"))
    axes[0, n - 1].legend(handles=handles, loc="upper right", frameon=False, fontsize=10)
    if stats_text:
        axes[1, n - 1].text(1.0, 0.85, stats_text, ha="right", va="top", fontsize=9.5, color=INK2,
                            transform=axes[1, n - 1].transAxes)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=130)
    plt.close(fig)
