"""trend_plots.py - scatter + fitted trend for district pairs, labelled by district."""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import linregress, theilslopes, spearmanr

# (outcome, predictor, title, x label, y label)
PAIRS = [
    ("avg_tickets_per_household", "median_income",      "Per-household rate vs. income",         "Median household income ($)", "Tickets per household per year"),
    ("avg_tickets_per_household", "avg_household_size", "Per-household rate vs. household size", "Average household size",      "Tickets per household per year"),
    ("avg_tickets_per_capita",    "avg_household_size", "Per-capita rate vs. household size",    "Average household size",      "Tickets per person per year"),
]
HOUSING_PAIRS = [
    ("avg_tickets_per_household", "renter_share",          "Per-household rate vs. renter share",     "Renter share of households",             "Tickets per household per year"),
    ("avg_tickets_per_household", "multiunit_share",       "Per-household rate vs. multi-unit share", "Households in multi-unit buildings",    "Tickets per household per year"),
    ("avg_tickets_per_household", "transit_commute_share", "Per-household rate vs. transit share",    "Workers commuting by transit",           "Tickets per household per year"),
    ("avg_tickets_per_household", "rent_burden_share",     "Per-household rate vs. rent burden",      "Renters paying 30%+ of income on rent",  "Tickets per household per year"),
]
PERCENT_X = {"renter_share", "multiunit_share", "transit_commute_share", "rent_burden_share", "severe_rent_burden_share"}
INK, MUTED, GRID, POINT, LINE = "#1f2328", "#6b7280", "#e5e7eb", "#2f6db5", "#b45309"


def trend_plots(summary, pairs=PAIRS, path="trend_plots.png", ncols=None):
    """One scatter panel per (outcome, predictor) pair with OLS and Theil-Sen trend lines."""
    ncols = ncols or min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.8 * nrows), facecolor="white", squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    for i, (ax, (y, x, title, xl, yl)) in enumerate(zip(axes, pairs)):
        d = summary[[x, y]].dropna()
        xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
        xs = np.linspace(xv.min(), xv.max(), 50)

        ols = linregress(xv, yv)          # y = a + b*x
        ts = theilslopes(yv, xv)          # (slope, intercept, low, high); median of pairwise slopes
        rho = spearmanr(xv, yv)[0]

        ax.plot(xs, ols.intercept + ols.slope * xs, color=LINE, lw=2, label="OLS trend")
        ax.plot(xs, ts[1] + ts[0] * xs, color=LINE, lw=1.5, ls="--", label="Theil-Sen (robust)")
        ax.scatter(xv, yv, s=60, color=POINT, edgecolor="white", linewidth=1.5, zorder=3)
        for dist, xi, yi in zip(d.index, xv, yv):       # label each point with its district
            ax.annotate(str(dist), (xi, yi), xytext=(5, 4), textcoords="offset points", fontsize=8, color=MUTED)

        ax.set_xlabel(xl, color=MUTED, fontsize=9)
        ax.set_ylabel(yl, color=MUTED, fontsize=9)
        ax.set_title(f"rho = {rho:.2f}   R² = {ols.rvalue**2:.2f}   n = {len(d)}",
                     loc="left", fontsize=8.5, color=MUTED, pad=4)
        ax.text(0, 1.085, title, transform=ax.transAxes, fontsize=11, color=INK, fontweight="bold")

        if x == "median_income":
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
        elif x in PERCENT_X:
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        if i == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="best")

        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return path