"""
descriptive_tests.py

Descriptive tests across the 15 council districts (Spearman rank correlation, permutation p-values).

Test families (declared before looking at results):

  PRIMARY   (4 tests, Bonferroni-corrected within the family):
      level : avg tickets per household vs median income / average household size
      trend : % change in tickets 2022->2025 vs median income / average household size

  SECONDARY (reported, NOT corrected):
      diagnostics : tickets per capita and per vehicle vs household size / income
      rent        : outcomes vs 2025 rent, only districts with enough rent coverage

  TERTIARY  (10 tests, Bonferroni-corrected within the family):
      housing / transit : tickets per household and % change in tickets vs renter share,
                          multi-unit share, rent burden, severe rent burden, transit commute share

With n = 15 districts every result is fragile, so each test also reports the range of the
correlation when any single district is dropped (leave-one-out).

Usage:
    summary = district_summary(panel)
    results = run_tests(summary)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

N_PERM = 10_000
SEED = 20260101
ALPHA = 0.05


def district_summary(panel: pd.DataFrame, min_rent_coverage: float = 0.7) -> pd.DataFrame:
    """One row per district: pooled level, growth, and the predictors."""
    first, last = panel["year"].min(), panel["year"].max()
    g = panel.groupby("District")

    s = pd.DataFrame({
        "avg_tickets_per_household": g["tickets_per_household"].mean(),
        "avg_tickets_per_capita": g["tickets_per_capita"].mean(),
    })
    t = panel.pivot(index="District", columns="year", values="tickets")
    s["pct_change_tickets"] = t[last] / t[first] - 1.0           # denominators are constant, so this is also the rate change

    keep = ["median_income", "avg_household_size", "rent_2025", "coverage_2025", "agg_vehicles_total",
            "renter_share", "multiunit_share", "rent_burden_share", "severe_rent_burden_share",
            "transit_commute_share"]
    s = s.join(panel.drop_duplicates("District").set_index("District")[keep])
    # residents' vehicles (ACS B25046) as the exposure denominator; excludes commuters' and visitors' cars
    s["avg_tickets_per_vehicle"] = g["tickets"].mean() / s["agg_vehicles_total"]
    s["rent_2025_masked"] = s["rent_2025"].where(s["coverage_2025"] >= min_rent_coverage)
    return s


def spearman_perm(x, y, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """Spearman rho, two-sided permutation p-value, and leave-one-out range of rho."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)

    xr, yr = rankdata(x), rankdata(y)
    xc, yc = xr - xr.mean(), yr - yr.mean()
    denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    rho = float((xc * yc).sum() / denom)

    rng = np.random.default_rng(seed)
    perms = np.argsort(rng.random((n_perm, n)), axis=1)         # n_perm random orderings of the districts
    null = (xc[perms] * yc).sum(axis=1) / denom                 # shuffle one variable, keep the other
    p = float((1 + (np.abs(null) >= abs(rho) - 1e-12).sum()) / (1 + n_perm))

    loo = [spearmanr(np.delete(x, i), np.delete(y, i))[0] for i in range(n)]
    return {"n": n, "rho": rho, "p_perm": p, "loo_min": float(np.min(loo)), "loo_max": float(np.max(loo))}


PRIMARY = [
    ("avg_tickets_per_household", "median_income"),
    ("avg_tickets_per_household", "avg_household_size"),
    ("pct_change_tickets", "median_income"),
    ("pct_change_tickets", "avg_household_size"),
]
SECONDARY = [
    ("avg_tickets_per_capita", "avg_household_size"),      # the diagnostic
    ("avg_tickets_per_capita", "median_income"),
    ("avg_tickets_per_household", "rent_2025_masked"),
    ("pct_change_tickets", "rent_2025_masked"),
    ("avg_tickets_per_vehicle", "avg_household_size"),     # per-vehicle version of the diagnostic
    ("avg_tickets_per_vehicle", "median_income"),
]

HOUSING = ["renter_share", "multiunit_share", "rent_burden_share",
           "severe_rent_burden_share", "transit_commute_share"]
TERTIARY = [(o, h) for o in ("avg_tickets_per_household", "pct_change_tickets") for h in HOUSING]
FAMILIES = [("primary", PRIMARY), ("secondary", SECONDARY), ("tertiary", TERTIARY)]
CORRECTED = ("primary", "tertiary")        # families that get a Bonferroni correction


def _check_tests(summary: pd.DataFrame) -> None:
    """Every test must be an (outcome, predictor) pair of columns that exist in the summary."""
    for family, tests in FAMILIES:
        for t in tests:
            if not (isinstance(t, tuple) and len(t) == 2):
                raise ValueError(f"{family} test {t!r} must be an (outcome, predictor) tuple")
            missing = [c for c in t if c not in summary.columns]
            if missing:
                raise KeyError(f"{family} test {t}: {missing} not in the summary. Available: {sorted(summary.columns)}")


def run_tests(summary: pd.DataFrame, n_perm: int = N_PERM) -> pd.DataFrame:
    _check_tests(summary)
    rows = []
    for family, tests in FAMILIES:
        for outcome, predictor in tests:
            r = spearman_perm(summary[predictor], summary[outcome], n_perm=n_perm)
            rows.append({"family": family, "outcome": outcome, "predictor": predictor, **r})
    res = pd.DataFrame(rows)

    m = res.groupby("family")["p_perm"].transform("size")          # tests in that row's own family
    corr = res["family"].isin(CORRECTED)
    res["p_bonferroni"] = np.where(corr, np.minimum(1.0, res["p_perm"] * m), np.nan)
    res["significant_corrected"] = np.where(corr, res["p_bonferroni"] < ALPHA, np.nan)
    res["sign_stable_loo"] = (res["loo_min"] * res["loo_max"]) > 0     # rho keeps its sign when any one district is dropped
    return res