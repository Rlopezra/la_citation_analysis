"""trend_fit.py - fitted trend lines across the 15 districts (cross-sectional, not over time)."""
import numpy as np
import pandas as pd
from scipy.stats import linregress, theilslopes

PAIRS = [
    ("avg_tickets_per_household", "median_income"),
    ("avg_tickets_per_household", "avg_household_size"),
    ("avg_tickets_per_capita",   "avg_household_size"),
]

def trend_fit(summary, pairs=PAIRS, n_perm=10_000, seed=20260101):
    rng = np.random.default_rng(seed)
    rows = []
    for y_col, x_col in pairs:
        d = summary[[y_col, x_col]].dropna()
        x, y = d[x_col].to_numpy(float), d[y_col].to_numpy(float)
        ols = linregress(x, y)                                   # y = a + b*x
        ts_slope, ts_int, ts_lo, ts_hi = theilslopes(y, x)       # median of pairwise slopes
        el = linregress(np.log(x), np.log(y))                    # ln y = a + b ln x  (elasticity)
        # permutation p-value for the OLS slope: shuffle y, keep x
        xc = x - x.mean()
        null = np.array([(xc * (rng.permutation(y) - y.mean())).sum() for _ in range(n_perm)]) / (xc ** 2).sum()
        p = (1 + (np.abs(null) >= abs(ols.slope) - 1e-12).sum()) / (1 + n_perm)
        rows.append({"outcome": y_col, "predictor": x_col, "n": len(d),
                     "ols_slope": ols.slope, "ols_intercept": ols.intercept, "r2": ols.rvalue ** 2, "p_perm_slope": p,
                     "theil_sen_slope": ts_slope, "ts_ci_low": ts_lo, "ts_ci_high": ts_hi,
                     "elasticity": el.slope})
    return pd.DataFrame(rows)