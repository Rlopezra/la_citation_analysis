"""nb_model.py - negative binomial rate model: tickets ~ housing predictor(s), offset log(households), SEs clustered by district."""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
import statsmodels.formula.api as smf

def prep(panel):
    d = panel.copy()
    d["renter_10pt"] = d["renter_share"] * 10          # per 10 percentage points
    d["income_10k"] = d["median_income"] / 10_000      # per $10k
    d["log_hh"] = np.log(d["total_household"])
    return d

def fit_nb(d, formula, year_effects=False):
    if year_effects:
        formula += " + C(year)"
    # 1) estimate the dispersion alpha by maximum likelihood. The built-in NegativeBinomial optimizer fails to
    #    converge here (counts ~1e5, huge exposure), so profile it: for each alpha fit the GLM, keep the best log-likelihood.
    def neg_ll(log_alpha):
        m = smf.glm(formula, data=d, family=sm.families.NegativeBinomial(alpha=np.exp(log_alpha)),
                    offset=d["log_hh"]).fit()
        return -m.llf
    opt = minimize_scalar(neg_ll, bounds=(np.log(1e-4), np.log(10)), method="bounded", options={"xatol": 1e-6})
    alpha = float(np.exp(opt.x))
    # 2) refit as a GLM with that alpha fixed, SEs clustered by district
    glm = smf.glm(formula, data=d, family=sm.families.NegativeBinomial(alpha=alpha),
                  offset=d["log_hh"]).fit(cov_type="cluster", cov_kwds={"groups": d["District"]})
    return alpha, glm

def fit_poisson(d, formula, year_effects=False):
    if year_effects:
        formula += " + C(year)"
    glm = smf.glm(formula, data=d, family=sm.families.Poisson(), offset=d["log_hh"]).fit(cov_type="cluster", cov_kwds={"groups": d["District"]})
    return glm

def tiddy(glm,label, alpha=None):
    ci = glm.conf_int()
    pearson_dispersion = np.sum(glm.resid_pearson ** 2) / glm.df_resid
    
    out = pd.DataFrame({
        "model": label,
        "alpha": alpha if alpha is not None else np.nan,
        "coef": glm.params,
        "std_error": glm.bse,
        "z_value": glm.tvalues,          # or glm.zvalues
        "p_value": glm.pvalues,          # standardized naming
        "rate_ratio": np.exp(glm.params),
        "rr_low": np.exp(ci[0]),
        "rr_high": np.exp(ci[1]),
        "aic": glm.aic,
        "deviance": glm.deviance,
        "df_residuals": glm.df_resid,
        "deviance_ratio": glm.deviance / glm.df_resid,
        "pearson_dispersion": pearson_dispersion,
        "family": "Negative Binomial" if alpha is not None else "Poisson"
    })
    
    # Safely drop intercept if present
    if "Intercept" in out.index:
        out = out.drop(index="Intercept")
        
    return out.reset_index(names="term")
