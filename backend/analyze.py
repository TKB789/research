#!/usr/bin/env python3
"""
Run the statistical meta-analysis on a table of extracted study effects.

Input:  data/studies.csv  with columns:
          study, year, effect, se        (effect = effect size, se = standard error)
Output: data/forest.png, data/funnel.png, data/meta_results.json

Implements (DerSimonian-Laird random-effects, plus fixed-effect):
  - pooled effect + 95% CI
  - heterogeneity: Q, I^2, tau^2
  - Egger's test for funnel-plot asymmetry (publication bias)
  - meta-regression with publication YEAR as moderator (the "outdated papers" check)
  - cumulative meta-analysis (chronological)

Pure numpy/scipy/matplotlib. No R. Runs in seconds on a laptop or in Actions.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "studies.csv"


def load_studies():
    rows = []
    with open(CSV_PATH) as f:
        header = [h.strip() for h in f.readline().split(",")]
        idx = {h: i for i, h in enumerate(header)}
        for line in f:
            if not line.strip():
                continue
            c = [x.strip() for x in line.split(",")]
            rows.append({
                "study": c[idx["study"]],
                "year": int(c[idx["year"]]),
                "effect": float(c[idx["effect"]]),
                "se": float(c[idx["se"]]),
            })
    return rows


def random_effects(effects, ses):
    """DerSimonian-Laird random-effects pooling."""
    y = np.asarray(effects, float)
    v = np.asarray(ses, float) ** 2
    w = 1.0 / v
    fixed = np.sum(w * y) / np.sum(w)          # fixed-effect mean
    Q = np.sum(w * (y - fixed) ** 2)
    df = len(y) - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0

    w_re = 1.0 / (v + tau2)
    pooled = np.sum(w_re * y) / np.sum(w_re)
    se_pooled = np.sqrt(1.0 / np.sum(w_re))
    ci = (pooled - 1.96 * se_pooled, pooled + 1.96 * se_pooled)
    z = pooled / se_pooled
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "pooled": pooled, "se": se_pooled, "ci_low": ci[0], "ci_high": ci[1],
        "z": z, "p": pval, "Q": Q, "df": df, "I2": I2, "tau2": tau2,
        "p_heterogeneity": 1 - stats.chi2.cdf(Q, df) if df > 0 else None,
    }


def eggers_test(effects, ses):
    """Regress standard normal deviate on precision; intercept != 0 => asymmetry."""
    y = np.asarray(effects, float)
    s = np.asarray(ses, float)
    snd = y / s          # standard normal deviate
    prec = 1.0 / s       # precision
    slope, intercept, r, p, se = stats.linregress(prec, snd)
    return {"intercept": intercept, "p": p, "interpretation":
            "Possible small-study/publication bias" if p < 0.05
            else "No strong evidence of funnel asymmetry"}


def year_moderator(studies, res_overall):
    """Meta-regression: does effect drift with publication year? (outdated-paper check)"""
    years = np.array([s["year"] for s in studies], float)
    y = np.array([s["effect"] for s in studies], float)
    w = 1.0 / (np.array([s["se"] for s in studies], float) ** 2 + res_overall["tau2"])
    # weighted least squares of effect ~ year
    X = np.column_stack([np.ones_like(years), years - years.mean()])
    W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    cov = np.linalg.inv(X.T @ W @ X)
    slope, slope_se = beta[1], np.sqrt(cov[1, 1])
    z = slope / slope_se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"slope_per_year": slope, "p": p, "interpretation":
            "Effect size drifts significantly over time — older papers may be outdated"
            if p < 0.05 else
            "No significant time trend — old and new studies broadly agree"}


def cumulative(studies):
    ordered = sorted(studies, key=lambda s: s["year"])
    out = []
    for i in range(1, len(ordered) + 1):
        sub = ordered[:i]
        r = random_effects([s["effect"] for s in sub], [s["se"] for s in sub])
        out.append({"through_year": ordered[i - 1]["year"],
                    "n_studies": i, "pooled": r["pooled"],
                    "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return out


def forest_plot(studies, res):
    n = len(studies)
    fig, ax = plt.subplots(figsize=(8, 0.5 * n + 2))
    ys = np.arange(n)[::-1]
    for s, y in zip(studies, ys):
        lo, hi = s["effect"] - 1.96 * s["se"], s["effect"] + 1.96 * s["se"]
        ax.plot([lo, hi], [y, y], color="#3d5a80", lw=1.5)
        ax.plot(s["effect"], y, "s", color="#3d5a80", ms=6)
        ax.text(-0.01, y, f"{s['study']} ({s['year']})", ha="right", va="center",
                fontsize=8, transform=ax.get_yaxis_transform())
    # pooled diamond
    yp = -1
    p, lo, hi = res["pooled"], res["ci_low"], res["ci_high"]
    ax.fill([lo, p, hi, p], [yp, yp + 0.3, yp, yp - 0.3], color="#ee6c4d")
    ax.text(-0.01, yp, "Pooled (RE)", ha="right", va="center", fontsize=9,
            fontweight="bold", transform=ax.get_yaxis_transform())
    ax.axvline(0, color="#999", ls="--", lw=0.8)
    ax.set_yticks([])
    ax.set_xlabel("Effect size (95% CI)")
    ax.set_title(f"Forest plot  ·  I²={res['I2']:.0f}%  τ²={res['tau2']:.3f}")
    ax.set_ylim(yp - 1, n)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "forest.png", dpi=130)
    plt.close()


def funnel_plot(studies, res):
    fig, ax = plt.subplots(figsize=(6, 5))
    eff = [s["effect"] for s in studies]
    se = [s["se"] for s in studies]
    ax.scatter(eff, se, color="#3d5a80", zorder=3)
    ax.axvline(res["pooled"], color="#ee6c4d", lw=1.5, label="pooled")
    ax.set_ylim(max(se) * 1.1, 0)  # invert: precise studies on top
    ax.set_xlabel("Effect size")
    ax.set_ylabel("Standard error")
    ax.set_title("Funnel plot")
    ax.legend()
    plt.tight_layout()
    plt.savefig(DATA_DIR / "funnel.png", dpi=130)
    plt.close()


def main():
    studies = load_studies()
    if len(studies) < 2:
        raise SystemExit("Need >= 2 studies in data/studies.csv")
    res = random_effects([s["effect"] for s in studies], [s["se"] for s in studies])
    egg = eggers_test([s["effect"] for s in studies], [s["se"] for s in studies])
    yr = year_moderator(studies, res)
    cum = cumulative(studies)
    forest_plot(studies, res)
    funnel_plot(studies, res)

    out = {
        "n_studies": len(studies),
        "pooled_effect": round(res["pooled"], 4),
        "ci_95": [round(res["ci_low"], 4), round(res["ci_high"], 4)],
        "p_value": round(res["p"], 5),
        "heterogeneity": {"I2_percent": round(res["I2"], 1),
                          "tau2": round(res["tau2"], 4),
                          "Q": round(res["Q"], 3),
                          "p": res["p_heterogeneity"]},
        "publication_bias_egger": egg,
        "year_moderator": yr,
        "cumulative": cum,
        "plots": ["forest.png", "funnel.png"],
    }
    (DATA_DIR / "meta_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
