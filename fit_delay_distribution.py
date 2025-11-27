#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fit parametric delay distributions for Telegram star cascades.

Inputs:
  Results/model_inputs/delays/*.npy         # per-cascade delay arrays (seconds)
  Results/model_inputs/model_inputs_summary.csv (optional, for ordering/size)

Outputs:
  Results/delay_fit/
    per_cascade_fit.csv
    pooled_fit.json
    figures/
      qq_lognormal_<cid>.png
      qq_weibull_<cid>.png
      pdf_cdf_compare_<cid>.png
      pooled_pdf_cdf.png
      hist_mu_sigma.png
      hist_weibull_k_lambda.png

Usage:
  python fit_delay_distribution.py \
      --delays_dir Results/model_inputs/delays \
      --summary_csv Results/model_inputs/model_inputs_summary.csv \
      --out_dir Results/delay_fit \
      --plot_n 48
"""

import os
import json
import argparse
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import kstest


# -------------------------
# Utils
# -------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_delays(path):
    d = np.load(path)
    d = d.astype(np.float64)
    d = d[np.isfinite(d)]
    d = d[d > 0]  # strictly positive for log/Weibull
    return d

def aic_bic(logL, n_params, n):
    aic = 2 * n_params - 2 * logL
    bic = n_params * np.log(n) - 2 * logL
    return float(aic), float(bic)

def plot_qq(dist_name, sample, dist, params, out_path, title):
    """
    QQ plot vs fitted distribution.
    dist: scipy.stats distribution object
    params: fitted parameters tuple
    """
    sample_sorted = np.sort(sample)
    n = len(sample_sorted)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo_q = dist.ppf(probs, *params)

    plt.figure(figsize=(5.5,5.5))
    plt.plot(theo_q, sample_sorted, marker=".", linestyle="none")
    # 45-degree line
    minv = min(theo_q.min(), sample_sorted.min())
    maxv = max(theo_q.max(), sample_sorted.max())
    plt.plot([minv, maxv], [minv, maxv], linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.title(title)
    plt.xlabel(f"Theoretical quantiles ({dist_name})")
    plt.ylabel("Empirical quantiles")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_pdf_cdf_compare(sample, ln_params, wb_params, out_path, title):
    """
    Overlay empirical CDF + fitted CDF/PDF (lognormal + Weibull)
    """
    sample = np.sort(sample)
    n = len(sample)

    # empirical cdf
    y_emp = np.arange(1, n + 1) / n
    x = sample

    # fitted distributions
    ln_dist = stats.lognorm(s=ln_params["sigma"], scale=np.exp(ln_params["mu"]))
    wb_dist = stats.weibull_min(c=wb_params["k"], scale=wb_params["lam"])

    # cdf
    cdf_ln = ln_dist.cdf(x)
    cdf_wb = wb_dist.cdf(x)

    # pdf on log grid for smoothness
    x_pdf = np.logspace(np.log10(x.min()), np.log10(x.max()), 300)
    pdf_ln = ln_dist.pdf(x_pdf)
    pdf_wb = wb_dist.pdf(x_pdf)

    fig, ax1 = plt.subplots(figsize=(7,5))
    ax1.plot(x, y_emp, label="Empirical CDF")
    ax1.plot(x, cdf_ln, label="Lognormal CDF")
    ax1.plot(x, cdf_wb, label="Weibull CDF")
    ax1.set_xscale("log")
    ax1.set_ylabel("CDF")
    ax1.set_xlabel("Delay (sec)")
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x_pdf, pdf_ln, label="Lognormal PDF", linestyle="--")
    ax2.plot(x_pdf, pdf_wb, label="Weibull PDF", linestyle="--")
    ax2.set_yscale("log")
    ax2.set_ylabel("PDF (log scale)")

    # combine legends
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="best")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -------------------------
# Fit per cascade
# -------------------------

def fit_lognormal(sample):
    """
    Fit lognormal by MLE on log-space.
    Returns mu, sigma, logL, KS
    """
    y = np.log(sample)
    mu = y.mean()
    sigma = y.std(ddof=0)

    dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    logL = np.sum(dist.logpdf(sample))

    # KS test
    D, p = kstest(sample, dist.cdf)

    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "logL": float(logL),
        "ks_D": float(D),
        "ks_p": float(p)
    }

def fit_weibull(sample):
    """
    Fit Weibull_min by MLE.
    Fix loc=0 for stability.
    Returns k(shape), lam(scale), logL, KS
    """
    k, loc, lam = stats.weibull_min.fit(sample, floc=0)
    dist = stats.weibull_min(c=k, scale=lam)
    logL = np.sum(dist.logpdf(sample))

    D, p = kstest(sample, dist.cdf)

    return {
        "k": float(k),
        "lam": float(lam),
        "logL": float(logL),
        "ks_D": float(D),
        "ks_p": float(p)
    }


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delays_dir", type=str, default="Results/model_inputs/delays")
    ap.add_argument("--summary_csv", type=str, default=None,
                    help="Optional summary to order by size")
    ap.add_argument("--out_dir", type=str, default="Results/delay_fit")
    ap.add_argument("--plot_n", type=int, default=48,
                    help="How many cascades to plot (largest by size if summary provided)")
    ap.add_argument("--min_size", type=int, default=50,
                    help="Skip cascades with fewer delays")
    args = ap.parse_args()

    delays_dir = args.delays_dir
    out_dir = args.out_dir
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    # Collect delay files
    files = sorted(glob(os.path.join(delays_dir, "*.npy")))
    if not files:
        raise FileNotFoundError("No .npy delays found in " + delays_dir)

    # If summary given, order by size for plotting
    size_map = {}
    if args.summary_csv and os.path.exists(args.summary_csv):
        df_sum = pd.read_csv(args.summary_csv)
        if "cascade_id" in df_sum.columns and "size" in df_sum.columns:
            size_map = dict(zip(df_sum["cascade_id"], df_sum["size"]))

    def file_key(fp):
        cid = os.path.splitext(os.path.basename(fp))[0]
        return -(size_map.get(cid, 0)), cid

    files.sort(key=file_key)

    per_rows = []
    pooled_delays = []

    print(f"Fitting {len(files)} cascades ...")

    for fp in files:
        cid = os.path.splitext(os.path.basename(fp))[0]
        sample = load_delays(fp)

        if len(sample) < args.min_size:
            continue

        pooled_delays.append(sample)

        # Fit both
        ln = fit_lognormal(sample)
        wb = fit_weibull(sample)

        aic_ln, bic_ln = aic_bic(ln["logL"], n_params=2, n=len(sample))
        aic_wb, bic_wb = aic_bic(wb["logL"], n_params=2, n=len(sample))

        best = "lognormal" if aic_ln < aic_wb else "weibull"

        per_rows.append({
            "cascade_id": cid,
            "size": int(len(sample)),
            # lognormal
            "ln_mu": ln["mu"],
            "ln_sigma": ln["sigma"],
            "ln_logL": ln["logL"],
            "ln_AIC": aic_ln,
            "ln_BIC": bic_ln,
            "ln_ks_D": ln["ks_D"],
            "ln_ks_p": ln["ks_p"],
            # weibull
            "wb_k": wb["k"],
            "wb_lam": wb["lam"],
            "wb_logL": wb["logL"],
            "wb_AIC": aic_wb,
            "wb_BIC": bic_wb,
            "wb_ks_D": wb["ks_D"],
            "wb_ks_p": wb["ks_p"],
            # best
            "best_by_AIC": best
        })

    df_fit = pd.DataFrame(per_rows)
    per_path = os.path.join(out_dir, "per_cascade_fit.csv")
    df_fit.to_csv(per_path, index=False)
    print("Saved per-cascade fit:", per_path)

    # -------------------------
    # pooled fit
    # -------------------------
    pooled = np.concatenate(pooled_delays) if pooled_delays else np.array([])
    pooled = pooled[pooled > 0]

    pooled_ln = fit_lognormal(pooled)
    pooled_wb = fit_weibull(pooled)
    pooled_aic_ln, pooled_bic_ln = aic_bic(pooled_ln["logL"], 2, len(pooled))
    pooled_aic_wb, pooled_bic_wb = aic_bic(pooled_wb["logL"], 2, len(pooled))

    pooled_summary = {
        "n_delays": int(len(pooled)),
        "lognormal": {**pooled_ln, "AIC": pooled_aic_ln, "BIC": pooled_bic_ln},
        "weibull":   {**pooled_wb, "AIC": pooled_aic_wb, "BIC": pooled_bic_wb},
        "best_by_AIC": "lognormal" if pooled_aic_ln < pooled_aic_wb else "weibull"
    }

    pooled_path = os.path.join(out_dir, "pooled_fit.json")
    with open(pooled_path, "w", encoding="utf-8") as f:
        json.dump(pooled_summary, f, indent=2)
    print("Saved pooled fit:", pooled_path)

    # -------------------------
    # plots for top cascades
    # -------------------------
    top_plot = df_fit.sort_values("size", ascending=False).head(args.plot_n)
    print("Plotting QQ/PDF/CDF for", len(top_plot), "largest cascades...")

    for _, row in top_plot.iterrows():
        cid = row["cascade_id"]
        sample = load_delays(os.path.join(delays_dir, f"{cid}.npy"))

        ln_params = {"mu": row["ln_mu"], "sigma": row["ln_sigma"]}
        wb_params = {"k": row["wb_k"], "lam": row["wb_lam"]}

        ln_dist = stats.lognorm(s=ln_params["sigma"], scale=np.exp(ln_params["mu"]))
        wb_dist = stats.weibull_min(c=wb_params["k"], scale=wb_params["lam"])

        plot_qq(
            "Lognormal", sample, ln_dist, (),
            out_path=os.path.join(fig_dir, f"qq_lognormal_{cid}.png"),
            title=f"QQ plot (lognormal) - cascade {cid}"
        )

        plot_qq(
            "Weibull", sample, wb_dist, (),
            out_path=os.path.join(fig_dir, f"qq_weibull_{cid}.png"),
            title=f"QQ plot (Weibull) - cascade {cid}"
        )

        plot_pdf_cdf_compare(
            sample,
            ln_params=ln_params,
            wb_params=wb_params,
            out_path=os.path.join(fig_dir, f"pdf_cdf_compare_{cid}.png"),
            title=f"CDF/PDF comparison - cascade {cid}"
        )

    # pooled comparison figure
    print("Plotting pooled comparison...")
    # Use first 1e6 delays for plotting smoothness if huge
    pooled_plot = pooled
    if len(pooled_plot) > 1_000_000:
        pooled_plot = np.random.default_rng(0).choice(pooled_plot, 1_000_000, replace=False)

    ln_params = {"mu": pooled_ln["mu"], "sigma": pooled_ln["sigma"]}
    wb_params = {"k": pooled_wb["k"], "lam": pooled_wb["lam"]}

    plot_pdf_cdf_compare(
        pooled_plot,
        ln_params=ln_params,
        wb_params=wb_params,
        out_path=os.path.join(fig_dir, "pooled_pdf_cdf.png"),
        title="Pooled delays: empirical vs fitted Lognormal/Weibull"
    )

    # hist of parameters
    plt.figure(figsize=(7,5))
    plt.hist(df_fit["ln_mu"], bins=40)
    plt.title("Distribution of lognormal μ across cascades")
    plt.xlabel("μ")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hist_mu.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.hist(df_fit["ln_sigma"], bins=40)
    plt.title("Distribution of lognormal σ across cascades")
    plt.xlabel("σ")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hist_sigma.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.hist(df_fit["wb_k"], bins=40)
    plt.title("Distribution of Weibull shape k across cascades")
    plt.xlabel("k")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hist_weibull_k.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.hist(df_fit["wb_lam"], bins=40)
    plt.title("Distribution of Weibull scale λ across cascades")
    plt.xlabel("λ")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hist_weibull_lambda.png"), dpi=220)
    plt.close()

    print("DONE. Figures in:", fig_dir)


if __name__ == "__main__":
    main()
