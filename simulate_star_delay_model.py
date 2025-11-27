#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulate and validate Delay-Driven Star Diffusion model on Telegram cascades.

Observable model:
  D(t) - delay distribution (lognormal or weibull)
  C_pred(t) = N * F_D(t)
  lambda_pred(t) = N * f_D(t)

Inputs:
  Results/model_inputs/curves/*_C.csv
  Results/model_inputs/curves/*_lambda.csv
  Results/delay_fit/per_cascade_fit.csv
  Results/delay_fit/pooled_fit.json

Outputs:
  Results/star_model_sim/
    sim_results.csv
    figures/
      <cid>_C_pred_vs_obs.png
      <cid>_lambda_pred_vs_obs.png
      pooled_C_pred_vs_obs.png
      pooled_lambda_pred_vs_obs.png

Usage:
  python simulate_star_delay_model.py \
      --curves_dir Results/model_inputs/curves \
      --fit_csv Results/delay_fit/per_cascade_fit.csv \
      --pooled_json Results/delay_fit/pooled_fit.json \
      --out_dir Results/star_model_sim \
      --mode per_cascade_best \
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


# -------------------------
# Utils
# -------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_curve(path):
    return pd.read_csv(path)

def get_cascade_id_from_path(path):
    base = os.path.basename(path)
    return base.replace("_C.csv", "").replace("_lambda.csv", "")

def weibull_F(t, k, lam):
    return stats.weibull_min(c=k, scale=lam).cdf(t)

def weibull_f(t, k, lam):
    return stats.weibull_min(c=k, scale=lam).pdf(t)

def lognorm_F(t, mu, sigma):
    return stats.lognorm(s=sigma, scale=np.exp(mu)).cdf(t)

def lognorm_f(t, mu, sigma):
    return stats.lognorm(s=sigma, scale=np.exp(mu)).pdf(t)

def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a-b)**2)))

def mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a-b)))

def safe_interp(x_src, y_src, x_tgt):
    # assumes x_src sorted
    return np.interp(x_tgt, x_src, y_src, left=np.nan, right=np.nan)

def plot_pred_vs_obs(t, obs, pred, out_path, title, ylabel):
    plt.figure(figsize=(7,5))
    plt.plot(t, obs, label="Observed")
    plt.plot(t, pred, label="Predicted", linestyle="--")
    plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.title(title)
    plt.xlabel("Delay (seconds)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -------------------------
# Model prediction per cascade
# -------------------------

def predict_curves(t_grid, N, params, family):
    """
    Return C_pred(t), lambda_pred(t)
    """
    if family == "weibull":
        k = params["k"]; lam = params["lam"]
        F = weibull_F(t_grid, k, lam)
        f = weibull_f(t_grid, k, lam)
    elif family == "lognormal":
        mu = params["mu"]; sigma = params["sigma"]
        F = lognorm_F(t_grid, mu, sigma)
        f = lognorm_f(t_grid, mu, sigma)
    else:
        raise ValueError("Unknown family")

    C_pred = N * F
    lam_pred = N * f
    return C_pred, lam_pred


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves_dir", type=str, default="Results/model_inputs/curves")
    ap.add_argument("--fit_csv", type=str, default="Results/delay_fit/per_cascade_fit.csv")
    ap.add_argument("--pooled_json", type=str, default="Results/delay_fit/pooled_fit.json")
    ap.add_argument("--out_dir", type=str, default="Results/star_model_sim")
    ap.add_argument("--mode", type=str, default="per_cascade_best",
                    choices=["per_cascade_best", "per_cascade_weibull", "pooled_weibull"],
                    help="Which parameters to use for prediction")
    ap.add_argument("--plot_n", type=int, default=48)
    args = ap.parse_args()

    curves_dir = args.curves_dir
    out_dir = args.out_dir
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    # Load fits
    df_fit = pd.read_csv(args.fit_csv)
    fit_map = df_fit.set_index("cascade_id").to_dict(orient="index")

    with open(args.pooled_json, "r", encoding="utf-8") as f:
        pooled = json.load(f)
    pooled_params = {
        "k": pooled["weibull"]["k"],
        "lam": pooled["weibull"]["lam"]
    }

    # Collect curve files
    C_files = sorted(glob(os.path.join(curves_dir, "*_C.csv")))
    if not C_files:
        raise FileNotFoundError("No *_C.csv files found in " + curves_dir)

    # Order for plotting by size from fit
    def key_fp(fp):
        cid = get_cascade_id_from_path(fp)
        size = fit_map.get(cid, {}).get("size", 0)
        return (-size, cid)
    C_files.sort(key=key_fp)

    sim_rows = []

    # For pooled global curves
    pooled_obs_delays = []
    pooled_obs_counts = []
    pooled_t_grid = None
    pooled_N_total = 0

    print(f"Simulating {len(C_files)} cascades using mode={args.mode} ...")

    for fp_C in C_files:
        cid = get_cascade_id_from_path(fp_C)
        fp_L = os.path.join(curves_dir, f"{cid}_lambda.csv")
        if not os.path.exists(fp_L):
            continue
        if cid not in fit_map:
            continue

        fit_row = fit_map[cid]
        N = int(fit_row["size"])
        if N <= 0:
            continue

        # observed curves
        dfC = load_curve(fp_C)
        dfL = load_curve(fp_L)

        tC = dfC["t_sec"].values.astype(float)
        C_obs = dfC["C_t"].values.astype(float)

        tL = dfL["t_sec"].values.astype(float)
        lam_obs = dfL["lambda_t_per_sec"].values.astype(float)

        # unify time grid (use C grid for predictions)
        t_grid = tC

        # choose family + params
        if args.mode == "pooled_weibull":
            family = "weibull"
            params = pooled_params
        else:
            if args.mode == "per_cascade_weibull":
                family = "weibull"
            else:
                family = fit_row["best_by_AIC"]

            if family == "weibull":
                params = {"k": fit_row["wb_k"], "lam": fit_row["wb_lam"]}
            else:
                params = {"mu": fit_row["ln_mu"], "sigma": fit_row["ln_sigma"]}

        # predict
        C_pred, lam_pred = predict_curves(t_grid, N, params, family)

        # evaluate errors on common grids
        # lambda observed is on tL, interpolate pred to tL
        lam_pred_on_tL = safe_interp(t_grid, lam_pred, tL)

        C_rmse = rmse(C_obs, C_pred)
        C_mae  = mae(C_obs, C_pred)

        lam_rmse = rmse(lam_obs, lam_pred_on_tL)
        lam_mae  = mae(lam_obs, lam_pred_on_tL)

        sim_rows.append({
            "cascade_id": cid,
            "N": N,
            "family_used": family,
            "C_rmse": C_rmse,
            "C_mae": C_mae,
            "lambda_rmse": lam_rmse,
            "lambda_mae": lam_mae,
            "k": params.get("k"),
            "lam": params.get("lam"),
            "mu": params.get("mu"),
            "sigma": params.get("sigma")
        })

        # accumulate for pooled observed C(t)
        pooled_obs_delays.append(tC)
        pooled_obs_counts.append(C_obs)
        pooled_N_total += N
        pooled_t_grid = tC if pooled_t_grid is None else pooled_t_grid

    df_sim = pd.DataFrame(sim_rows).sort_values("N", ascending=False)
    sim_path = os.path.join(out_dir, "sim_results.csv")
    df_sim.to_csv(sim_path, index=False)
    print("Saved simulation results:", sim_path)

    # -------------------------
    # Plot top cascades
    # -------------------------
    top_plot = df_sim.head(args.plot_n)
    print("Plotting predicted vs observed for", len(top_plot), "cascades...")

    for _, row in top_plot.iterrows():
        cid = row["cascade_id"]
        N = int(row["N"])
        family = row["family_used"]

        fp_C = os.path.join(curves_dir, f"{cid}_C.csv")
        fp_L = os.path.join(curves_dir, f"{cid}_lambda.csv")
        if not os.path.exists(fp_C) or not os.path.exists(fp_L):
            continue

        dfC = load_curve(fp_C)
        dfL = load_curve(fp_L)

        t_grid = dfC["t_sec"].values.astype(float)
        C_obs  = dfC["C_t"].values.astype(float)

        tL = dfL["t_sec"].values.astype(float)
        lam_obs = dfL["lambda_t_per_sec"].values.astype(float)

        if args.mode == "pooled_weibull":
            params = pooled_params
        else:
            fit_row = fit_map[cid]
            if family == "weibull":
                params = {"k": fit_row["wb_k"], "lam": fit_row["wb_lam"]}
            else:
                params = {"mu": fit_row["ln_mu"], "sigma": fit_row["ln_sigma"]}

        C_pred, lam_pred = predict_curves(t_grid, N, params, family)
        lam_pred_on_tL = safe_interp(t_grid, lam_pred, tL)

        plot_pred_vs_obs(
            t_grid, C_obs, C_pred,
            out_path=os.path.join(fig_dir, f"{cid}_C_pred_vs_obs.png"),
            title=f"C(t): observed vs predicted ({family}) - {cid}",
            ylabel="C(t) cumulative forwards"
        )

        plot_pred_vs_obs(
            tL, lam_obs, lam_pred_on_tL,
            out_path=os.path.join(fig_dir, f"{cid}_lambda_pred_vs_obs.png"),
            title=f"λ(t): observed vs predicted ({family}) - {cid}",
            ylabel="λ(t) forwards/sec"
        )

    # -------------------------
    # Pooled comparison (global)
    # -------------------------
    print("Building pooled observed curve...")
    # pool by summing interpolated C(t) on common grid
    if pooled_t_grid is not None:
        C_pool_obs = np.zeros_like(pooled_t_grid, dtype=float)

        for tC, C_obs in zip(pooled_obs_delays, pooled_obs_counts):
            C_interp = safe_interp(tC, C_obs, pooled_t_grid)
            C_interp = np.nan_to_num(C_interp)
            C_pool_obs += C_interp

        # pooled prediction uses pooled Weibull
        C_pool_pred, lam_pool_pred = predict_curves(
            pooled_t_grid, pooled_N_total, pooled_params, "weibull"
        )

        plot_pred_vs_obs(
            pooled_t_grid, C_pool_obs, C_pool_pred,
            out_path=os.path.join(fig_dir, "pooled_C_pred_vs_obs.png"),
            title="Pooled C(t): observed vs predicted (Weibull pooled)",
            ylabel="C(t) cumulative forwards"
        )

        # pooled lambda observed: approximate derivative of pooled C by bins
        # compute finite differences on grid
        dt = np.diff(pooled_t_grid, prepend=pooled_t_grid[0])
        dt[dt == 0] = np.nan
        lam_pool_obs = np.diff(C_pool_obs, prepend=0) / dt
        lam_pool_obs = np.nan_to_num(lam_pool_obs)

        plot_pred_vs_obs(
            pooled_t_grid, lam_pool_obs, lam_pool_pred,
            out_path=os.path.join(fig_dir, "pooled_lambda_pred_vs_obs.png"),
            title="Pooled λ(t): observed vs predicted (Weibull pooled)",
            ylabel="λ(t) forwards/sec"
        )

    print("DONE. Figures in:", fig_dir)


if __name__ == "__main__":
    main()
