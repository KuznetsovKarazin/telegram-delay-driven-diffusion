#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Competing Delay-Driven Star Diffusion for Telegram cascades.

We derive predicted star curves for each cascade:
  C_i(t) = N_i * F_i(t)
  lambda_i(t) = N_i * f_i(t)

Then define competing intensities with shared attention:
  lambdaA_tilde(t) = lambdaA(t) * wA(t) / (wA(t)+wB(t))
  lambdaB_tilde(t) = lambdaB(t) * wB(t) / (wA(t)+wB(t))

Default w = lambda (speed-based competition).

Inputs:
  Results/model_inputs/curves/*_C.csv
  Results/delay_fit/per_cascade_fit.csv

Outputs:
  Results/competing_diffusion/
    competing_pairs.csv
    figures/
      pair_<cidA>__<cidB>_pred.png
      pair_<cidA>__<cidB>_obs.png
      pair_<cidA>__<cidB>_pred_vs_obs.png

Usage:
  python simulate_competing_diffusion.py \
      --curves_dir Results/model_inputs/curves \
      --fit_csv Results/delay_fit/per_cascade_fit.csv \
      --out_dir Results/competing_diffusion \
      --n_pairs 30 \
      --overlap_days 120 \
      --min_size 80 \
      --grid_points 300 \
      --alpha 1.0
"""

import os, argparse
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def cid_from_path(p):
    return os.path.basename(p).replace("_C.csv","")

def load_C_curve(fp):
    df = pd.read_csv(fp)
    t = df["t_sec"].to_numpy(dtype=float)
    C = df["C_t"].to_numpy(dtype=float)
    # ensure monotone nondecreasing for safety (empirical may have tiny drops)
    C = np.maximum.accumulate(C)
    return t, C

def make_time_grid(t_min, t_max, n=300):
    t_min = max(t_min, 1.0)
    return np.logspace(np.log10(t_min), np.log10(t_max), n)

def interp(x_src, y_src, x_tgt):
    return np.interp(x_tgt, x_src, y_src, left=0.0, right=y_src[-1])

# --- fitted star curves
def star_curves(t, N, family, params):
    if family == "weibull":
        dist = stats.weibull_min(c=params["k"], scale=params["lam"])
    else:
        dist = stats.lognorm(s=params["sigma"], scale=np.exp(params["mu"]))
    F = dist.cdf(t)
    f = dist.pdf(t)
    C = N * F
    lam = N * f
    return C, lam

# --- competition
def compete(lamA, lamB, CA=None, CB=None, alpha=1.0):
    """
    alpha=1 -> w=lambda
    alpha=0 -> w=C
    """
    if CA is None: CA = np.cumsum(lamA)
    if CB is None: CB = np.cumsum(lamB)

    wA = alpha*lamA + (1-alpha)*CA
    wB = alpha*lamB + (1-alpha)*CB
    denom = wA + wB + 1e-12
    lamA_t = lamA * (wA/denom)
    lamB_t = lamB * (wB/denom)
    CA_t = np.cumsum(lamA_t)
    CB_t = np.cumsum(lamB_t)
    return CA_t, CB_t, lamA_t, lamB_t

def intersection_time(t, yA, yB):
    d = yA - yB
    s = np.sign(d)
    idx = np.where(s[:-1]*s[1:] < 0)[0]
    if len(idx)==0:
        return np.nan
    i = idx[0]
    # linear interpolation in log-time
    t1,t2 = t[i], t[i+1]
    d1,d2 = d[i], d[i+1]
    if d2==d1:
        return float(t1)
    return float(t1 + (t2-t1)*(-d1/(d2-d1)))

# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves_dir", type=str, default="Results/model_inputs/curves")
    ap.add_argument("--fit_csv", type=str, default="Results/delay_fit/per_cascade_fit.csv")
    ap.add_argument("--out_dir", type=str, default="Results/competing_diffusion")
    ap.add_argument("--n_pairs", type=int, default=30)
    ap.add_argument("--overlap_days", type=float, default=120,
                    help="minimum overlap window in days between cascades")
    ap.add_argument("--min_size", type=int, default=80)
    ap.add_argument("--grid_points", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="competition weight mix (1=lambda, 0=C)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    fig_dir = os.path.join(args.out_dir, "figures")
    ensure_dir(fig_dir)

    df_fit = pd.read_csv(args.fit_csv)
    df_fit = df_fit[df_fit["size"] >= args.min_size].copy()
    fit_map = df_fit.set_index("cascade_id").to_dict(orient="index")

    C_files = sorted(glob(os.path.join(args.curves_dir, "*_C.csv")))
    cids = [cid_from_path(f) for f in C_files if cid_from_path(f) in fit_map]

    # Load observed spans
    spans = {}
    for f in C_files:
        cid = cid_from_path(f)
        if cid not in fit_map: 
            continue
        t, C = load_C_curve(f)
        spans[cid] = (float(t.min()), float(t.max()))

    overlap_sec = args.overlap_days * 24 * 3600

    # Build candidate pairs by overlap
    pairs = []
    cids_sorted = sorted(cids, key=lambda c: -fit_map[c]["size"])
    for i, a in enumerate(cids_sorted):
        for b in cids_sorted[i+1:]:
            a0,a1 = spans[a]
            b0,b1 = spans[b]
            left = max(a0,b0); right = min(a1,b1)
            if right - left >= overlap_sec:
                pairs.append((a,b,right-left))
        if len(pairs) > 10_000:
            break

    if not pairs:
        print("No overlapping pairs found. Try reducing --overlap_days.")
        return

    # Rank by mixture of size and overlap
    def pair_score(p):
        a,b,ov = p
        return (fit_map[a]["size"] + fit_map[b]["size"]) * ov

    pairs = sorted(pairs, key=pair_score, reverse=True)[:args.n_pairs]

    out_rows = []
    print(f"Simulating {len(pairs)} competing pairs...")

    for a,b,ov in pairs:
        fa = fit_map[a]["best_by_AIC"]
        fb = fit_map[b]["best_by_AIC"]
        Na = fit_map[a]["size"]
        Nb = fit_map[b]["size"]

        # params
        if fa == "weibull":
            pa = {"k": fit_map[a]["wb_k"], "lam": fit_map[a]["wb_lam"]}
        else:
            pa = {"mu": fit_map[a]["ln_mu"], "sigma": fit_map[a]["ln_sigma"]}
        if fb == "weibull":
            pb = {"k": fit_map[b]["wb_k"], "lam": fit_map[b]["wb_lam"]}
        else:
            pb = {"mu": fit_map[b]["ln_mu"], "sigma": fit_map[b]["ln_sigma"]}

        # observed curves
        ta, Ca_obs = load_C_curve(os.path.join(args.curves_dir, f"{a}_C.csv"))
        tb, Cb_obs = load_C_curve(os.path.join(args.curves_dir, f"{b}_C.csv"))

        t_min = max(ta.min(), tb.min())
        t_max = min(ta.max(), tb.max())

        t_grid = make_time_grid(t_min, t_max, args.grid_points)

        # predicted star curves
        Ca_pred, la_pred = star_curves(t_grid, Na, fa, pa)
        Cb_pred, lb_pred = star_curves(t_grid, Nb, fb, pb)

        Ca_comp_pred, Cb_comp_pred, la_comp_pred, lb_comp_pred = compete(
            la_pred, lb_pred, Ca_pred, Cb_pred, alpha=args.alpha
        )

        # observed competition (derive empirical lambda by diff)
        Ca_obs_g = interp(ta, Ca_obs, t_grid)
        Cb_obs_g = interp(tb, Cb_obs, t_grid)
        dt = np.diff(t_grid, prepend=t_grid[0])
        dt[dt==0] = np.nan
        la_obs = np.nan_to_num(np.diff(Ca_obs_g, prepend=0)/dt)
        lb_obs = np.nan_to_num(np.diff(Cb_obs_g, prepend=0)/dt)

        Ca_comp_obs, Cb_comp_obs, la_comp_obs, lb_comp_obs = compete(
            la_obs, lb_obs, Ca_obs_g, Cb_obs_g, alpha=args.alpha
        )

        # metrics
        t_cross_pred = intersection_time(t_grid, Ca_comp_pred, Cb_comp_pred)
        t_cross_obs  = intersection_time(t_grid, Ca_comp_obs,  Cb_comp_obs)

        winner_pred = "A" if Ca_comp_pred[-1] > Cb_comp_pred[-1] else "B"
        winner_obs  = "A" if Ca_comp_obs[-1]  > Cb_comp_obs[-1]  else "B"

        out_rows.append({
            "cid_A": a, "cid_B": b,
            "size_A": Na, "size_B": Nb,
            "family_A": fa, "family_B": fb,
            "overlap_days": ov/(24*3600),
            "t_cross_pred_sec": t_cross_pred,
            "t_cross_obs_sec": t_cross_obs,
            "winner_pred": winner_pred,
            "winner_obs": winner_obs
        })

        # ---- plots
        # predicted
        plt.figure(figsize=(7,5))
        plt.plot(t_grid, Ca_comp_pred, label=f"A comp (pred)")
        plt.plot(t_grid, Cb_comp_pred, label=f"B comp (pred)")
        plt.xscale("log"); plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.xlabel("Delay (sec)"); plt.ylabel("Competing C(t)")
        plt.title(f"Predicted competing diffusion\nA={a} vs B={b}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"pair_{a}__{b}_pred.png"), dpi=220)
        plt.close()

        # observed
        plt.figure(figsize=(7,5))
        plt.plot(t_grid, Ca_comp_obs, label=f"A comp (obs)")
        plt.plot(t_grid, Cb_comp_obs, label=f"B comp (obs)")
        plt.xscale("log"); plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.xlabel("Delay (sec)"); plt.ylabel("Competing C(t)")
        plt.title(f"Observed competing diffusion\nA={a} vs B={b}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"pair_{a}__{b}_obs.png"), dpi=220)
        plt.close()

        # predicted vs observed overlay
        plt.figure(figsize=(7,5))
        plt.plot(t_grid, Ca_comp_obs, label="A obs")
        plt.plot(t_grid, Cb_comp_obs, label="B obs")
        plt.plot(t_grid, Ca_comp_pred, label="A pred", linestyle="--")
        plt.plot(t_grid, Cb_comp_pred, label="B pred", linestyle="--")
        plt.xscale("log"); plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.xlabel("Delay (sec)"); plt.ylabel("Competing C(t)")
        plt.title(f"Competing diffusion: pred vs obs\nA={a} vs B={b}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"pair_{a}__{b}_pred_vs_obs.png"), dpi=220)
        plt.close()

    df_out = pd.DataFrame(out_rows)
    out_path = os.path.join(args.out_dir, "competing_pairs.csv")
    df_out.to_csv(out_path, index=False)
    print("DONE. Saved:", out_path)
    print("Figures in:", fig_dir)

if __name__ == "__main__":
    main()
