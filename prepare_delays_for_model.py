#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare delay-driven star diffusion inputs from top-K cascades.

Inputs:
  Results/cascades/edges_topK.csv.gz
    columns:
      cascade_peer,cascade_post,parent_node,child_node,delay_sec,child_time,child_channel

Outputs:
  Results/model_inputs/
    delays/                         # per-cascade delays arrays (.npy)
    curves/                         # per-cascade C(t) and lambda(t) curves (.csv)
    figures/                        # publication-ready plots
    model_inputs_summary.csv        # per-cascade metrics

Usage:
  python prepare_delays_for_model.py \
      --edges Results/cascades/edges_topK.csv.gz \
      --out_dir Results/model_inputs \
      --bin_hours 6 \
      --plot_n 48
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Utils
# -------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def cascade_id(peer, post):
    return f"{int(peer)}_{int(post)}"


def compute_curves(delays, bin_seconds):
    """
    Given delays (seconds), compute:
      - C(t): cumulative count vs time (bin centers)
      - lambda(t): rate (count per bin) / bin_seconds
    """
    delays = np.asarray(delays, dtype=np.int64)
    delays = delays[delays >= 0]
    if len(delays) == 0:
        return None

    t_max = delays.max()
    # bin edges from 0 to t_max
    bins = np.arange(0, t_max + bin_seconds, bin_seconds)
    if len(bins) < 2:
        bins = np.array([0, t_max + bin_seconds])

    counts, edges = np.histogram(delays, bins=bins)
    cum = np.cumsum(counts)

    # bin centers for plotting
    centers = (edges[:-1] + edges[1:]) / 2.0
    lamb = counts / bin_seconds

    return centers, cum, lamb, counts


def compute_metrics(delays, centers, cum, lamb):
    """
    Extract per-cascade summary metrics.
    """
    delays = np.asarray(delays, dtype=np.int64)
    delays = delays[delays >= 0]
    n = len(delays)
    if n == 0:
        return {}

    delays_sorted = np.sort(delays)

    def q(p):
        return float(np.percentile(delays_sorted, p))

    # peak speed time
    if lamb is not None and len(lamb) > 0:
        peak_idx = int(np.argmax(lamb))
        peak_time = float(centers[peak_idx])
        peak_speed = float(lamb[peak_idx])
    else:
        peak_time = None
        peak_speed = None

    metrics = {
        "size": n,
        "temporal_span_sec": float(delays_sorted[-1] - delays_sorted[0]) if n > 1 else 0.0,
        "delay_mean_sec": float(np.mean(delays_sorted)),
        "delay_median_sec": float(np.median(delays_sorted)),
        "delay_p90_sec": q(90),
        "delay_p99_sec": q(99),
        "T10_sec": q(10),
        "T50_sec": q(50),
        "T90_sec": q(90),
        "peak_speed_time_sec": peak_time,
        "peak_speed_per_sec": peak_speed,
    }
    return metrics


def plot_cdf(delays, out_path, title):
    delays = np.asarray(delays)
    delays = delays[delays >= 0]
    if len(delays) == 0:
        return
    delays.sort()
    y = np.arange(1, len(delays)+1)/len(delays)

    plt.figure(figsize=(7,5))
    plt.plot(delays, y)
    plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.title(title)
    plt.xlabel("Delay (seconds)")
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_curve(x, y, out_path, title, xlabel, ylabel, logx=True):
    plt.figure(figsize=(7,5))
    plt.plot(x, y)
    if logx:
        plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=str, default="Results/cascades/edges_topK.csv.gz")
    ap.add_argument("--out_dir", type=str, default="Results/model_inputs")
    ap.add_argument("--bin_hours", type=float, default=6.0,
                    help="Bin size for lambda(t) in hours")
    ap.add_argument("--plot_n", type=int, default=48,
                    help="How many cascades to plot (largest ones)")
    args = ap.parse_args()

    edges_path = args.edges
    out_dir = args.out_dir
    bin_seconds = int(args.bin_hours * 3600)

    if not os.path.exists(edges_path):
        raise FileNotFoundError(edges_path)

    # output folders
    delays_dir  = os.path.join(out_dir, "delays")
    curves_dir  = os.path.join(out_dir, "curves")
    figures_dir = os.path.join(out_dir, "figures")
    ensure_dir(delays_dir)
    ensure_dir(curves_dir)
    ensure_dir(figures_dir)

    print("Reading edges...")
    df = pd.read_csv(edges_path, compression="gzip")

    # group delays by cascade
    print("Grouping delays by cascade...")
    grouped = df.groupby(["cascade_peer", "cascade_post"])

    summaries = []
    cascade_delays = {}

    for (peer, post), g in grouped:
        delays = g["delay_sec"].dropna().astype(np.int64).values
        delays = delays[delays >= 0]
        if len(delays) == 0:
            continue

        cid = cascade_id(peer, post)
        cascade_delays[cid] = delays

        # curves
        res = compute_curves(delays, bin_seconds)
        if res is None:
            continue
        centers, cum, lamb, counts = res

        # save delays
        np.save(os.path.join(delays_dir, f"{cid}.npy"), delays)

        # save curves
        pd.DataFrame({"t_sec": centers, "C_t": cum}).to_csv(
            os.path.join(curves_dir, f"{cid}_C.csv"), index=False
        )
        pd.DataFrame({"t_sec": centers, "lambda_t_per_sec": lamb, "bin_count": counts}).to_csv(
            os.path.join(curves_dir, f"{cid}_lambda.csv"), index=False
        )

        # metrics
        m = compute_metrics(delays, centers, cum, lamb)
        m.update({"cascade_id": cid, "cascade_peer": int(peer), "cascade_post": int(post)})
        summaries.append(m)

    df_sum = pd.DataFrame(summaries).sort_values("size", ascending=False)
    sum_path = os.path.join(out_dir, "model_inputs_summary.csv")
    df_sum.to_csv(sum_path, index=False)

    print("Saved summary:", sum_path)
    print("Total cascades with delays:", len(df_sum))

    # -------------------------
    # Plot representative cascades
    # -------------------------
    top_plot = df_sum.head(args.plot_n)["cascade_id"].tolist()
    print("Plotting top cascades:", len(top_plot))

    for cid in top_plot:
        delays = cascade_delays[cid]

        res = compute_curves(delays, bin_seconds)
        if res is None:
            continue
        centers, cum, lamb, counts = res

        plot_curve(
            centers, cum,
            os.path.join(figures_dir, f"{cid}_C.png"),
            title=f"Cumulative forwards C(t) - cascade {cid}",
            xlabel="Delay (seconds)",
            ylabel="C(t): cumulative forwards",
            logx=True
        )

        plot_curve(
            centers, lamb,
            os.path.join(figures_dir, f"{cid}_lambda.png"),
            title=f"Forwarding rate λ(t) - cascade {cid}",
            xlabel="Delay (seconds)",
            ylabel="λ(t) (forwards/sec)",
            logx=True
        )

        plot_cdf(
            delays,
            os.path.join(figures_dir, f"{cid}_cdf.png"),
            title=f"CDF of delays - cascade {cid}"
        )

    # global plots for paper
    all_delays = np.concatenate(list(cascade_delays.values())) if cascade_delays else np.array([])
    if len(all_delays) > 0:
        plot_cdf(
            all_delays,
            os.path.join(figures_dir, "global_cdf_delays_topK.png"),
            title="Global CDF of forwarding delays (top-K cascades)"
        )

        # global lambda via pooling
        resg = compute_curves(all_delays, bin_seconds)
        if resg is not None:
            centers_g, cum_g, lamb_g, _ = resg
            plot_curve(
                centers_g, cum_g,
                os.path.join(figures_dir, "global_C_topK.png"),
                title="Global cumulative forwards C(t) (top-K cascades)",
                xlabel="Delay (seconds)",
                ylabel="C(t)",
                logx=True
            )
            plot_curve(
                centers_g, lamb_g,
                os.path.join(figures_dir, "global_lambda_topK.png"),
                title=f"Global forwarding rate λ(t) (bin={args.bin_hours}h)",
                xlabel="Delay (seconds)",
                ylabel="λ(t) (forwards/sec)",
                logx=True
            )

    print("DONE. Outputs in:", out_dir)


if __name__ == "__main__":
    main()
