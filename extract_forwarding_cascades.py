#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pushshift Telegram Dataset - Forwarding Cascades Extraction (2-pass)
Pass 1: count sizes of all forwarding cascades
Pass 2: extract top-K / size-filtered cascades, compute delays + summary stats

Outputs:
 Results/cascades/
   edges_topK.csv.gz            # parent->child edges for selected cascades
   nodes_topK.csv.gz            # node metadata (timestamps, channel)
   cascades_summary_topK.csv    # per-cascade metrics (size, depth proxy, delays)
   figures/                     # histograms/CDFs for paper
"""

import os
import sys
import json
import gzip
import argparse
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import zstandard as zstd
import orjson
from dateutil import parser as dateparser
import networkx as nx


# -------------------------
# Utilities (same style as previous script)
# -------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def open_zst_ndjson(path: str, chunk_bytes: int = 1 << 20):
    """Stream NDJSON.ZST line by line."""
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            buffer = b""
            while True:
                chunk = reader.read(chunk_bytes)
                if not chunk:
                    break
                buffer += chunk
                while True:
                    nl = buffer.find(b"\n")
                    if nl == -1:
                        break
                    line = buffer[:nl]
                    buffer = buffer[nl+1:]
                    if not line.strip():
                        continue
                    try:
                        yield orjson.loads(line)
                    except orjson.JSONDecodeError:
                        continue
            if buffer.strip():
                try:
                    yield orjson.loads(buffer)
                except orjson.JSONDecodeError:
                    pass

def safe_parse_date(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            return datetime.utcfromtimestamp(x)
        except Exception:
            return None
    if isinstance(x, str):
        try:
            return dateparser.parse(x)
        except Exception:
            return None
    return None


def extract_peer_id(peer):
    """
    Peer can be:
      - int
      - dict with channel_id/chat_id/user_id
    """
    if peer is None:
        return None
    if isinstance(peer, int):
        return peer
    if isinstance(peer, dict):
        return peer.get("channel_id") or peer.get("chat_id") or peer.get("user_id")
    return None


def get_forward_key(msg):
    """
    Build cascade key for forwarded messages.
    We look into:
      msg["fwd_from"]["from_id"] (original peer)
      msg["fwd_from"]["channel_post"] (original msg id, optional)
      msg["fwd_from"]["date"] (original date)
    """
    fwd = msg.get("fwd_from") or msg.get("forward")
    if not isinstance(fwd, dict):
        return None

    orig_peer = extract_peer_id(fwd.get("from_id") or fwd.get("peer_id"))
    orig_post = fwd.get("channel_post") or fwd.get("saved_from_msg_id")
    orig_date = safe_parse_date(fwd.get("date"))

    if orig_peer is None:
        return None

    if orig_post is not None:
        return (orig_peer, int(orig_post))
    if orig_date is not None:
        # fallback to original date (seconds resolution)
        return (orig_peer, int(orig_date.timestamp()))
    return None


def get_message_id(msg):
    return msg.get("id") or msg.get("_id") or msg.get("message_id")

def get_channel_id(msg):
    cid = msg.get("peer_id") or msg.get("to_id") or msg.get("chat_id") or msg.get("channel_id")
    return extract_peer_id(cid)

# -------------------------
# Pass 1: count cascade sizes
# -------------------------

def pass1_count_cascades(messages_path):
    cascade_counts = Counter()

    for msg in tqdm(open_zst_ndjson(messages_path), desc="Pass1 counting cascades"):
        key = get_forward_key(msg)
        if key is not None:
            cascade_counts[key] += 1

    return cascade_counts


# -------------------------
# Pass 2: extract top-K cascades
# -------------------------

def pass2_extract(messages_path, selected_keys, out_dir,
                  max_children_per_cascade=None):
    """
    Extract edges and node metadata for selected cascades.
    We treat original message as root node (virtual).
    For each forwarded message:
        root -> forwarded_copy
    If you want multi-hop trees later, we can refine using forward chains.
    """
    edges_path = os.path.join(out_dir, "edges_topK.csv.gz")
    nodes_path = os.path.join(out_dir, "nodes_topK.csv.gz")

    # We'll stream-write CSV to gzip to avoid RAM blowup
    edges_f = gzip.open(edges_path, "wt", encoding="utf-8")
    nodes_f = gzip.open(nodes_path, "wt", encoding="utf-8")

    edges_f.write("cascade_peer,cascade_post,parent_node,child_node,delay_sec,child_time,child_channel\n")
    nodes_f.write("node_id,cascade_peer,cascade_post,node_type,time,channel_id\n")

    # Count children per cascade while extracting (optional cap)
    children_seen = Counter()

    for msg in tqdm(open_zst_ndjson(messages_path), desc="Pass2 extracting cascades"):
        key = get_forward_key(msg)
        if key is None or key not in selected_keys:
            continue

        orig_peer, orig_post = key

        # cap very huge cascades if desired
        if max_children_per_cascade is not None:
            if children_seen[key] >= max_children_per_cascade:
                continue

        child_id = get_message_id(msg)
        child_cid = get_channel_id(msg)
        child_time = safe_parse_date(msg.get("date"))
        if child_id is None or child_time is None:
            continue

        # root node id (virtual stable id)
        root_id = f"root_{orig_peer}_{orig_post}"

        delay_sec = None
        # delay relative to original date if we have it
        fwd = msg.get("fwd_from") or msg.get("forward")
        if isinstance(fwd, dict):
            orig_date = safe_parse_date(fwd.get("date"))
            if orig_date:
                delay_sec = int((child_time - orig_date).total_seconds())

        # write edge
        edges_f.write(
            f"{orig_peer},{orig_post},{root_id},{child_id},{delay_sec},{child_time.isoformat()},{child_cid}\n"
        )

        # write nodes (root once, child always)
        if children_seen[key] == 0:
            nodes_f.write(f"{root_id},{orig_peer},{orig_post},root,,\n")

        nodes_f.write(
            f"{child_id},{orig_peer},{orig_post},forward,{child_time.isoformat()},{child_cid}\n"
        )

        children_seen[key] += 1

    edges_f.close()
    nodes_f.close()

    return edges_path, nodes_path


# -------------------------
# Build per-cascade summaries + figures
# -------------------------

def build_summaries(edges_path, out_dir):
    df_edges = pd.read_csv(edges_path, compression="gzip")

    # per cascade stats
    grouped = df_edges.groupby(["cascade_peer", "cascade_post"])
    summary = []

    delays_all = []

    for (peer, post), g in grouped:
        size = len(g)

        delays = g["delay_sec"].dropna().astype(int).values
        if len(delays) > 0:
            delays_all.extend(delays.tolist())
            d_mean = float(np.mean(delays))
            d_med  = float(np.median(delays))
            d_p90  = float(np.percentile(delays, 90))
        else:
            d_mean = d_med = d_p90 = None

        # depth proxy: since our extraction is root->child (star),
        # depth=1 always; but we keep "temporal span" as useful property
        times = pd.to_datetime(g["child_time"], errors="coerce")
        t_span = (times.max() - times.min()).total_seconds() if times.notna().any() else None

        summary.append({
            "cascade_peer": peer,
            "cascade_post": post,
            "size": size,
            "delay_mean_sec": d_mean,
            "delay_median_sec": d_med,
            "delay_p90_sec": d_p90,
            "temporal_span_sec": t_span
        })

    df_sum = pd.DataFrame(summary).sort_values("size", ascending=False)
    sum_path = os.path.join(out_dir, "cascades_summary_topK.csv")
    df_sum.to_csv(sum_path, index=False)

    # Figures for paper
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    # size distribution
    plt.figure(figsize=(7,5))
    plt.hist(df_sum["size"], bins=50)
    plt.yscale("log")
    plt.title("Cascade size distribution (top-K selected)")
    plt.xlabel("Cascade size (# forwards)")
    plt.ylabel("Count (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hist_cascade_sizes_topK.png"), dpi=220)
    plt.close()

    if len(delays_all) > 0:
        delays_all = np.array(delays_all)
        delays_all = delays_all[delays_all >= 0]

        plt.figure(figsize=(7,5))
        plt.hist(delays_all, bins=80)
        plt.yscale("log")
        plt.title("Delay distribution for forwarded messages (top-K)")
        plt.xlabel("Delay (seconds)")
        plt.ylabel("Count (log)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "hist_delays_topK.png"), dpi=220)
        plt.close()

        # CDF delays
        delays_all.sort()
        y = np.arange(1, len(delays_all)+1)/len(delays_all)

        plt.figure(figsize=(7,5))
        plt.plot(delays_all, y)
        plt.xscale("log")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.title("CDF of forwarding delays (top-K)")
        plt.xlabel("Delay (seconds)")
        plt.ylabel("CDF")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "cdf_delays_topK.png"), dpi=220)
        plt.close()

    return sum_path


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="Data")
    ap.add_argument("--out_dir", type=str, default="Results/cascades")
    ap.add_argument("--top_k", type=int, default=5000,
                    help="Select top-K largest cascades")
    ap.add_argument("--min_size", type=int, default=50,
                    help="Also keep cascades with size >= min_size")
    ap.add_argument("--max_children", type=int, default=None,
                    help="Optional cap of children per cascade (e.g., 20000)")
    args = ap.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)

    messages_path = os.path.join(data_dir, "messages.ndjson.zst")
    if not os.path.exists(messages_path):
        print("ERROR: messages.ndjson.zst not found.")
        sys.exit(1)

    print("\n=== PASS 1: Counting cascades ===")
    cascade_counts = pass1_count_cascades(messages_path)

    # select cascades
    top_items = cascade_counts.most_common(args.top_k)
    selected = {k for k, _ in top_items}
    # add all >= min_size
    selected |= {k for k, c in cascade_counts.items() if c >= args.min_size}

    print(f"Total cascades observed: {len(cascade_counts)}")
    print(f"Selected cascades:       {len(selected)}")

    # save cascade counts table
    df_counts = pd.DataFrame(
        [{"cascade_peer": k[0], "cascade_post": k[1], "size": c}
         for k, c in cascade_counts.items()]
    )
    df_counts.to_csv(os.path.join(out_dir, "all_cascade_sizes.csv"), index=False)

    print("\n=== PASS 2: Extracting selected cascades ===")
    edges_path, nodes_path = pass2_extract(
        messages_path,
        selected_keys=selected,
        out_dir=out_dir,
        max_children_per_cascade=args.max_children
    )

    print("\n=== Building summaries & figures ===")
    sum_path = build_summaries(edges_path, out_dir)

    print("\nDONE.")
    print(f"Edges:    {edges_path}")
    print(f"Nodes:    {nodes_path}")
    print(f"Summary:  {sum_path}")
    print(f"Figures:  {os.path.join(out_dir,'figures')}")


if __name__ == "__main__":
    main()
