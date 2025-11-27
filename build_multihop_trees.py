#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pushshift Telegram Dataset - Multi-hop Forwarding Trees
Build real diffusion trees for previously selected top-K cascades.

Inputs:
  Data/messages.ndjson.zst
  Results/cascades/cascades_summary_topK.csv   (or all_cascade_sizes.csv)

Outputs:
  Results/trees/
    edges_multihop_topK.csv.gz
    nodes_multihop_topK.csv.gz
    trees_summary_topK.csv
    figures/
"""

import os
import sys
import json
import gzip
import argparse
from collections import defaultdict, Counter
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
# Utils
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
    if peer is None:
        return None
    if isinstance(peer, int):
        return peer
    if isinstance(peer, dict):
        return peer.get("channel_id") or peer.get("chat_id") or peer.get("user_id")
    return None

def get_message_id(msg):
    return msg.get("id") or msg.get("_id") or msg.get("message_id")

def get_channel_id(msg):
    cid = msg.get("peer_id") or msg.get("to_id") or msg.get("chat_id") or msg.get("channel_id")
    return extract_peer_id(cid)

def get_forward_key(msg):
    """
    Cascade key = (orig_peer, orig_post_or_orig_date_ts)
    same as prev script to stay consistent.
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
        return (orig_peer, int(orig_date.timestamp()))
    return None

def get_forward_parent_locator(msg):
    """
    Locator of the *immediate source* message of this forward:
      (parent_peer, parent_post_or_parent_date_ts)
    This may refer to root OR another forwarded node.
    """
    fwd = msg.get("fwd_from") or msg.get("forward")
    if not isinstance(fwd, dict):
        return None

    p_peer = extract_peer_id(fwd.get("from_id") or fwd.get("peer_id"))
    p_post = fwd.get("channel_post") or fwd.get("saved_from_msg_id")
    p_date = safe_parse_date(fwd.get("date"))

    if p_peer is None:
        return None
    if p_post is not None:
        return (p_peer, int(p_post))
    if p_date is not None:
        return (p_peer, int(p_date.timestamp()))
    return None


# -------------------------
# Load selected cascades
# -------------------------

def load_selected_keys(cascades_csv, top_k=None):
    df = pd.read_csv(cascades_csv)
    if top_k is not None and "size" in df.columns:
        df = df.sort_values("size", ascending=False).head(top_k)
    keys = set(zip(df["cascade_peer"].astype(int), df["cascade_post"].astype(int)))
    return keys


# -------------------------
# Main extraction (single pass)
# -------------------------

def extract_forward_nodes(messages_path, selected_keys):
    """
    Collect all forwarded nodes for selected cascades.
    Return per-cascade lists of nodes with parent locators.
    """
    per_cascade = defaultdict(list)

    for msg in tqdm(open_zst_ndjson(messages_path), desc="Collecting forwarded nodes"):
        key = get_forward_key(msg)
        if key is None or key not in selected_keys:
            continue

        child_id = get_message_id(msg)
        child_cid = get_channel_id(msg)
        child_time = safe_parse_date(msg.get("date"))
        if child_id is None or child_time is None:
            continue

        parent_loc = get_forward_parent_locator(msg)

        per_cascade[key].append({
            "child_id": child_id,
            "child_time": child_time,
            "child_channel": child_cid,
            "parent_locator": parent_loc  # (peer,post/date_ts) of immediate source
        })

    return per_cascade


def build_tree_for_cascade(key, items):
    """
    Build multi-hop tree:
      - virtual root = root_peer/root_post
      - if parent_locator matches any forwarded node -> edge to that node
      - else edge from root
    Returns networkx.DiGraph and stats.
    """
    root_peer, root_post = key
    root_id = f"root_{root_peer}_{root_post}"

    # map locator -> node id for forwarded nodes
    # We don't know locator of forwarded copy directly, but we can approximate by
    # its own (channel_id, message_id) if present.
    locator_to_node = {}

    for it in items:
        # locator of this forwarded copy (where it lives)
        # Use (child_channel, child_id) as a locator candidate.
        # This allows multi-hop linking when a forward references another forward by id.
        loc = (it["child_channel"], int(it["child_id"])) if it["child_channel"] is not None else None
        if loc is not None:
            locator_to_node[loc] = it["child_id"]

    G = nx.DiGraph()
    G.add_node(root_id, node_type="root", time=None, channel_id=None)

    multi_hop_edges = 0
    delays_edges = []

    # add forwarded nodes
    for it in items:
        cid = it["child_id"]
        ctime = it["child_time"]
        cchan = it["child_channel"]
        G.add_node(cid, node_type="forward", time=ctime, channel_id=cchan)

    for it in items:
        cid = it["child_id"]
        ctime = it["child_time"]
        parent = root_id

        ploc = it["parent_locator"]
        if ploc is not None:
            # If parent_locator refers to a forwarded copy,
            # it should match (child_channel, child_id) of some node.
            # But parent_locator is (peer,post/date_ts) from fwd_from.
            # We attempt two matches:
            # 1) exact matching of post id in same channel (peer==channel)
            # 2) fallback by root
            p_peer, p_post = ploc
            match1 = (p_peer, p_post)
            if match1 in locator_to_node:
                parent = locator_to_node[match1]
                multi_hop_edges += 1

        G.add_edge(parent, cid)

        # delay on edge:
        if parent == root_id:
            # delay wrt root time (if known in fwd_from)
            ploc = it["parent_locator"]
            if ploc is not None:
                # parent locator time could be root time; not always accessible
                pass
        else:
            ptime = G.nodes[parent].get("time")
            if ptime is not None:
                delays_edges.append((ctime - ptime).total_seconds())

    # compute depth
    try:
        depths = nx.single_source_shortest_path_length(G, root_id)
        depth_vals = [d for n, d in depths.items() if n != root_id]
        max_depth = max(depth_vals) if depth_vals else 0
        avg_depth = float(np.mean(depth_vals)) if depth_vals else 0.0
    except Exception:
        max_depth = avg_depth = 0

    # branching factors
    out_degs = [G.out_degree(n) for n in G.nodes() if n != root_id]
    branching_mean = float(np.mean(out_degs)) if out_degs else 0.0

    # temporal span
    times = [it["child_time"] for it in items]
    t_span = (max(times) - min(times)).total_seconds() if times else None

    stats = {
        "cascade_peer": root_peer,
        "cascade_post": root_post,
        "size": len(items),
        "multi_hop_edges": multi_hop_edges,
        "multi_hop_share": multi_hop_edges / len(items) if items else 0.0,
        "max_depth": max_depth,
        "avg_depth": avg_depth,
        "branching_mean": branching_mean,
        "temporal_span_sec": t_span,
        "edge_delay_mean_sec": float(np.mean(delays_edges)) if delays_edges else None,
        "edge_delay_median_sec": float(np.median(delays_edges)) if delays_edges else None,
    }

    return G, stats


def save_global_outputs(per_cascade_graphs, per_cascade_stats, out_dir):
    edges_path = os.path.join(out_dir, "edges_multihop_topK.csv.gz")
    nodes_path = os.path.join(out_dir, "nodes_multihop_topK.csv.gz")
    summary_path = os.path.join(out_dir, "trees_summary_topK.csv")
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    # write edges/nodes
    ef = gzip.open(edges_path, "wt", encoding="utf-8")
    nf = gzip.open(nodes_path, "wt", encoding="utf-8")

    ef.write("cascade_peer,cascade_post,parent,child\n")
    nf.write("node_id,cascade_peer,cascade_post,node_type,time,channel_id\n")

    for key, G in per_cascade_graphs.items():
        peer, post = key
        for n, data in G.nodes(data=True):
            t = data.get("time")
            nf.write(f"{n},{peer},{post},{data.get('node_type')},{t.isoformat() if t else ''},{data.get('channel_id')}\n")
        for u, v in G.edges():
            ef.write(f"{peer},{post},{u},{v}\n")

    ef.close()
    nf.close()

    df_sum = pd.DataFrame(per_cascade_stats).sort_values("size", ascending=False)
    df_sum.to_csv(summary_path, index=False)

    # -------- figures for paper --------

    # depth distribution
    plt.figure(figsize=(7,5))
    plt.hist(df_sum["max_depth"], bins=30)
    plt.yscale("log")
    plt.title("Max depth distribution of multi-hop forwarding trees (top-K)")
    plt.xlabel("Max depth")
    plt.ylabel("Count (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hist_max_depth_topK.png"), dpi=220)
    plt.close()

    # multi-hop share
    plt.figure(figsize=(7,5))
    plt.hist(df_sum["multi_hop_share"], bins=40)
    plt.title("Share of multi-hop forwards per cascade (top-K)")
    plt.xlabel("Multi-hop share")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hist_multihop_share_topK.png"), dpi=220)
    plt.close()

    # branching
    plt.figure(figsize=(7,5))
    plt.hist(df_sum["branching_mean"], bins=40)
    plt.yscale("log")
    plt.title("Mean branching factor distribution (top-K)")
    plt.xlabel("Mean out-degree")
    plt.ylabel("Count (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hist_branching_mean_topK.png"), dpi=220)
    plt.close()

    return edges_path, nodes_path, summary_path, fig_dir


# -------------------------
# Driver
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="Data")
    ap.add_argument("--cascades_csv", type=str, default="Results/cascades/cascades_summary_topK.csv",
                    help="CSV with selected top-K cascades")
    ap.add_argument("--out_dir", type=str, default="Results/trees")
    ap.add_argument("--top_k", type=int, default=None,
                    help="Optionally re-apply top_k on cascades_csv")
    args = ap.parse_args()

    messages_path = os.path.join(args.data_dir, "messages.ndjson.zst")
    if not os.path.exists(messages_path):
        print("ERROR: messages.ndjson.zst not found")
        sys.exit(1)
    if not os.path.exists(args.cascades_csv):
        print("ERROR: cascades_csv not found:", args.cascades_csv)
        sys.exit(1)

    ensure_dir(args.out_dir)

    print("Loading selected cascades...")
    selected_keys = load_selected_keys(args.cascades_csv, top_k=args.top_k)
    print("Selected keys:", len(selected_keys))

    print("\n=== Pass: collecting forwarded nodes for selected cascades ===")
    per_cascade_items = extract_forward_nodes(messages_path, selected_keys)

    print("\n=== Building multi-hop trees ===")
    per_cascade_graphs = {}
    per_cascade_stats = []

    for key, items in tqdm(per_cascade_items.items(), desc="Building trees"):
        G, stats = build_tree_for_cascade(key, items)
        per_cascade_graphs[key] = G
        per_cascade_stats.append(stats)

    print("\n=== Saving outputs ===")
    edges_path, nodes_path, summary_path, fig_dir = save_global_outputs(
        per_cascade_graphs, per_cascade_stats, args.out_dir
    )

    print("\nDONE.")
    print("Edges:", edges_path)
    print("Nodes:", nodes_path)
    print("Summary:", summary_path)
    print("Figures:", fig_dir)


if __name__ == "__main__":
    main()
