#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pushshift Telegram Dataset - basic profiling & visualization
Author: Oleksandr Kuznetsov (project continuation)
Goal:
  - Stream-read *.ndjson.zst files
  - Compute core dataset stats:
      * #channels, #accounts, #messages
      * users per channel distribution
      * messages per channel distribution
      * time series: messages per month
      * forwarded messages share
      * message length distribution
  - Save publication-ready figures (.png) and tables (.csv/.json)

Usage:
  python analysis_telegram_pushshift.py --data_dir Data --out_dir Results
"""

import os
import re
import sys
import json
import math
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


# -------------------------
# Utilities
# -------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def human(n: int) -> str:
    """Human-readable integer."""
    for u in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.1f}{u}"
        n /= 1000
    return f"{n:.1f}P"

def open_zst_ndjson(path: str, chunk_bytes: int = 1 << 20):
    """
    Stream NDJSON.ZST line by line.
    Yields decoded JSON dict.
    """
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
                        # skip corrupted line
                        continue
            if buffer.strip():
                try:
                    yield orjson.loads(buffer)
                except orjson.JSONDecodeError:
                    pass


def save_df(df: pd.DataFrame, out_csv: str, out_json: str = None):
    df.to_csv(out_csv, index=False)
    if out_json:
        df.to_json(out_json, orient="records", indent=2)


def plot_cdf(values, title, xlabel, out_path, logx=True):
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    values = values[values >= 0]
    values.sort()
    if len(values) == 0:
        return
    y = np.arange(1, len(values) + 1) / len(values)

    plt.figure(figsize=(7,5))
    plt.plot(values, y)
    if logx:
        plt.xscale("log")
    plt.ylim(0, 1.0)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_timeseries_monthly(counter_by_month: Counter, title, ylabel, out_path):
    # counter keys like "2019-08"
    items = sorted(counter_by_month.items())
    if not items:
        return
    months = [k for k, _ in items]
    counts = [v for _, v in items]

    plt.figure(figsize=(9,4.8))
    plt.plot(months, counts)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, ls="--", alpha=0.3)
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def safe_parse_date(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        # unix seconds
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


# -------------------------
# Profiling steps
# -------------------------

def profile_channels(channels_path: str, out_tables: str):
    """
    Read channels.ndjson.zst
    Basic stats: count, creation month distribution, member_count, etc.
    """
    n_channels = 0
    created_by_month = Counter()
    members = []

    for row in tqdm(open_zst_ndjson(channels_path), desc="Reading channels"):
        n_channels += 1
        # fields depend on Telethon schema
        dt = safe_parse_date(row.get("date") or row.get("created_at"))
        if dt:
            created_by_month[dt.strftime("%Y-%m")] += 1

        mc = row.get("participants_count") or row.get("members_count") or row.get("users") or row.get("user_count")
        if isinstance(mc, int):
            members.append(mc)

    df_month = pd.DataFrame(sorted(created_by_month.items()), columns=["month", "channels_created"])
    save_df(df_month,
            os.path.join(out_tables, "channels_created_by_month.csv"),
            os.path.join(out_tables, "channels_created_by_month.json"))

    stats = {
        "n_channels": n_channels,
        "members_mean": float(np.mean(members)) if members else None,
        "members_median": float(np.median(members)) if members else None,
    }
    with open(os.path.join(out_tables, "channels_summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats, members, created_by_month


def profile_accounts(accounts_path: str, out_tables: str):
    """
    Read accounts.ndjson.zst
    Basic stats: accounts count, bot share, verified share, etc.
    """
    n_accounts = 0
    bots = 0
    verified = 0

    for row in tqdm(open_zst_ndjson(accounts_path), desc="Reading accounts"):
        n_accounts += 1
        if row.get("bot") is True:
            bots += 1
        if row.get("verified") is True:
            verified += 1

    stats = {
        "n_accounts": n_accounts,
        "bots": bots,
        "bots_share": bots / n_accounts if n_accounts else 0.0,
        "verified": verified,
        "verified_share": verified / n_accounts if n_accounts else 0.0,
    }
    with open(os.path.join(out_tables, "accounts_summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def profile_messages(messages_path: str, out_tables: str,
                     max_channels_for_full_counts: int = 3_000_000,
                     sample_channel_counts: bool = False,
                     sample_rate: float = 0.1):
    """
    Read messages.ndjson.zst (huge)
    We compute:
      - total messages
      - messages per month
      - messages per channel distribution
      - forwarded messages share
      - message length distribution (chars)
    To keep memory safe:
      - channel_counts stored in dict; number of channels ~27k so ok
      - length sample stored in list (cap)
    """
    n_messages = 0
    forwarded = 0
    empty_text = 0

    messages_by_month = Counter()
    channel_counts = defaultdict(int)

    lengths = []
    max_len_samples = 2_000_000  # cap to avoid huge RAM

    rng = np.random.default_rng(42)

    for row in tqdm(open_zst_ndjson(messages_path), desc="Reading messages"):
        n_messages += 1

        # channel id
        cid = row.get("peer_id") or row.get("to_id") or row.get("chat_id") or row.get("channel_id")
        if isinstance(cid, dict):
            cid = cid.get("channel_id") or cid.get("chat_id") or cid.get("user_id")
        if cid is not None:
            channel_counts[cid] += 1

        # timestamp -> month
        dt = safe_parse_date(row.get("date"))
        if dt:
            messages_by_month[dt.strftime("%Y-%m")] += 1

        # forwarded?
        if row.get("fwd_from") is not None or row.get("forward") is not None:
            forwarded += 1

        # text length
        txt = row.get("message") or row.get("text")
        if txt is None or txt == "":
            empty_text += 1
            l = 0
        else:
            l = len(txt)

        # sampling for length distribution
        if len(lengths) < max_len_samples:
            if (not sample_channel_counts) or (rng.random() < sample_rate):
                lengths.append(l)

        if n_messages % 5_000_000 == 0:
            # lightweight progress snapshot
            pass

    # Save tables
    df_month = pd.DataFrame(sorted(messages_by_month.items()), columns=["month", "messages"])
    save_df(df_month,
            os.path.join(out_tables, "messages_by_month.csv"),
            os.path.join(out_tables, "messages_by_month.json"))

    df_counts = pd.DataFrame(
        [{"channel_id": k, "messages": v} for k, v in channel_counts.items()]
    )
    save_df(df_counts,
            os.path.join(out_tables, "messages_per_channel.csv"))

    stats = {
        "n_messages": n_messages,
        "n_forwarded": forwarded,
        "forwarded_share": forwarded / n_messages if n_messages else 0.0,
        "empty_text": empty_text,
        "empty_text_share": empty_text / n_messages if n_messages else 0.0,
        "n_channels_observed": len(channel_counts),
        "msgs_per_channel_mean": float(df_counts["messages"].mean()) if len(df_counts) else None,
        "msgs_per_channel_median": float(df_counts["messages"].median()) if len(df_counts) else None,
    }
    with open(os.path.join(out_tables, "messages_summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats, channel_counts, messages_by_month, lengths


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="Data", help="Folder with *.ndjson.zst files")
    ap.add_argument("--out_dir", type=str, default="Results", help="Output folder for figures/tables")
    args = ap.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir

    figures_dir = os.path.join(out_dir, "figures")
    tables_dir  = os.path.join(out_dir, "tables")
    logs_dir    = os.path.join(out_dir, "logs")
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)
    ensure_dir(logs_dir)

    accounts_path = os.path.join(data_dir, "accounts.ndjson.zst")
    channels_path = os.path.join(data_dir, "channels.ndjson.zst")
    messages_path = os.path.join(data_dir, "messages.ndjson.zst")

    if not os.path.exists(accounts_path) or not os.path.exists(channels_path) or not os.path.exists(messages_path):
        print("ERROR: Dataset files not found in data_dir.")
        print(accounts_path, channels_path, messages_path)
        sys.exit(1)

    # 1) channels
    ch_stats, members, created_by_month = profile_channels(channels_path, tables_dir)

    # Figures for channels
    plot_timeseries_monthly(
        created_by_month,
        title="Number of Telegram channels created per month",
        ylabel="# channels created",
        out_path=os.path.join(figures_dir, "channels_created_by_month.png")
    )

    if members:
        plot_cdf(
            members,
            title="CDF of the number of registered users per channel",
            xlabel="# users per channel",
            out_path=os.path.join(figures_dir, "cdf_users_per_channel.png"),
            logx=True
        )

    # 2) accounts
    acc_stats = profile_accounts(accounts_path, tables_dir)

    # 3) messages (heavy)
    msg_stats, channel_counts, messages_by_month, lengths = profile_messages(messages_path, tables_dir)

    # Figures for messages
    plot_timeseries_monthly(
        messages_by_month,
        title="Monthly number of Telegram messages (non-status + status)",
        ylabel="# messages",
        out_path=os.path.join(figures_dir, "messages_by_month.png")
    )

    plot_cdf(
        list(channel_counts.values()),
        title="CDF of the number of messages per channel",
        xlabel="# messages per channel",
        out_path=os.path.join(figures_dir, "cdf_messages_per_channel.png"),
        logx=True
    )

    if lengths:
        plot_cdf(
            lengths,
            title="CDF of message length (characters)",
            xlabel="# characters per message",
            out_path=os.path.join(figures_dir, "cdf_message_length.png"),
            logx=True
        )

    # 4) global summary table for paper
    global_summary = {
        **ch_stats,
        **acc_stats,
        **msg_stats
    }
    df_global = pd.DataFrame([global_summary])
    save_df(df_global,
            os.path.join(tables_dir, "global_summary.csv"),
            os.path.join(tables_dir, "global_summary.json"))

    print("\nDONE.")
    print(f"Figures saved to: {figures_dir}")
    print(f"Tables saved to:  {tables_dir}")
    print("Key stats:")
    for k, v in global_summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
