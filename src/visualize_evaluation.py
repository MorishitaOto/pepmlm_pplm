#!/usr/bin/env python
# ============================================================
# Evaluation summary CSV → PDF 可視化スクリプト
# ページ構成 (11ページ、話題ごとにまとめた順序):
#   1. Figure Guide
#   2. Summary Table
#   --- ① 信頼度スコア ---
#   3. Confidence Scores (confidence / iptm / receptor→peptide iptm / peptide pLDDT)
#   --- ② 指定位置 (cryptic) の近くにペプチドが来ているか ---
#   4. Cryptic Pocket Binding (recall / f1)
#   5. Distance Metrics (centroid dist / min dist / holo ligand centroid dist)
#   --- ③ Crypticity (クリプティックになっているか) ---
#   6. Pocket Volume at Peptide Binding Site (apo / predicted / delta)
#   7. RSA (apo / predicted / delta)
#   8. Backbone RMSD + CryptoBank Crypticity Score
#   --- 補助 ---
#   9. Scatter Plots (recall vs iptm / min dist vs f1)
#  10. Interface Residue Heatmap
#  11. Radar Chart
# ============================================================

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from pypdf import PdfWriter, PdfReader

BLUE   = "#4477AA"
RED    = "#EE6677"
GREEN  = "#228833"
ORANGE = "#CCBB44"
PURPLE = "#AA3377"
GREY   = "#BBBBBB"
DARK   = "#222222"
BG     = "#F8F8F8"
PALETTE = [BLUE, RED, GREEN, ORANGE, PURPLE, "#66CCEE", "#AA3377",
           "#BBBBBB", "#44BB99", "#DDCC77"]


def short_name(name: str) -> str:
    parts = name.split("_")
    for p in parts:
        if p.startswith("top"):
            return p
    return name


def page_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": BG,
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def add_page_title(fig, title, subtitle=""):
    fig.text(0.5, 0.97, title, ha="center", va="top",
             fontsize=14, fontweight="bold", color=DARK)
    if subtitle:
        fig.text(0.5, 0.945, subtitle, ha="center", va="top",
                 fontsize=9, color="#666666")


def hbar(ax, labels, values, colors, xlabel, title, vlines=None, xlim=None):
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, pad=6)
    if xlim:
        ax.set_xlim(xlim)
    if vlines:
        for v in vlines:
            ax.axvline(v, color=DARK, lw=0.8, ls="--", alpha=0.5)
    xmax = ax.get_xlim()[1]
    for bar, val in zip(bars, values):
        if np.isfinite(val):
            ax.text(bar.get_width() + xmax * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7, color=DARK)
    ax.invert_yaxis()


# ── Page 1: Figure Guide ────────────────────────────────────
def page_guide_fig():
    fig = plt.figure(figsize=(14, 10))
    page_style()
    fig.patch.set_facecolor("white")
    add_page_title(fig, "Figure Guide  —  How to Read Each Figure")

    guide_text = [
        ("Page 2  Summary Table",
         "Overview of key metrics. Conf=Confidence Score, ipTM=interface pTM, "
         "Rec→Pep ipTM=pair_chains_iptm[0→1], Crypt.Recall=fraction of cryptic residues contacted."),

        ("─── ① Confidence Scores ───", ""),
        ("Page 3  Confidence Scores",
         "Boltz-2 structural confidence. confidence_score / ipTM (overall) / "
         "ipTM (receptor→peptide) [pair_chains_iptm 0→1]: binding position validity as seen from receptor / "
         "Peptide pLDDT mean. Range 0-1; higher is better."),

        ("─── ② Does the peptide bind near the cryptic site? ───", ""),
        ("Page 4  Cryptic Pocket Binding Metrics",
         "Recall = fraction of cryptic residues contacted by the peptide. "
         "F1 = harmonic mean of recall and precision. "
         "Range 0-1; higher is better."),

        ("Page 5  Distance Metrics",
         "Centroid dist: distance between peptide centroid and cryptic pocket centroid. "
         "Min dist: shortest atom-atom distance to cryptic pocket. "
         "Holo ligand centroid dist: distance between peptide centroid and holo ligand centroid "
         "(after superimposing receptor chains). Smaller is better."),

        ("─── ③ Crypticity: is the binding site really cryptic? ───", ""),
        ("Page 6  Pocket Volume at Peptide Binding Site",
         "fpocket pocket volume, selected by maximum overlap with the peptide interface residues "
         "(i.e., where the peptide actually binds). Apo / Predicted / Delta. "
         "ΔVolume > 0 in predicted = pocket opens up at the peptide binding site."),

        ("Page 7  Cryptic Residue RSA",
         "RSA = ASA / MAX_ASA (Tien 2013 theoretical values). "
         "0 = fully buried, 1 = fully exposed. "
         "Delta RSA > 0 means more exposed in predicted (pocket opens)."),

        ("Page 8  Backbone RMSD & CryptoBank Crypticity Score",
         "Left: backbone (N,CA,C,O) RMSD of cryptic residues after global CA superimpose vs apo. "
         "Larger = more conformational change. "
         "Right: CryptoBank crypticity score (0–1) computed by the concentric-shell model. "
         "Score ≥ 0.5 (dashed line) = cryptic pocket detected. "
         "The score quantifies how much the protein environment around the peptide (as ligand) "
         "differs between the apo and predicted states."),

        ("─── Supplementary ───", ""),
        ("Page 9  Scatter Plots",
         "Left: Cryptic Recall vs ipTM (color=Confidence). Upper-right is ideal. "
         "Right: Min distance to cryptic pocket vs Cryptic F1 (color=ipTM). Upper-left is ideal."),

        ("Page 10  Interface Residue Heatmap",
         "X-axis = target residue ID, Y-axis = peptide. Blue = contact. "
         "Orange border = cryptic pocket residue."),

        ("Page 11  Radar Chart",
         "Multi-axis comparison normalized 0-1. "
         "Axes where lower is better (RMSD, distance) are inverted. Larger area = better overall."),
    ]

    ax = fig.add_axes([0.03, 0.02, 0.94, 0.88])
    ax.axis("off")
    y = 0.97
    for title, desc in guide_text:
        if not desc:
            # セクション見出し: 太字＋やや大きめ、説明枠なし
            ax.text(0.0, y, title, transform=ax.transAxes,
                    fontsize=10, fontweight="bold", color=DARK, va="top")
            y -= 0.040
            continue
        ax.text(0.0, y, title, transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=BLUE, va="top")
        y -= 0.030
        ax.text(0.02, y, desc, transform=ax.transAxes,
                fontsize=7.5, color=DARK, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#F0F4FF",
                          edgecolor="#CCDDFF", linewidth=0.5))
        y -= 0.048
    return fig


# ── Page 2: Summary Table ────────────────────────────────────
def page_summary_table_fig(df, run_title):
    fig = plt.figure(figsize=(14, 10))
    page_style()
    fig.text(0.5, 0.94, "Peptide-Target Prediction Evaluation Report",
             ha="center", fontsize=18, fontweight="bold", color=DARK)
    fig.text(0.5, 0.90, run_title, ha="center", fontsize=12, color="#444444")
    fig.text(0.5, 0.87, f"n = {len(df)} peptides",
             ha="center", fontsize=10, color="#666666")

    cols_display = {
        "name": "Name",
        "peptide_sequence": "Sequence",
        "confidence_score": "Conf.",
        "iptm": "ipTM",
        "receptor_to_peptide_iptm": "Rec→Pep ipTM",
        "cryptic_recall": "Recall",
        "cryptic_f1": "F1",
        "min_dist_peptide_to_cryptic": "MinDist(Å)",
        "peptide_centroid_to_holo_ligand_centroid_dist": "HoloDist(Å)",
        "delta_volume": "ΔVol(Å³)",
        "delta_cryptic_rsa": "ΔRSA",
        "cryptic_backbone_rmsd_vs_apo": "BBRMSD",
        "crypticity_score": "CrypticScore",
    }
    sub = df[[c for c in cols_display if c in df.columns]].copy()
    sub.columns = [cols_display[c] for c in sub.columns]
    for col in sub.columns:
        try:
            sub[col] = sub[col].apply(
                lambda x: f"{x:.3f}" if isinstance(x, float) else x)
        except Exception:
            pass

    ax = fig.add_axes([0.02, 0.05, 0.96, 0.78])
    ax.axis("off")
    tbl = ax.table(cellText=sub.values, colLabels=sub.columns,
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EEF2FF")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#CCCCCC")
    return fig


# ── Page 3: Confidence Scores ────────────────────────────────
def page_confidence_fig(df):
    metrics = [
        ("confidence_score",         "Confidence Score",           (0, 1)),
        ("iptm",                     "ipTM (overall)",             (0, 1)),
        ("receptor_to_peptide_iptm", "ipTM (receptor→peptide)",    (0, 1)),
        ("peptide_plddt_mean",       "Peptide pLDDT (mean)",       (0, 1)),
    ]
    available = [(c, t, xl) for c, t, xl in metrics
                 if c in df.columns and not df[c].isnull().all()]
    n = len(available)
    if n == 0:
        return None

    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
    page_style()
    add_page_title(fig, "Structural Confidence Scores",
                   "receptor→peptide ipTM: Boltz pair_chains_iptm[0→1] "
                   "— peptide binding position validity as seen from receptor")
    fig.subplots_adjust(hspace=0.45, wspace=0.4, top=0.88)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    labels = [short_name(n_) for n_ in df["name"]]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    for ax, (col, title, xlim) in zip(axes_flat, available):
        vals = df[col].fillna(0).values
        hbar(ax, labels, vals, colors, "", title, vlines=[0.7, 0.8], xlim=xlim)
    for ax in axes_flat[len(available):]:
        ax.set_visible(False)
    return fig


# ── Page 4: Cryptic Pocket Binding (recall / f1) ─────────────
def page_cryptic_metrics_fig(df):
    metrics = [
        ("cryptic_recall", "Cryptic Pocket Recall", "Recall"),
        ("cryptic_f1",     "Cryptic Pocket F1",     "F1"),
    ]
    available = [(c, t, xl) for c, t, xl in metrics if c in df.columns]
    if not available:
        return None

    fig, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 6))
    page_style()
    add_page_title(fig, "Cryptic Pocket Binding Metrics",
                   "How well does the peptide target the cryptic pocket residues?")
    fig.subplots_adjust(top=0.85, wspace=0.4)
    if len(available) == 1:
        axes = [axes]
    labels = [short_name(n_) for n_ in df["name"]]

    def green_colors(vals):
        cmap = plt.get_cmap("Greens")
        norm = plt.Normalize(0, 1)
        return [cmap(norm(v)) for v in vals]

    for ax, (col, title, xlabel) in zip(axes, available):
        vals = df[col].fillna(0).values
        hbar(ax, labels, vals, green_colors(vals), xlabel, title, xlim=(0, 1))
    return fig


# ── Page 5: Distance Metrics ─────────────────────────────────
def page_distances_fig(df):
    metrics = [
        ("peptide_centroid_to_cryptic_centroid_dist",
         "Centroid-Cryptic Distance",            "Distance (Å)"),
        ("min_dist_peptide_to_cryptic",
         "Min Peptide-Cryptic Distance",          "Min Distance (Å)"),
        ("peptide_centroid_to_holo_ligand_centroid_dist",
         "Centroid-Holo Ligand Distance",         "Distance (Å)"),
    ]
    available = [(c, t, xl) for c, t, xl in metrics
                 if c in df.columns and not df[c].isnull().all()]
    if not available:
        return None

    ncols = len(available)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    page_style()
    add_page_title(fig, "Peptide-Pocket Distance Metrics",
                   "Smaller = peptide closer to target site  |  "
                   "Holo ligand dist: peptide centroid vs ligand centroid after receptor superimpose")
    fig.subplots_adjust(top=0.85, wspace=0.45)
    if ncols == 1:
        axes = [axes]
    labels = [short_name(n_) for n_ in df["name"]]

    def dist_colors(vals):
        cmap = plt.get_cmap("RdYlGn_r")
        mn, mx = np.nanmin(vals), np.nanmax(vals)
        norm = plt.Normalize(mn, mx)
        return [cmap(norm(v)) for v in vals]

    for ax, (col, title, xlabel) in zip(axes, available):
        vals = df[col].fillna(np.nan).values
        hbar(ax, labels, vals, dist_colors(vals), xlabel, title, vlines=[5.0])
    return fig


# ── Page: Pocket Volume (predicted / apo / delta) ────────────
def page_pocket_volume_fig(df):
    """
    fpocket pocket volume を「ペプチド界面残基ベース」で抽出した結果を表示。
    apo / predicted / delta の3パネル構成。
    """
    cols = ["fpocket_volume_apo", "fpocket_volume_predicted", "delta_volume"]
    if all(c not in df.columns or df[c].isnull().all() for c in cols):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    page_style()
    add_page_title(fig, "Pocket Volume at Peptide Binding Site",
                   "fpocket pocket selected by max overlap with peptide interface residues  |  "
                   "ΔV = predicted − apo (positive = pocket opens)")
    fig.subplots_adjust(top=0.82, wspace=0.45)
    labels = [short_name(n_) for n_ in df["name"]]

    apo_v   = df["fpocket_volume_apo"].fillna(0).values       if "fpocket_volume_apo"       in df.columns else np.zeros(len(df))
    pred_v  = df["fpocket_volume_predicted"].fillna(0).values if "fpocket_volume_predicted" in df.columns else np.zeros(len(df))
    delta_v = df["delta_volume"].fillna(0).values             if "delta_volume"             in df.columns else np.zeros(len(df))

    # Apo
    axes[0].barh(np.arange(len(labels)), apo_v, color=BLUE, alpha=0.8,
                 edgecolor="white", height=0.6)
    axes[0].set_yticks(np.arange(len(labels))); axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Pocket Volume (Å³)")
    axes[0].set_title("Apo Pocket Volume", fontsize=11)
    axes[0].invert_yaxis()

    # Predicted (apoより大きければ緑、小さければ赤)
    colors1 = [GREEN if p >= a else RED for p, a in zip(pred_v, apo_v)]
    axes[1].barh(np.arange(len(labels)), pred_v, color=colors1, alpha=0.8,
                 edgecolor="white", height=0.6)
    axes[1].set_yticks(np.arange(len(labels))); axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("Pocket Volume (Å³)")
    axes[1].set_title("Predicted Pocket Volume", fontsize=11)
    axes[1].invert_yaxis()

    # Delta
    axes[2].barh(np.arange(len(labels)), delta_v,
                 color=[GREEN if v >= 0 else RED for v in delta_v],
                 alpha=0.8, edgecolor="white", height=0.6)
    axes[2].set_yticks(np.arange(len(labels))); axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].axvline(0, color=DARK, lw=1)
    axes[2].set_xlabel("ΔVolume (Å³)")
    axes[2].set_title("Delta Volume (Predicted − Apo)", fontsize=11)
    axes[2].invert_yaxis()

    # 数値ラベル
    for ax, vals in zip(axes, [apo_v, pred_v, delta_v]):
        xmin, xmax = ax.get_xlim()
        span = xmax - xmin if xmax > xmin else 1.0
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(v + span * 0.01, i, f"{v:.1f}",
                        va="center", fontsize=7, color=DARK)

    return fig



# ── Page 7: RSA ─────────────────────────────────────────────
def page_rsa_fig(df):
    rsa_cols = ["cryptic_rsa_apo", "cryptic_rsa_predicted", "delta_cryptic_rsa"]
    available_any = any(c in df.columns and not df[c].isnull().all() for c in rsa_cols)
    if not available_any:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    page_style()
    add_page_title(fig, "Cryptic Residue RSA (Relative SASA)",
                   "RSA = ASA / MAX_ASA (Tien 2013)  |  "
                   "ΔRSA > 0 = more exposed in predicted (pocket opens)")
    fig.subplots_adjust(top=0.82, wspace=0.45)
    labels = [short_name(n_) for n_ in df["name"]]

    apo_v  = df["cryptic_rsa_apo"].fillna(0).values       if "cryptic_rsa_apo"       in df.columns else np.zeros(len(df))
    pred_v = df["cryptic_rsa_predicted"].fillna(0).values  if "cryptic_rsa_predicted"  in df.columns else np.zeros(len(df))
    delta  = df["delta_cryptic_rsa"].fillna(0).values      if "delta_cryptic_rsa"      in df.columns else np.zeros(len(df))

    axes[0].barh(np.arange(len(labels)), apo_v, color=BLUE, alpha=0.8,
                 edgecolor="white", height=0.6)
    axes[0].set_yticks(np.arange(len(labels))); axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("RSA (0–1)"); axes[0].set_title("Apo RSA", fontsize=11)
    axes[0].set_xlim(0, 1); axes[0].invert_yaxis()

    colors1 = [GREEN if p >= a else RED for p, a in zip(pred_v, apo_v)]
    axes[1].barh(np.arange(len(labels)), pred_v, color=colors1, alpha=0.8,
                 edgecolor="white", height=0.6)
    axes[1].set_yticks(np.arange(len(labels))); axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("RSA (0–1)"); axes[1].set_title("Predicted RSA", fontsize=11)
    axes[1].set_xlim(0, 1); axes[1].invert_yaxis()

    axes[2].barh(np.arange(len(labels)), delta,
                 color=[GREEN if v >= 0 else RED for v in delta],
                 alpha=0.8, edgecolor="white", height=0.6)
    axes[2].set_yticks(np.arange(len(labels))); axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].axvline(0, color=DARK, lw=1)
    axes[2].set_xlabel("ΔRSA"); axes[2].set_title("Delta RSA (Predicted − Apo)", fontsize=11)
    axes[2].invert_yaxis()
    return fig


# ── Page 8: Backbone RMSD + CryptoBank Crypticity Score ──────
def page_rmsd_and_contact_fig(df):
    has_rmsd    = "cryptic_backbone_rmsd_vs_apo" in df.columns and not df["cryptic_backbone_rmsd_vs_apo"].isnull().all()
    has_crypticity = "crypticity_score" in df.columns and not df["crypticity_score"].isnull().all()
    if not has_rmsd and not has_crypticity:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    page_style()
    add_page_title(fig, "Backbone RMSD & CryptoBank Crypticity Score",
                   "Left: cryptic residue backbone RMSD (predicted vs apo) after global CA alignment  |  "
                   "Right: CryptoBank concentric-shell crypticity score (≥ 0.5 = cryptic)")
    fig.subplots_adjust(top=0.85, wspace=0.45)
    labels = [short_name(n_) for n_ in df["name"]]

    # ── 左: backbone RMSD ────────────────────────────────────────
    ax1 = axes[0]
    if has_rmsd:
        vals = df["cryptic_backbone_rmsd_vs_apo"].fillna(np.nan).values
        cmap = plt.get_cmap("YlOrRd")
        vmax = max(np.nanmax(vals), 1.0)
        norm = plt.Normalize(0, vmax)
        colors = [cmap(norm(v)) if np.isfinite(v) else GREY for v in vals]
        hbar(ax1, labels, vals, colors, "RMSD (Å)", "Backbone RMSD vs Apo")
    else:
        ax1.set_visible(False)

    # ── 右: CryptoBank crypticity score (連続値 [0, 1]) ──────────
    ax2 = axes[1]
    if has_crypticity:
        scores = df["crypticity_score"].fillna(np.nan).values

        # 閾値 0.5 を境に色を変える（≥0.5 = cryptic → 緑、< 0.5 → 赤、NaN → 灰）
        bar_colors = []
        for v in scores:
            if not np.isfinite(v):
                bar_colors.append(GREY)
            elif v >= 0.5:
                bar_colors.append(GREEN)
            else:
                bar_colors.append(RED)

        y = np.arange(len(labels))
        bars = ax2.barh(y, np.where(np.isfinite(scores), scores, 0),
                        color=bar_colors, edgecolor="white", height=0.6)
        ax2.set_yticks(y)
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlim(0, 1)
        ax2.axvline(0.5, color=DARK, lw=1.2, ls="--", alpha=0.7,
                    label="Cryptic threshold (0.5)")
        ax2.set_xlabel("Crypticity Score (0–1)")
        ax2.set_title("CryptoBank Crypticity Score\n(≥ 0.5 = cryptic pocket detected)", fontsize=11)
        ax2.invert_yaxis()

        # 数値ラベル
        for bar, v in zip(bars, scores):
            if np.isfinite(v):
                ax2.text(min(v + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                         f"{v:.2f}", va="center", fontsize=7, color=DARK)

        ax2.legend(handles=[
            mpatches.Patch(color=GREEN, label="≥ 0.5 : cryptic"),
            mpatches.Patch(color=RED,   label="< 0.5 : non-cryptic"),
            mpatches.Patch(color=GREY,  label="N/A"),
        ], loc="lower right", fontsize=8)
    else:
        ax2.set_visible(False)

    return fig


# ── Page 9: Scatter Plots ────────────────────────────────────
def page_scatter_fig(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    page_style()
    add_page_title(fig, "Multi-metric Scatter Plots")
    fig.subplots_adjust(top=0.88, wspace=0.4)

    ax1 = axes[0]
    x1 = df["cryptic_recall"].fillna(0).values if "cryptic_recall" in df.columns else np.zeros(len(df))
    y1 = df["iptm"].fillna(0).values if "iptm" in df.columns else np.zeros(len(df))
    c1 = df["confidence_score"].fillna(0).values if "confidence_score" in df.columns else np.zeros(len(df))
    sc1 = ax1.scatter(x1, y1, c=c1, cmap="viridis", s=120,
                      edgecolors=DARK, linewidths=0.5, vmin=0, vmax=1)
    for i, row in df.iterrows():
        ax1.annotate(short_name(row["name"]), (x1[i], y1[i]),
                     textcoords="offset points", xytext=(5, 3), fontsize=7, color=DARK)
    plt.colorbar(sc1, ax=ax1, label="Confidence Score")
    ax1.set_xlabel("Cryptic Recall"); ax1.set_ylabel("ipTM")
    ax1.set_title("Cryptic Recall vs ipTM\n(color = Confidence)", fontsize=11)
    ax1.set_xlim(-0.05, 1.05); ax1.set_ylim(-0.05, 1.05)

    ax2 = axes[1]
    x2 = df["min_dist_peptide_to_cryptic"].fillna(np.nan).values if "min_dist_peptide_to_cryptic" in df.columns else np.full(len(df), np.nan)
    y2 = df["cryptic_f1"].fillna(0).values if "cryptic_f1" in df.columns else np.zeros(len(df))
    c2 = df["iptm"].fillna(0).values if "iptm" in df.columns else np.zeros(len(df))
    sc2 = ax2.scatter(x2, y2, c=c2, cmap="plasma", s=120,
                      edgecolors=DARK, linewidths=0.5, vmin=0, vmax=1)
    for i, row in df.iterrows():
        ax2.annotate(short_name(row["name"]), (x2[i], y2[i]),
                     textcoords="offset points", xytext=(5, 3), fontsize=7, color=DARK)
    plt.colorbar(sc2, ax=ax2, label="ipTM")
    ax2.axvline(5.0, color=RED, lw=1, ls="--", alpha=0.6, label="5Å threshold")
    ax2.set_xlabel("Min Distance to Cryptic Pocket (Å)"); ax2.set_ylabel("Cryptic F1")
    ax2.set_title("Min Distance vs Cryptic F1\n(color = ipTM)", fontsize=11)
    ax2.legend(fontsize=8)
    return fig



# ── Page 11: Interface Residue Heatmap ──────────────────────
def page_interface_heatmap_fig(df, cryptic_residues=None):
    all_res = set()
    iface_sets = []
    for val in df["interface_residues"]:
        if pd.isna(val) or val == "":
            iface_sets.append(set())
            continue
        s = {int(x) for x in str(val).split(";") if x.strip()}
        iface_sets.append(s)
        all_res |= s
    if not all_res:
        return None

    sorted_res = sorted(all_res)
    labels = [short_name(n_) for n_ in df["name"]]
    mat = np.zeros((len(df), len(sorted_res)))
    for i, s in enumerate(iface_sets):
        for j, r in enumerate(sorted_res):
            mat[i, j] = 1.0 if r in s else 0.0

    cryptic_cols = set()
    if cryptic_residues:
        for j, r in enumerate(sorted_res):
            if r in cryptic_residues:
                cryptic_cols.add(j)

    fig_w = max(10, len(sorted_res) * 0.22 + 2)
    fig_h = max(5, len(df) * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    page_style()
    add_page_title(fig, "Interface Residue Heatmap",
                   "Blue = contact  |  Orange border = cryptic pocket residue")
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.12, right=0.98)

    cmap = LinearSegmentedColormap.from_list("iface", ["white", BLUE])
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(np.arange(len(sorted_res)))
    ax.set_xticklabels(sorted_res, rotation=90, fontsize=6)
    ax.set_xlabel("Target Residue ID"); ax.set_ylabel("Peptide")

    for j in cryptic_cols:
        ax.add_patch(mpatches.FancyBboxPatch(
            (j - 0.5, -0.5), 1, len(df),
            boxstyle="square,pad=0",
            linewidth=1.5, edgecolor=ORANGE, facecolor="none", zorder=3))

    ax.legend(handles=[
        mpatches.Patch(color=BLUE,   label="Contact"),
        mpatches.Patch(color="white", ec=GREY, label="No contact"),
        mpatches.Patch(color=ORANGE, label="Cryptic residue (border)"),
    ], loc="upper right", fontsize=8, framealpha=0.9)
    return fig


# ── Page 12: Radar Chart ─────────────────────────────────────
def page_radar_fig(df):
    axes_def = [
        ("iptm",                     "ipTM",            True,  0,    1),
        ("receptor_to_peptide_iptm", "Rec→Pep\nipTM",   True,  0,    1),
        ("cryptic_recall",           "Recall",           True,  0,    1),
        ("cryptic_f1",               "F1",               True,  0,    1),
        ("crypticity_score",         "Crypticity\nScore",True,  0,    1),
        ("delta_volume",             "ΔVol",             True, -200, 200),
        ("delta_cryptic_rsa",        "ΔRSA",             True, -0.3,  0.3),
        ("cryptic_backbone_rmsd_vs_apo",                "BB RMSD\n(inv)", False, 0, 15),
        ("min_dist_peptide_to_cryptic",                 "MinDist\n(inv)", False, 0, 35),
        ("peptide_centroid_to_holo_ligand_centroid_dist","HoloDist\n(inv)", False, 0, 40),
    ]
    available = [(col, lab, hb, mn, mx) for col, lab, hb, mn, mx in axes_def
                 if col in df.columns and not df[col].isnull().all()]
    if len(available) < 3:
        return None

    N = len(available)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ncols = 3
    nrows = int(np.ceil(len(df) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows),
                             subplot_kw=dict(polar=True))
    page_style()
    add_page_title(fig, "Multi-Axis Radar Chart",
                   "Normalized 0-1  |  inverted axes (inv): lower value = better")
    fig.subplots_adjust(hspace=0.6, wspace=0.5, top=0.92)
    axes_flat = np.array(axes).flatten()

    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes_flat[idx]
        vals_norm = []
        for col, _, hb, mn, mx in available:
            v = row[col] if not pd.isna(row[col]) else mn
            normed = np.clip((v - mn) / (mx - mn) if mx != mn else 0.5, 0, 1)
            if not hb:
                normed = 1 - normed
            vals_norm.append(normed)
        vals_norm += vals_norm[:1]
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(angles, vals_norm, color=color, lw=2)
        ax.fill(angles, vals_norm, color=color, alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([d[1] for d in available], fontsize=6.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=5)
        ax.set_title(f"{short_name(row['name'])}\n{row['peptide_sequence']}",
                     fontsize=8, pad=10)
        ax.spines["polar"].set_color("#CCCCCC")

    for idx in range(len(df), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    return fig


# ── メイン ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",  required=True)
    parser.add_argument("--output_pdf", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--cryptic_residues", default=None,
                        help="カンマ区切りの cryptic 残基番号 (interface heatmap の orange border 用)")
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path).reset_index(drop=True)
    if df.empty:
        sys.exit("CSV is empty")

    run_title = args.title or csv_path.parent.parent.name
    cryptic_residues = None
    if args.cryptic_residues:
        cryptic_residues = [int(x.strip()) for x in args.cryptic_residues.split(",")
                            if x.strip()]

    out_path = Path(args.output_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    page_funcs = [
        # メタ
        lambda: page_guide_fig(),
        lambda: page_summary_table_fig(df, run_title),
        # ① 信頼度スコア
        lambda: page_confidence_fig(df),
        # ② 指定位置 (cryptic) の近くにリガンド (ペプチド) が来ているか
        lambda: page_cryptic_metrics_fig(df),
        lambda: page_distances_fig(df),
        # ③ Crypticity (クリプティックになっているか)
        lambda: page_pocket_volume_fig(df),
        lambda: page_rsa_fig(df),
        lambda: page_rmsd_and_contact_fig(df),
        # 補助
        lambda: page_scatter_fig(df),
        lambda: page_interface_heatmap_fig(df, cryptic_residues),
        lambda: page_radar_fig(df),
    ]

    writer = PdfWriter()
    for func in page_funcs:
        fig = func()
        if fig is None:
            continue
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf", bbox_inches="tight",
                    metadata={"Creator": "visualize_evaluation.py"})
        plt.close(fig)
        buf.seek(0)
        reader = PdfReader(buf)
        for page in reader.pages:
            writer.add_page(page)

    with open(out_path, "wb") as f:
        writer.write(f)

    print(f"Saved: {out_path}  ({len(writer.pages)} pages)")


if __name__ == "__main__":
    main()
