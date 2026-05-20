#!/usr/bin/env python
# ============================================================
# Evaluation summary CSV → PDF 可視化スクリプト
# ============================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import io
import tempfile

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


def hbar(ax, labels, values, colors, xlabel, title,
         vlines=None, xlim=None):
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


# ── Page 1: 表紙 + サマリーテーブル ─────────────────────────
def page_cover_and_table(pdf, df, run_title):
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
        "cryptic_recall": "Crypt.Recall",
        "cryptic_f1": "Crypt.F1",
        "min_dist_peptide_to_cryptic": "MinDist(A)",
        "delta_druggability": "DDrugg.",
        "delta_cryptic_sasa": "DSASA",
        "cryptic_backbone_rmsd_vs_apo": "BBRMSD(apo)",
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

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 2: 信頼度スコア ──────────────────────────────────────
def page_confidence(pdf, df):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    page_style()
    add_page_title(fig, "Structural Confidence Scores")
    fig.subplots_adjust(hspace=0.45, wspace=0.4, top=0.88)
    labels = [short_name(n) for n in df["name"]]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    metrics = [
        ("confidence_score",    "Confidence Score",        (0, 1)),
        ("ptm",                 "pTM",                     (0, 1)),
        ("iptm",                "ipTM",                    (0, 1)),
        ("complex_plddt",       "Complex pLDDT",           (0, 1)),
        ("peptide_plddt_mean",  "Peptide pLDDT (mean)",    (0, 1)),
        ("interface_plddt_mean","Interface pLDDT (mean)",  (0, 1)),
    ]
    for ax, (col, title, xlim) in zip(axes.flat, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].fillna(0).values
        hbar(ax, labels, vals, colors, col.replace("_", " "),
             title, vlines=[0.7, 0.8], xlim=xlim)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 3: Cryptic pocket 一致指標 ──────────────────────────
def page_cryptic_metrics(pdf, df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    page_style()
    add_page_title(fig, "Cryptic Pocket Binding Metrics",
                   "How well does the peptide target the cryptic pocket residues?")
    fig.subplots_adjust(hspace=0.45, wspace=0.4, top=0.88)
    labels = [short_name(n) for n in df["name"]]

    def green_colors(vals):
        cmap = plt.get_cmap("Greens")
        norm = plt.Normalize(0, 1)
        return [cmap(norm(v)) for v in vals]

    for ax, (col, title, xlabel) in zip(axes.flat, [
        ("cryptic_recall",    "Cryptic Pocket Recall",    "Recall"),
        ("cryptic_precision", "Cryptic Pocket Precision", "Precision"),
        ("cryptic_jaccard",   "Cryptic Pocket Jaccard",   "Jaccard"),
        ("cryptic_f1",        "Cryptic Pocket F1",        "F1"),
    ]):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].fillna(0).values
        hbar(ax, labels, vals, green_colors(vals), xlabel, title, xlim=(0, 1))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 4: 距離指標 ──────────────────────────────────────────
def page_distances(pdf, df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    page_style()
    add_page_title(fig, "Peptide-Pocket Distance Metrics",
                   "Smaller = peptide closer to cryptic pocket")
    fig.subplots_adjust(top=0.88, wspace=0.4)
    labels = [short_name(n) for n in df["name"]]

    def dist_colors(vals):
        cmap = plt.get_cmap("RdYlGn_r")
        mn, mx = np.nanmin(vals), np.nanmax(vals)
        norm = plt.Normalize(mn, mx)
        return [cmap(norm(v)) for v in vals]

    for ax, col, title, xlabel in [
        (ax1, "peptide_centroid_to_cryptic_centroid_dist",
         "Centroid-Centroid Distance", "Distance (A)"),
        (ax2, "min_dist_peptide_to_cryptic",
         "Minimum Peptide-Pocket Distance", "Min Distance (A)"),
    ]:
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].fillna(np.nan).values
        hbar(ax, labels, vals, dist_colors(vals), xlabel, title, vlines=[5.0])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 5: fpocket druggability & volume ─────────────────────
def page_fpocket(pdf, df):
    fig = plt.figure(figsize=(14, 10))
    page_style()
    add_page_title(fig, "Fpocket Druggability & Volume",
                   "Cryptic pocket opening: apo vs predicted  "
                   "(Delta > 0 = pocket opens)")
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.5, wspace=0.4, top=0.88, bottom=0.08)
    labels = [short_name(n) for n in df["name"]]
    x = np.arange(len(labels))
    w = 0.35

    # druggability grouped bar
    ax_grp = fig.add_subplot(gs[0, :2])
    apo_d  = df["fpocket_druggability_apo"].fillna(0).values
    pred_d = df["fpocket_druggability_predicted"].fillna(0).values
    ax_grp.bar(x - w/2, apo_d,  w, label="apo",       color=BLUE,  alpha=0.8)
    ax_grp.bar(x + w/2, pred_d, w, label="predicted",  color=GREEN, alpha=0.8)
    ax_grp.set_xticks(x)
    ax_grp.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_grp.set_ylabel("Druggability Score")
    ax_grp.set_title("Druggability: Apo vs Predicted", fontsize=11)
    ax_grp.legend(fontsize=8)
    ax_grp.set_ylim(0, max(np.max(apo_d), np.max(pred_d)) * 1.2 + 0.05)

    # delta druggability
    ax_dd = fig.add_subplot(gs[0, 2])
    delta_d = df["delta_druggability"].fillna(0).values
    ax_dd.barh(np.arange(len(labels)), delta_d,
               color=[GREEN if v >= 0 else RED for v in delta_d],
               edgecolor="white", height=0.6)
    ax_dd.set_yticks(np.arange(len(labels)))
    ax_dd.set_yticklabels(labels, fontsize=8)
    ax_dd.axvline(0, color=DARK, lw=1)
    ax_dd.set_xlabel("Delta Druggability")
    ax_dd.set_title("Delta Druggability\n(predicted - apo)", fontsize=11)
    ax_dd.invert_yaxis()

    # volume grouped bar
    ax_vol = fig.add_subplot(gs[1, :2])
    apo_v  = df["fpocket_volume_apo"].fillna(0).values
    pred_v = df["fpocket_volume_predicted"].fillna(0).values
    ax_vol.bar(x - w/2, apo_v,  w, label="apo",      color=BLUE,  alpha=0.8)
    ax_vol.bar(x + w/2, pred_v, w, label="predicted", color=GREEN, alpha=0.8)
    ax_vol.set_xticks(x)
    ax_vol.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_vol.set_ylabel("Pocket Volume (A3)")
    ax_vol.set_title("Pocket Volume: Apo vs Predicted", fontsize=11)
    ax_vol.legend(fontsize=8)

    # delta volume
    ax_dv = fig.add_subplot(gs[1, 2])
    delta_v = df["delta_volume"].fillna(0).values
    ax_dv.barh(np.arange(len(labels)), delta_v,
               color=[GREEN if v >= 0 else RED for v in delta_v],
               edgecolor="white", height=0.6)
    ax_dv.set_yticks(np.arange(len(labels)))
    ax_dv.set_yticklabels(labels, fontsize=8)
    ax_dv.axvline(0, color=DARK, lw=1)
    ax_dv.set_xlabel("Delta Volume (A3)")
    ax_dv.set_title("Delta Volume\n(predicted - apo)", fontsize=11)
    ax_dv.invert_yaxis()

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 6: SASA ─────────────────────────────────────────────
def page_sasa(pdf, df):
    sasa_cols = ["cryptic_sasa_apo", "cryptic_sasa_predicted", "delta_cryptic_sasa"]
    if all(c not in df.columns or df[c].isnull().all() for c in sasa_cols):
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    page_style()
    add_page_title(fig, "Cryptic Residue SASA",
                   "Solvent Accessible Surface Area  |  "
                   "Delta SASA > 0 = more exposed in predicted")
    fig.subplots_adjust(top=0.85, wspace=0.45)
    labels = [short_name(n) for n in df["name"]]
    apo_v  = df["cryptic_sasa_apo"].fillna(0).values
    pred_v = df["cryptic_sasa_predicted"].fillna(0).values
    delta  = df["delta_cryptic_sasa"].fillna(0).values

    axes[0].barh(np.arange(len(labels)), apo_v, color=BLUE, alpha=0.8,
                 edgecolor="white", height=0.6)
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("SASA (A2)")
    axes[0].set_title("Cryptic SASA (Apo)", fontsize=11)
    axes[0].invert_yaxis()

    colors1 = [GREEN if p >= a else RED for p, a in zip(pred_v, apo_v)]
    axes[1].barh(np.arange(len(labels)), pred_v, color=colors1, alpha=0.8,
                 edgecolor="white", height=0.6)
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("SASA (A2)")
    axes[1].set_title("Cryptic SASA (Predicted)", fontsize=11)
    axes[1].invert_yaxis()

    axes[2].barh(np.arange(len(labels)), delta,
                 color=[GREEN if v >= 0 else RED for v in delta],
                 alpha=0.8, edgecolor="white", height=0.6)
    axes[2].set_yticks(np.arange(len(labels)))
    axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].axvline(0, color=DARK, lw=1)
    axes[2].set_xlabel("Delta SASA (A2)")
    axes[2].set_title("Delta SASA (Predicted - Apo)", fontsize=11)
    axes[2].invert_yaxis()

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 7: RMSD ─────────────────────────────────────────────
def page_rmsd(pdf, df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    page_style()
    add_page_title(fig, "Cryptic Residue RMSD",
                   "Structural change at the cryptic pocket  |  "
                   "Larger = more conformational change")
    fig.subplots_adjust(hspace=0.45, wspace=0.4, top=0.88)
    labels = [short_name(n) for n in df["name"]]

    def rmsd_colors(vals):
        cmap = plt.get_cmap("YlOrRd")
        vmax = max(np.nanmax(vals), 1.0)
        norm = plt.Normalize(0, vmax)
        return [cmap(norm(v)) if np.isfinite(v) else GREY for v in vals]

    for ax, (col, title) in zip(axes.flat, [
        ("cryptic_backbone_rmsd_vs_apo",   "Backbone RMSD vs Apo"),
        ("cryptic_sidechain_rmsd_vs_apo",  "Sidechain RMSD vs Apo"),
        ("cryptic_backbone_rmsd_vs_holo",  "Backbone RMSD vs Holo"),
        ("cryptic_sidechain_rmsd_vs_holo", "Sidechain RMSD vs Holo"),
    ]):
        if col not in df.columns or df[col].isnull().all():
            ax.set_visible(False)
            continue
        vals = df[col].fillna(np.nan).values
        hbar(ax, labels, vals, rmsd_colors(vals), "RMSD (A)", title)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 8: Holo リガンド一致 ────────────────────────────────
def page_holo_ligand(pdf, df):
    holo_cols = ["holo_ligand_recall", "holo_ligand_precision", "holo_ligand_jaccard"]
    if all(c not in df.columns or df[c].isnull().all() for c in holo_cols):
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    page_style()
    add_page_title(fig, "Holo Ligand Binding Site Overlap",
                   "Does the peptide bind where the known ligand binds?")
    fig.subplots_adjust(top=0.85, wspace=0.45)
    labels = [short_name(n) for n in df["name"]]
    cmap = plt.get_cmap("Blues")
    norm = plt.Normalize(0, 1)
    for ax, (col, title) in zip(axes, [
        ("holo_ligand_recall",    "Recall"),
        ("holo_ligand_precision", "Precision"),
        ("holo_ligand_jaccard",   "Jaccard"),
    ]):
        vals = df[col].fillna(0).values if col in df.columns else np.zeros(len(df))
        colors = [cmap(norm(v)) for v in vals]
        hbar(ax, labels, vals, colors, title, title, xlim=(0, 1))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 9: 散布図 ───────────────────────────────────────────
def page_scatter(pdf, df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    page_style()
    add_page_title(fig, "Multi-metric Scatter Plots")
    fig.subplots_adjust(top=0.88, wspace=0.4)

    # 散布図1: cryptic_recall vs iptm
    ax1 = axes[0]
    x1 = df["cryptic_recall"].fillna(0).values
    y1 = df["iptm"].fillna(0).values
    c1 = df["confidence_score"].fillna(0).values
    sc1 = ax1.scatter(x1, y1, c=c1, cmap="viridis", s=120,
                      edgecolors=DARK, linewidths=0.5, vmin=0, vmax=1)
    for i, row in df.iterrows():
        ax1.annotate(short_name(row["name"]), (x1[i], y1[i]),
                     textcoords="offset points", xytext=(5, 3),
                     fontsize=7, color=DARK)
    plt.colorbar(sc1, ax=ax1, label="Confidence Score")
    ax1.set_xlabel("Cryptic Recall")
    ax1.set_ylabel("ipTM")
    ax1.set_title("Cryptic Recall vs ipTM\n(color = Confidence)", fontsize=11)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)

    # 散布図2: min_dist vs cryptic_f1
    ax2 = axes[1]
    x2 = df["min_dist_peptide_to_cryptic"].fillna(np.nan).values
    y2 = df["cryptic_f1"].fillna(0).values
    c2 = df["iptm"].fillna(0).values
    sc2 = ax2.scatter(x2, y2, c=c2, cmap="plasma", s=120,
                      edgecolors=DARK, linewidths=0.5, vmin=0, vmax=1)
    for i, row in df.iterrows():
        ax2.annotate(short_name(row["name"]), (x2[i], y2[i]),
                     textcoords="offset points", xytext=(5, 3),
                     fontsize=7, color=DARK)
    plt.colorbar(sc2, ax=ax2, label="ipTM")
    ax2.axvline(5.0, color=RED, lw=1, ls="--", alpha=0.6, label="5A threshold")
    ax2.set_xlabel("Min Distance to Cryptic Pocket (A)")
    ax2.set_ylabel("Cryptic F1")
    ax2.set_title("Min Distance vs Cryptic F1\n(color = ipTM)", fontsize=11)
    ax2.legend(fontsize=8)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 10: Interface 残基ヒートマップ ──────────────────────
def page_interface_heatmap(pdf, df, cryptic_residues=None):
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
        return

    sorted_res = sorted(all_res)
    labels = [short_name(n) for n in df["name"]]
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
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(np.arange(len(sorted_res)))
    ax.set_xticklabels(sorted_res, rotation=90, fontsize=6)
    ax.set_xlabel("Target Residue ID")
    ax.set_ylabel("Peptide")

    for j in cryptic_cols:
        ax.add_patch(mpatches.FancyBboxPatch(
            (j - 0.5, -0.5), 1, len(df),
            boxstyle="square,pad=0",
            linewidth=1.5, edgecolor=ORANGE, facecolor="none", zorder=3))

    ax.legend(handles=[
        mpatches.Patch(color=BLUE, label="Contact"),
        mpatches.Patch(color="white", ec=GREY, label="No contact"),
        mpatches.Patch(color=ORANGE, label="Cryptic residue (border)"),
    ], loc="upper right", fontsize=8, framealpha=0.9)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 11: レーダーチャート ────────────────────────────────
def page_radar(pdf, df):
    axes_def = [
        ("iptm",              "ipTM",          True,  0, 1),
        ("cryptic_recall",    "Crypt.Recall",  True,  0, 1),
        ("cryptic_f1",        "Crypt.F1",      True,  0, 1),
        ("holo_ligand_recall","Holo Recall",   True,  0, 1),
        ("delta_druggability","DDrugg.",        True, -0.5, 0.5),
        ("delta_cryptic_sasa","DSASA(norm)",   True, -500, 500),
        ("cryptic_backbone_rmsd_vs_apo", "BB RMSD\n(inv)", False, 0, 15),
        ("min_dist_peptide_to_cryptic",  "MinDist\n(inv)", False, 0, 35),
    ]
    available = [(col, lab, hb, mn, mx)
                 for col, lab, hb, mn, mx in axes_def
                 if col in df.columns and not df[col].isnull().all()]
    if len(available) < 3:
        return

    N = len(available)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ncols = 3
    nrows = int(np.ceil(len(df) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows),
                             subplot_kw=dict(polar=True))
    page_style()
    add_page_title(fig, "Multi-Axis Radar Chart",
                   "Normalized 0-1  |  inverted axes: lower value = better")
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
        ax.set_yticklabels(["0.25","0.5","0.75","1.0"], fontsize=5)
        ax.set_title(f"{short_name(row['name'])}\n{row['peptide_sequence']}",
                     fontsize=8, pad=10)
        ax.spines["polar"].set_color("#CCCCCC")

    for idx in range(len(df), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── fig を返すラッパー (pypdf 結合用) ─────────────────────────
def _fig_wrapper(func, pdf_arg_func):
    """pdf 引数を受け取る関数を fig を返す関数に変換するヘルパー"""
    import io as _io
    figs = []
    class _Collector:
        def savefig(self, fig, **kw):
            figs.append(fig)
    collector = _Collector()
    # 関数を呼ぶと内部で pdf.savefig(fig) が呼ばれる想定だが、
    # 現状の実装は plt.close(fig) を呼んでしまうので直接 fig を作る

def page_guide_fig():
    """Guide page explaining how to read each figure"""
    fig = plt.figure(figsize=(14, 10))
    page_style()
    fig.patch.set_facecolor("white")
    add_page_title(fig, "Figure Guide  -  How to Read Each Figure", "")

    guide_text = [
        ("Page 2  Summary Table",
         "Overview of all key metrics for each peptide. "
         "Conf.=Confidence Score, ipTM=interface pTM (overall), "
         "Crypt.Recall=fraction of cryptic residues contacted."),

        ("Page 3  Confidence Scores",
         "Boltz-2 structural confidence. confidence_score / pTM / ipTM (overall) / "
         "ipTM (receptor to peptide): validity of peptide binding position as seen from receptor "
         "[pair_chains_iptm 0 to 1]. All range 0-1; higher is better."),

        ("Page 4  Cryptic Pocket Binding Metrics",
         "Overlap between peptide interface residues and defined cryptic pocket residues. "
         "Recall = fraction of cryptic residues contacted. "
         "Precision = fraction of interface residues that are cryptic. "
         "F1 = harmonic mean. All range 0-1; higher is better."),

        ("Page 5  Distance Metrics",
         "Distance between peptide and cryptic pocket. "
         "Centroid dist = distance between centroids. "
         "Min distance = shortest atom-atom distance. "
         "Smaller is better. Red dashed line = 5 A contact threshold."),

        ("Page 6  Fpocket Druggability & Volume",
         "Druggability of the cryptic pocket predicted by fpocket. "
         "Druggability: 0-1 (higher = more druggable). "
         "Delta Druggability > 0 means pocket opens in predicted structure (evidence of crypticity). "
         "Volume in cubic angstroms."),

        ("Page 7  Cryptic Residue RSA",
         "RSA = ASA / MAX_ASA (residue type). "
         "Relative Solvent Accessibility normalized by Tien 2013 theoretical values. "
         "0 = fully buried, 1 = fully exposed. "
         "Delta RSA > 0 means more exposed in predicted structure (pocket opens)."),

        ("Page 8  Cryptic Residue RMSD",
         "Structural change at cryptic residues. "
         "Global CA alignment first, then RMSD computed only for cryptic residues. "
         "Backbone RMSD: main chain change. Sidechain RMSD: side chain change (key for pocket formation). "
         "vs apo: change from apo. vs holo: similarity to holo structure."),

        ("Page 9  Holo Ligand Overlap",
         "Overlap between peptide interface residues and known ligand binding residues. "
         "Recall = fraction of known binding residues contacted by peptide. "
         "Higher means peptide binds at the same site as the known ligand."),

        ("Page 10  Scatter Plots",
         "Left: Cryptic Recall vs ipTM (color = Confidence). Upper-right is ideal. "
         "Right: Min distance vs Cryptic F1 (color = ipTM). Upper-left is ideal (close and high F1)."),

        ("Page 11  Correlation Analysis",
         "Spearman rank correlation heatmap. |r| > 0.6 = strong correlation. "
         "Bottom: scatter plots of the top correlated metric pairs with regression lines."),

        ("Page 12  Interface Residue Heatmap",
         "X-axis = target residue ID, Y-axis = peptide. Blue = contact. "
         "Orange border = cryptic pocket residue. "
         "Vertical clusters of blue indicate binding at the cryptic pocket."),

        ("Page 13  Radar Chart",
         "Multi-axis comparison of each peptide. All axes normalized 0-1 "
         "(axes where lower is better are inverted). "
         "Larger area = better overall performance."),
    ]

    ax = fig.add_axes([0.03, 0.02, 0.94, 0.88])
    ax.axis("off")

    y = 0.97
    for title, desc in guide_text:
        ax.text(0.0, y, title, transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=BLUE, va="top")
        y -= 0.038
        ax.text(0.02, y, desc, transform=ax.transAxes,
                fontsize=8, color=DARK, va="top", wrap=True,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#F0F4FF",
                          edgecolor="#CCDDFF", linewidth=0.5))
        y -= 0.062

    return fig


def page_cover_and_table_fig(df, run_title):
    fig = plt.figure(figsize=(14, 10))
    page_style()
    fig.text(0.5, 0.94, "Peptide-Target Prediction Evaluation Report",
             ha="center", fontsize=18, fontweight="bold", color=DARK)
    fig.text(0.5, 0.90, run_title, ha="center", fontsize=12, color="#444444")
    fig.text(0.5, 0.87, f"n = {len(df)} peptides",
             ha="center", fontsize=10, color="#666666")
    cols_display = {
        "name": "Name", "peptide_sequence": "Sequence",
        "confidence_score": "Conf.", "iptm": "ipTM",
        "cryptic_recall": "Crypt.Recall", "cryptic_f1": "Crypt.F1",
        "min_dist_peptide_to_cryptic": "MinDist(A)",
        "delta_druggability": "DDrugg.", "delta_cryptic_sasa": "DSASA",
        "cryptic_backbone_rmsd_vs_apo": "BBRMSD(apo)",
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


def page_confidence_fig(df):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    page_style()
    add_page_title(fig, "Structural Confidence Scores")
    fig.subplots_adjust(hspace=0.45, wspace=0.4, top=0.88)
    labels = [short_name(n) for n in df["name"]]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    metrics = [
        ("confidence_score",           "Confidence Score",               (0, 1)),
        ("ptm",                        "pTM",                            (0, 1)),
        ("iptm",                       "ipTM (overall)",                 (0, 1)),
        ("receptor_to_peptide_iptm",   "ipTM (receptor→peptide)",        (0, 1)),
        ("complex_plddt",              "Complex pLDDT",                  (0, 1)),
        ("peptide_plddt_mean",         "Peptide pLDDT (mean)",           (0, 1)),
        ("interface_plddt_mean",       "Interface pLDDT (mean)",         (0, 1)),
    ]
    # 7指標なので 2x4 に拡張
    fig.clf()
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    page_style()
    add_page_title(fig, "Structural Confidence Scores",
                   "receptor→peptide ipTM: Boltz の pair_chains_iptm[0→1] — "
                   "validity of peptide binding position as seen from receptor")
    fig.subplots_adjust(hspace=0.45, wspace=0.4, top=0.88)
    labels_new = [short_name(n) for n in df["name"]]
    colors_new = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    for ax, (col, title, xlim) in zip(axes.flat, metrics):
        if col not in df.columns or df[col].isnull().all():
            ax.set_visible(False)
            continue
        vals = df[col].fillna(0).values
        hbar(ax, labels_new, vals, colors_new, "", title, vlines=[0.7, 0.8], xlim=xlim)
    # 余った軸を非表示
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    return fig


def page_cryptic_metrics_fig(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    page_style()
    add_page_title(fig, "Cryptic Pocket Binding Metrics",
                   "How well does the peptide target the cryptic pocket residues?")
    fig.subplots_adjust(hspace=0.45, wspace=0.4, top=0.88)
    labels = [short_name(n) for n in df["name"]]
    def green_colors(vals):
        cmap = plt.get_cmap("Greens")
        norm = plt.Normalize(0, 1)
        return [cmap(norm(v)) for v in vals]
    for ax, (col, title, xlabel) in zip(axes.flat, [
        ("cryptic_recall",    "Cryptic Pocket Recall",    "Recall"),
        ("cryptic_precision", "Cryptic Pocket Precision", "Precision"),
        ("cryptic_jaccard",   "Cryptic Pocket Jaccard",   "Jaccard"),
        ("cryptic_f1",        "Cryptic Pocket F1",        "F1"),
    ]):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].fillna(0).values
        hbar(ax, labels, vals, green_colors(vals), xlabel, title, xlim=(0, 1))
    return fig


def page_distances_fig(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    page_style()
    add_page_title(fig, "Peptide-Pocket Distance Metrics",
                   "Smaller = peptide closer to cryptic pocket")
    fig.subplots_adjust(top=0.88, wspace=0.4)
    labels = [short_name(n) for n in df["name"]]
    def dist_colors(vals):
        cmap = plt.get_cmap("RdYlGn_r")
        mn, mx = np.nanmin(vals), np.nanmax(vals)
        norm = plt.Normalize(mn, mx)
        return [cmap(norm(v)) for v in vals]
    for ax, col, title, xlabel in [
        (ax1, "peptide_centroid_to_cryptic_centroid_dist",
         "Centroid-Centroid Distance", "Distance (A)"),
        (ax2, "min_dist_peptide_to_cryptic",
         "Minimum Peptide-Pocket Distance", "Min Distance (A)"),
    ]:
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].fillna(np.nan).values
        hbar(ax, labels, vals, dist_colors(vals), xlabel, title, vlines=[5.0])
    return fig


def page_fpocket_fig(df):
    fig = plt.figure(figsize=(14, 10))
    page_style()
    add_page_title(fig, "Fpocket Druggability & Volume",
                   "Cryptic pocket opening: apo vs predicted  "
                   "(Delta > 0 = pocket opens)")
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.5, wspace=0.4, top=0.88, bottom=0.08)
    labels = [short_name(n) for n in df["name"]]
    x = np.arange(len(labels))
    w = 0.35
    apo_d  = df["fpocket_druggability_apo"].fillna(0).values
    pred_d = df["fpocket_druggability_predicted"].fillna(0).values
    ax_grp = fig.add_subplot(gs[0, :2])
    ax_grp.bar(x - w/2, apo_d,  w, label="apo",      color=BLUE,  alpha=0.8)
    ax_grp.bar(x + w/2, pred_d, w, label="predicted", color=GREEN, alpha=0.8)
    ax_grp.set_xticks(x); ax_grp.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_grp.set_ylabel("Druggability Score")
    ax_grp.set_title("Druggability: Apo vs Predicted", fontsize=11)
    ax_grp.legend(fontsize=8)
    ax_grp.set_ylim(0, max(np.max(apo_d), np.max(pred_d)) * 1.2 + 0.05)
    ax_dd = fig.add_subplot(gs[0, 2])
    delta_d = df["delta_druggability"].fillna(0).values
    ax_dd.barh(np.arange(len(labels)), delta_d,
               color=[GREEN if v >= 0 else RED for v in delta_d],
               edgecolor="white", height=0.6)
    ax_dd.set_yticks(np.arange(len(labels))); ax_dd.set_yticklabels(labels, fontsize=8)
    ax_dd.axvline(0, color=DARK, lw=1)
    ax_dd.set_xlabel("Delta Druggability")
    ax_dd.set_title("Delta Druggability\n(predicted - apo)", fontsize=11)
    ax_dd.invert_yaxis()
    apo_v  = df["fpocket_volume_apo"].fillna(0).values
    pred_v = df["fpocket_volume_predicted"].fillna(0).values
    ax_vol = fig.add_subplot(gs[1, :2])
    ax_vol.bar(x - w/2, apo_v,  w, label="apo",      color=BLUE,  alpha=0.8)
    ax_vol.bar(x + w/2, pred_v, w, label="predicted", color=GREEN, alpha=0.8)
    ax_vol.set_xticks(x); ax_vol.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_vol.set_ylabel("Pocket Volume (A3)")
    ax_vol.set_title("Pocket Volume: Apo vs Predicted", fontsize=11)
    ax_vol.legend(fontsize=8)
    ax_dv = fig.add_subplot(gs[1, 2])
    delta_v = df["delta_volume"].fillna(0).values
    ax_dv.barh(np.arange(len(labels)), delta_v,
               color=[GREEN if v >= 0 else RED for v in delta_v],
               edgecolor="white", height=0.6)
    ax_dv.set_yticks(np.arange(len(labels))); ax_dv.set_yticklabels(labels, fontsize=8)
    ax_dv.axvline(0, color=DARK, lw=1)
    ax_dv.set_xlabel("Delta Volume (A3)")
    ax_dv.set_title("Delta Volume\n(predicted - apo)", fontsize=11)
    ax_dv.invert_yaxis()
    return fig


def page_rsa_fig(df):
    """RSA (相対SASA) を優先表示、なければ絶対SASAを表示"""
    rsa_cols  = ["cryptic_rsa_apo", "cryptic_rsa_predicted", "delta_cryptic_rsa"]
    sasa_cols = ["cryptic_sasa_apo", "cryptic_sasa_predicted", "delta_cryptic_sasa"]
    use_rsa = any(c in df.columns and not df[c].isnull().all() for c in rsa_cols)
    use_sasa = any(c in df.columns and not df[c].isnull().all() for c in sasa_cols)
    if not use_rsa and not use_sasa:
        return None

    if use_rsa:
        cols  = rsa_cols
        xlabel_unit = "RSA (0–1)"
        delta_unit  = "ΔRSA"
        subtitle = ("Relative Solvent Accessibility of cryptic pocket residues  |  "
                    "Normalized by Tien 2013 theoretical max ASA  |  "
                    "ΔRSA > 0 = more exposed in predicted (pocket opens)")
        main_title = "Cryptic Residue RSA (Relative SASA)"
    else:
        cols  = sasa_cols
        xlabel_unit = "SASA (Å²)"
        delta_unit  = "ΔSASA (Å²)"
        subtitle = ("Absolute SASA of cryptic pocket residues  |  "
                    "ΔSASA > 0 = more exposed in predicted")
        main_title = "Cryptic Residue SASA (Absolute)"

    apo_col, pred_col, delta_col = cols

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    page_style()
    add_page_title(fig, main_title, subtitle)
    fig.subplots_adjust(top=0.82, wspace=0.45)
    labels = [short_name(n) for n in df["name"]]
    apo_v  = df[apo_col].fillna(0).values  if apo_col  in df.columns else np.zeros(len(df))
    pred_v = df[pred_col].fillna(0).values if pred_col in df.columns else np.zeros(len(df))
    delta  = df[delta_col].fillna(0).values if delta_col in df.columns else np.zeros(len(df))

    axes[0].barh(np.arange(len(labels)), apo_v, color=BLUE, alpha=0.8,
                 edgecolor="white", height=0.6)
    axes[0].set_yticks(np.arange(len(labels))); axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel(xlabel_unit); axes[0].set_title("Apo", fontsize=11)
    axes[0].invert_yaxis()

    colors1 = [GREEN if p >= a else RED for p, a in zip(pred_v, apo_v)]
    axes[1].barh(np.arange(len(labels)), pred_v, color=colors1, alpha=0.8,
                 edgecolor="white", height=0.6)
    axes[1].set_yticks(np.arange(len(labels))); axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel(xlabel_unit); axes[1].set_title("Predicted", fontsize=11)
    axes[1].invert_yaxis()

    axes[2].barh(np.arange(len(labels)), delta,
                 color=[GREEN if v >= 0 else RED for v in delta],
                 alpha=0.8, edgecolor="white", height=0.6)
    axes[2].set_yticks(np.arange(len(labels))); axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].axvline(0, color=DARK, lw=1)
    axes[2].set_xlabel(delta_unit); axes[2].set_title("Delta (Predicted - Apo)", fontsize=11)
    axes[2].invert_yaxis()
    return fig


def page_sasa_fig(df):
    sasa_cols = ["cryptic_sasa_apo", "cryptic_sasa_predicted", "delta_cryptic_sasa"]
    if all(c not in df.columns or df[c].isnull().all() for c in sasa_cols):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    page_style()
    add_page_title(fig, "Cryptic Residue SASA",
                   "Solvent Accessible Surface Area  |  "
                   "Delta SASA > 0 = more exposed in predicted")
    fig.subplots_adjust(top=0.85, wspace=0.45)
    labels = [short_name(n) for n in df["name"]]
    apo_v  = df["cryptic_sasa_apo"].fillna(0).values
    pred_v = df["cryptic_sasa_predicted"].fillna(0).values
    delta  = df["delta_cryptic_sasa"].fillna(0).values
    axes[0].barh(np.arange(len(labels)), apo_v, color=BLUE, alpha=0.8,
                 edgecolor="white", height=0.6)
    axes[0].set_yticks(np.arange(len(labels))); axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("SASA (A2)"); axes[0].set_title("Cryptic SASA (Apo)", fontsize=11)
    axes[0].invert_yaxis()
    colors1 = [GREEN if p >= a else RED for p, a in zip(pred_v, apo_v)]
    axes[1].barh(np.arange(len(labels)), pred_v, color=colors1, alpha=0.8,
                 edgecolor="white", height=0.6)
    axes[1].set_yticks(np.arange(len(labels))); axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("SASA (A2)"); axes[1].set_title("Cryptic SASA (Predicted)", fontsize=11)
    axes[1].invert_yaxis()
    axes[2].barh(np.arange(len(labels)), delta,
                 color=[GREEN if v >= 0 else RED for v in delta],
                 alpha=0.8, edgecolor="white", height=0.6)
    axes[2].set_yticks(np.arange(len(labels))); axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].axvline(0, color=DARK, lw=1)
    axes[2].set_xlabel("Delta SASA (A2)"); axes[2].set_title("Delta SASA (Predicted - Apo)", fontsize=11)
    axes[2].invert_yaxis()
    return fig


def page_rmsd_fig(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    page_style()
    add_page_title(fig, "Cryptic Residue RMSD",
                   "Structural change at the cryptic pocket  |  "
                   "Larger = more conformational change")
    fig.subplots_adjust(hspace=0.45, wspace=0.4, top=0.88)
    labels = [short_name(n) for n in df["name"]]
    def rmsd_colors(vals):
        cmap = plt.get_cmap("YlOrRd")
        vmax = max(np.nanmax(vals), 1.0)
        norm = plt.Normalize(0, vmax)
        return [cmap(norm(v)) if np.isfinite(v) else GREY for v in vals]
    for ax, (col, title) in zip(axes.flat, [
        ("cryptic_backbone_rmsd_vs_apo",   "Backbone RMSD vs Apo"),
        ("cryptic_sidechain_rmsd_vs_apo",  "Sidechain RMSD vs Apo"),
        ("cryptic_backbone_rmsd_vs_holo",  "Backbone RMSD vs Holo"),
        ("cryptic_sidechain_rmsd_vs_holo", "Sidechain RMSD vs Holo"),
    ]):
        if col not in df.columns or df[col].isnull().all():
            ax.set_visible(False)
            continue
        vals = df[col].fillna(np.nan).values
        hbar(ax, labels, vals, rmsd_colors(vals), "RMSD (A)", title)
    return fig


def page_holo_ligand_fig(df):
    holo_cols = ["holo_ligand_recall", "holo_ligand_precision", "holo_ligand_jaccard"]
    if all(c not in df.columns or df[c].isnull().all() for c in holo_cols):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    page_style()
    add_page_title(fig, "Holo Ligand Binding Site Overlap",
                   "Does the peptide bind where the known ligand binds?")
    fig.subplots_adjust(top=0.85, wspace=0.45)
    labels = [short_name(n) for n in df["name"]]
    cmap = plt.get_cmap("Blues")
    norm = plt.Normalize(0, 1)
    for ax, (col, title) in zip(axes, [
        ("holo_ligand_recall",    "Recall"),
        ("holo_ligand_precision", "Precision"),
        ("holo_ligand_jaccard",   "Jaccard"),
    ]):
        vals = df[col].fillna(0).values if col in df.columns else np.zeros(len(df))
        hbar(ax, labels, vals, [cmap(norm(v)) for v in vals], title, title, xlim=(0, 1))
    return fig


def page_scatter_fig(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    page_style()
    add_page_title(fig, "Multi-metric Scatter Plots")
    fig.subplots_adjust(top=0.88, wspace=0.4)
    ax1 = axes[0]
    x1 = df["cryptic_recall"].fillna(0).values
    y1 = df["iptm"].fillna(0).values
    c1 = df["confidence_score"].fillna(0).values
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
    x2 = df["min_dist_peptide_to_cryptic"].fillna(np.nan).values
    y2 = df["cryptic_f1"].fillna(0).values
    c2 = df["iptm"].fillna(0).values
    sc2 = ax2.scatter(x2, y2, c=c2, cmap="plasma", s=120,
                      edgecolors=DARK, linewidths=0.5, vmin=0, vmax=1)
    for i, row in df.iterrows():
        ax2.annotate(short_name(row["name"]), (x2[i], y2[i]),
                     textcoords="offset points", xytext=(5, 3), fontsize=7, color=DARK)
    plt.colorbar(sc2, ax=ax2, label="ipTM")
    ax2.axvline(5.0, color=RED, lw=1, ls="--", alpha=0.6, label="5A threshold")
    ax2.set_xlabel("Min Distance to Cryptic Pocket (A)"); ax2.set_ylabel("Cryptic F1")
    ax2.set_title("Min Distance vs Cryptic F1\n(color = ipTM)", fontsize=11)
    ax2.legend(fontsize=8)
    return fig


def page_correlation_fig(df):
    """Spearman相関ヒートマップ + 主要ペア散布図"""
    from scipy import stats as _stats

    numeric_cols = {
        "confidence_score": "Confidence",
        "iptm": "ipTM",
        "receptor_to_peptide_iptm": "Rec→Pep ipTM",
        "cryptic_recall": "Crypt.Recall",
        "cryptic_f1": "Crypt.F1",
        "min_dist_peptide_to_cryptic": "Min.Dist",
        "delta_druggability": "ΔDrugg.",
        "delta_cryptic_rsa": "ΔRSA",
        "delta_cryptic_sasa": "ΔSASA",
        "cryptic_backbone_rmsd_vs_apo": "BB RMSD(apo)",
        "cryptic_backbone_rmsd_vs_holo": "BB RMSD(holo)",
        "holo_ligand_recall": "Holo Recall",
    }
    # 存在かつ非全欠損の列だけ
    available = {k: v for k, v in numeric_cols.items()
                 if k in df.columns and not df[k].isnull().all()}
    if len(available) < 3:
        return None

    cols = list(available.keys())
    labels = [available[c] for c in cols]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")

    # Spearman 相関行列
    n = len(cols)
    corr_mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            xi = sub[cols[i]].dropna()
            xj = sub[cols[j]].dropna()
            common = xi.index.intersection(xj.index)
            if len(common) >= 3:
                r, _ = _stats.spearmanr(xi[common], xj[common])
                corr_mat[i, j] = r
            else:
                corr_mat[i, j] = 0.0

    fig = plt.figure(figsize=(14, 12))
    page_style()
    add_page_title(fig, "Correlation Analysis",
                   "Spearman rank correlation  |  |r| > 0.6 = strong correlation")
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.55, wspace=0.45, top=0.88, bottom=0.07)

    # ヒートマップ (上段全体)
    ax_heat = fig.add_subplot(gs[0, :])
    cmap = plt.get_cmap("RdBu_r")
    im = ax_heat.imshow(corr_mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax_heat.set_xticks(range(n)); ax_heat.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax_heat.set_yticks(range(n)); ax_heat.set_yticklabels(labels, fontsize=7)
    ax_heat.set_title("Spearman Correlation Matrix", fontsize=11)
    plt.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02, label="r")
    for i in range(n):
        for j in range(n):
            v = corr_mat[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 0.6 else DARK
                ax_heat.text(j, i, f"{v:.2f}", ha="center", va="center",
                             fontsize=5.5, color=color)

    # 下段: |r|>0.5 の上位3ペア散布図
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            r = corr_mat[i, j]
            if not np.isnan(r):
                pairs.append((abs(r), r, cols[i], cols[j],
                               available[cols[i]], available[cols[j]]))
    pairs.sort(reverse=True)

    axes_scatter = [fig.add_subplot(gs[1, k]) for k in range(3)]
    for k, ax in enumerate(axes_scatter):
        if k >= len(pairs):
            ax.set_visible(False)
            continue
        _, r, cx, cy, lx, ly = pairs[k]
        x = sub[cx].values.astype(float)
        y = sub[cy].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=80, color=PALETTE[k],
                   edgecolors=DARK, linewidths=0.5, alpha=0.9)
        for i, row in df.iterrows():
            if mask[i]:
                ax.annotate(short_name(row["name"]), (x[i], y[i]),
                            textcoords="offset points", xytext=(4, 2),
                            fontsize=6, color=DARK)
        # 回帰線
        if mask.sum() >= 3:
            slope, intercept, *_ = _stats.linregress(x[mask], y[mask])
            xfit = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 50)
            ax.plot(xfit, slope * xfit + intercept, color=RED, lw=1.2, ls="--")
        ax.set_xlabel(lx, fontsize=8)
        ax.set_ylabel(ly, fontsize=8)
        ax.set_title(f"{lx} vs {ly}\nr = {r:.3f}", fontsize=9)

    return fig


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
    labels = [short_name(n) for n in df["name"]]
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
        mpatches.Patch(color=BLUE, label="Contact"),
        mpatches.Patch(color="white", ec=GREY, label="No contact"),
        mpatches.Patch(color=ORANGE, label="Cryptic residue (border)"),
    ], loc="upper right", fontsize=8, framealpha=0.9)
    return fig


def page_radar_fig(df):
    axes_def = [
        ("iptm",              "ipTM",          True,  0, 1),
        ("cryptic_recall",    "Crypt.Recall",  True,  0, 1),
        ("cryptic_f1",        "Crypt.F1",      True,  0, 1),
        ("holo_ligand_recall","Holo Recall",   True,  0, 1),
        ("delta_druggability","DDrugg.",        True, -0.5, 0.5),
        ("delta_cryptic_sasa","DSASA(norm)",   True, -500, 500),
        ("cryptic_backbone_rmsd_vs_apo", "BB RMSD\n(inv)", False, 0, 15),
        ("min_dist_peptide_to_cryptic",  "MinDist\n(inv)", False, 0, 35),
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
                   "Normalized 0-1  |  inverted axes: lower value = better")
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
        ax.set_yticklabels(["0.25","0.5","0.75","1.0"], fontsize=5)
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
    parser.add_argument("--cryptic_residues", default=None)
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

    # 各ページを PNG として一時ファイルに書き出し、pypdf で結合
    # → macOS Preview / Adobe Reader で確実に開けるPDFを生成
    page_funcs = [
        lambda: page_guide_fig(),
        lambda: page_cover_and_table_fig(df, run_title),
        lambda: page_confidence_fig(df),
        lambda: page_cryptic_metrics_fig(df),
        lambda: page_distances_fig(df),
        lambda: page_fpocket_fig(df),
        lambda: page_rsa_fig(df),
        lambda: page_rmsd_fig(df),
        lambda: page_holo_ligand_fig(df),
        lambda: page_scatter_fig(df),
        lambda: page_correlation_fig(df),
        lambda: page_interface_heatmap_fig(df, cryptic_residues),
        lambda: page_radar_fig(df),
    ]

    writer = PdfWriter()
    tmp_pdfs = []

    for i, func in enumerate(page_funcs):
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

    n_pages = len(writer.pages)
    print(f"Saved: {out_path}  ({n_pages} pages)")


if __name__ == "__main__":
    main()
