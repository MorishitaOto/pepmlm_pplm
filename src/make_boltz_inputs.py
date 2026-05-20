#!/usr/bin/env python
# ============================================================
# Top10 ペプチド抽出 → YAML ファイル生成スクリプト
#
# 入力CSV例:
#   index,peptide,pep_len,affinity
#   0,RWYWR,5,-8.688
#   ...
#
# 出力ファイル名例: 6XI7_top01_affm9.23.yaml
# ============================================================

import argparse
import csv
import os
from pathlib import Path


def make_yaml_content(protein_a_seq: str, peptide_seq: str) -> str:
    """YAML 文字列を組み立てる（インデントを保持するため手書き）"""
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {protein_a_seq}\n"
        "      msa: empty\n"
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: {peptide_seq}\n"
        "      msa: empty\n"
    )


def format_affinity_for_filename(aff: float) -> str:
    """
    affinity 値をファイル名用の文字列にフォーマット。
    小数点以下2桁（100分の1の位）まで保持。
    マイナス記号は 'm' に置換。

    例: -9.234567 → 'm9.23'
        7.521000  → '7.52'
    """
    rounded = f"{aff:.2f}"
    if rounded.startswith("-"):
        return "m" + rounded[1:]
    return rounded


def main():
    parser = argparse.ArgumentParser(
        description="CSVからtop10ペプチドを抽出してYAMLファイル群を生成"
    )
    parser.add_argument("--input_csv", required=True, help="入力CSVのパス")
    parser.add_argument("--output_dir", required=True, help="YAML出力フォルダ")
    parser.add_argument("--target_sequence", required=True,
                        help="ターゲットタンパク質の配列（プロテインA）")
    parser.add_argument("--pdb_id", required=True,
                        help="PDB ID（ファイル名に反映される）")
    parser.add_argument("--top_n", type=int, default=10,
                        help="抽出する上位件数 (default: 10)")
    args = parser.parse_args()

    # =========================
    # CSV 読み込み
    # =========================
    rows = []
    with open(args.input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("affinity") or not row.get("peptide"):
                continue
            try:
                aff = float(row["affinity"])
            except ValueError:
                continue
            rows.append({
                "peptide": row["peptide"].strip(),
                "affinity": aff,
            })

    if not rows:
        raise ValueError(f"CSVから有効なエントリが取得できませんでした: {args.input_csv}")

    # =========================
    # affinity 昇順ソート（小さいほど結合が強い）
    # =========================
    rows.sort(key=lambda r: r["affinity"])
    top_rows = rows[: args.top_n]

    print(f"=== Top {len(top_rows)} peptides ===")
    for i, r in enumerate(top_rows):
        print(f"  {i+1:2d}. {r['peptide']:<15s}  affinity={r['affinity']:.4f}")

    # =========================
    # 出力フォルダ作成
    # =========================
    os.makedirs(args.output_dir, exist_ok=True)

    width = max(2, len(str(args.top_n)))

    for rank, row in enumerate(top_rows, start=1):
        aff_str = format_affinity_for_filename(row["affinity"])
        filename = f"{args.pdb_id}_top{rank:0{width}d}_aff{aff_str}.yaml"
        filepath = Path(args.output_dir) / filename

        yaml_content = make_yaml_content(
            protein_a_seq=args.target_sequence,
            peptide_seq=row["peptide"],
        )

        with open(filepath, "w") as f:
            f.write(yaml_content)

        print(f"  saved: {filepath}")

    print(f"\n✅ 完了: {args.output_dir} に {len(top_rows)} 個のYAMLを生成しました")


if __name__ == "__main__":
    main()
