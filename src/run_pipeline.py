#!/usr/bin/env python
# ============================================================
# run_pipeline.py
# ------------------------------------------------------------
# AF2BIND → PepMLM+PPLM の一連の流れを統括するスクリプト。
#
# 【やること】
#   1. config.json を読む
#   2. config の PDB_PATH / CHAIN を使って run_af2bind.py を実行
#      （af2bind conda 環境で subprocess 実行）
#      → AF2BIND_MATRIX_PATH に .npy を生成
#   3. config に AF2BIND_MATRIX_PATH を動的に追加して
#      pepmlm_pplm.py を実行
#
# 【使い方】
#   conda activate pplm_esm
#   python run_pipeline.py --config config.json
#
# 【config.json に必要な追加フィールド】
#   "PDB_PATH"          : "/home/morishita/af2bind/1YCR.pdb"
#   "PDB_CHAIN"         : "A"
#   "AF2BIND_CONDA_ENV" : "af2bind"
#   "AF2BIND_SCRIPT"    : "/home/morishita/af2bind/run_af2bind.py"
#   "AF2BIND_PARAMS_DIR": "/home/morishita/af2bind/af2bind_params"
#   "AF2BIND_AF_PARAMS_DIR": "/home/morishita/af2bind/params"
#   "AF2BIND_OUT_DIR"   : "/mnt/hdd/morishita/af2bind_outputs"
#   "AF2BIND_SEED"      : 0            # 省略可、デフォルト0
#   "LAMBDA_AF2BIND"    : 1.0          # 省略可、0なら prior 無効
#   "PBIND_BIAS_SCALE"  : 5.0          # 省略可
#
# 【既存の config フィールド（変更不要）】
#   "TARGET_SEQUENCE", "PEPTIDE_LEN", "NUM_ITERATIONS", ... など
# ============================================================

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace


# =============================================================
# config loader（pepmlm_pplm.py と共通）
# =============================================================
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_config(cfg_dict: dict, path: str):
    with open(path, "w") as f:
        json.dump(cfg_dict, f, indent=2)


# =============================================================
# Step 1: AF2BIND の実行
# =============================================================
def run_af2bind(cfg_dict: dict) -> Path:
    """
    run_af2bind.py を af2bind conda 環境で subprocess 実行し、
    生成された .npy のパスを返す。

    すでに .npy が存在する場合はスキップ。

    Returns
    -------
    Path: 生成された .npy ファイルのパス
    """
    pdb_path = Path(cfg_dict["PDB_PATH"])
    chain = cfg_dict.get("PDB_CHAIN", "A")
    af2bind_script = Path(cfg_dict["AF2BIND_SCRIPT"])
    params_dir = Path(cfg_dict["AF2BIND_PARAMS_DIR"])
    af_params_dir = Path(cfg_dict["AF2BIND_AF_PARAMS_DIR"])
    out_dir = Path(cfg_dict["AF2BIND_OUT_DIR"])
    seed = cfg_dict.get("AF2BIND_SEED", 0)
    conda_env = cfg_dict.get("AF2BIND_CONDA_ENV", "af2bind")

    # 出力パスを決定
    pdb_stem = pdb_path.stem
    out_npy = out_dir / f"{pdb_stem}_{chain}.npy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # すでに生成済みならスキップ
    if out_npy.exists():
        print(f"[pipeline] AF2BIND output already exists: {out_npy}")
        print(f"[pipeline] Skipping AF2BIND. Delete the file to re-run.")
        return out_npy

    # ターゲット配列との整合性チェック（メタ情報があれば確認）
    meta_path = out_npy.with_suffix("").with_suffix(".meta.json")
    # ↑ .npy なし でも .meta.json だけある場合はないはずだが念のため

    print(f"\n{'='*60}")
    print(f"[pipeline] Step 1: Running AF2BIND")
    print(f"{'='*60}")
    print(f"  PDB:          {pdb_path}")
    print(f"  Chain:        {chain}")
    print(f"  Seed:         {seed}")
    print(f"  Output:       {out_npy}")
    print(f"  Conda env:    {conda_env}")

    # conda run で af2bind 環境を使って実行
    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", str(af2bind_script),
        "--pdb",            str(pdb_path),
        "--chain",          chain,
        "--out",            str(out_npy),
        "--params-dir",     str(params_dir),
        "--af-params-dir",  str(af_params_dir),
        "--seed",           str(seed),
    ]

    mask_sidechains = cfg_dict.get("AF2BIND_MASK_SIDECHAINS", True)
    if not mask_sidechains:
        cmd.append("--no-mask-sidechains")

    print(f"\n  Running: {' '.join(cmd)}\n")

    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[ERROR] AF2BIND failed (exit code {result.returncode})")
        print("→ af2bind conda 環境と run_af2bind.py の設定を確認してください")
        sys.exit(1)

    if not out_npy.exists():
        print(f"\n[ERROR] AF2BIND completed but output not found: {out_npy}")
        sys.exit(1)

    print(f"\n[pipeline] AF2BIND done in {elapsed:.1f}s")
    print(f"[pipeline] Output: {out_npy}")
    return out_npy


# =============================================================
# Step 2: ターゲット配列と AF2BIND 行列の整合性チェック
# =============================================================
def check_target_sequence(cfg_dict: dict, npy_path: Path):
    """
    config の TARGET_SEQUENCE と AF2BIND の出力配列が一致するか確認。
    """
    meta_path = npy_path.with_suffix("").parent / (npy_path.stem + ".meta.json")
    if not meta_path.exists():
        print(f"[pipeline] WARNING: meta.json not found at {meta_path}, skipping check")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    af2bind_seq = meta.get("target_seq", "")
    config_seq = cfg_dict.get("TARGET_SEQUENCE", "")

    if af2bind_seq == config_seq:
        print(f"[pipeline] ✓ TARGET_SEQUENCE matches AF2BIND output ({len(config_seq)} residues)")
    else:
        print(f"\n[pipeline] ⚠ TARGET_SEQUENCE MISMATCH!")
        print(f"  config:  {config_seq}")
        print(f"  AF2BIND: {af2bind_seq}")
        print(f"  → config の TARGET_SEQUENCE を AF2BIND の出力配列に合わせてください")
        print(f"  → または PDB_PATH を正しいファイルに変更してください")
        print()

        # 自動修正するか確認
        resp = input("AF2BIND の配列を TARGET_SEQUENCE として使用しますか？ [y/N]: ").strip().lower()
        if resp == "y":
            cfg_dict["TARGET_SEQUENCE"] = af2bind_seq
            print(f"[pipeline] TARGET_SEQUENCE を更新しました: {af2bind_seq}")
        else:
            print("[pipeline] 中断します。config の TARGET_SEQUENCE を修正してください。")
            sys.exit(1)


# =============================================================
# Step 3: pepmlm_pplm.py を実行
# =============================================================
def run_pepmlm(cfg_dict: dict, npy_path: Path, original_config_path: str):
    """
    AF2BIND の結果を config に追加して pepmlm_pplm.py を実行。
    """
    # config に AF2BIND のパスを追加
    cfg_dict["AF2BIND_MATRIX_PATH"] = str(npy_path)

    # LAMBDA_AF2BIND のデフォルト
    if "LAMBDA_AF2BIND" not in cfg_dict:
        cfg_dict["LAMBDA_AF2BIND"] = 1.0
        print(f"[pipeline] LAMBDA_AF2BIND not set, using default: 1.0")
    if "PBIND_BIAS_SCALE" not in cfg_dict:
        cfg_dict["PBIND_BIAS_SCALE"] = 5.0

    # 一時的な config ファイルを書き出す（pepmlm_pplm.py が --config で読む）
    tmp_config_path = Path(original_config_path).with_suffix(".pipeline.json")
    save_config(cfg_dict, str(tmp_config_path))
    print(f"[pipeline] Saved updated config: {tmp_config_path}")

    print(f"\n{'='*60}")
    print(f"[pipeline] Step 2: Running PepMLM+PPLM")
    print(f"{'='*60}")
    print(f"  AF2BIND matrix: {npy_path}")
    print(f"  LAMBDA_AF2BIND: {cfg_dict['LAMBDA_AF2BIND']}")
    print(f"  PBIND_BIAS_SCALE: {cfg_dict['PBIND_BIAS_SCALE']}")
    print(f"  TARGET_SEQUENCE: {cfg_dict['TARGET_SEQUENCE'][:30]}... ({len(cfg_dict['TARGET_SEQUENCE'])} aa)")

    # pepmlm_pplm.py のパス（このスクリプトと同じディレクトリを想定）
    pepmlm_script = Path(__file__).parent / "pepmlm_pplm.py"
    if not pepmlm_script.exists():
        # 同じディレクトリになければ PATH から探す
        pepmlm_script = Path(cfg_dict.get("PEPMLM_SCRIPT", "pepmlm_pplm.py"))

    # 現在の Python 環境でそのまま実行
    cmd = [
        sys.executable,
        str(pepmlm_script),
        "--config", str(tmp_config_path),
    ]

    print(f"\n  Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\n[ERROR] pepmlm_pplm.py failed (exit code {result.returncode})")
        sys.exit(1)

    print(f"\n[pipeline] PepMLM+PPLM done.")


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser(
        description="AF2BIND → PepMLM+PPLM パイプライン"
    )
    parser.add_argument("--config", required=True, help="path to config.json")
    parser.add_argument(
        "--skip-af2bind", action="store_true",
        help="AF2BIND をスキップして既存の .npy を使う（AF2BIND_MATRIX_PATH が config に必要）"
    )
    parser.add_argument(
        "--af2bind-only", action="store_true",
        help="AF2BIND だけ実行して PepMLM は実行しない"
    )
    args = parser.parse_args()

    cfg_dict = load_config(args.config)

    # =========================================================
    # Step 1: AF2BIND
    # =========================================================
    if args.skip_af2bind:
        # スキップ: config に直接 AF2BIND_MATRIX_PATH が書いてある前提
        npy_path = Path(cfg_dict.get("AF2BIND_MATRIX_PATH", ""))
        if not npy_path.exists():
            print(f"[ERROR] --skip-af2bind が指定されましたが、"
                  f"AF2BIND_MATRIX_PATH が見つかりません: {npy_path}")
            sys.exit(1)
        print(f"[pipeline] Skipping AF2BIND, using: {npy_path}")
    else:
        # 必須フィールドの確認
        required_af2bind_fields = [
            "PDB_PATH", "AF2BIND_SCRIPT",
            "AF2BIND_PARAMS_DIR", "AF2BIND_AF_PARAMS_DIR", "AF2BIND_OUT_DIR",
        ]
        missing = [k for k in required_af2bind_fields if k not in cfg_dict]
        if missing:
            print(f"[ERROR] config に以下のフィールドが不足しています:")
            for k in missing:
                print(f"  - {k}")
            print("\n以下を config.json に追加してください:")
            print(json.dumps({
                "PDB_PATH": "/home/morishita/af2bind/1YCR.pdb",
                "PDB_CHAIN": "A",
                "AF2BIND_CONDA_ENV": "af2bind",
                "AF2BIND_SCRIPT": "/home/morishita/af2bind/run_af2bind.py",
                "AF2BIND_PARAMS_DIR": "/home/morishita/af2bind/af2bind_params",
                "AF2BIND_AF_PARAMS_DIR": "/home/morishita/af2bind/params",
                "AF2BIND_OUT_DIR": "/mnt/hdd/morishita/af2bind_outputs",
                "AF2BIND_SEED": 0,
                "AF2BIND_MASK_SIDECHAINS": True,
            }, indent=2))
            sys.exit(1)

        npy_path = run_af2bind(cfg_dict)

    # =========================================================
    # ターゲット配列の整合性チェック
    # =========================================================
    check_target_sequence(cfg_dict, npy_path)

    if args.af2bind_only:
        print(f"\n[pipeline] --af2bind-only が指定されたので PepMLM はスキップします。")
        print(f"[pipeline] 生成された .npy: {npy_path}")
        return

    # =========================================================
    # Step 2: PepMLM + PPLM
    # =========================================================
    run_pepmlm(cfg_dict, npy_path, args.config)

    print(f"\n{'='*60}")
    print(f"[pipeline] All done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
