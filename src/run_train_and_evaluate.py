#!/usr/bin/env python
# ============================================================
# run_train_and_evaluate.py
#
# PepMLM 学習 (pepmlm_pplm_cryptic.py) →
# 推論〜Boltz-2 評価 (run_full_pipeline.py) を一括実行する統合スクリプト。
#
# 使い方:
#   # 学習〜評価まで全実行
#   python src/run_train_and_evaluate.py \
#       --pipeline_config config/pipeline_config/pipeline_20260520_4LDJ_cryptic_iter100.json
#
#   # 学習をスキップして推論以降だけ実行
#   python src/run_train_and_evaluate.py \
#       --pipeline_config config/pipeline_config/pipeline_20260520_4LDJ_cryptic_iter100.json \
#       --skip train
#
# pipeline_config の追加キー（run_full_pipeline.py の既存キーに加えて）:
#   TRAIN_SCRIPT  : 学習スクリプトのパス (default: src/pepmlm_pplm_cryptic.py)
#   TRAIN_CONFIG  : 学習用 config JSON のパス (必須)
#   SKIP_TRAIN    : true にすると --skip train と同等 (default: false)
# ============================================================

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =========================
# Config loading
# =========================
def load_pipeline_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def resolve_path(p: str | None, base_dir: Path) -> Path | None:
    """相対パスをスクリプト実行ディレクトリ基準で解決"""
    if p is None:
        return None
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base_dir / pp).resolve()


# =========================
# Step 1: 学習
# =========================
def run_train(cfg: dict, base_dir: Path, python_exe: str) -> int:
    train_script = resolve_path(
        cfg.get("TRAIN_SCRIPT", "src/pepmlm_pplm_cryptic.py"), base_dir
    )
    train_config = resolve_path(cfg.get("TRAIN_CONFIG"), base_dir)

    if train_config is None:
        log.error("TRAIN_CONFIG is required in pipeline_config")
        return 1
    if not train_script.exists():
        log.error(f"TRAIN_SCRIPT not found: {train_script}")
        return 1
    if not train_config.exists():
        log.error(f"TRAIN_CONFIG not found: {train_config}")
        return 1

    cmd = [python_exe, str(train_script), "--config", str(train_config)]  # --config に修正
    log.info("=" * 60)
    log.info("STEP: Training PepMLM")
    log.info(f"  Script : {train_script}")
    log.info(f"  Config : {train_config}")
    log.info(f"  Command: {' '.join(cmd)}")
    log.info("=" * 60)

    env = os.environ.copy()
    env["TRANSFORMERS_OFFLINE"] = "1"  # スパコン環境: HuggingFace オフライン

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(base_dir), env=env)
    elapsed = time.time() - t0
    log.info(f"Training finished in {elapsed:.1f}s (returncode={result.returncode})")
    return result.returncode


# =========================
# Step 2: 推論〜評価 (run_full_pipeline.py に委譲)
# =========================
def run_full_pipeline(
    cfg: dict,
    pipeline_config_path: str,
    base_dir: Path,
    python_exe: str,
    skip_steps: list[str],
) -> int:
    pipeline_script = resolve_path("src/run_full_pipeline.py", base_dir)
    if not pipeline_script.exists():
        log.error(f"run_full_pipeline.py not found: {pipeline_script}")
        return 1

    cmd = [python_exe, str(pipeline_script), "--pipeline_config", pipeline_config_path]
    if skip_steps:
        cmd += ["--skip"] + skip_steps

    log.info("=" * 60)
    log.info("STEP: Inference + Evaluation (run_full_pipeline.py)")
    log.info(f"  Skip steps: {skip_steps}")
    log.info(f"  Command: {' '.join(cmd)}")
    log.info("=" * 60)

    env = os.environ.copy()
    env["TRANSFORMERS_OFFLINE"] = "1"

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(base_dir), env=env)
    elapsed = time.time() - t0
    log.info(f"Pipeline finished in {elapsed:.1f}s (returncode={result.returncode})")
    return result.returncode


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="PepMLM 学習 → 推論 → 評価 を一括実行する統合スクリプト"
    )
    parser.add_argument(
        "--pipeline_config",
        required=True,
        help="pipeline_config JSON のパス",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="STEP",
        help=(
            "スキップするステップを空白区切りで指定。"
            "'train' を指定すると学習をスキップ。"
            "推論以降のスキップは run_full_pipeline.py の --skip と同じ"
            " (inference make_yaml boltz evaluate visualize)。"
        ),
    )
    args = parser.parse_args()

    pipeline_config_path = args.pipeline_config
    if not Path(pipeline_config_path).exists():
        log.error(f"pipeline_config not found: {pipeline_config_path}")
        sys.exit(1)

    cfg = load_pipeline_config(pipeline_config_path)

    # base_dir: config/pipeline_config/ の3つ上 = プロジェクトルート
    base_dir = Path(pipeline_config_path).parent.parent.parent.resolve()
    log.info(f"Base directory: {base_dir}")

    # Python 実行ファイル (pplm 環境)
    python_exe = cfg.get("PYTHON_EXECUTABLE", sys.executable)

    # --skip train が指定されているか、または config の SKIP_TRAIN=true
    skip_steps = list(args.skip) if args.skip else []
    if cfg.get("SKIP_TRAIN", False) and "train" not in skip_steps:
        skip_steps.insert(0, "train")

    skip_train = "train" in skip_steps
    # run_full_pipeline.py に渡すスキップリスト (train は除外)
    pipeline_skip = [s for s in skip_steps if s != "train"]

    # ── Step 1: 学習 ──────────────────────────────────────────
    if not skip_train:
        rc = run_train(cfg, base_dir, python_exe)
        if rc != 0:
            log.error(f"Training failed (returncode={rc}). Aborting.")
            sys.exit(rc)
    else:
        log.info("Skipping training (--skip train or SKIP_TRAIN=true)")

    # ── Step 2: 推論〜評価 ────────────────────────────────────
    rc = run_full_pipeline(
        cfg=cfg,
        pipeline_config_path=str(Path(pipeline_config_path).resolve()),
        base_dir=base_dir,
        python_exe=python_exe,
        skip_steps=pipeline_skip,
    )
    if rc != 0:
        log.error(f"Pipeline failed (returncode={rc}).")
        sys.exit(rc)

    log.info("All steps completed successfully.")


if __name__ == "__main__":
    main()
