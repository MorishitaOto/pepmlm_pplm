#!/usr/bin/env python
# ============================================================
# Full pipeline: checkpoint → inference → Boltz yaml → Boltz predict → evaluate
#
# 前提:
#   - pepmlm_pplm.py (or *_cryptic.py) で学習済みの checkpoint があること
#   - apo PDB (必須) と holo PDB (任意) が pepmlm_pplm/data/pdb/ にあること
#   - cryptic スコア JSON が pepmlm_pplm/cryptic_json/ にあること
#
# 使い方:
#   python run_full_pipeline.py --pipeline_config <pipeline.json>
#
# pipeline_config 例 (pepmlm_pplm/config/pipeline_*.json):
# {
#   "TRAIN_CONFIG": "pepmlm_pplm/config/20260502_6XI7_cryptic.json",
#   "ITERATION": 100,
#   "PDB_ID": "6XI7",
#   "NUM_SAMPLES": 100,
#   "TOP_N": 10,
#   "APO_PDB": "pepmlm_pplm/data/pdb/6XI7_apo.pdb",
#   "HOLO_PDB": "pepmlm_pplm/data/pdb/6XI7_holo.pdb",
#   "CRYPTIC_SCORES_JSON": "pepmlm_pplm/cryptic_json/cryptic_6XI7_A.json",
#   "CRYPTIC_THRESHOLD": 0.3,
#   "CRYPTIC_RESIDUES": null,
#   "TARGET_CHAIN": "A",
#   "PEPTIDE_CHAIN": "B",
#   "APO_CHAIN": "A",
#   "HOLO_CHAIN": "A",
#   "HOLO_LIGAND_NAME": null,
#   "INTERFACE_CUTOFF": 5.0,
#   "PEPMLM_PPLM_ROOT": "/home/users/gds/pepmlm_pplm",
#   "BOLTZ_OPTIONS": {
#       "output_format": "pdb",
#       "use_msa_server": false,
#       "diffusion_samples": 1,
#       "recycling_steps": 3
#   },
#   "SKIP_STEPS": []
# }
#
# SKIP_STEPS で "inference", "make_yaml", "boltz", "evaluate" のいずれかを
# 指定するとそのステップを飛ばせる (再実行・部分実行用)
# ============================================================

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# =========================
# Pipeline 設定読み込み
# =========================
def load_pipeline_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = json.load(f)

    required = ["TRAIN_CONFIG", "ITERATION", "PDB_ID", "APO_PDB"]
    missing = [k for k in required if k not in cfg or cfg[k] is None]
    if missing:
        raise ValueError(f"Missing required keys in pipeline config: {missing}")

    cfg.setdefault("NUM_SAMPLES", 100)
    cfg.setdefault("TOP_N", 10)
    cfg.setdefault("TARGET_CHAIN", "A")
    cfg.setdefault("PEPTIDE_CHAIN", "B")
    cfg.setdefault("APO_CHAIN", "A")
    cfg.setdefault("HOLO_CHAIN", "A")
    cfg.setdefault("HOLO_PDB", None)
    cfg.setdefault("HOLO_LIGAND_NAME", None)
    cfg.setdefault("CRYPTIC_SCORES_JSON", None)
    cfg.setdefault("CRYPTIC_THRESHOLD", None)
    cfg.setdefault("CRYPTIC_RESIDUES", None)
    cfg.setdefault("INTERFACE_CUTOFF", 5.0)
    cfg.setdefault("BOLTZ_OPTIONS", {})
    cfg.setdefault("SKIP_STEPS", [])
    cfg.setdefault("PEPMLM_PPLM_ROOT", None)
    cfg.setdefault("PYTHON_EXECUTABLE", None)  # None = sys.executable (実行環境の Python)
    cfg.setdefault("BOLTZ_PYTHON_EXECUTABLE", None)  # Boltz 専用環境の Python
    cfg.setdefault("VISUALIZE", True)   # False にすると可視化をスキップ

    # cryptic 指定の整合性チェック
    if cfg["CRYPTIC_RESIDUES"] is None:
        if cfg["CRYPTIC_SCORES_JSON"] is None or cfg["CRYPTIC_THRESHOLD"] is None:
            raise ValueError(
                "Must specify either CRYPTIC_RESIDUES or "
                "(CRYPTIC_SCORES_JSON + CRYPTIC_THRESHOLD)"
            )

    return cfg


def load_train_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


# =========================
# Path helpers
# =========================
def resolve_root(pipeline_cfg: Dict[str, Any]) -> Path:
    """pepmlm_pplm/ のルートを決める"""
    if pipeline_cfg.get("PEPMLM_PPLM_ROOT"):
        return Path(pipeline_cfg["PEPMLM_PPLM_ROOT"]).resolve()
    # フォールバック: このスクリプトの親ディレクトリ
    return Path(__file__).resolve().parent.parent


def get_run_name(train_cfg: Dict[str, Any], train_cfg_path: str) -> str:
    """
    学習configの SAVE_DIR から run_name を抽出。
    例: SAVE_DIR='.../result/20260502_6XI7_cryptic' → '20260502_6XI7_cryptic'
    なければ config ファイル名をフォールバック。
    """
    if "SAVE_DIR" in train_cfg and train_cfg["SAVE_DIR"]:
        return os.path.basename(train_cfg["SAVE_DIR"].rstrip("/"))
    return Path(train_cfg_path).stem


def get_python_executable(pipeline_cfg: Dict[str, Any]) -> str:
    """
    pipeline_cfg["PYTHON_EXECUTABLE"] が指定されていればそれを使う。
    None / 未指定の場合は sys.executable (run_full_pipeline.py を実行した Python)。
    """
    explicit = pipeline_cfg.get("PYTHON_EXECUTABLE")
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(
                f"PYTHON_EXECUTABLE not found: {explicit}\n"
                "conda activate <env> 後に "
                "`which python` で確認してください"
            )
        return str(p)
    return sys.executable


def get_boltz_python_executable(pipeline_cfg: Dict[str, Any]) -> str:
    """
    Boltz 実行用の Python。
    BOLTZ_PYTHON_EXECUTABLE が指定されていればそれを使い、
    なければ PYTHON_EXECUTABLE → sys.executable の順にフォールバック。
    """
    explicit = pipeline_cfg.get("BOLTZ_PYTHON_EXECUTABLE")
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(
                f"BOLTZ_PYTHON_EXECUTABLE not found: {explicit}\n"
                "conda activate boltz 後に `which python` で確認してください"
            )
        return str(p)
    # フォールバック: PYTHON_EXECUTABLE or sys.executable
    return get_python_executable(pipeline_cfg)


def build_paths(
    pipeline_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    run_name: str,
) -> Dict[str, Path]:
    root = resolve_root(pipeline_cfg)
    iteration = pipeline_cfg["ITERATION"]

    run_iter_dir = root / "boltz" / f"{run_name}_iter{iteration}"
    return {
        "root": root,
        "src_dir": Path(__file__).resolve().parent,
        "inference_csv": root / "inference_result" / f"{run_name}_iter{iteration}.csv",
        "run_iter_dir": run_iter_dir,
        "input_yaml_dir": run_iter_dir / "input_yaml",
        "boltz_output_dir": run_iter_dir / "boltz_output",
        "boltz_ini": run_iter_dir / "boltz_config.ini",
        "evaluation_dir": run_iter_dir / "evaluation",
        "evaluation_pdf": run_iter_dir / "evaluation" / "evaluation_report.pdf",
        "log_file": run_iter_dir / "pipeline.log",
    }


# =========================
# サブプロセス実行
# =========================
def run_cmd(cmd: List[str], log_file: Optional[Path] = None,
            cwd: Optional[Path] = None,
            extra_env: Optional[Dict[str, str]] = None) -> None:
    """コマンドを実行し、stdout/stderr をリアルタイムで出力 + ログ記録"""
    pretty = " ".join(str(c) for c in cmd)
    print(f"\n$ {pretty}\n", flush=True)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as lf:
            lf.write(f"\n$ {pretty}\n")

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log_file is not None:
            with open(log_file, "a") as lf:
                lf.write(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {pretty}")


# =========================
# Step 1: Inference
# =========================
def step_inference(
    pipeline_cfg: Dict[str, Any],
    paths: Dict[str, Path],
) -> None:
    print("\n" + "=" * 60)
    print("STEP 1: Inference from checkpoint")
    print("=" * 60)

    inference_script = paths["src_dir"] / "inference_from_checkpoint.py"
    if not inference_script.exists():
        raise FileNotFoundError(f"inference script not found: {inference_script}")

    paths["inference_csv"].parent.mkdir(parents=True, exist_ok=True)

    python = get_python_executable(pipeline_cfg)
    cmd = [
        python, str(inference_script),
        "--config", pipeline_cfg["TRAIN_CONFIG"],
        "--iteration", str(pipeline_cfg["ITERATION"]),
        "--num_samples", str(pipeline_cfg["NUM_SAMPLES"]),
        "--use_pplm",
        "--output_csv", str(paths["inference_csv"]),
    ]
    # スパコン等でネットワーク不可の場合 HuggingFace キャッシュを使う
    offline_env = {
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    run_cmd(cmd, log_file=paths["log_file"], extra_env=offline_env)

    if not paths["inference_csv"].exists():
        raise RuntimeError(f"Inference CSV not produced: {paths['inference_csv']}")
    print(f"✅ Inference CSV: {paths['inference_csv']}")


# =========================
# Step 2: yaml 生成
# =========================
def step_make_yaml(
    pipeline_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    paths: Dict[str, Path],
) -> None:
    print("\n" + "=" * 60)
    print("STEP 2: Generate Boltz yaml inputs")
    print("=" * 60)

    yaml_script = paths["src_dir"] / "make_boltz_inputs.py"
    if not yaml_script.exists():
        raise FileNotFoundError(f"make_boltz_inputs.py not found: {yaml_script}")

    target_seq = train_cfg.get("TARGET_SEQUENCE")
    if not target_seq:
        raise ValueError("TARGET_SEQUENCE missing in train config")

    paths["input_yaml_dir"].mkdir(parents=True, exist_ok=True)

    python = get_python_executable(pipeline_cfg)
    cmd = [
        python, str(yaml_script),
        "--input_csv", str(paths["inference_csv"]),
        "--output_dir", str(paths["input_yaml_dir"]),
        "--target_sequence", target_seq,
        "--pdb_id", pipeline_cfg["PDB_ID"],
        "--top_n", str(pipeline_cfg["TOP_N"]),
    ]
    run_cmd(cmd, log_file=paths["log_file"])

    yamls = list(paths["input_yaml_dir"].glob("*.yaml"))
    if not yamls:
        raise RuntimeError(f"No yaml produced in {paths['input_yaml_dir']}")
    print(f"✅ Generated {len(yamls)} yaml file(s) in {paths['input_yaml_dir']}")


# =========================
# Step 3: Boltz config 生成 + 実行
# =========================
def write_boltz_ini(
    ini_path: Path,
    input_dir: Path,
    output_dir: Path,
    options: Dict[str, Any],
) -> None:
    """
    Boltz の .ini を書き出す。
    既知セクション: paths / model / hardware / prediction / msa / affinity
    options で渡された値を該当セクションに振り分け、未指定は空欄。
    """
    section_keys = {
        "paths": ["input_dir", "output_dir", "cache_dir"],
        "model": ["checkpoint", "model", "method"],
        "hardware": ["devices", "accelerator", "num_workers", "preprocessing_threads"],
        "prediction": [
            "recycling_steps", "sampling_steps", "diffusion_samples",
            "max_parallel_samples", "step_scale",
            "write_full_pae", "write_full_pde",
            "output_format", "override", "seed",
        ],
        "msa": [
            "use_msa_server", "msa_server_url", "msa_pairing_strategy",
            "msa_server_username", "msa_server_password",
        ],
        "affinity": [
            "use_potentials", "affinity_mw_correction",
            "sampling_steps_affinity", "diffusion_samples_affinity",
            "affinity_checkpoint", "max_msa_seqs", "subsample_msa",
            "num_subsampled_msa", "no_kernels", "write_embeddings",
        ],
    }

    parser = configparser.ConfigParser(allow_no_value=True)
    # ConfigParser はデフォルトで keys を小文字化するが、明示的に保持
    parser.optionxform = str

    # paths は必須
    parser["paths"] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "cache_dir": str(options.get("cache_dir", "") or ""),
    }

    for section, keys in section_keys.items():
        if section == "paths":
            continue
        parser[section] = {}
        for k in keys:
            v = options.get(k, "")
            if v is None:
                v = ""
            elif isinstance(v, bool):
                v = "True" if v else "False"
            else:
                v = str(v)
            parser[section][k] = v

    ini_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ini_path, "w") as f:
        parser.write(f)


def parse_boltz_ini(ini_path: Path) -> Dict[str, Dict[str, str]]:
    parser = configparser.ConfigParser(allow_no_value=True)
    parser.optionxform = str
    parser.read(ini_path)
    return {s: dict(parser[s]) for s in parser.sections()}


def ini_to_boltz_args(ini_data: Dict[str, Dict[str, str]]) -> List[str]:
    """
    .ini の内容を `boltz predict` のコマンドライン引数に変換する。
    位置引数 input_dir / out_dir 以外はすべてオプション。

    boolean は True なら --flag、False なら無視。
    空欄は無視。
    """
    args: List[str] = []
    input_dir = ini_data.get("paths", {}).get("input_dir", "").strip()
    output_dir = ini_data.get("paths", {}).get("output_dir", "").strip()
    cache_dir = ini_data.get("paths", {}).get("cache_dir", "").strip()

    if not input_dir or not output_dir:
        raise ValueError("ini must specify paths.input_dir and paths.output_dir")

    # 位置引数: input_dir
    args.append(input_dir)
    # 出力先
    args += ["--out_dir", output_dir]
    if cache_dir:
        args += ["--cache", cache_dir]

    # boolean フラグ扱いのキー (Boltz CLI が --flag 形式で受けるもの)
    bool_flags = {
        "use_msa_server", "write_full_pae", "write_full_pde",
        "override", "use_potentials", "affinity_mw_correction",
        "no_kernels", "write_embeddings",
    }

    # CLI に渡すときの引数名マップ (.ini キー → CLI フラグ)
    cli_name = {
        # model
        "checkpoint": "--checkpoint",
        "model": "--model",
        "method": "--method",
        # hardware
        "devices": "--devices",
        "accelerator": "--accelerator",
        "num_workers": "--num_workers",
        "preprocessing_threads": "--preprocessing-threads",
        # prediction
        "recycling_steps": "--recycling_steps",
        "sampling_steps": "--sampling_steps",
        "diffusion_samples": "--diffusion_samples",
        "max_parallel_samples": "--max_parallel_samples",
        "step_scale": "--step_scale",
        "write_full_pae": "--write_full_pae",
        "write_full_pde": "--write_full_pde",
        "output_format": "--output_format",
        "override": "--override",
        "seed": "--seed",
        # msa
        "use_msa_server": "--use_msa_server",
        "msa_server_url": "--msa_server_url",
        "msa_pairing_strategy": "--msa_pairing_strategy",
        "msa_server_username": "--msa_server_username",
        "msa_server_password": "--msa_server_password",
        # affinity
        "use_potentials": "--use_potentials",
        "affinity_mw_correction": "--affinity_mw_correction",
        "sampling_steps_affinity": "--sampling_steps_affinity",
        "diffusion_samples_affinity": "--diffusion_samples_affinity",
        "affinity_checkpoint": "--affinity_checkpoint",
        "max_msa_seqs": "--max_msa_seqs",
        "subsample_msa": "--subsample_msa",
        "num_subsampled_msa": "--num_subsampled_msa",
        "no_kernels": "--no_kernels",
        "write_embeddings": "--write_embeddings",
    }

    for section_name in ("model", "hardware", "prediction", "msa", "affinity"):
        for k, v in ini_data.get(section_name, {}).items():
            v = (v or "").strip()
            if not v:
                continue
            flag = cli_name.get(k)
            if flag is None:
                continue

            if k in bool_flags:
                if v.lower() in ("true", "1", "yes"):
                    args.append(flag)
                # False / 0 は何もしない
            else:
                args += [flag, v]

    return args


def step_boltz_predict(
    pipeline_cfg: Dict[str, Any],
    paths: Dict[str, Path],
) -> None:
    print("\n" + "=" * 60)
    print("STEP 3: Boltz predict")
    print("=" * 60)

    boltz_python = get_boltz_python_executable(pipeline_cfg)
    # `boltz` コマンドを直接探す前に、boltz_python の同ディレクトリも確認する
    boltz_bin = Path(boltz_python).parent / "boltz"
    if boltz_bin.exists():
        boltz_cmd = str(boltz_bin)
    elif shutil.which("boltz"):
        boltz_cmd = "boltz"
    else:
        # python -m boltz にフォールバック
        boltz_cmd = None
    if boltz_cmd is None and not shutil.which("boltz"):
        raise RuntimeError(
            "`boltz` command not found.\n"
            f"  boltz_python={boltz_python}\n"
            "  BOLTZ_PYTHON_EXECUTABLE の conda 環境に boltz がインストールされているか確認してください"
        )

    # .ini を生成
    boltz_options = dict(pipeline_cfg.get("BOLTZ_OPTIONS") or {})
    # output_format のデフォルトは pdb
    boltz_options.setdefault("output_format", "pdb")
    # use_msa_server のデフォルトは False (yaml で msa: empty を指定している前提)
    boltz_options.setdefault("use_msa_server", False)

    write_boltz_ini(
        ini_path=paths["boltz_ini"],
        input_dir=paths["input_yaml_dir"],
        output_dir=paths["boltz_output_dir"],
        options=boltz_options,
    )
    print(f"✅ Wrote Boltz config: {paths['boltz_ini']}")

    # .ini を読んで CLI 引数に変換
    ini_data = parse_boltz_ini(paths["boltz_ini"])
    boltz_args = ini_to_boltz_args(ini_data)

    paths["boltz_output_dir"].mkdir(parents=True, exist_ok=True)

    if boltz_cmd:
        cmd = [boltz_cmd, "predict"] + boltz_args
    else:
        cmd = [boltz_python, "-m", "boltz", "predict"] + boltz_args
    run_cmd(cmd, log_file=paths["log_file"])

    # 出力の存在確認
    results = list(paths["boltz_output_dir"].glob("boltz_results_*"))
    if not results:
        raise RuntimeError(
            f"No Boltz output found under {paths['boltz_output_dir']}"
        )
    print(f"✅ Boltz produced {len(results)} result dir(s) under "
          f"{paths['boltz_output_dir']}")


# =========================
# Step 4: 評価
# =========================
def step_evaluate(
    pipeline_cfg: Dict[str, Any],
    paths: Dict[str, Path],
) -> None:
    print("\n" + "=" * 60)
    print("STEP 4: Evaluate predictions")
    print("=" * 60)

    eval_script = paths["src_dir"] / "evaluate_boltz_predictions.py"
    if not eval_script.exists():
        raise FileNotFoundError(
            f"evaluate_boltz_predictions.py not found: {eval_script}"
        )

    paths["evaluation_dir"].mkdir(parents=True, exist_ok=True)

    python = get_python_executable(pipeline_cfg)
    cmd = [
        python, str(eval_script),
        "--boltz_output_root", str(paths["boltz_output_dir"]),
        "--apo_pdb", pipeline_cfg["APO_PDB"],
        "--output_dir", str(paths["evaluation_dir"]),
        "--target_chain", pipeline_cfg["TARGET_CHAIN"],
        "--peptide_chain", pipeline_cfg["PEPTIDE_CHAIN"],
        "--apo_chain", pipeline_cfg["APO_CHAIN"],
        "--holo_chain", pipeline_cfg["HOLO_CHAIN"],
        "--interface_cutoff", str(pipeline_cfg["INTERFACE_CUTOFF"]),
    ]
    if pipeline_cfg.get("HOLO_PDB"):
        cmd += ["--holo_pdb", pipeline_cfg["HOLO_PDB"]]
    if pipeline_cfg.get("HOLO_LIGAND_NAME"):
        cmd += ["--holo_ligand_name", pipeline_cfg["HOLO_LIGAND_NAME"]]

    if pipeline_cfg.get("CRYPTIC_RESIDUES"):
        cmd += ["--cryptic_residues", pipeline_cfg["CRYPTIC_RESIDUES"]]
    else:
        cmd += [
            "--cryptic_scores_json", pipeline_cfg["CRYPTIC_SCORES_JSON"],
            "--cryptic_threshold", str(pipeline_cfg["CRYPTIC_THRESHOLD"]),
        ]

    run_cmd(cmd, log_file=paths["log_file"])

    summary = paths["evaluation_dir"] / "evaluation_summary.csv"
    if not summary.exists():
        raise RuntimeError(f"Evaluation summary not produced: {summary}")
    print(f"✅ Evaluation summary: {summary}")


# =========================
# Step 5: 可視化
# =========================
def step_visualize(
    pipeline_cfg: Dict[str, Any],
    paths: Dict[str, Path],
) -> None:
    print("\n" + "=" * 60)
    print("STEP 5: Visualize evaluation results")
    print("=" * 60)

    viz_script = paths["src_dir"] / "visualize_evaluation.py"
    if not viz_script.exists():
        raise FileNotFoundError(
            f"visualize_evaluation.py not found: {viz_script}"
        )

    summary_csv = paths["evaluation_dir"] / "evaluation_summary.csv"
    if not summary_csv.exists():
        raise RuntimeError(f"evaluation_summary.csv not found: {summary_csv}")

    # cryptic 残基の文字列を構築 (明示リスト優先、なければ JSON + threshold から自動生成)
    cryptic_residues_str = pipeline_cfg.get("CRYPTIC_RESIDUES")
    if not cryptic_residues_str and pipeline_cfg.get("CRYPTIC_SCORES_JSON"):
        import json as _json
        scores_path = Path(pipeline_cfg["CRYPTIC_SCORES_JSON"])
        threshold   = float(pipeline_cfg.get("CRYPTIC_THRESHOLD", 0.3))
        if scores_path.exists():
            with open(scores_path) as f:
                scores = _json.load(f)
            ids = [
                str(e["residue_id"]) for e in scores
                if (e.get("displayed_score") or e.get("raw_score") or 0) >= threshold
            ]
            cryptic_residues_str = ",".join(ids)

    # タイトル: run_name + iter
    run_name  = paths["run_iter_dir"].name  # 例: 20260502_6XI7_cryptic_iter100
    run_title = run_name.replace("_", " ")

    python = get_python_executable(pipeline_cfg)
    cmd = [
        python, str(viz_script),
        "--input_csv",  str(summary_csv),
        "--output_pdf", str(paths["evaluation_pdf"]),
        "--title",      run_title,
    ]
    if cryptic_residues_str:
        cmd += ["--cryptic_residues", cryptic_residues_str]

    run_cmd(cmd, log_file=paths["log_file"])

    if not paths["evaluation_pdf"].exists():
        raise RuntimeError(f"PDF not produced: {paths['evaluation_pdf']}")
    print(f"✅ Evaluation PDF: {paths['evaluation_pdf']}")


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="Full pipeline: inference → yaml → Boltz → evaluate → visualize"
    )
    ap.add_argument("--pipeline_config", required=True,
                    help="Pipeline config JSON")
    ap.add_argument("--skip", nargs="*", default=None,
                    help="Skip steps: inference / make_yaml / boltz / evaluate / visualize")
    args = ap.parse_args()

    t0 = time.time()
    pipeline_cfg = load_pipeline_config(args.pipeline_config)
    if args.skip:
        pipeline_cfg["SKIP_STEPS"] = list(
            set((pipeline_cfg.get("SKIP_STEPS") or []) + args.skip)
        )

    # 入力ファイル存在チェック
    if not Path(pipeline_cfg["TRAIN_CONFIG"]).exists():
        sys.exit(f"TRAIN_CONFIG not found: {pipeline_cfg['TRAIN_CONFIG']}")
    if not Path(pipeline_cfg["APO_PDB"]).exists():
        sys.exit(f"APO_PDB not found: {pipeline_cfg['APO_PDB']}")
    if pipeline_cfg.get("HOLO_PDB") and not Path(pipeline_cfg["HOLO_PDB"]).exists():
        print(f"⚠ HOLO_PDB not found, ignoring: {pipeline_cfg['HOLO_PDB']}")
        pipeline_cfg["HOLO_PDB"] = None
    if (pipeline_cfg.get("CRYPTIC_SCORES_JSON")
            and not Path(pipeline_cfg["CRYPTIC_SCORES_JSON"]).exists()):
        sys.exit(
            f"CRYPTIC_SCORES_JSON not found: "
            f"{pipeline_cfg['CRYPTIC_SCORES_JSON']}"
        )

    train_cfg = load_train_config(pipeline_cfg["TRAIN_CONFIG"])
    run_name = get_run_name(train_cfg, pipeline_cfg["TRAIN_CONFIG"])
    paths = build_paths(pipeline_cfg, train_cfg, run_name)

    # run ディレクトリと初回ログ
    paths["run_iter_dir"].mkdir(parents=True, exist_ok=True)
    with open(paths["log_file"], "a") as lf:
        lf.write(f"\n{'=' * 60}\n")
        lf.write(f"Pipeline start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"run_name = {run_name}, iteration = {pipeline_cfg['ITERATION']}\n")
        lf.write(f"pipeline_config = {args.pipeline_config}\n")
        lf.write(f"{'=' * 60}\n")

    # pipeline config のスナップショットを残す
    snapshot = paths["run_iter_dir"] / "pipeline_config.snapshot.json"
    with open(snapshot, "w") as f:
        json.dump(pipeline_cfg, f, indent=2)

    skip = set(pipeline_cfg.get("SKIP_STEPS") or [])

    print(f"\n=== Pipeline plan ===")
    print(f"  run_name        : {run_name}")
    print(f"  iteration       : {pipeline_cfg['ITERATION']}")
    print(f"  output base     : {paths['run_iter_dir']}")
    print(f"  skip            : {sorted(skip) if skip else '(none)'}")
    print(f"  visualize       : {'yes' if pipeline_cfg.get('VISUALIZE', True) else 'no'}")
    print(f"=====================")

    # 1. Inference
    if "inference" not in skip:
        step_inference(pipeline_cfg, paths)
    else:
        print("\n[skipped] inference")
        if not paths["inference_csv"].exists():
            sys.exit(
                f"inference skipped but CSV not found: {paths['inference_csv']}"
            )

    # 2. yaml 生成
    if "make_yaml" not in skip:
        step_make_yaml(pipeline_cfg, train_cfg, paths)
    else:
        print("\n[skipped] make_yaml")
        if not list(paths["input_yaml_dir"].glob("*.yaml")):
            sys.exit(
                f"make_yaml skipped but no yaml in {paths['input_yaml_dir']}"
            )

    # 3. Boltz predict
    if "boltz" not in skip:
        step_boltz_predict(pipeline_cfg, paths)
    else:
        print("\n[skipped] boltz")
        if not list(paths["boltz_output_dir"].glob("boltz_results_*")):
            sys.exit(
                f"boltz skipped but no boltz_results_* in "
                f"{paths['boltz_output_dir']}"
            )

    # 4. 評価
    if "evaluate" not in skip:
        step_evaluate(pipeline_cfg, paths)
    else:
        print("\n[skipped] evaluate")

    # 5. 可視化
    if "visualize" not in skip and pipeline_cfg.get("VISUALIZE", True):
        step_visualize(pipeline_cfg, paths)
    else:
        print("\n[skipped] visualize")

    elapsed = time.time() - t0
    print(f"\n🎉 Pipeline complete in {elapsed:.1f}s")
    print(f"   Results: {paths['run_iter_dir']}")


if __name__ == "__main__":
    main()
