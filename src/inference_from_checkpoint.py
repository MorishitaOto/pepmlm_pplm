#!/usr/bin/env python
# ============================================================
# Inference script using saved PepMLM checkpoint (.pt)
# - 指定したiterationのモデルをロード
# - ペプチド生成（Gibbs sampling）
# - 結果をCSVに保存
# - 固定長 / 可変長 (PEPTIDE_LEN_MIN〜PEPTIDE_LEN_MAX) どちらにも対応
# ============================================================

import os
import json
import argparse
import random
import csv
from types import SimpleNamespace
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ---- external evaluator (optional) ----
from compute_PPLM_affinity import PPLMAffinityPredictor


# =========================
# Config
# =========================
def load_config(path: str):
    with open(path, "r") as f:
        cfg_dict = json.load(f)

    cfg = SimpleNamespace(**cfg_dict)
    cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.DTYPE = getattr(torch, cfg.DTYPE)

    return cfg


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Load model
# =========================
def load_model_from_checkpoint(cfg, checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.PEPM_LM_NAME,
        local_files_only=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.PEPM_LM_NAME,
        local_files_only=True,
    )
    state_dict = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    model.load_state_dict(state_dict)

    model = model.to(cfg.DEVICE).to(cfg.DTYPE)
    model.eval()

    return tokenizer, model


# =========================
# Gibbs sampling
# =========================
@torch.no_grad()
def gibbs_sample_peptide(
    model,
    tokenizer,
    target_seq: str,
    pep_len: int,
    num_steps: int,
    temperature: float,
    device: str,
) -> str:
    mask_token_id = tokenizer.mask_token_id
    cls_token_id  = tokenizer.cls_token_id
    sep_token_id  = tokenizer.sep_token_id
    eos_token_id  = tokenizer.eos_token_id
    pad_token_id  = tokenizer.pad_token_id or 0

    # sep が None の場合は eos を末尾トークンとして使う（PepMLM-650M 対応）
    end_token_id = sep_token_id if sep_token_id is not None else eos_token_id

    # ターゲット配列をIDレベルで事前エンコード（特殊トークンなし）
    target_ids: List[int] = tokenizer(
        target_seq,
        add_special_tokens=False,
    ).input_ids

    # 固定prefix: [CLS] target_ids [EOS]
    gibbs_prefix = torch.tensor(
        [cls_token_id] + target_ids + [end_token_id],
        dtype=torch.long, device=device,
    )
    gibbs_pep_start = 1 + len(target_ids) + 1

    peptide_token_ids = torch.full(
        (pep_len,), mask_token_id, dtype=torch.long, device=device
    )

    for _ in range(num_steps):
        positions = list(range(pep_len))
        random.shuffle(positions)

        for pos in positions:
            peptide_token_ids[pos] = mask_token_id

            # IDレベルで入力構築: [CLS] target_ids [EOS] pep_ids [EOS]
            input_ids = torch.cat([
                gibbs_prefix,
                peptide_token_ids,
                torch.tensor([end_token_id], dtype=torch.long, device=device),
            ]).unsqueeze(0)

            logits = model(input_ids=input_ids).logits[0]
            token_logits = logits[gibbs_pep_start + pos] / temperature

            # 特殊トークンを除外
            for tid in [cls_token_id, sep_token_id, eos_token_id, pad_token_id, mask_token_id]:
                if tid is not None:
                    token_logits[tid] = -float("inf")

            probs = torch.softmax(token_logits, dim=-1)
            peptide_token_ids[pos] = torch.multinomial(probs, 1).item()

    tokens = tokenizer.convert_ids_to_tokens(peptide_token_ids.tolist())
    return tokenizer.convert_tokens_to_string(tokens).replace(" ", "")


# =========================
# PPLM evaluator
# =========================
class PPLMEvaluator:
    def __init__(self, pplm_script):
        self.predictor = PPLMAffinityPredictor(
            pplm_script=pplm_script,
            verbose=False,
        )

    def score(self, target, peptide, step):
        return float(
            self.predictor.predict_affinity(
                protein_seq=target,
                peptide_seq=peptide,
                step=step,
            )
        )


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--use_pplm", action="store_true")
    parser.add_argument("--output_csv", default="results.csv")

    # コマンドラインからの上書き（任意）
    parser.add_argument("--pep_len", type=int, default=None,
                        help="固定長を指定（指定すると config を上書き）")
    parser.add_argument("--pep_len_min", type=int, default=None,
                        help="可変長の下限（指定すると config を上書き）")
    parser.add_argument("--pep_len_max", type=int, default=None,
                        help="可変長の上限（指定すると config を上書き）")

    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.RANDOM_SEED)

    # =========================
    # ペプチド長の設定（コマンドライン引数 > config）
    # =========================
    pep_len_min = args.pep_len_min if args.pep_len_min is not None else getattr(cfg, "PEPTIDE_LEN_MIN", None)
    pep_len_max = args.pep_len_max if args.pep_len_max is not None else getattr(cfg, "PEPTIDE_LEN_MAX", None)
    pep_len_fixed = args.pep_len if args.pep_len is not None else getattr(cfg, "PEPTIDE_LEN", None)

    if pep_len_min is not None and pep_len_max is not None:
        pep_len_min = int(pep_len_min)
        pep_len_max = int(pep_len_max)
        if pep_len_min > pep_len_max:
            raise ValueError(
                f"PEPTIDE_LEN_MIN ({pep_len_min}) > PEPTIDE_LEN_MAX ({pep_len_max})"
            )
        print(f"[PepLen] 可変長モード: {pep_len_min}〜{pep_len_max}")
        variable_length = True
    elif pep_len_fixed is not None:
        pep_len_fixed = int(pep_len_fixed)
        print(f"[PepLen] 固定長モード: {pep_len_fixed}")
        variable_length = False
    else:
        raise ValueError(
            "PEPTIDE_LEN または PEPTIDE_LEN_MIN/PEPTIDE_LEN_MAX を "
            "config またはコマンドラインで指定してください"
        )

    # =========================
    # checkpoint path
    # =========================
    folder_name = os.path.basename(cfg.SAVE_DIR)
    checkpoint_folder = os.path.join(
        "/home/users/gds/pepmlm_pplm/checkpoint",
        folder_name
    )
    ckpt_path = os.path.join(
        checkpoint_folder,
        f"pepmlm_iter_{args.iteration}.pt"
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")

    tokenizer, model = load_model_from_checkpoint(cfg, ckpt_path)

    evaluator = None
    if args.use_pplm:
        evaluator = PPLMEvaluator(cfg.PPLM_SCRIPT)

    # =========================
    # Inference & collect results
    # =========================
    results = []

    for i in range(args.num_samples):
        # 長さを決定（可変長 or 固定長）
        if variable_length:
            current_pep_len = random.randint(pep_len_min, pep_len_max)
        else:
            current_pep_len = pep_len_fixed

        peptide = gibbs_sample_peptide(
            model,
            tokenizer,
            cfg.TARGET_SEQUENCE,
            current_pep_len,
            cfg.NUM_GIBBS_STEPS,
            cfg.TEMPERATURE,
            cfg.DEVICE,
        )

        row = {
            "index": i,
            "peptide": peptide,
            "pep_len": current_pep_len,
        }

        if evaluator:
            score = evaluator.score(cfg.TARGET_SEQUENCE, peptide, step=i)
            row["affinity"] = score

        results.append(row)

    # =========================
    # Save to CSV
    # =========================
    keys = results[0].keys()

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to: {args.output_csv}")


if __name__ == "__main__":
    main()
