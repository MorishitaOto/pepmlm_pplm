#!/usr/bin/env python
# ============================================================
# PepMLM + GFlowNet (Trajectory Balance) - MEMORY FIX VERSION
# ============================================================

import os
import csv
import json
import argparse
import random
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW

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


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_csv_logger(path: str):
    header = [
        "iteration", "sample_idx", "peptide",
        "affinity", "reward", "logprob", "loss"
    ]
    exists = os.path.exists(path)
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=header)
    if not exists:
        writer.writeheader()
    return f, writer


# =========================
# Load model
# =========================
def load_pepmlm(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.PEPM_LM_NAME)
    model = AutoModelForMaskedLM.from_pretrained(cfg.PEPM_LM_NAME)
    model = model.to(cfg.DEVICE)
    model.train()
    return tokenizer, model


# =========================
# Gibbs sampling（軽量版）
# =========================
def gibbs_sample_peptide(
    model,
    tokenizer,
    target_seq,
    pep_len,
    num_steps,
    temperature,
    device,
):
    mask_token_id = tokenizer.mask_token_id
    peptide_token_ids = [mask_token_id] * pep_len

    logprob = torch.tensor(0.0, device=device)

    for step in range(num_steps):
        positions = list(range(pep_len))
        random.shuffle(positions)

        for pos in positions:

            peptide_token_ids[pos] = mask_token_id

            full_seq = target_seq + tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(peptide_token_ids)
            )

            inputs = tokenizer(full_seq, return_tensors="pt").to(device)

            # 🔥 最終stepのみgrad、それ以外は完全no_grad
            if step == num_steps - 1:
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
                    logits = outputs.logits[0]
            else:
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits[0]

            pep_start = inputs.input_ids.size(1) - pep_len - 1
            pos_idx = pep_start + pos

            token_logits = logits[pos_idx] / temperature

            for tid in [
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
                tokenizer.mask_token_id,
            ]:
                if tid is not None:
                    token_logits[tid] = -float("inf")

            probs = torch.softmax(token_logits, dim=-1)

            token = torch.multinomial(probs, 1)
            prob = probs[token]

            # 🔥 logprobはgrad保持（最後stepのみ有効）
            logprob = logprob + torch.log(prob + 1e-9)

            peptide_token_ids[pos] = token.item()

    tokens = tokenizer.convert_ids_to_tokens(peptide_token_ids)
    peptide = tokenizer.convert_tokens_to_string(tokens).replace(" ", "")

    return peptide, logprob


# =========================
# Evaluator
# =========================
class PPLMEvaluator:
    def __init__(self, pplm_script: str):
        self.predictor = PPLMAffinityPredictor(
            pplm_script=pplm_script,
            verbose=False,
        )

    def score(self, target: str, peptide: str, step: int) -> float:
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
    args = parser.parse_args()

    cfg = load_config(args.config)

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    set_seed(cfg.RANDOM_SEED)

    log_file, csv_writer = init_csv_logger(
        os.path.join(cfg.SAVE_DIR, cfg.LOG_CSV)
    )

    tokenizer, model = load_pepmlm(cfg)

    evaluator = PPLMEvaluator(cfg.PPLM_SCRIPT)

    logZ = torch.nn.Parameter(torch.zeros(1, device=cfg.DEVICE))

    optimizer = AdamW(
        list(model.parameters()) + [logZ],
        lr=cfg.LEARNING_RATE,
    )

    scaler = torch.cuda.amp.GradScaler()

    for it in range(cfg.NUM_ITERATIONS):
        print(f"\n=== Iteration {it+1} ===")

        optimizer.zero_grad()  # 🔥 ここで1回だけ

        total_loss_val = 0.0  # 表示用だけ（Tensorにしない）

        for i in range(cfg.NUM_SAMPLES_PER_ITER):

            peptide, logprob = gibbs_sample_peptide(
                model,
                tokenizer,
                cfg.TARGET_SEQUENCE,
                cfg.PEPTIDE_LEN,
                cfg.NUM_GIBBS_STEPS,
                cfg.TEMPERATURE,
                cfg.DEVICE,
            )

            affinity = evaluator.score(
                cfg.TARGET_SEQUENCE,
                peptide,
                step=i,
            )

            log_reward = torch.tensor(
                -affinity / getattr(cfg, "REWARD_TEMPERATURE", 1.0),
                device=cfg.DEVICE,
            ).clamp(-50, 50)

            loss = (logZ + logprob - log_reward) ** 2

            # 🔥 逐次 backward（これが核心）
            scaler.scale(loss).backward()

            total_loss_val += loss.item()

            csv_writer.writerow({
                "iteration": it + 1,
                "sample_idx": i,
                "peptide": peptide,
                "affinity": affinity,
                "reward": torch.exp(log_reward).item(),
                "logprob": logprob.item(),
                "loss": loss.item(),
            })

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        print(f"  loss: {total_loss_val:.4f}")

        ckpt_path = os.path.join(
            cfg.SAVE_DIR,
            f"gflownet_iter_{it+1}.pt"
        )

        torch.save({
            "model": model.state_dict(),
            "logZ": logZ.detach().cpu(),
        }, ckpt_path)

        print(f"  saved: {ckpt_path}")

    log_file.close()


if __name__ == "__main__":
    main()