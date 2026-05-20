#!/usr/bin/env python
# ============================================================
# PepMLM + MLM fine-tuning for peptide generation
# + PPLM affinity evaluation (black-box)
# ============================================================

import os
import csv
import json
import argparse
import random
from types import SimpleNamespace
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AdamW,
)

# ---- external evaluator (black-box) ----
from compute_PPLM_affinity import PPLMAffinityPredictor
from mmseq2_clustering import mmseqs_cluster_from_sequences

# ---- AF2BIND prior（オプション） ----
try:
    from af2bind_prior import AF2BindPrior, load_af2bind_matrix
    _AF2BIND_AVAILABLE = True
except ImportError:
    _AF2BIND_AVAILABLE = False

# =========================
# Config loader
# =========================
def load_config(path: str):
    with open(path, "r") as f:
        cfg_dict = json.load(f)

    cfg = SimpleNamespace(**cfg_dict)

    # device / dtype は実行環境依存なのでここで解決
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
        "iteration",
        "sample_idx",
        "peptide",
        "affinity",
        "kept",
        "gibbs_steps",
        "temperature",
        "epoch",
        "mlm_loss",
    ]
    exists = os.path.exists(path)
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=header)
    if not exists:
        writer.writeheader()
    return f, writer


# =========================
# PepMLM initialization
# =========================
def load_pepmlm(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.PEPM_LM_NAME)
    model = AutoModelForMaskedLM.from_pretrained(cfg.PEPM_LM_NAME)
    model = model.to(cfg.DEVICE).to(cfg.DTYPE)
    model.eval()
    return tokenizer, model


# =========================
# Conditional Gibbs sampling
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
    af2bind_prior: "AF2BindPrior | None" = None,
    lambda_af2bind: float = 0.0,
) -> str:
    mask_token_id = tokenizer.mask_token_id

    peptide_token_ids = [mask_token_id] * pep_len

    for _ in range(num_steps):
        positions = list(range(pep_len))
        random.shuffle(positions)

        for pos in positions:
            peptide_token_ids[pos] = mask_token_id

            full_seq = target_seq + tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(peptide_token_ids)
            )

            inputs = tokenizer(
                full_seq,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits[0]

            pep_start = inputs.input_ids.size(1) - pep_len - 1
            pos_idx = pep_start + pos

            token_logits = logits[pos_idx] / temperature

            # ---- AF2BIND prior の加算 ----
            if af2bind_prior is not None and lambda_af2bind > 0.0:
                prior_vocab = af2bind_prior.compute_prior_vocab(
                    peptide_token_ids, mask_token_id
                )   # (L_pep, vocab_size)
                pos_prior = prior_vocab[pos].to(token_logits.dtype)
                token_logits = token_logits + lambda_af2bind * pos_prior
            # --------------------------------

            for tid in [
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
                tokenizer.mask_token_id,
            ]:
                if tid is not None:
                    token_logits[tid] = -float("inf")

            probs = torch.softmax(token_logits, dim=-1)
            peptide_token_ids[pos] = torch.multinomial(probs, 1).item()

    tokens = tokenizer.convert_ids_to_tokens(peptide_token_ids)
    return tokenizer.convert_tokens_to_string(tokens).replace(" ", "")


# =========================
# PPLM evaluator
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
# Top-K filtering
# =========================
def filter_top_k(samples: List[Tuple[str, float]], top_k_percent: float) -> List[str]:
    samples = sorted(samples, key=lambda x: x[1])
    k = max(1, int(len(samples) * top_k_percent))
    return [s[0] for s in samples[:k]]


# =========================
# MLM dataset
# =========================
class MLMDataset(Dataset):
    def __init__(self, pairs, tokenizer, mask_prob):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target, peptide = self.pairs[idx]
        full_seq = target + peptide

        enc = self.tokenizer(
            full_seq,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = enc.input_ids.squeeze(0)
        labels = input_ids.clone()

        pep_start = input_ids.size(0) - len(peptide) - 1

        for i in range(len(peptide)):
            if random.random() < self.mask_prob:
                input_ids[pep_start + i] = self.tokenizer.mask_token_id
            else:
                labels[pep_start + i] = -100

        return input_ids, labels


# =========================
# MLM fine-tuning
# =========================
def finetune_mlm(model, dataset, optimizer, cfg, csv_writer, iteration):
    model.train()
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    for epoch in range(cfg.NUM_EPOCHS):
        total_loss = 0.0
        for input_ids, labels in loader:
            input_ids = input_ids.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        csv_writer.writerow({
            "iteration": iteration,
            "sample_idx": "",
            "peptide": "",
            "affinity": "",
            "kept": "",
            "gibbs_steps": "",
            "temperature": "",
            "epoch": epoch + 1,
            "mlm_loss": avg_loss,
        })
        print(f"    Epoch {epoch+1}: MLM loss = {avg_loss:.4f}")

    model.eval()


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    set_seed(cfg.RANDOM_SEED)

    log_path = os.path.join(cfg.SAVE_DIR, cfg.LOG_CSV)
    folder_name = os.path.basename(cfg.SAVE_DIR)
    log_file, csv_writer = init_csv_logger(log_path)

    tokenizer, model = load_pepmlm(cfg)
    model = model.to(cfg.DEVICE).half()

    evaluator = PPLMEvaluator(cfg.PPLM_SCRIPT)
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

    # =========================
    # AF2BIND prior の初期化
    # =========================
    af2bind_prior = None
    lambda_af2bind = float(getattr(cfg, "LAMBDA_AF2BIND", 0.0))
    af2bind_path = getattr(cfg, "AF2BIND_MATRIX_PATH", None)

    if af2bind_path and lambda_af2bind > 0.0:
        if not _AF2BIND_AVAILABLE:
            print("[AF2BIND] WARNING: af2bind_prior.py が見つかりません。prior を無効化します。")
        else:
            print(f"\n[AF2BIND] loading matrix from {af2bind_path}")
            af2bind_matrix = load_af2bind_matrix(af2bind_path)
            print(f"[AF2BIND] matrix shape: {af2bind_matrix.shape}")

            if af2bind_matrix.shape[0] != len(cfg.TARGET_SEQUENCE):
                print(
                    f"[AF2BIND] WARNING: matrix rows ({af2bind_matrix.shape[0]}) "
                    f"!= target length ({len(cfg.TARGET_SEQUENCE)})"
                )

            af2bind_prior = AF2BindPrior(
                af2bind_matrix=af2bind_matrix,
                tokenizer=tokenizer,
                device=cfg.DEVICE,
                dtype=torch.float32,
                pbind_bias_scale=float(getattr(cfg, "PBIND_BIAS_SCALE", 5.0)),
            )
            print(f"[AF2BIND] prior enabled  lambda={lambda_af2bind}  pbind_bias={af2bind_prior.pbind_bias_scale}")
    else:
        print("\n[AF2BIND] prior disabled (baseline PepMLM only)")

    for it in range(cfg.NUM_ITERATIONS):
        print(f"\n=== Iteration {it+1} ===")
        samples = []

        for i in range(cfg.NUM_SAMPLES_PER_ITER):
            peptide = gibbs_sample_peptide(
                model,
                tokenizer,
                cfg.TARGET_SEQUENCE,
                cfg.PEPTIDE_LEN,
                cfg.NUM_GIBBS_STEPS,
                cfg.TEMPERATURE,
                cfg.DEVICE,
                af2bind_prior=af2bind_prior,
                lambda_af2bind=lambda_af2bind,
            )

            score = evaluator.score(cfg.TARGET_SEQUENCE, peptide, step=i)
            samples.append((peptide, score))

        kept = set(filter_top_k(samples, cfg.TOP_K_PERCENT))

        for idx, (pep, score) in enumerate(samples):
            csv_writer.writerow({
                "iteration": it + 1,
                "sample_idx": idx,
                "peptide": pep,
                "affinity": score,
                "kept": int(pep in kept),
                "gibbs_steps": cfg.NUM_GIBBS_STEPS,
                "temperature": cfg.TEMPERATURE,
                "epoch": "",
                "mlm_loss": "",
            })

        dataset = MLMDataset(
            [(cfg.TARGET_SEQUENCE, p) for p in kept],
            tokenizer,
            cfg.MLM_MASK_PROB,
        )


        finetune_mlm(model, dataset, optimizer, cfg, csv_writer, it + 1)
        
        # チェックポイントを保存するフォルダを作成
        checkpoint_folder = os.path.join("/mnt/hdd/morishita/pepmlm_pplm/checkpoint",folder_name)
        os.makedirs(checkpoint_folder, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_folder, f"pepmlm_iter_{it+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  saved: {ckpt_path}")
        
        """ 
        # ---- 追加: 多様性チェック ----
        all_peptides = [p for p, _ in samples]

        cluster_count = mmseqs_cluster_from_sequences(
            all_peptides,
            min_seq_id=cfg.MMSEQS_MIN_SEQ_ID,
            coverage=cfg.MMSEQS_COVERAGE
        )

        print(f"  clusters: {cluster_count}")

        if cluster_count == 1:
            print("  Early stopping: converged to one cluster")
            break
            log_file.close()

        """
if __name__ == "__main__":
    main()
