#!/usr/bin/env python
# ============================================================
# PepMLM + MLM fine-tuning for peptide generation
# + PPLM affinity evaluation (black-box)
# + SMC-guided Gibbs sampling for cryptic pocket targeting
# ============================================================

import os
import csv
import json
import argparse
import random
from types import SimpleNamespace
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Cryptic scores loader
# =========================
def load_cryptic_scores(path: str, target_seq: str) -> List[float]:
    with open(path, "r") as f:
        data = json.load(f)

    data_sorted = sorted(data, key=lambda x: x["residue_id"])

    if len(data_sorted) != len(target_seq):
        raise ValueError(
            f"cryptic_scores の残基数 ({len(data_sorted)}) が "
            f"TARGET_SEQUENCE の長さ ({len(target_seq)}) と一致しません"
        )

    scores = []
    for entry in data_sorted:
        score = entry.get("displayed_score",
                entry.get("normalized_score",
                entry.get("raw_score", 0.0)))
        scores.append(float(score))

    return scores


# =========================
# Config loader
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
        "iteration", "sample_idx", "peptide", "affinity",
        "kept", "gibbs_steps", "temperature", "epoch", "mlm_loss",
    ]
    exists = os.path.exists(path)
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=header)
    if not exists:
        writer.writeheader()
    return f, writer


# =========================
# 標準AAの vocab ID セットを構築
# =========================
def build_valid_token_ids(tokenizer) -> torch.Tensor:
    """
    20種類の標準アミノ酸の vocab ID だけを集めた 1D LongTensor を返す。
    Gibbsサンプリング時に「これ以外は -inf」にするために使う。
    <null_1>, X, <unk> などの非標準トークンを全て除外できる。
    """
    STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
    ids = []
    for aa in STANDARD_AAS:
        tid = tokenizer.convert_tokens_to_ids(aa)
        if tid is not None and tid != tokenizer.unk_token_id:
            ids.append(tid)
    return torch.tensor(sorted(set(ids)), dtype=torch.long)


def mask_non_standard_tokens(
    token_logits: torch.Tensor,
    valid_ids: torch.Tensor,
) -> torch.Tensor:
    """
    token_logits (vocab_size,) に対して、valid_ids 以外を -inf にして返す。
    """
    masked = torch.full_like(token_logits, -float("inf"))
    masked[valid_ids] = token_logits[valid_ids]
    return masked


# =========================
# PepMLM initialization
# =========================
def load_pepmlm(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.PEPM_LM_NAME, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(cfg.PEPM_LM_NAME, local_files_only=True)
    model = model.to(cfg.DEVICE).to(cfg.DTYPE)
    model.eval()
    return tokenizer, model


# =========================
# Conditional Gibbs sampling (original)
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
    valid_token_ids: torch.Tensor,
    af2bind_prior: "AF2BindPrior | None" = None,
    lambda_af2bind: float = 0.0,
) -> str:
    mask_token_id = tokenizer.mask_token_id
    cls_token_id  = tokenizer.cls_token_id
    sep_token_id  = tokenizer.sep_token_id
    eos_token_id  = tokenizer.eos_token_id

    end_token_id = sep_token_id if sep_token_id is not None else eos_token_id

    target_ids: List[int] = tokenizer(
        target_seq, add_special_tokens=False,
    ).input_ids

    gibbs_prefix = torch.tensor(
        [cls_token_id] + target_ids,
        dtype=torch.long, device=device,
    )
    gibbs_pep_start = 1 + len(target_ids)

    peptide_token_ids = torch.full(
        (pep_len,), mask_token_id, dtype=torch.long, device=device
    )

    valid_ids = valid_token_ids.to(device)

    for _ in range(num_steps):
        positions = list(range(pep_len))
        random.shuffle(positions)

        for pos in positions:
            peptide_token_ids[pos] = mask_token_id

            input_ids = torch.cat([
                gibbs_prefix,
                peptide_token_ids,
                torch.tensor([end_token_id], dtype=torch.long, device=device),
            ]).unsqueeze(0)

            logits = model(input_ids=input_ids).logits[0]
            token_logits = logits[gibbs_pep_start + pos] / temperature

            # AF2BIND prior の加算
            if af2bind_prior is not None and lambda_af2bind > 0.0:
                prior_vocab = af2bind_prior.compute_prior_vocab(
                    peptide_token_ids.tolist(), mask_token_id
                )
                pos_prior = prior_vocab[pos].to(token_logits.dtype)
                token_logits = token_logits + lambda_af2bind * pos_prior

            # ★ 標準AA以外を全て -inf にする（<null_1>, X, <unk> 等を除外）
            token_logits = mask_non_standard_tokens(token_logits, valid_ids)

            probs = torch.softmax(token_logits, dim=-1)
            peptide_token_ids[pos] = torch.multinomial(probs, 1).item()

    tokens = tokenizer.convert_ids_to_tokens(peptide_token_ids.tolist())
    return tokenizer.convert_tokens_to_string(tokens).replace(" ", "")


# =========================
# SMC-guided Gibbs sampling (new)
# =========================
@torch.no_grad()
def gibbs_sample_peptide_smc(
    model,
    tokenizer,
    target_seq: str,
    pep_len: int,
    num_steps: int,
    temperature: float,
    device: str,
    cryptic_scores: List[float],
    valid_token_ids: torch.Tensor,
    cryptic_threshold: float = 0.5,
    num_particles: int = 8,
    smc_lambda: float = 1.0,
    af2bind_prior: "AF2BindPrior | None" = None,
    lambda_af2bind: float = 0.0,
) -> str:
    mask_token_id = tokenizer.mask_token_id
    cls_token_id  = tokenizer.cls_token_id
    sep_token_id  = tokenizer.sep_token_id
    eos_token_id  = tokenizer.eos_token_id

    end_token_id = sep_token_id if sep_token_id is not None else eos_token_id
    valid_ids = valid_token_ids.to(device)

    # ------------------------------------------------------------------
    # ステップ1: 重要残基の特定
    # ------------------------------------------------------------------
    important_indices = [
        i for i, s in enumerate(cryptic_scores)
        if s >= cryptic_threshold
    ]

    use_smc = (len(important_indices) > 0) and (smc_lambda > 0.0)
    if not use_smc:
        print("[SMC] WARNING: 重要残基が見つからないか smc_lambda=0。通常Gibbsにフォールバックします。")
        return gibbs_sample_peptide(
            model, tokenizer, target_seq, pep_len, num_steps,
            temperature, device, valid_token_ids, af2bind_prior, lambda_af2bind
        )

    effective_particles = num_particles if smc_lambda > 0.0 else 1

    # ------------------------------------------------------------------
    # ステップ2: 事前計算
    # ------------------------------------------------------------------
    target_token_ids: List[int] = tokenizer(
        target_seq, add_special_tokens=False
    ).input_ids

    correct_token_ids_for_mask: List[int] = [
        target_token_ids[idx]
        for idx in important_indices
        if idx < len(target_token_ids)
    ]
    correct_ids_tensor = torch.tensor(
        correct_token_ids_for_mask, dtype=torch.long, device=device
    )

    masked_target_token_ids: List[int] = list(target_token_ids)
    for idx in important_indices:
        if idx < len(masked_target_token_ids):
            masked_target_token_ids[idx] = mask_token_id

    target_offset_in_recovery = 1 + pep_len

    recovery_mask_abs_positions: List[int] = [
        target_offset_in_recovery + local_idx
        for local_idx, tid in enumerate(masked_target_token_ids)
        if tid == mask_token_id
    ]
    recovery_mask_pos_tensor = torch.tensor(
        recovery_mask_abs_positions, dtype=torch.long, device=device
    )

    recovery_fixed_suffix = torch.tensor(
        masked_target_token_ids + [end_token_id],
        dtype=torch.long, device=device
    )

    gibbs_pep_start = 1 + len(target_token_ids)

    gibbs_prefix = torch.tensor(
        [cls_token_id] + target_token_ids,
        dtype=torch.long, device=device
    )

    # ------------------------------------------------------------------
    # ベースラインrecovery log prob
    # ------------------------------------------------------------------
    def compute_baseline_recovery() -> float:
        all_mask_pep = torch.full(
            (pep_len,), mask_token_id, dtype=torch.long, device=device
        )
        row = torch.cat([
            torch.tensor([cls_token_id], dtype=torch.long, device=device),
            all_mask_pep,
            recovery_fixed_suffix,
        ])
        out = model(input_ids=row.unsqueeze(0)).logits[0]
        logits_at_mask = out[recovery_mask_pos_tensor]
        log_probs = F.log_softmax(logits_at_mask.float(), dim=-1)
        correct_log_probs = log_probs[
            torch.arange(len(correct_token_ids_for_mask), device=device),
            correct_ids_tensor
        ]
        return correct_log_probs.mean().item()

    # ------------------------------------------------------------------
    # Gibbsサンプリング本体
    # ------------------------------------------------------------------
    peptide_token_ids_tensor = torch.full(
        (pep_len,), mask_token_id, dtype=torch.long, device=device
    )

    for _ in range(num_steps):
        baseline_recovery = compute_baseline_recovery()

        positions = list(range(pep_len))
        random.shuffle(positions)

        for pos in positions:
            peptide_token_ids_tensor[pos] = mask_token_id

            # ----------------------------------------------------------
            # 3-1. 通常Gibbs forward
            # ----------------------------------------------------------
            gibbs_input = torch.cat([
                gibbs_prefix,
                peptide_token_ids_tensor,
                torch.tensor([end_token_id], dtype=torch.long, device=device),
            ]).unsqueeze(0)

            logits = model(input_ids=gibbs_input).logits[0]
            token_logits = logits[gibbs_pep_start + pos] / temperature

            # AF2BIND prior の加算
            if af2bind_prior is not None and lambda_af2bind > 0.0:
                prior_vocab = af2bind_prior.compute_prior_vocab(
                    peptide_token_ids_tensor.tolist(), mask_token_id
                )
                token_logits = token_logits + lambda_af2bind * prior_vocab[pos].to(token_logits.dtype)

            # ★ 標準AA以外を全て -inf にする
            token_logits = mask_non_standard_tokens(token_logits, valid_ids)

            log_probs_base = F.log_softmax(token_logits, dim=-1)
            candidates = torch.multinomial(
                torch.exp(log_probs_base), effective_particles, replacement=True
            )

            candidate_log_probs = log_probs_base[candidates]

            # ----------------------------------------------------------
            # 3-2. 差分SMCスコア計算
            # ----------------------------------------------------------
            if smc_lambda > 0.0 and len(recovery_mask_abs_positions) > 0:
                batch_pep = peptide_token_ids_tensor.unsqueeze(0).expand(
                    effective_particles, -1
                ).clone()
                batch_pep[:, pos] = candidates

                cls_col = torch.full(
                    (effective_particles, 1), cls_token_id,
                    dtype=torch.long, device=device
                )
                suffix_expanded = recovery_fixed_suffix.unsqueeze(0).expand(
                    effective_particles, -1
                )

                batch_input = torch.cat([cls_col, batch_pep, suffix_expanded], dim=1)
                batch_logits = model(input_ids=batch_input).logits

                logits_at_mask = batch_logits[:, recovery_mask_pos_tensor, :]
                log_probs_at_mask = F.log_softmax(logits_at_mask.float(), dim=-1)

                correct_ids_expanded = correct_ids_tensor.unsqueeze(0).unsqueeze(-1).expand(
                    effective_particles, -1, 1
                )
                correct_log_probs = log_probs_at_mask.gather(2, correct_ids_expanded).squeeze(-1)

                recovery_scores = correct_log_probs.mean(dim=1)
                delta_scores = recovery_scores - baseline_recovery

            else:
                delta_scores = torch.zeros(effective_particles, device=device)

            # ----------------------------------------------------------
            # 3-3. 重み付きリサンプリング
            # ----------------------------------------------------------
            log_weights = candidate_log_probs + smc_lambda * delta_scores
            weights = torch.softmax(log_weights, dim=-1)
            selected_idx = torch.multinomial(weights, 1).item()
            peptide_token_ids_tensor[pos] = candidates[selected_idx]

    pep_ids = peptide_token_ids_tensor.tolist()
    tokens = tokenizer.convert_ids_to_tokens(pep_ids)
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
        attention_mask = enc.attention_mask.squeeze(0)
        labels = input_ids.clone()

        pep_start = input_ids.size(0) - len(peptide) - 1

        labels[:pep_start] = -100
        labels[pep_start + len(peptide):] = -100

        for i in range(len(peptide)):
            if random.random() < self.mask_prob:
                input_ids[pep_start + i] = self.tokenizer.mask_token_id
            else:
                labels[pep_start + i] = -100

        return input_ids, attention_mask, labels


# =========================
# Collate function
# =========================
def make_mlm_collate_fn(pad_token_id: int):
    def collate(batch):
        input_ids_list, attn_list, labels_list = zip(*batch)
        max_len = max(t.size(0) for t in input_ids_list)

        def pad_to(t, value):
            if t.size(0) >= max_len:
                return t
            pad = torch.full((max_len - t.size(0),), value, dtype=t.dtype)
            return torch.cat([t, pad])

        input_ids = torch.stack([pad_to(t, pad_token_id) for t in input_ids_list])
        attention_mask = torch.stack([pad_to(t, 0) for t in attn_list])
        labels = torch.stack([pad_to(t, -100) for t in labels_list])
        return input_ids, attention_mask, labels

    return collate


# =========================
# MLM fine-tuning
# =========================
def finetune_mlm(model, dataset, optimizer, cfg, csv_writer, iteration, tokenizer):
    model.train()
    pad_id = tokenizer.pad_token_id or 0
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=make_mlm_collate_fn(pad_id),
    )

    for epoch in range(cfg.NUM_EPOCHS):
        total_loss = 0.0
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(cfg.DEVICE)
            attention_mask = attention_mask.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
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

    # ★ 標準AAの vocab ID セットを事前に構築（ループ外で1回だけ）
    valid_token_ids = build_valid_token_ids(tokenizer)
    print(f"[Tokenizer] 標準AA vocab IDs: {len(valid_token_ids)}種 確認済み")

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
            print(
                f"[AF2BIND] prior enabled  lambda={lambda_af2bind} "
                f" pbind_bias={af2bind_prior.pbind_bias_scale}"
            )
    else:
        print("\n[AF2BIND] prior disabled (baseline PepMLM only)")

    # =========================
    # SMC / Cryptic scores の設定
    # =========================
    use_smc_flag = getattr(cfg, "USE_SMC", None)
    cryptic_scores_path = getattr(cfg, "CRYPTIC_SCORES_PATH", None)
    cryptic_threshold = float(getattr(cfg, "CRYPTIC_THRESHOLD", 0.5))
    smc_num_particles = int(getattr(cfg, "SMC_NUM_PARTICLES", 8))
    smc_lambda = float(getattr(cfg, "SMC_LAMBDA", 1.0))

    cryptic_scores: Optional[List[float]] = None
    use_smc = False

    if use_smc_flag is False:
        print("\n[SMC] USE_SMC=false が指定されています。通常Gibbsを使用します。")
    elif use_smc_flag is True:
        if cryptic_scores_path and os.path.exists(cryptic_scores_path):
            print(f"\n[SMC] USE_SMC=true: cryptic scores を読み込み中: {cryptic_scores_path}")
            cryptic_scores = load_cryptic_scores(cryptic_scores_path, cfg.TARGET_SEQUENCE)
            n_important = sum(1 for s in cryptic_scores if s >= cryptic_threshold)
            print(f"[SMC] 重要残基数: {n_important} / {len(cryptic_scores)} (threshold={cryptic_threshold})")
            print(f"[SMC] smc_lambda={smc_lambda}, num_particles={smc_num_particles}")
            use_smc = True
        else:
            raise ValueError(
                "[SMC] USE_SMC=true ですが CRYPTIC_SCORES_PATH が未設定または存在しません。"
                f" path={cryptic_scores_path}"
            )
    else:
        if cryptic_scores_path and os.path.exists(cryptic_scores_path):
            print(f"\n[SMC] cryptic scores を読み込み中: {cryptic_scores_path}")
            cryptic_scores = load_cryptic_scores(cryptic_scores_path, cfg.TARGET_SEQUENCE)
            n_important = sum(1 for s in cryptic_scores if s >= cryptic_threshold)
            print(f"[SMC] 重要残基数: {n_important} / {len(cryptic_scores)} (threshold={cryptic_threshold})")
            print(f"[SMC] smc_lambda={smc_lambda}, num_particles={smc_num_particles}")
            use_smc = True
        else:
            if cryptic_scores_path:
                print(
                    f"\n[SMC] WARNING: CRYPTIC_SCORES_PATH '{cryptic_scores_path}' が見つかりません。"
                    " 通常Gibbsを使用します。"
                )
            else:
                print("\n[SMC] CRYPTIC_SCORES_PATH が未設定。通常Gibbsを使用します。")

    # =========================
    # ペプチド長の設定
    # =========================
    pep_len_min = getattr(cfg, "PEPTIDE_LEN_MIN", None)
    pep_len_max = getattr(cfg, "PEPTIDE_LEN_MAX", None)
    pep_len_fixed = getattr(cfg, "PEPTIDE_LEN", None)

    if pep_len_min is not None and pep_len_max is not None:
        pep_len_min = int(pep_len_min)
        pep_len_max = int(pep_len_max)
        if pep_len_min > pep_len_max:
            raise ValueError(
                f"PEPTIDE_LEN_MIN ({pep_len_min}) > PEPTIDE_LEN_MAX ({pep_len_max})"
            )
        print(f"\n[PepLen] 可変長モード: {pep_len_min}〜{pep_len_max}")
        variable_length = True
    elif pep_len_fixed is not None:
        print(f"\n[PepLen] 固定長モード: {pep_len_fixed}")
        variable_length = False
    else:
        raise ValueError(
            "config に PEPTIDE_LEN または PEPTIDE_LEN_MIN/PEPTIDE_LEN_MAX を設定してください"
        )

    # =========================
    # サンプリング関数の選択
    # =========================
    def sample_peptide() -> str:
        current_pep_len = (
            random.randint(pep_len_min, pep_len_max) if variable_length else pep_len_fixed
        )

        if use_smc and cryptic_scores is not None:
            return gibbs_sample_peptide_smc(
                model=model,
                tokenizer=tokenizer,
                target_seq=cfg.TARGET_SEQUENCE,
                pep_len=current_pep_len,
                num_steps=cfg.NUM_GIBBS_STEPS,
                temperature=cfg.TEMPERATURE,
                device=cfg.DEVICE,
                cryptic_scores=cryptic_scores,
                valid_token_ids=valid_token_ids,
                cryptic_threshold=cryptic_threshold,
                num_particles=smc_num_particles,
                smc_lambda=smc_lambda,
                af2bind_prior=af2bind_prior,
                lambda_af2bind=lambda_af2bind,
            )
        else:
            return gibbs_sample_peptide(
                model=model,
                tokenizer=tokenizer,
                target_seq=cfg.TARGET_SEQUENCE,
                pep_len=current_pep_len,
                num_steps=cfg.NUM_GIBBS_STEPS,
                temperature=cfg.TEMPERATURE,
                device=cfg.DEVICE,
                valid_token_ids=valid_token_ids,
                af2bind_prior=af2bind_prior,
                lambda_af2bind=lambda_af2bind,
            )

    for it in range(cfg.NUM_ITERATIONS):
        print(f"\n=== Iteration {it+1} ===")
        samples = []

        for i in range(cfg.NUM_SAMPLES_PER_ITER):
            peptide = sample_peptide()
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

        finetune_mlm(model, dataset, optimizer, cfg, csv_writer, it + 1, tokenizer)

        checkpoint_folder = os.path.join(
            "/home/users/gds/pepmlm_pplm/checkpoint", folder_name
        )
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
