#!/usr/bin/env python
# ============================================================
# Inference script using saved PepMLM checkpoint (.pt)
# - 指定したiterationのモデルをロード
# - ペプチド生成（Gibbs sampling、通常 or SMC誘導）
# - 結果をCSVに保存
# - 固定長 / 可変長 (PEPTIDE_LEN_MIN〜PEPTIDE_LEN_MAX) どちらにも対応
#
# Gibbs 入力構造 (学習スクリプトと統一):
#   [CLS] target_ids pep_ids [EOS]
#   ※ target と pep の間に EOS は入れない
#
# 推論時 SMC ガイダンス:
#   config の INFER_USE_SMC か CLI の --use_smc を有効化すると、学習時と同じく
#   cryptic 残基を復元できるペプチドを優先的に生成する。
#   通常は学習済みモデルの重みに cryptic バイアスが焼き込まれているため
#   不要だが、追加で推論時にもガイダンスを入れたい場合に使う。
#
# config キー (推論用):
#   INFER_USE_SMC          : bool. 推論時に SMC を使うか
#   INFER_SMC_NUM_PARTICLES: int.  推論時の SMC パーティクル数 (default: 8)
#   INFER_SMC_LAMBDA       : float.推論時の SMC 重み (default: 1.0)
#
# 共有キー (学習・推論で同じ値を使う):
#   CRYPTIC_SCORES_PATH    : str.  cryptic スコア JSON のパス
#   CRYPTIC_THRESHOLD      : float.重要残基判定の閾値 (default: 0.5)
#
# 旧キー (deprecated, fallback あり):
#   USE_SMC / SMC_NUM_PARTICLES / SMC_LAMBDA は警告を出しつつ使用可能
# ============================================================

import os
import json
import argparse
import random
import csv
from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ---- external evaluator (optional) ----
from compute_PPLM_affinity import PPLMAffinityPredictor

# ---- AF2BIND prior (optional) ----
# 学習側で AF2BIND prior を使っている場合、推論側でも同じ prior を加えると
# 学習時と一貫した分布からサンプルできる。なくても動作する。
try:
    from af2bind_prior import AF2BindPrior, load_af2bind_matrix
    _AF2BIND_AVAILABLE = True
except ImportError:
    _AF2BIND_AVAILABLE = False


# =========================
# Cryptic scores loader (学習スクリプトと同じ実装)
# =========================
def load_cryptic_scores(path: str, target_seq: str) -> List[float]:
    """cryptic スコア JSON を読み込み、target_seq の各残基に対応する
    スコアリストを返す (residue_id でソート)。"""
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
# 標準AAの vocab ID セットを構築 (学習スクリプトと同じ実装)
# =========================
def build_valid_token_ids(tokenizer) -> torch.Tensor:
    """
    20種類の標準アミノ酸の vocab ID だけを集めた 1D LongTensor を返す。
    Gibbsサンプリング時に「これ以外は -inf」にするために使う。
    <null_1>, X, <unk>, B, U, Z, O などの非標準トークンを全て除外できる。
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
# Gibbs sampling (通常版)
# 入力構造: [CLS] target_ids pep_ids [EOS]
# ※ 学習スクリプト (pepmlm_pplm_cryptic.py) と完全に同じ構造に統一
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

    # 固定prefix: [CLS] target_ids
    # ★ 学習側と統一: target と pep の間に EOS は入れない
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

            # 入力: [CLS] target_ids pep_ids [EOS]
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

            # 標準AA以外を全て -inf にする
            token_logits = mask_non_standard_tokens(token_logits, valid_ids)

            probs = torch.softmax(token_logits, dim=-1)
            peptide_token_ids[pos] = torch.multinomial(probs, 1).item()

    tokens = tokenizer.convert_ids_to_tokens(peptide_token_ids.tolist())
    return tokenizer.convert_tokens_to_string(tokens).replace(" ", "")


# =========================
# SMC-guided Gibbs sampling (推論用)
# 入力構造: 学習スクリプトと完全に同じ
#   - Gibbs forward:    [CLS] target_ids pep_ids [EOS]
#   - Recovery forward: [CLS] pep_ids masked_target_ids [EOS]
# ※ 学習スクリプト (pepmlm_pplm_cryptic.py) の gibbs_sample_peptide_smc()
#   をそのまま流用 (no_grad で純粋に推論用途)
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

    # ステップ1: 重要残基の特定
    important_indices = [
        i for i, s in enumerate(cryptic_scores)
        if s >= cryptic_threshold
    ]

    use_smc = (len(important_indices) > 0) and (smc_lambda > 0.0)
    if not use_smc:
        print("[SMC] WARNING: 重要残基が見つからないか smc_lambda=0。通常Gibbsにフォールバックします。")
        return gibbs_sample_peptide(
            model, tokenizer, target_seq, pep_len, num_steps,
            temperature, device, valid_token_ids,
            af2bind_prior=af2bind_prior, lambda_af2bind=lambda_af2bind,
        )

    effective_particles = num_particles if smc_lambda > 0.0 else 1

    # ステップ2: 事前計算
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

    # 学習側と統一: target と pep の間に EOS は入れない
    gibbs_prefix = torch.tensor(
        [cls_token_id] + target_token_ids,
        dtype=torch.long, device=device
    )

    # ベースラインrecovery log prob
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

    # Gibbsサンプリング本体
    peptide_token_ids_tensor = torch.full(
        (pep_len,), mask_token_id, dtype=torch.long, device=device
    )

    for _ in range(num_steps):
        baseline_recovery = compute_baseline_recovery()

        positions = list(range(pep_len))
        random.shuffle(positions)

        for pos in positions:
            peptide_token_ids_tensor[pos] = mask_token_id

            # 3-1. 通常Gibbs forward
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

            # 標準AA以外を全て -inf にする
            token_logits = mask_non_standard_tokens(token_logits, valid_ids)

            log_probs_base = F.log_softmax(token_logits, dim=-1)
            candidates = torch.multinomial(
                torch.exp(log_probs_base), effective_particles, replacement=True
            )

            candidate_log_probs = log_probs_base[candidates]

            # 3-2. 差分SMCスコア計算
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

            # 3-3. 重み付きリサンプリング
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

    # ★ 新規: SMC 推論の有効化 (CLI で config を上書き)
    parser.add_argument("--use_smc", action="store_true",
                        help="推論時に SMC ガイダンスを使う (config の INFER_USE_SMC を上書き有効化)")
    parser.add_argument("--no_smc", action="store_true",
                        help="推論時に SMC ガイダンスを強制的に無効化")

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

    # ★ 標準AAの vocab ID セットを構築（学習スクリプトと同じ実装）
    valid_token_ids = build_valid_token_ids(tokenizer)
    print(f"[Tokenizer] 標準AA vocab IDs: {len(valid_token_ids)}種 確認済み")

    # =========================
    # AF2BIND prior の初期化 (任意)
    # 学習時と同じ条件で推論したい場合に使う
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
        print("[AF2BIND] prior disabled (推論時)")

    # =========================
    # SMC / Cryptic scores の設定 (推論用)
    # 優先順位:
    #   CLI (--no_smc > --use_smc)
    #   > config INFER_USE_SMC
    #   > config の旧 USE_SMC (deprecation warning)
    #   > デフォルト (None = 通常Gibbs)
    # =========================

    # フォールバック付き取得ヘルパ (旧キー使用時は warning)
    def _get_infer_smc_key(cfg, new_key, old_key, default=None, cast=None):
        if hasattr(cfg, new_key):
            v = getattr(cfg, new_key)
            return cast(v) if cast and v is not None else v
        if hasattr(cfg, old_key):
            v = getattr(cfg, old_key)
            print(f"[SMC] WARNING: config key '{old_key}' は非推奨です。"
                  f" '{new_key}' に置き換えてください (推論用)。")
            return cast(v) if cast and v is not None else v
        return default

    # --- USE_SMC フラグの決定 ---
    if args.no_smc:
        use_smc_flag = False
        print("[SMC] CLI --no_smc により SMC を強制無効化")
    elif args.use_smc:
        use_smc_flag = True
        print("[SMC] CLI --use_smc により SMC を有効化")
    else:
        use_smc_flag = _get_infer_smc_key(cfg, "INFER_USE_SMC", "USE_SMC", default=None)
        if use_smc_flag is True:
            print("[SMC] config の INFER_USE_SMC=true により SMC を有効化")
        elif use_smc_flag is False:
            print("[SMC] config の INFER_USE_SMC=false により SMC を無効化")
        else:
            print("[SMC] config / CLI 共に INFER_USE_SMC 未指定 → 通常Gibbs (デフォルト)")

    # --- SMC ハイパラの取得 (INFER_* 優先、旧 SMC_* を fallback) ---
    smc_num_particles = _get_infer_smc_key(
        cfg, "INFER_SMC_NUM_PARTICLES", "SMC_NUM_PARTICLES",
        default=8, cast=int,
    )
    smc_lambda = _get_infer_smc_key(
        cfg, "INFER_SMC_LAMBDA", "SMC_LAMBDA",
        default=1.0, cast=float,
    )

    # --- cryptic 共通設定 (学習・推論共有) ---
    cryptic_scores_path = getattr(cfg, "CRYPTIC_SCORES_PATH", None)
    cryptic_threshold = float(getattr(cfg, "CRYPTIC_THRESHOLD", 0.5))

    cryptic_scores: Optional[List[float]] = None
    use_smc = False

    if use_smc_flag is True:
        if cryptic_scores_path and os.path.exists(cryptic_scores_path):
            print(f"[SMC] cryptic scores を読み込み中: {cryptic_scores_path}")
            cryptic_scores = load_cryptic_scores(cryptic_scores_path, cfg.TARGET_SEQUENCE)
            n_important = sum(1 for s in cryptic_scores if s >= cryptic_threshold)
            print(f"[SMC] 重要残基数: {n_important} / {len(cryptic_scores)} (threshold={cryptic_threshold})")
            print(f"[SMC] smc_lambda={smc_lambda}, num_particles={smc_num_particles}")
            use_smc = True
        else:
            raise ValueError(
                f"[SMC] INFER_USE_SMC が有効ですが CRYPTIC_SCORES_PATH が未設定または存在しません。"
                f" path={cryptic_scores_path}"
            )

    evaluator = None
    if args.use_pplm:
        evaluator = PPLMEvaluator(cfg.PPLM_SCRIPT)

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
            ), current_pep_len
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
            ), current_pep_len

    # =========================
    # Inference & collect results
    # =========================
    results = []

    for i in range(args.num_samples):
        peptide, current_pep_len = sample_peptide()

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
