#!/usr/bin/env python
# ============================================================
# AF2BIND prior for PepMLM Gibbs sampling
# ============================================================
#
# AF2BIND から得られる (L_target, 20) の行列を使って、
# PepMLM の logits に加算する prior を計算する。
#
# 現在の実装: Strategy C-2 の簡易版
#   - ペプチドの現在状態を one-hot で表現
#   - AF2BIND 行列との内積で attention を計算
#   - P(bind) で attention にバイアスをかける
#   - attention で AF2BIND 行列を加重平均 → 各ペプチド位置の prior
#
# 将来的な拡張:
#   - one-hot を PepMLM hidden state に置き換え
#   - 学習可能な projection を追加
# ============================================================

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# AF2BIND 出力の列順（アルファベット順、20アミノ酸）
# ★実際のAF2BINDコードに合わせて要確認★
AF2BIND_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


class AF2BindPrior:
    """
    AF2BIND の (L_target, 20) 行列を保持し、
    ペプチドの現在状態から position-dependent prior を計算する。
    """

    def __init__(
        self,
        af2bind_matrix: np.ndarray | torch.Tensor,
        tokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        pbind_bias_scale: float = 5.0,
        eps: float = 1e-8,
    ):
        """
        Parameters
        ----------
        af2bind_matrix : (L_target, 20) の行列
            AF2BIND の出力。各行がターゲット残基、各列が20アミノ酸への activation。
        tokenizer : PepMLM の tokenizer
        device : 計算デバイス
        dtype : 計算精度
        pbind_bias_scale : P(bind) を attention logits に加算するときのスケール
                           大きいほどポケット残基に強く注目
        eps : log(0) 回避用
        """
        # tensor化
        if isinstance(af2bind_matrix, np.ndarray):
            af2bind_matrix = torch.from_numpy(af2bind_matrix)
        
        # ★ p_bind_aa は sigmoid 前の logit 値で保存されているので、
        #   ここで sigmoid を適用して [0, 1] の確率値に変換する
        af2bind_matrix = torch.sigmoid(af2bind_matrix.float())
        
        self.af2bind = af2bind_matrix.to(device=device, dtype=dtype)   # (L_t, 20)

        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.pbind_bias_scale = pbind_bias_scale
        self.eps = eps

        # P(bind): 各残基の結合サイト度合い
        # ここでは「20アミノ酸の中で最も好むものの activation」を使う
        # （AF2BIND論文の P(bind) とは厳密には異なるが、代理として機能）
        self.p_bind = self.af2bind.max(dim=1).values                   # (L_t,)

        # vocab → AF2BIND 20軸のマッピングを作成
        self.vocab_size = tokenizer.vocab_size
        self._build_aa_to_vocab_mapping()

    def _build_aa_to_vocab_mapping(self):
        """
        AF2BIND の20アミノ酸順序 → PepMLM vocab インデックスへの対応表を作る。
        また逆方向（vocab → 20軸）のマスクも作る。
        """
        # aa_idx (0-19) → vocab_id
        self.aa20_to_vocab = []
        for aa in AF2BIND_AA_ORDER:
            vid = self.tokenizer.convert_tokens_to_ids(aa)
            if vid is None or vid == self.tokenizer.unk_token_id:
                raise ValueError(f"amino acid {aa} not found in tokenizer vocab")
            self.aa20_to_vocab.append(vid)
        self.aa20_to_vocab = torch.tensor(self.aa20_to_vocab, device=self.device)

        # vocab_id → aa_idx (0-19) または -1（対応なし）
        vocab_to_aa20 = torch.full((self.vocab_size,), -1, dtype=torch.long,
                                    device=self.device)
        for aa_idx, vid in enumerate(self.aa20_to_vocab.tolist()):
            vocab_to_aa20[vid] = aa_idx
        self.vocab_to_aa20 = vocab_to_aa20

    def peptide_to_onehot20(self, peptide_token_ids: list[int],
                             mask_token_id: int) -> torch.Tensor:
        """
        ペプチドの token_ids を (L_pep, 20) の one-hot に変換。
        MASK や未知トークンは uniform (1/20) として扱う。

        Returns
        -------
        onehot : (L_pep, 20)
        """
        L_pep = len(peptide_token_ids)
        onehot = torch.zeros(L_pep, 20, device=self.device, dtype=self.dtype)

        for i, tid in enumerate(peptide_token_ids):
            if tid == mask_token_id:
                # MASK の場合は「何でもあり」を表現（uniform）
                onehot[i] = 1.0 / 20
            else:
                aa_idx = self.vocab_to_aa20[tid].item()
                if aa_idx >= 0:
                    onehot[i, aa_idx] = 1.0
                else:
                    # 特殊トークン等 → uniform
                    onehot[i] = 1.0 / 20
        return onehot

    @torch.no_grad()
    def compute_prior(
        self,
        peptide_token_ids: list[int],
        mask_token_id: int,
    ) -> torch.Tensor:
        """
        現在のペプチド状態から、各位置の 20次元 prior を計算する。

        【重要】 各位置 pos の prior を計算するとき、その位置自身を除外した
        「周囲のペプチド状態」を query にする。こうすることで、位置ごとに
        異なる attention が生まれる。

        例: ペプチド [K, _, _, _, _]
            位置1 の prior 計算 → query = 位置0,2,3,4 の状態の集約
                                ≈ [K の one-hot + 3個の uniform] を正規化したもの
            位置0 の prior 計算 → query = 位置1,2,3,4 の状態 (全部 MASK)
                                ≈ uniform
        これにより、「埋まっている残基の隣」と「全MASKに囲まれた位置」で
        異なる prior が出る。

        Returns
        -------
        prior_20 : (L_pep, 20)
            各ペプチド位置について、20アミノ酸それぞれの prior log-probability。
        """
        # ペプチドの現在状態 (L_pep, 20)
        pep_onehot = self.peptide_to_onehot20(peptide_token_ids, mask_token_id)
        L_pep = pep_onehot.size(0)

        # ---- 各位置で「自分を除外した周囲の平均」を計算 ----
        # 全体の和: (20,)
        total = pep_onehot.sum(dim=0)                                  # (20,)
        # 各位置 pos の query = (total - pep_onehot[pos]) / (L_pep - 1)
        #   → pos 以外の平均状態
        if L_pep > 1:
            query = (total.unsqueeze(0) - pep_onehot) / (L_pep - 1)     # (L_pep, 20)
        else:
            # L_pep == 1 の場合は自分自身を使う（除外できないため）
            query = pep_onehot                                          # (1, 20)

        # 類似度: (L_pep, L_target)
        #   query[i] @ target[j] = 「位置 i の周囲のアミノ酸構成を、残基 j がどれだけ好むか」
        similarity = query @ self.af2bind.T                             # (L_pep, L_t)

        # P(bind) バイアス: ポケット残基に注目を集める
        pbind_bias = self.pbind_bias_scale * self.p_bind                # (L_t,)
        attn_logits = similarity + pbind_bias.unsqueeze(0)              # (L_pep, L_t)

        # attention
        attn = F.softmax(attn_logits, dim=1)                            # (L_pep, L_t)

        # prior: (L_pep, 20)
        #   各ペプチド位置が注目している残基群の 20アミノ酸プロファイルを加重平均
        prior_raw = attn @ self.af2bind                                 # (L_pep, 20)

        # log-probability に変換
        # (prior_raw は非負、正規化されていない → 正規化してから log)
        prior_norm = prior_raw / (prior_raw.sum(dim=1, keepdim=True) + self.eps)
        prior_log = torch.log(prior_norm + self.eps)                    # (L_pep, 20)

        return prior_log

    def prior_to_vocab_logits(self, prior_log_20: torch.Tensor) -> torch.Tensor:
        """
        (L_pep, 20) の prior を (L_pep, vocab_size) に展開する。
        20アミノ酸以外の vocab 位置は -inf にして、logits に加算しても影響しないようにする。

        Parameters
        ----------
        prior_log_20 : (L_pep, 20)

        Returns
        -------
        prior_log_vocab : (L_pep, vocab_size)
        """
        L_pep = prior_log_20.size(0)
        prior_vocab = torch.full(
            (L_pep, self.vocab_size),
            -float("inf"),
            device=self.device,
            dtype=prior_log_20.dtype,
        )
        # 20アミノ酸の位置だけ埋める
        prior_vocab[:, self.aa20_to_vocab] = prior_log_20
        return prior_vocab

    @torch.no_grad()
    def compute_prior_vocab(
        self,
        peptide_token_ids: list[int],
        mask_token_id: int,
    ) -> torch.Tensor:
        """
        compute_prior + prior_to_vocab_logits をまとめて実行。

        Returns
        -------
        prior_vocab : (L_pep, vocab_size)
        """
        prior_20 = self.compute_prior(peptide_token_ids, mask_token_id)
        return self.prior_to_vocab_logits(prior_20)


# =========================
# ロード用ユーティリティ
# =========================
def load_af2bind_matrix(path: str | Path) -> np.ndarray:
    """
    AF2BIND 行列を .npy または .txt からロードする。

    想定フォーマット:
      - .npy: shape (L_target, 20) の numpy array
      - .txt / .csv: 各行が "aa1 aa2 ... aa20" のスペース or カンマ区切り
    """
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix in (".txt", ".tsv"):
        arr = np.loadtxt(path)
    elif path.suffix == ".csv":
        arr = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"unsupported format: {path.suffix}")

    if arr.ndim != 2 or arr.shape[1] != 20:
        raise ValueError(
            f"expected shape (L_target, 20), got {arr.shape}"
        )
    return arr.astype(np.float32)
