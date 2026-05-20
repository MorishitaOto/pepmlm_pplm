#!/usr/bin/env python
# ============================================================
# af2bind_prior.py の単体動作確認
# ============================================================
#
# AF2BIND行列をロードしなくても、ダミー行列で動作確認できるようにしてある。
# 実際のAF2BIND行列ができたら AF2BIND_PATH を指定して実行。
# ============================================================

import numpy as np
import torch
from transformers import AutoTokenizer

from af2bind_prior import AF2BindPrior, AF2BIND_AA_ORDER, load_af2bind_matrix


def test_with_dummy():
    """ダミーのAF2BIND行列で動作確認"""

    # ===== 設定 =====
    PEPM_LM_NAME = "ChatterjeeLab/PepMLM-650M"   # 要変更: 実際のモデル名
    L_target = 100
    L_pep = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== ダミーのAF2BIND行列を作る =====
    np.random.seed(42)
    # ほとんどの残基は低い値、一部の残基（ポケット）だけ高い値
    af2bind = np.random.rand(L_target, 20) * 0.05   # ベースラインは低い

    # 「疎水性ポケット」残基42,43,44: F,Y,W,L を高くする
    hydrophobic_aa = ['F', 'Y', 'W', 'L']
    for pocket_res in [42, 43, 44]:
        for aa in hydrophobic_aa:
            idx = AF2BIND_AA_ORDER.index(aa)
            af2bind[pocket_res, idx] = 0.3 + np.random.rand() * 0.2

    # 「荷電ポケット」残基78,79,80: K,R,H を高くする
    charged_aa = ['K', 'R', 'H']
    for pocket_res in [78, 79, 80]:
        for aa in charged_aa:
            idx = AF2BIND_AA_ORDER.index(aa)
            af2bind[pocket_res, idx] = 0.3 + np.random.rand() * 0.2

    print(f"dummy af2bind matrix shape: {af2bind.shape}")
    print(f"疎水性ポケット残基42のプロファイル:")
    for aa in AF2BIND_AA_ORDER:
        idx = AF2BIND_AA_ORDER.index(aa)
        print(f"  {aa}: {af2bind[42, idx]:.3f}")

    # ===== tokenizer & prior =====
    try:
        tokenizer = AutoTokenizer.from_pretrained(PEPM_LM_NAME)
    except Exception as e:
        print(f"tokenizer のロード失敗: {e}")
        print("→ PEPM_LM_NAME を正しいモデルに変更してください")
        return

    prior = AF2BindPrior(
        af2bind_matrix=af2bind,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.float32,
        pbind_bias_scale=5.0,
    )
    mask_id = tokenizer.mask_token_id

    # ===== テスト1: 全部 MASK の場合 =====
    print("\n=== Test 1: all MASK ===")
    peptide_ids = [mask_id] * L_pep
    prior_20 = prior.compute_prior(peptide_ids, mask_id)
    print(f"prior_20 shape: {prior_20.shape}")   # (L_pep, 20)
    print(f"位置0の prior (log-prob):")
    for i, aa in enumerate(AF2BIND_AA_ORDER):
        print(f"  {aa}: {prior_20[0, i].item():.3f}")
    # 全位置で同じ値になるはず（全部MASKなので区別なし）

    # ===== テスト2: 位置0にKを入れた場合 =====
    print("\n=== Test 2: position 0 = K (荷電) ===")
    peptide_ids = [mask_id] * L_pep
    peptide_ids[0] = tokenizer.convert_tokens_to_ids('K')
    prior_20 = prior.compute_prior(peptide_ids, mask_id)

    print("位置1の prior (Kの隣):")
    # 位置1は周囲のポケット状況を反映するはず
    for aa in ['K', 'R', 'H', 'F', 'Y', 'W']:
        idx = AF2BIND_AA_ORDER.index(aa)
        print(f"  {aa}: {prior_20[1, idx].item():.3f}")
    # K,R,Hが高めに出てほしい（荷電ポケットを向くため）

    # ===== テスト3: vocab_logits への変換 =====
    print("\n=== Test 3: vocab logits conversion ===")
    prior_vocab = prior.compute_prior_vocab(peptide_ids, mask_id)
    print(f"prior_vocab shape: {prior_vocab.shape}")   # (L_pep, vocab_size)

    # 20AAの位置は有限値、それ以外は -inf になっているはず
    K_vocab_id = tokenizer.convert_tokens_to_ids('K')
    print(f"K の vocab index {K_vocab_id} の値: {prior_vocab[1, K_vocab_id].item():.3f}")
    print(f"MASK token の値 (should be -inf): {prior_vocab[1, mask_id].item()}")

    print("\n✓ 全テスト完了")


def test_with_real():
    """実際のAF2BIND出力ファイルで動作確認"""
    AF2BIND_PATH = "./af2bind_matrix.npy"    # 要変更
    PEPM_LM_NAME = "ChatterjeeLab/PepMLM-650M"

    try:
        af2bind = load_af2bind_matrix(AF2BIND_PATH)
    except FileNotFoundError:
        print(f"ファイルが見つからない: {AF2BIND_PATH}")
        print("→ AF2BIND行列ができたら再実行してください")
        return

    tokenizer = AutoTokenizer.from_pretrained(PEPM_LM_NAME)
    prior = AF2BindPrior(
        af2bind_matrix=af2bind,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # P(bind) の分布を確認
    print(f"P(bind) stats:")
    print(f"  mean: {prior.p_bind.mean().item():.3f}")
    print(f"  max:  {prior.p_bind.max().item():.3f}")
    print(f"  top10 residues: {prior.p_bind.argsort(descending=True)[:10].tolist()}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running dummy test...")
    print("=" * 60)
    test_with_dummy()

    # print("\n" + "=" * 60)
    # print("Running real test...")
    # print("=" * 60)
    # test_with_real()
