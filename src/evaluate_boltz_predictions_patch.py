"""
evaluate_boltz_predictions.py の `cryptic_contact_with_apo_ligand` 指標を
CryptoBank のスコアリング関数 (scoring_function.py) で差し替えるパッチ。

■ 差し替え前（旧）
    apo に holo リガンドをスーパーインポーズし、cryptic 残基のいずれかの
    原子が ligand から 3Å 以内にあるか（True / False）

■ 差し替え後（新）
    CryptoBank の concentric-shell モデルで apo ↔ predicted の
    "crypticity score" を計算し、≥ 0.5 を cryptic と判定する連続量スコア。

使い方
------
1. evaluate_boltz_predictions.py の先頭付近に以下の import を追加：
       from evaluate_boltz_predictions_patch import (
           compute_cryptobank_score,
           CryptoBankScorer,
       )

2. `cryptic_contact_with_apo_ligand` を計算していた箇所を
   `compute_cryptobank_crypticity` の呼び出しで置き換える（詳細は下記）。

3. pipeline_config.json に "CRYPTOBANK_DIR" キーを追加し、
   Gervasiolab/CryptoBank リポジトリへのパスを指定する。

依存
----
    pip install MDAnalysis scipy numpy biopython
    # CryptoBank リポジトリの scoring_function.py が PYTHONPATH 上にあること
    # または CRYPTOBANK_DIR で指定すること
"""

from __future__ import annotations

import os
import sys
import tempfile
import importlib.util
import warnings
import numpy as np
from pathlib import Path
from typing import Optional

# ── BioPython (superimpose) ────────────────────────────────────────────────
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Superimposer import Superimposer


# ═══════════════════════════════════════════════════════════════════════════
# CryptoBank scoring_function の動的ロード
# ═══════════════════════════════════════════════════════════════════════════

def _load_scoring_function(cryptobank_dir: str):
    """
    CryptoBank リポジトリの scoring_function.py を動的にインポートして返す。

    Parameters
    ----------
    cryptobank_dir : str
        Gervasiolab/CryptoBank リポジトリのルートディレクトリ。
        scoring_function.py と *.npy / *.pkl の重みファイルがここにある。

    Returns
    -------
    module : types.ModuleType
        インポートされた scoring_function モジュール。
    """
    sf_path = Path(cryptobank_dir) / "scoring_function.py"
    if not sf_path.exists():
        raise FileNotFoundError(
            f"CryptoBank の scoring_function.py が見つかりません: {sf_path}\n"
            "CRYPTOBANK_DIR を正しく設定してください。"
        )

    spec = importlib.util.spec_from_file_location("cryptobank_scoring", sf_path)
    module = importlib.util.module_from_spec(spec)

    # scoring_function.py は SCRIPT_DIR を使って重みファイルを探すため、
    # __file__ を差し替えておく
    module.__file__ = str(sf_path)
    spec.loader.exec_module(module)
    return module


# ═══════════════════════════════════════════════════════════════════════════
# XYZ ファイル生成ユーティリティ
# ═══════════════════════════════════════════════════════════════════════════

def _atoms_to_xyz_block(label: str, atoms) -> str:
    """
    BioPython の Atom イテレータから CryptoBank 形式の XYZ ブロックを生成する。

    CryptoBank の XYZ フォーマット:
        <n_atoms>
        <label>
        <element> <x> <y> <z>
        ...
    """
    lines = []
    for atom in atoms:
        elem = atom.element.strip() if atom.element else atom.get_name()[0]
        x, y, z = atom.get_vector()
        lines.append(f"{elem} {x:.4f} {y:.4f} {z:.4f}")
    return f"{len(lines)}\n{label}\n" + "\n".join(lines) + "\n"


def _write_combined_xyz(
    xyz_path: str,
    holo_id: str,
    apo_id: str,
    lig_id: int,
    lig_atoms,        # holo リガンド（ペプチド）の原子イテレータ
    holo_prot_atoms,  # holo（predicted）の receptor チェーン原子イテレータ
    apo_prot_atoms,   # apo の receptor チェーン原子イテレータ
) -> None:
    """
    CryptoBank の `parse_xyz_file` が期待するフォーマットで XYZ ファイルを書く。

    フォーマット仕様（scoring_function.parse_xyz_file から逆算）:
        lig_label  = f'holo_{holo}_{lig_id}'
        prot_label = f'holo_{holo}'  (for holo)  /  f'apo_{apo}'  (for apo)

    1 ファイルにすべてのブロックを連結して書き込む。
    """
    lig_label       = f"holo_{holo_id}_{lig_id}"
    holo_prot_label = f"holo_{holo_id}"
    apo_prot_label  = f"apo_{apo_id}"

    content = ""
    content += _atoms_to_xyz_block(lig_label,       lig_atoms)
    content += _atoms_to_xyz_block(holo_prot_label, holo_prot_atoms)
    content += _atoms_to_xyz_block(apo_prot_label,  apo_prot_atoms)

    with open(xyz_path, "w") as f:
        f.write(content)


# ═══════════════════════════════════════════════════════════════════════════
# 構造アライメントユーティリティ
# ═══════════════════════════════════════════════════════════════════════════

def _get_ca_atoms(chain, residue_ids: Optional[set] = None):
    """chain から CA 原子のリストを取得する。residue_ids 指定時は絞り込む。"""
    cas = []
    for res in chain.get_residues():
        if residue_ids and res.get_id()[1] not in residue_ids:
            continue
        if "CA" in res:
            cas.append(res["CA"])
    return cas


def _superimpose_chains(mobile_chain, fixed_chain):
    """
    mobile_chain を fixed_chain に CA でスーパーインポーズし、
    mobile_chain の全原子に変換行列を適用する。

    Returns
    -------
    rms : float  — スーパーインポーズ後の RMSD
    """
    fixed_cas = _get_ca_atoms(fixed_chain)
    mobile_cas = _get_ca_atoms(mobile_chain)

    # 共通残基のみで合わせる
    fixed_ids  = {a.get_parent().get_id()[1] for a in fixed_cas}
    mobile_ids = {a.get_parent().get_id()[1] for a in mobile_cas}
    common_ids = fixed_ids & mobile_ids

    fixed_ca_common  = [a for a in fixed_cas  if a.get_parent().get_id()[1] in common_ids]
    mobile_ca_common = [a for a in mobile_cas if a.get_parent().get_id()[1] in common_ids]

    if len(fixed_ca_common) < 3:
        warnings.warn("スーパーインポーズに使える共通 CA が 3 残基未満です。")
        return None

    sup = Superimposer()
    sup.set_atoms(fixed_ca_common, mobile_ca_common)
    sup.apply(mobile_chain.get_atoms())  # mobile_chain の全原子を動かす
    return sup.rms


# ═══════════════════════════════════════════════════════════════════════════
# CryptoBankScorer: スコアリング関数のラッパー
# ═══════════════════════════════════════════════════════════════════════════

class CryptoBankScorer:
    """
    CryptoBank の scoring_function を Biopython PDB 構造から直接呼べるラッパー。

    Parameters
    ----------
    cryptobank_dir : str
        Gervasiolab/CryptoBank リポジトリのルートパス。
    n_lig_splits : int
        リガンドの分割数。CryptoBank デフォルト = 3（大きなリガンド向け）。
        ペプチドが短い（< 10 残基）場合は 1 を推奨。
    """

    def __init__(self, cryptobank_dir: str, n_lig_splits: int = 1):
        self._sf = _load_scoring_function(cryptobank_dir)
        self.n_lig_splits = n_lig_splits

    def score(
        self,
        predicted_pdb: str,
        apo_pdb: str,
        predicted_receptor_chain_id: str = "A",
        predicted_peptide_chain_id:  str = "B",
        apo_receptor_chain_id:       str = "A",
        holo_id: str = "predicted",
        apo_id:  str = "apo",
        lig_resid: int = 0,
    ) -> dict:
        """
        Predicted 構造（ペプチド + receptor）と apo 構造を受け取り、
        CryptoBank のスコアを計算して返す。

        ペプチドを「リガンド」、receptor を「タンパク質」として扱う。
        apo の receptor を predicted の receptor に CA アライン後、
        apo 側のタンパク質座標を変換して使う。

        Parameters
        ----------
        predicted_pdb : str
            Boltz-2 出力の予測構造 PDB パス。
        apo_pdb : str
            リガンド未結合の apo 構造 PDB パス。
        predicted_receptor_chain_id : str
            予測構造の receptor チェーン ID。
        predicted_peptide_chain_id : str
            予測構造のペプチドチェーン ID。
        apo_receptor_chain_id : str
            apo 構造の receptor チェーン ID。
        holo_id / apo_id : str
            XYZ ファイル内ラベル用の識別子（任意の文字列）。
        lig_resid : int
            XYZ ファイル内のリガンド ID（デフォルト 0 で問題なし）。

        Returns
        -------
        dict with keys:
            crypticity_score : float   [0, 1]。≥ 0.5 で cryptic 判定。
            is_cryptic       : bool    crypticity_score >= 0.5 かどうか。
            n_lig_splits     : int     使用したリガンド分割数。
            rms_apo_align    : float   apo アライメント RMSD（参考値）。
        """
        parser = PDBParser(QUIET=True)
        pred_struct = parser.get_structure("predicted", predicted_pdb)
        apo_struct  = parser.get_structure("apo",       apo_pdb)

        pred_model = pred_struct[0]
        apo_model  = apo_struct[0]

        # ── チェーンの取得 ──────────────────────────────────────────────
        pred_receptor = pred_model[predicted_receptor_chain_id]
        pred_peptide  = pred_model[predicted_peptide_chain_id]
        apo_receptor  = apo_model[apo_receptor_chain_id]

        # ── apo receptor → predicted receptor に CA アライン ────────────
        rms = _superimpose_chains(
            mobile_chain=apo_receptor,
            fixed_chain=pred_receptor,
        )

        # ── 一時 XYZ ファイルに書き出し ─────────────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            # holo XYZ: ペプチド（リガンド）+ predicted receptor（タンパク質）
            holo_xyz = os.path.join(tmpdir, "holo.xyz")
            _write_combined_xyz(
                xyz_path        = holo_xyz,
                holo_id         = holo_id,
                apo_id          = apo_id,
                lig_id          = lig_resid,
                lig_atoms       = list(pred_peptide.get_atoms()),
                holo_prot_atoms = list(pred_receptor.get_atoms()),
                apo_prot_atoms  = list(apo_receptor.get_atoms()),  # アライン済み
            )

            # apo XYZ: ペプチド（リガンド）は holo と同じ座標のまま使う
            #   ※ "apo にリガンドが存在しない" ことを模倣するため、
            #     apo_matrices はペプチド atoms との距離行列になる
            apo_xyz = os.path.join(tmpdir, "apo.xyz")
            _write_combined_xyz(
                xyz_path        = apo_xyz,
                holo_id         = holo_id,
                apo_id          = apo_id,
                lig_id          = lig_resid,
                lig_atoms       = list(pred_peptide.get_atoms()),   # 同じリガンド
                holo_prot_atoms = list(pred_receptor.get_atoms()),  # 同じ holo タンパク質
                apo_prot_atoms  = list(apo_receptor.get_atoms()),   # アライン済み apo
            )

            # ── CryptoBank スコアリング ─────────────────────────────────
            try:
                score, probs, lig_around_apo, lig_around_holo = self._sf.get_score(
                    xyz_apo    = apo_xyz,
                    xyz_holo   = holo_xyz,
                    apo_id     = apo_id,
                    holo_id    = holo_id,
                    lig_resid  = lig_resid,
                    n_lig_splits = self.n_lig_splits,
                )
            except Exception as e:
                warnings.warn(f"CryptoBank スコアリング中にエラーが発生しました: {e}")
                score = float("nan")
                probs = []
                lig_around_apo = lig_around_holo = 0

        return {
            "crypticity_score": float(score),
            "is_cryptic":       bool(float(score) >= 0.5),
            "segment_probs":    [float(p) for p in probs],
            "n_lig_splits":     self.n_lig_splits,
            "rms_apo_align":    float(rms) if rms is not None else None,
            "lig_around_apo":   int(lig_around_apo),
            "lig_around_holo":  int(lig_around_holo),
        }


# ═══════════════════════════════════════════════════════════════════════════
# evaluate_boltz_predictions.py への差し込み関数
# ═══════════════════════════════════════════════════════════════════════════

def compute_cryptobank_crypticity(
    predicted_pdb:   str,
    apo_pdb:         str,
    cryptobank_dir:  str,
    predicted_receptor_chain_id: str = "A",
    predicted_peptide_chain_id:  str = "B",
    apo_receptor_chain_id:       str = "A",
    n_lig_splits:    int = 1,
) -> dict:
    """
    evaluate_boltz_predictions.py から直接呼び出せる関数。

    旧実装 (cryptic_contact_with_apo_ligand) を以下で置き換える:

        # 旧コード（evaluate_boltz_predictions.py 内）:
        result["cryptic_contact_with_apo_ligand"] = _check_apo_ligand_contact(
            predicted_pdb, apo_pdb, holo_pdb, cryptic_residue_ids
        )

        # 新コード（差し替え後）:
        cb_result = compute_cryptobank_crypticity(
            predicted_pdb  = predicted_pdb,
            apo_pdb        = apo_pdb,
            cryptobank_dir = pipeline_config["CRYPTOBANK_DIR"],
        )
        result["crypticity_score"]         = cb_result["crypticity_score"]
        result["is_cryptic"]               = cb_result["is_cryptic"]
        result["crypticity_rms_apo_align"] = cb_result["rms_apo_align"]

    Parameters
    ----------
    predicted_pdb : str
        Boltz-2 出力 PDB（receptor = chain A, peptide = chain B を想定）。
    apo_pdb : str
        Apo 構造 PDB（receptor のみ）。
    cryptobank_dir : str
        Gervasiolab/CryptoBank リポジトリのルートパス。
    predicted_receptor_chain_id : str  default "A"
    predicted_peptide_chain_id  : str  default "B"
    apo_receptor_chain_id       : str  default "A"
    n_lig_splits : int
        リガンド分割数。ペプチドが短い場合は 1 を推奨（デフォルト）。
        ペプチドが 20 残基を超える場合は 3 を試すとよい。

    Returns
    -------
    dict:
        crypticity_score         : float  — [0, 1]（≥ 0.5 で cryptic）
        is_cryptic               : bool
        segment_probs            : list[float]  — 各セグメントの確率
        n_lig_splits             : int
        rms_apo_align            : float | None  — アライメント RMSD（Å）
        lig_around_apo           : int  — 0/1 フラグ
        lig_around_holo          : int  — 0/1 フラグ
    """
    scorer = CryptoBankScorer(
        cryptobank_dir=cryptobank_dir,
        n_lig_splits=n_lig_splits,
    )
    return scorer.score(
        predicted_pdb                = predicted_pdb,
        apo_pdb                      = apo_pdb,
        predicted_receptor_chain_id  = predicted_receptor_chain_id,
        predicted_peptide_chain_id   = predicted_peptide_chain_id,
        apo_receptor_chain_id        = apo_receptor_chain_id,
    )


# ═══════════════════════════════════════════════════════════════════════════
# evaluate_boltz_predictions.py の修正箇所まとめ（コメントで示す）
# ═══════════════════════════════════════════════════════════════════════════
#
# [1] pipeline_config.json に追加するキー:
#
#       "CRYPTOBANK_DIR": "/path/to/Gervasiolab/CryptoBank",
#       "CRYPTOBANK_N_LIG_SPLITS": 1
#
# [2] evaluate_boltz_predictions.py の import セクションに追加:
#
#       from evaluate_boltz_predictions_patch import compute_cryptobank_crypticity
#
# [3] 旧 _check_apo_ligand_contact の呼び出しを以下で置き換え:
#
#       # ── 旧 ──────────────────────────────────────────────────────────
#       # result["cryptic_contact_with_apo_ligand"] = _check_apo_ligand_contact(
#       #     predicted_pdb, apo_pdb, holo_pdb, cryptic_residue_ids
#       # )
#
#       # ── 新 ──────────────────────────────────────────────────────────
#       _cryptobank_dir = config.get("CRYPTOBANK_DIR", "")
#       _n_splits       = config.get("CRYPTOBANK_N_LIG_SPLITS", 1)
#       if _cryptobank_dir and os.path.isdir(_cryptobank_dir):
#           cb = compute_cryptobank_crypticity(
#               predicted_pdb  = predicted_pdb,
#               apo_pdb        = apo_pdb,
#               cryptobank_dir = _cryptobank_dir,
#               n_lig_splits   = _n_splits,
#           )
#           result["crypticity_score"]         = cb["crypticity_score"]
#           result["is_cryptic"]               = cb["is_cryptic"]
#           result["crypticity_rms_apo_align"] = cb["rms_apo_align"]
#           result["crypticity_lig_around_apo"]  = cb["lig_around_apo"]
#           result["crypticity_lig_around_holo"] = cb["lig_around_holo"]
#       else:
#           warnings.warn("CRYPTOBANK_DIR が設定されていないためスキップします。")
#           result["crypticity_score"]         = None
#           result["is_cryptic"]               = None
#           result["crypticity_rms_apo_align"] = None
#
# [4] evaluation_summary.csv のカラムリストに新指標を追加:
#
#       NEW_COLS = [
#           "crypticity_score",
#           "is_cryptic",
#           "crypticity_rms_apo_align",
#           "crypticity_lig_around_apo",
#           "crypticity_lig_around_holo",
#       ]
#
# [5] visualize_evaluation.py の Page 8 (Backbone RMSD & Apo-Ligand Contact) を
#     以下のように更新:
#       - 棒グラフ: crypticity_score を各ペプチドで横並び
#       - 横破線: y=0.5（CryptoBank の cryptic 閾値）
#       - タイトル: "CryptoBank Crypticity Score (≥0.5 = cryptic)"
#       旧 cryptic_contact_with_apo_ligand (True/False バー) は削除または残す
#
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    """
    簡易動作確認。実際の PDB ファイルとリポジトリパスで試す。

    例:
        python evaluate_boltz_predictions_patch.py \
            --predicted  path/to/predicted.pdb \
            --apo        path/to/apo.pdb \
            --cryptobank /path/to/CryptoBank
    """
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("--predicted",  required=True)
    ap.add_argument("--apo",        required=True)
    ap.add_argument("--cryptobank", required=True)
    ap.add_argument("--n_splits",   type=int, default=1)
    args = ap.parse_args()

    result = compute_cryptobank_crypticity(
        predicted_pdb  = args.predicted,
        apo_pdb        = args.apo,
        cryptobank_dir = args.cryptobank,
        n_lig_splits   = args.n_splits,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
