#!/usr/bin/env python
# ============================================================
# Evaluate Boltz-2 predictions of peptide-target complexes
#
# 評価軸:
#   (a) 構造的信頼度
#       - confidence_score, ptm, iptm, complex_plddt (Boltz JSON より)
#       - peptide_plddt_mean / min, interface_plddt (CIF/PDB より)
#   (b) 狙った位置に結合しているか
#       - interface residues, cryptic recall/precision/jaccard/f1
#       - peptide-centroid vs cryptic-centroid 距離, min_dist_to_pocket
#       - holo PDB のリガンド結合残基との一致 (holo がある場合のみ)
#   (c) クリプティックポケットになっているか (CryptoBank / PocketMiner 流)
#       - fpocket druggability スコア (apo vs predicted)
#       - LIGSITE 風 pocket volume (fpocket の pocket volume で代用)
#       - cryptic 残基の SASA 変化 (apo vs predicted)
#       - cryptic 残基の backbone / sidechain RMSD (apo vs predicted)
#       - holo がある場合: predicted と holo の cryptic 残基 RMSD
#
# 入力構造の前提 (Boltz output_format=pdb):
#   boltz_output_root/
#     boltz_results_<name>/
#       predictions/
#         <name>/
#           <name>_model_0.pdb
#           confidence_<name>_model_0.json
#           plddt_<name>_model_0.npz
#
# 出力:
#   evaluation_dir/
#     evaluation_summary.csv
#     per_peptide/<name>.json
#     fpocket_cache/
#       apo/
#       predicted/<name>/
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, Superimposer, PDBIO, Selection
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

try:
    import freesasa
    FREESASA_AVAILABLE = True
except ImportError:
    FREESASA_AVAILABLE = False
    warnings.warn("freesasa not installed; SASA metrics will be skipped.")

warnings.filterwarnings("ignore", category=UserWarning, module="Bio")


# ---------- 標準アミノ酸 ----------
STANDARD_AAS = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}

# 水・イオン除外用
NON_LIGAND_HET = {
    "HOH", "WAT", "DOD",
    "NA", "K", "MG", "CA", "ZN", "FE", "MN", "CL", "BR", "I",
    "SO4", "PO4", "ACT", "GOL", "EDO", "PEG", "DMS", "TRS",
}


# =========================
# データクラス
# =========================
@dataclass
class CrypticDefinition:
    """cryptic 残基集合の定義"""
    residues: Set[int] = field(default_factory=set)  # 1-based residue id
    threshold: Optional[float] = None
    source: str = "unspecified"  # "explicit_list" or "threshold"

    def to_dict(self) -> dict:
        return {
            "residues": sorted(self.residues),
            "threshold": self.threshold,
            "source": self.source,
            "n_residues": len(self.residues),
        }


@dataclass
class EvaluationResult:
    """1ペプチドあたりの評価結果"""
    name: str
    peptide_sequence: str = ""

    # (a) 信頼度
    confidence_score: Optional[float] = None
    ptm: Optional[float] = None
    iptm: Optional[float] = None
    complex_plddt: Optional[float] = None
    complex_iplddt: Optional[float] = None
    complex_pde: Optional[float] = None
    complex_ipde: Optional[float] = None
    chains_ptm: Optional[dict] = None
    pair_chains_iptm: Optional[dict] = None
    peptide_plddt_mean: Optional[float] = None
    peptide_plddt_min: Optional[float] = None
    interface_plddt_mean: Optional[float] = None

    # (b) 狙った場所か
    n_interface_residues: int = 0
    interface_residues: List[int] = field(default_factory=list)
    cryptic_recall: Optional[float] = None
    cryptic_precision: Optional[float] = None
    cryptic_jaccard: Optional[float] = None
    cryptic_f1: Optional[float] = None
    peptide_centroid_to_cryptic_centroid_dist: Optional[float] = None
    min_dist_peptide_to_cryptic: Optional[float] = None
    holo_ligand_recall: Optional[float] = None
    holo_ligand_precision: Optional[float] = None
    holo_ligand_jaccard: Optional[float] = None

    # (c) クリプティックか
    fpocket_druggability_apo: Optional[float] = None
    fpocket_druggability_predicted: Optional[float] = None
    delta_druggability: Optional[float] = None
    fpocket_volume_apo: Optional[float] = None
    fpocket_volume_predicted: Optional[float] = None
    delta_volume: Optional[float] = None
    cryptic_sasa_apo: Optional[float] = None
    cryptic_sasa_predicted: Optional[float] = None
    delta_cryptic_sasa: Optional[float] = None
    cryptic_backbone_rmsd_vs_apo: Optional[float] = None
    cryptic_sidechain_rmsd_vs_apo: Optional[float] = None
    cryptic_backbone_rmsd_vs_holo: Optional[float] = None
    cryptic_sidechain_rmsd_vs_holo: Optional[float] = None

    # receptor→peptide 方向 ipTM
    receptor_to_peptide_iptm: Optional[float] = None

    # RSA (relative SASA)
    cryptic_rsa_apo: Optional[float] = None
    cryptic_rsa_predicted: Optional[float] = None
    delta_cryptic_rsa: Optional[float] = None

    # メタ
    errors: List[str] = field(default_factory=list)
    warnings_list: List[str] = field(default_factory=list)


# =========================
# I/O & ユーティリティ
# =========================
def load_cryptic_definition(
    explicit_residues: Optional[str],
    scores_json: Optional[str],
    threshold: Optional[float],
) -> CrypticDefinition:
    """
    cryptic 残基集合を作る。
    優先順位: 明示的リスト > scores_json + threshold
    """
    if explicit_residues:
        residues = parse_residue_list(explicit_residues)
        return CrypticDefinition(
            residues=residues,
            threshold=None,
            source="explicit_list",
        )

    if scores_json and threshold is not None:
        with open(scores_json, "r") as f:
            scores = json.load(f)

        residues: Set[int] = set()
        for entry in scores:
            score = entry.get("displayed_score")
            if score is None:
                score = entry.get("raw_score")
            if score is None:
                continue
            if score >= threshold:
                residues.add(int(entry["residue_id"]))

        return CrypticDefinition(
            residues=residues,
            threshold=threshold,
            source="threshold",
        )

    raise ValueError(
        "Cryptic residues unspecified: pass either --cryptic_residues or "
        "(--cryptic_scores_json + --cryptic_threshold)"
    )


def parse_residue_list(spec: str) -> Set[int]:
    """
    "45,67,89-95,102" のような表記を {45, 67, 89, 90, ..., 95, 102} に変換
    """
    out: Set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(token))
    return out


def load_structure(path: str, name: str = "s") -> Structure:
    """拡張子から自動でパース"""
    ext = Path(path).suffix.lower()
    if ext in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(name, path)


def get_chain(structure: Structure, chain_id: str) -> Chain:
    """指定 chain を返す。複数 model があれば model 0 を使う"""
    model = next(structure.get_models())
    if chain_id not in model:
        available = [c.id for c in model]
        raise ValueError(
            f"Chain '{chain_id}' not found in {structure.id}; available: {available}"
        )
    return model[chain_id]


def warn_multiple_chains(
    structure: Structure,
    target_chain: str,
    label: str,
    result: EvaluationResult,
) -> None:
    """複数 chain があれば警告を残す"""
    model = next(structure.get_models())
    protein_chains = []
    for ch in model:
        # 1残基以上のアミノ酸チェーン
        n_aa = sum(1 for r in ch if r.get_resname().strip() in STANDARD_AAS)
        if n_aa > 0:
            protein_chains.append(ch.id)
    if len(protein_chains) > 1:
        msg = (
            f"[{label}] multiple protein chains detected: {protein_chains}; "
            f"using chain '{target_chain}'"
        )
        print(f"  ⚠ {msg}")
        result.warnings_list.append(msg)


def get_residue_dict(chain: Chain) -> Dict[int, Residue]:
    """resseq → Residue (標準AAのみ、ヘテロ原子は除外)"""
    out: Dict[int, Residue] = {}
    for res in chain:
        if res.id[0] != " ":  # ヘテロ原子フラグ
            continue
        if res.get_resname().strip() not in STANDARD_AAS:
            continue
        out[res.id[1]] = res
    return out


def get_atoms_of_residues(chain: Chain, resseq_set: Set[int]) -> List:
    """指定残基集合の全原子"""
    atoms = []
    for res in chain:
        if res.id[1] in resseq_set and res.id[0] == " ":
            atoms.extend(res.get_atoms())
    return atoms


def get_all_atom_coords(atoms) -> np.ndarray:
    return np.array([a.get_coord() for a in atoms])


# =========================
# (a) 信頼度: Boltz JSON
# =========================
def parse_boltz_confidence(json_path: Path, result: EvaluationResult) -> None:
    if not json_path.exists():
        result.errors.append(f"confidence JSON not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # Boltz の confidence JSON のキー (v2 系) を吸収
    def pick(*keys):
        for k in keys:
            if k in data and data[k] is not None:
                return data[k]
        return None

    result.confidence_score = pick("confidence_score", "aggregate_score")
    result.ptm = pick("ptm")
    result.iptm = pick("iptm")
    result.complex_plddt = pick("complex_plddt")
    result.complex_iplddt = pick("complex_iplddt")
    result.complex_pde = pick("complex_pde")
    result.complex_ipde = pick("complex_ipde")
    result.chains_ptm = data.get("chains_ptm")
    result.pair_chains_iptm = data.get("pair_chains_iptm")

    # receptor(chain 0) → peptide(chain 1) 方向の ipTM
    pair = data.get("pair_chains_iptm")
    if pair:
        try:
            result.receptor_to_peptide_iptm = float(pair["0"]["1"])
        except (KeyError, TypeError, ValueError):
            pass


# =========================
# (a) per-residue pLDDT を npz から
# =========================
def parse_per_residue_plddt(
    plddt_npz: Path,
    structure: Structure,
    target_chain: str,
    peptide_chain: str,
    interface_target_resseqs: Set[int],
    result: EvaluationResult,
) -> None:
    """
    plddt_<name>_model_0.npz から per-residue pLDDT を抽出
    Boltz は通常 'plddt' キーで shape=(N,) を保存
    """
    if not plddt_npz.exists():
        # CIF/PDB の B-factor からフォールバック
        return _plddt_from_bfactor(structure, target_chain, peptide_chain,
                                   interface_target_resseqs, result)

    try:
        arr = np.load(plddt_npz)
    except Exception as e:
        result.warnings_list.append(f"failed to load plddt npz: {e}")
        return _plddt_from_bfactor(structure, target_chain, peptide_chain,
                                   interface_target_resseqs, result)

    # 'plddt' キーを優先、なければ最初の配列
    if "plddt" in arr:
        plddt = arr["plddt"]
    else:
        plddt = arr[list(arr.keys())[0]]
    plddt = np.asarray(plddt).flatten()

    # 構造の残基順に並べて、ペプチド/界面のインデックスを引く
    model = next(structure.get_models())
    idx_to_chain_res: List[Tuple[str, int]] = []
    for ch in model:
        for res in ch:
            if res.id[0] != " ":
                continue
            if res.get_resname().strip() not in STANDARD_AAS:
                continue
            idx_to_chain_res.append((ch.id, res.id[1]))

    if len(idx_to_chain_res) != len(plddt):
        result.warnings_list.append(
            f"pLDDT length ({len(plddt)}) != n_residues ({len(idx_to_chain_res)}); "
            "falling back to B-factor"
        )
        return _plddt_from_bfactor(structure, target_chain, peptide_chain,
                                   interface_target_resseqs, result)

    pep_vals = []
    iface_vals = []
    for i, (ch_id, _resseq) in enumerate(idx_to_chain_res):
        if ch_id == peptide_chain:
            pep_vals.append(plddt[i])
        if ch_id == target_chain and _resseq in interface_target_resseqs:
            iface_vals.append(plddt[i])

    if pep_vals:
        result.peptide_plddt_mean = float(np.mean(pep_vals))
        result.peptide_plddt_min = float(np.min(pep_vals))
    if iface_vals:
        result.interface_plddt_mean = float(np.mean(iface_vals))


def _plddt_from_bfactor(
    structure: Structure,
    target_chain: str,
    peptide_chain: str,
    interface_target_resseqs: Set[int],
    result: EvaluationResult,
) -> None:
    """フォールバック: CIF/PDB の B-factor を pLDDT として扱う"""
    model = next(structure.get_models())
    pep_vals, iface_vals = [], []

    if peptide_chain in model:
        for res in model[peptide_chain]:
            if res.id[0] != " ":
                continue
            ca = res["CA"] if "CA" in res else None
            if ca is not None:
                pep_vals.append(ca.get_bfactor())

    if target_chain in model:
        for res in model[target_chain]:
            if res.id[0] != " ":
                continue
            if res.id[1] not in interface_target_resseqs:
                continue
            ca = res["CA"] if "CA" in res else None
            if ca is not None:
                iface_vals.append(ca.get_bfactor())

    if pep_vals:
        result.peptide_plddt_mean = float(np.mean(pep_vals))
        result.peptide_plddt_min = float(np.min(pep_vals))
    if iface_vals:
        result.interface_plddt_mean = float(np.mean(iface_vals))


# =========================
# (b) interface 残基
# =========================
def compute_interface_residues(
    structure: Structure,
    target_chain: str,
    peptide_chain: str,
    cutoff: float = 5.0,
) -> Set[int]:
    """ペプチド原子から cutoff Å 以内のターゲット残基 (1-based resseq) を返す"""
    model = next(structure.get_models())
    if target_chain not in model or peptide_chain not in model:
        return set()

    pep_coords = []
    for res in model[peptide_chain]:
        if res.id[0] != " ":
            continue
        for atom in res:
            pep_coords.append(atom.get_coord())
    if not pep_coords:
        return set()
    pep_coords = np.array(pep_coords)

    interface = set()
    for res in model[target_chain]:
        if res.id[0] != " ":
            continue
        if res.get_resname().strip() not in STANDARD_AAS:
            continue
        for atom in res:
            d = np.linalg.norm(pep_coords - atom.get_coord(), axis=1)
            if np.any(d <= cutoff):
                interface.add(res.id[1])
                break
    return interface


def compute_overlap_metrics(
    pred_set: Set[int],
    ref_set: Set[int],
) -> Dict[str, Optional[float]]:
    """pred を ref と比較。ref=cryptic, pred=interface のように使う"""
    if not ref_set:
        return {"recall": None, "precision": None, "jaccard": None, "f1": None}
    inter = pred_set & ref_set
    union = pred_set | ref_set
    recall = len(inter) / len(ref_set) if ref_set else 0.0
    precision = len(inter) / len(pred_set) if pred_set else 0.0
    jaccard = len(inter) / len(union) if union else 0.0
    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0
    return {
        "recall": recall,
        "precision": precision,
        "jaccard": jaccard,
        "f1": f1,
    }


def compute_centroid_distance(
    structure: Structure,
    target_chain: str,
    peptide_chain: str,
    cryptic_resseqs: Set[int],
) -> Tuple[Optional[float], Optional[float]]:
    """ペプチド重心 ↔ cryptic ポケット重心の距離 と 最短原子間距離"""
    model = next(structure.get_models())
    if target_chain not in model or peptide_chain not in model:
        return None, None

    pep_coords = []
    for res in model[peptide_chain]:
        if res.id[0] != " ":
            continue
        for atom in res:
            pep_coords.append(atom.get_coord())

    poc_coords = []
    for res in model[target_chain]:
        if res.id[0] != " ":
            continue
        if res.id[1] not in cryptic_resseqs:
            continue
        for atom in res:
            poc_coords.append(atom.get_coord())

    if not pep_coords or not poc_coords:
        return None, None

    pep_coords = np.array(pep_coords)
    poc_coords = np.array(poc_coords)

    centroid_dist = float(np.linalg.norm(pep_coords.mean(0) - poc_coords.mean(0)))

    # 最短距離: ペアワイズで O(NM) だがペプチドサイズなら問題なし
    diffs = pep_coords[:, None, :] - poc_coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    min_dist = float(dists.min())

    return centroid_dist, min_dist


# =========================
# (b) holo リガンド結合残基
# =========================
def compute_holo_ligand_binding_residues(
    holo_structure: Structure,
    target_chain: str,
    cutoff: float = 5.0,
    ligand_name: Optional[str] = None,
) -> Set[int]:
    """
    holo PDB から、リガンド (water/ion 除く HETATM) から cutoff Å 以内の
    ターゲット残基集合を返す
    """
    model = next(holo_structure.get_models())
    if target_chain not in model:
        return set()

    # リガンド原子を収集
    ligand_coords = []
    for ch in model:
        for res in ch:
            hetflag = res.id[0]
            resname = res.get_resname().strip()
            if hetflag == " ":
                continue  # 標準残基
            if resname in NON_LIGAND_HET:
                continue
            if ligand_name is not None and resname != ligand_name:
                continue
            for atom in res:
                ligand_coords.append(atom.get_coord())

    if not ligand_coords:
        return set()
    ligand_coords = np.array(ligand_coords)

    binding = set()
    for res in model[target_chain]:
        if res.id[0] != " ":
            continue
        if res.get_resname().strip() not in STANDARD_AAS:
            continue
        for atom in res:
            d = np.linalg.norm(ligand_coords - atom.get_coord(), axis=1)
            if np.any(d <= cutoff):
                binding.add(res.id[1])
                break
    return binding


# =========================
# (c) fpocket
# =========================
def run_fpocket(input_pdb: Path, output_dir: Path) -> Optional[Path]:
    """
    fpocket を実行し、_out ディレクトリのパスを返す。
    出力ディレクトリは input_pdb と同じ場所に <stem>_out として作られる仕様なので、
    一度 output_dir にコピーしてから実行する。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_pdb = output_dir / input_pdb.name
    if target_pdb.resolve() != input_pdb.resolve():
        shutil.copy(input_pdb, target_pdb)

    expected_out = output_dir / f"{target_pdb.stem}_out"
    if expected_out.exists():
        return expected_out  # キャッシュ

    try:
        subprocess.run(
            ["fpocket", "-f", str(target_pdb)],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(output_dir),
        )
    except FileNotFoundError:
        warnings.warn("fpocket not found in PATH")
        return None
    except subprocess.CalledProcessError as e:
        warnings.warn(f"fpocket failed: {e.stderr}")
        return None

    if not expected_out.exists():
        return None
    return expected_out


def parse_fpocket_info(fpocket_out_dir: Path) -> List[Dict]:
    """
    <pdbname>_info.txt をパースして各 pocket のスコアを取得
    返り値: [{"pocket_id": 1, "druggability": 0.8, "volume": 500.2, ...}, ...]
    """
    info_files = list(fpocket_out_dir.glob("*_info.txt"))
    if not info_files:
        return []
    info_path = info_files[0]

    pockets = []
    current = None
    with open(info_path, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("Pocket "):
                # 例: "Pocket 1 :"
                if current is not None:
                    pockets.append(current)
                try:
                    pid = int(line.split()[1])
                except (IndexError, ValueError):
                    pid = len(pockets) + 1
                current = {"pocket_id": pid}
            elif current is not None and ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                # タブや余分な空白を除去してから変換
                val = val.replace("\t", "").strip()
                try:
                    val_f = float(val)
                except ValueError:
                    continue
                # よく使うキーを正規化
                key_map = {
                    "Druggability Score": "druggability",
                    "Score": "score",
                    # fpocket v3 系のキー名
                    "Volume": "volume",
                    "Total SASA": "total_sasa",
                    "Polar SASA": "polar_sasa",
                    "Apolar SASA": "apolar_sasa",
                    # 旧バージョン / 別表記
                    "Real volume (approximation)": "volume",
                    "Pocket volume (Monte Carlo)": "volume_mc",
                    "Pocket volume (convex hull)": "volume_convex",
                    "Number of Alpha Spheres": "n_alpha_spheres",
                    "Mean local hydrophobic density": "hydrophobic_density",
                    "Hydrophobicity score": "hydrophobicity",
                    "Polarity score": "polarity",
                    "Volume score": "volume_score",
                    "Charge score": "charge_score",
                    "Flexibility": "flexibility",
                }
                current[key_map.get(key, key)] = val_f
    if current is not None:
        pockets.append(current)
    return pockets


def get_pocket_residue_sets(
    fpocket_out_dir: Path,
    target_chain: str,
) -> Dict[int, Set[int]]:
    """
    各 pocket に属するターゲット残基集合を取得する。
    pockets/pocket<N>_atm.pdb を読んで残基を抽出。
    """
    pockets_dir = fpocket_out_dir / "pockets"
    if not pockets_dir.exists():
        return {}

    out: Dict[int, Set[int]] = {}
    for atm_pdb in pockets_dir.glob("pocket*_atm.pdb"):
        # ファイル名から pocket id を抽出
        stem = atm_pdb.stem  # e.g. "pocket1_atm"
        try:
            pid = int(stem.replace("pocket", "").replace("_atm", ""))
        except ValueError:
            continue

        residues: Set[int] = set()
        try:
            struct = load_structure(str(atm_pdb), name=f"pocket{pid}")
        except Exception:
            continue
        model = next(struct.get_models())
        if target_chain in model:
            for res in model[target_chain]:
                if res.id[0] == " " and res.get_resname().strip() in STANDARD_AAS:
                    residues.add(res.id[1])
        out[pid] = residues
    return out


def find_best_pocket_for_cryptic(
    pocket_residues: Dict[int, Set[int]],
    cryptic_resseqs: Set[int],
    pockets_info: List[Dict],
) -> Optional[Dict]:
    """
    cryptic 残基集合と最も overlap が大きい pocket を選び、その info を返す。
    overlap=0 のときは None。
    """
    if not pocket_residues or not cryptic_resseqs:
        return None

    best_pid = None
    best_overlap = 0
    for pid, residues in pocket_residues.items():
        overlap = len(residues & cryptic_resseqs)
        if overlap > best_overlap:
            best_overlap = overlap
            best_pid = pid

    if best_pid is None:
        return None

    for info in pockets_info:
        if info.get("pocket_id") == best_pid:
            return info
    return None


# =========================
# (c) ターゲット鎖のみ抽出した PDB を作る
# =========================
def extract_target_chain_pdb(
    input_structure: Path,
    output_pdb: Path,
    target_chain: str,
) -> Path:
    """
    ターゲット鎖のみを残した PDB を書き出す (fpocket / freesasa 用)。
    入力は .pdb / .cif どちらでも可。出力は常に PDB 形式。
    output_pdb の拡張子は .pdb に強制される。
    """
    output_pdb = output_pdb.with_suffix(".pdb")
    struct = load_structure(str(input_structure), name="extract")
    io = PDBIO()

    class ChainSelector:
        def accept_model(self, model):
            return 1 if model.id == next(struct.get_models()).id else 0

        def accept_chain(self, chain):
            return 1 if chain.id == target_chain else 0

        def accept_residue(self, res):
            return 1 if res.id[0] == " " else 0

        def accept_atom(self, atom):
            return 1

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    io.set_structure(struct)
    io.save(str(output_pdb), ChainSelector())
    return output_pdb



# =========================
# Relative SASA (RSA) 正規化テーブル
# Tien et al. (2013) 理論値 (Å²) を使用
# =========================
MAX_ASA_TIEN2013: Dict[str, float] = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0,
    "CYS": 167.0, "GLU": 223.0, "GLN": 225.0, "GLY": 104.0,
    "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0,
    "MET": 224.0, "PHE": 240.0, "PRO": 159.0, "SER": 155.0,
    "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

def compute_residue_rsa(
    pdb_path: Path,
    chain_id: str,
    structure_for_resnames: Optional[Structure] = None,
) -> Dict[int, float]:
    """
    各残基の RSA = ASA / MAX_ASA(residue_type) を返す。
    structure_for_resnames: 残基名の取得用構造 (省略時は pdb_path から再ロード)
    RSA は [0, 1] にクリップ (末端残基は >1 になることがあるため)。
    """
    if not FREESASA_AVAILABLE:
        return {}

    # ASA を取得
    struct_fs = freesasa.Structure(str(pdb_path))
    if hasattr(freesasa, "calc"):
        result = freesasa.calc(struct_fs)
    elif hasattr(freesasa, "Calc"):
        result = freesasa.Calc().calculate(struct_fs)
    else:
        raise RuntimeError("freesasa API not recognized")

    residue_areas = result.residueAreas()
    if chain_id not in residue_areas:
        return {}

    # 残基名の取得 (resseq → resname)
    resname_map: Dict[int, str] = {}
    if structure_for_resnames is not None:
        try:
            ch = get_chain(structure_for_resnames, chain_id)
            for res in ch:
                if res.id[0] == " ":
                    resname_map[res.id[1]] = res.get_resname().strip()
        except Exception:
            pass

    out: Dict[int, float] = {}
    for resnum_str, ra in residue_areas[chain_id].items():
        try:
            resnum = int(resnum_str)
            asa = float(ra.total)
            resname = resname_map.get(resnum, "")
            max_asa = MAX_ASA_TIEN2013.get(resname, None)
            if max_asa and max_asa > 0:
                rsa = min(asa / max_asa, 1.0)   # クリップ
            else:
                rsa = asa  # resname 不明の場合は絶対値で fallback
            out[resnum] = rsa
        except (ValueError, AttributeError):
            continue
    return out


# =========================
# (c) SASA (freesasa) — 後方互換 (絶対値)
# =========================
def compute_residue_sasa(pdb_path: Path, chain_id: str) -> Dict[int, float]:
    """指定鎖の各残基の SASA (Å²)"""
    if not FREESASA_AVAILABLE:
        return {}
    structure = freesasa.Structure(str(pdb_path))
    # freesasa v2.2+ では freesasa.calc(), 旧版では freesasa.Calc().calculate()
    if hasattr(freesasa, "calc"):
        result = freesasa.calc(structure)
    elif hasattr(freesasa, "Calc"):
        result = freesasa.Calc().calculate(structure)
    else:
        raise RuntimeError("freesasa API not recognized")
    residue_areas = result.residueAreas()
    out: Dict[int, float] = {}
    if chain_id not in residue_areas:
        return {}
    for resnum_str, ra in residue_areas[chain_id].items():
        try:
            out[int(resnum_str)] = float(ra.total)
        except (ValueError, AttributeError):
            continue
    return out


# =========================
# (c) RMSD (apo / holo vs predicted)
# =========================
def compute_rmsd_cryptic(
    apo_pdb: Path,
    predicted_pdb: Path,
    target_chain_apo: str,
    target_chain_pred: str,
    cryptic_resseqs: Set[int],
) -> Tuple[Optional[float], Optional[float]]:
    """
    cryptic 残基の backbone (N,CA,C,O) RMSD と side-chain heavy atom RMSD を返す。
    まず全 backbone CA でアラインしてから cryptic 残基だけで RMSD を計算。
    """
    if not cryptic_resseqs:
        return None, None

    s1 = load_structure(str(apo_pdb), name="apo")
    s2 = load_structure(str(predicted_pdb), name="pred")

    try:
        c1 = get_chain(s1, target_chain_apo)
        c2 = get_chain(s2, target_chain_pred)
    except ValueError:
        return None, None

    d1 = get_residue_dict(c1)
    d2 = get_residue_dict(c2)

    # 共通残基 (両方に CA を持つもの) で全体アライン
    common = sorted(set(d1.keys()) & set(d2.keys()))
    fixed_atoms, moving_atoms = [], []
    for rsq in common:
        if "CA" in d1[rsq] and "CA" in d2[rsq]:
            fixed_atoms.append(d1[rsq]["CA"])
            moving_atoms.append(d2[rsq]["CA"])

    if len(fixed_atoms) < 3:
        return None, None

    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)
    sup.apply(s2.get_atoms())

    # cryptic 残基の backbone / sidechain RMSD
    bb_names = {"N", "CA", "C", "O"}

    bb_d2 = []
    bb_d1 = []
    sc_d2 = []
    sc_d1 = []

    for rsq in sorted(cryptic_resseqs):
        if rsq not in d1 or rsq not in d2:
            continue
        r1, r2 = d1[rsq], d2[rsq]
        for atom_name in [a.name for a in r1]:
            if atom_name.startswith("H"):
                continue
            if atom_name not in r2:
                continue
            a1 = r1[atom_name]
            a2 = r2[atom_name]
            if atom_name in bb_names:
                bb_d1.append(a1.get_coord())
                bb_d2.append(a2.get_coord())
            else:
                sc_d1.append(a1.get_coord())
                sc_d2.append(a2.get_coord())

    bb_rmsd = _rmsd(bb_d1, bb_d2) if bb_d1 else None
    sc_rmsd = _rmsd(sc_d1, sc_d2) if sc_d1 else None
    return bb_rmsd, sc_rmsd


def _rmsd(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


# =========================
# ペプチド配列の抽出 (CIF/PDB から)
# =========================
ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def extract_peptide_sequence(structure: Structure, peptide_chain: str) -> str:
    model = next(structure.get_models())
    if peptide_chain not in model:
        return ""
    seq = ""
    for res in model[peptide_chain]:
        if res.id[0] != " ":
            continue
        seq += ONE_LETTER.get(res.get_resname().strip(), "X")
    return seq


# =========================
# 1ペプチドの完全評価
# =========================
def evaluate_one(
    name: str,
    predicted_pdb: Path,
    confidence_json: Path,
    plddt_npz: Path,
    apo_pdb: Path,           # .pdb or .cif
    holo_pdb: Optional[Path],  # .pdb or .cif
    cryptic_def: CrypticDefinition,
    target_chain: str,
    peptide_chain: str,
    fpocket_cache_dir: Path,
    interface_cutoff: float,
    apo_chain: str,
    holo_chain: str,
    holo_ligand_name: Optional[str],
) -> EvaluationResult:
    res = EvaluationResult(name=name)
    cryptic_resseqs = cryptic_def.residues

    if not predicted_pdb.exists():
        res.errors.append(f"predicted structure not found: {predicted_pdb}")
        return res

    pred_struct = load_structure(str(predicted_pdb), name=name)

    # 複数鎖チェック
    warn_multiple_chains(pred_struct, target_chain, "predicted", res)

    # ペプチド配列
    res.peptide_sequence = extract_peptide_sequence(pred_struct, peptide_chain)

    # interface 残基
    interface = compute_interface_residues(
        pred_struct, target_chain, peptide_chain, cutoff=interface_cutoff,
    )
    res.interface_residues = sorted(interface)
    res.n_interface_residues = len(interface)

    # (a) 信頼度
    parse_boltz_confidence(confidence_json, res)
    parse_per_residue_plddt(
        plddt_npz, pred_struct, target_chain, peptide_chain, interface, res,
    )

    # (b) cryptic overlap
    if cryptic_resseqs:
        m = compute_overlap_metrics(interface, cryptic_resseqs)
        res.cryptic_recall = m["recall"]
        res.cryptic_precision = m["precision"]
        res.cryptic_jaccard = m["jaccard"]
        res.cryptic_f1 = m["f1"]

        cd, mind = compute_centroid_distance(
            pred_struct, target_chain, peptide_chain, cryptic_resseqs,
        )
        res.peptide_centroid_to_cryptic_centroid_dist = cd
        res.min_dist_peptide_to_cryptic = mind

    # (b) holo リガンドとの一致 (オプション)
    if holo_pdb is not None and holo_pdb.exists():
        try:
            holo_struct = load_structure(str(holo_pdb), name="holo")
            warn_multiple_chains(holo_struct, holo_chain, "holo", res)
            holo_binding = compute_holo_ligand_binding_residues(
                holo_struct, holo_chain, cutoff=interface_cutoff,
                ligand_name=holo_ligand_name,
            )
            if holo_binding:
                m = compute_overlap_metrics(interface, holo_binding)
                res.holo_ligand_recall = m["recall"]
                res.holo_ligand_precision = m["precision"]
                res.holo_ligand_jaccard = m["jaccard"]
            else:
                res.warnings_list.append("no ligand found in holo PDB")
        except Exception as e:
            res.warnings_list.append(f"holo processing failed: {e}")

    # (c) クリプティック判定 ----------------------------------------
    # 予測構造からターゲット鎖だけを抽出した PDB を作って fpocket / SASA に流す
    target_only_dir = fpocket_cache_dir / "predicted" / name
    target_only_dir.mkdir(parents=True, exist_ok=True)
    target_only_pdb = target_only_dir / f"{name}_targetonly.pdb"
    try:
        target_only_pdb = extract_target_chain_pdb(
            predicted_pdb, target_only_pdb, target_chain
        )
    except Exception as e:
        res.warnings_list.append(f"target chain extraction failed: {e}")
        target_only_pdb = None

    # fpocket on predicted (target only)
    pred_drug, pred_vol = None, None
    if target_only_pdb is not None and target_only_pdb.exists():
        pred_fp_dir = run_fpocket(target_only_pdb, target_only_dir)
        if pred_fp_dir is not None:
            pockets_info = parse_fpocket_info(pred_fp_dir)
            pocket_residues = get_pocket_residue_sets(pred_fp_dir, target_chain)
            best = find_best_pocket_for_cryptic(
                pocket_residues, cryptic_resseqs, pockets_info,
            )
            if best is not None:
                pred_drug = best.get("druggability")
                pred_vol = best.get("volume") or best.get("volume_mc")
            else:
                # 最大スコアの pocket を fallback として記録
                if pockets_info:
                    best_any = max(
                        pockets_info,
                        key=lambda d: d.get("druggability", -1),
                    )
                    res.warnings_list.append(
                        f"no fpocket overlaps cryptic residues in predicted; "
                        f"reporting best-overall pocket {best_any.get('pocket_id')}"
                    )
                    pred_drug = best_any.get("druggability")
                    pred_vol = best_any.get("volume") or best_any.get("volume_mc")

    res.fpocket_druggability_predicted = pred_drug
    res.fpocket_volume_predicted = pred_vol

    # fpocket on apo (キャッシュ)
    apo_drug, apo_vol = None, None
    apo_cache_dir = fpocket_cache_dir / "apo"
    apo_cache_dir.mkdir(parents=True, exist_ok=True)
    apo_targetonly = apo_cache_dir / f"{apo_pdb.stem}_targetonly.pdb"  # always .pdb (CIF 入力でも変換される)
    if not apo_targetonly.exists():
        try:
            apo_targetonly = extract_target_chain_pdb(
                apo_pdb, apo_targetonly, apo_chain
            )
        except Exception as e:
            res.warnings_list.append(f"apo extraction failed: {e}")
            apo_targetonly = None

    if apo_targetonly is not None and apo_targetonly.exists():
        apo_fp_dir = run_fpocket(apo_targetonly, apo_cache_dir)
        if apo_fp_dir is not None:
            pockets_info = parse_fpocket_info(apo_fp_dir)
            pocket_residues = get_pocket_residue_sets(apo_fp_dir, apo_chain)
            best = find_best_pocket_for_cryptic(
                pocket_residues, cryptic_resseqs, pockets_info,
            )
            if best is not None:
                apo_drug = best.get("druggability")
                apo_vol = best.get("volume") or best.get("volume_mc")

    res.fpocket_druggability_apo = apo_drug
    res.fpocket_volume_apo = apo_vol

    if apo_drug is not None and pred_drug is not None:
        res.delta_druggability = pred_drug - apo_drug
    if apo_vol is not None and pred_vol is not None:
        res.delta_volume = pred_vol - apo_vol

    # SASA (絶対値) & RSA (相対値)
    if FREESASA_AVAILABLE and cryptic_resseqs:
        try:
            apo_struct_for_resnames = load_structure(str(apo_pdb), name="apo_rsa")

            # 絶対 SASA
            apo_sasa  = compute_residue_sasa(apo_pdb, apo_chain)
            pred_pdb_for_sasa = (target_only_pdb
                                 if target_only_pdb is not None and target_only_pdb.exists()
                                 else predicted_pdb)
            pred_sasa = compute_residue_sasa(pred_pdb_for_sasa, target_chain)

            apo_sum  = sum(apo_sasa.get(r, 0.0) for r in cryptic_resseqs if r in apo_sasa)
            pred_sum = sum(pred_sasa.get(r, 0.0) for r in cryptic_resseqs if r in pred_sasa)

            if any(r in apo_sasa for r in cryptic_resseqs):
                res.cryptic_sasa_apo = apo_sum
            if any(r in pred_sasa for r in cryptic_resseqs):
                res.cryptic_sasa_predicted = pred_sum
            if res.cryptic_sasa_apo is not None and res.cryptic_sasa_predicted is not None:
                res.delta_cryptic_sasa = res.cryptic_sasa_predicted - res.cryptic_sasa_apo

            # RSA (Tien2013 正規化、cryptic 残基の平均 RSA)
            apo_rsa  = compute_residue_rsa(apo_pdb, apo_chain, apo_struct_for_resnames)
            pred_struct_for_resnames = load_structure(str(pred_pdb_for_sasa), name="pred_rsa")
            pred_rsa = compute_residue_rsa(pred_pdb_for_sasa, target_chain,
                                           pred_struct_for_resnames)

            apo_rsa_vals  = [apo_rsa[r]  for r in cryptic_resseqs if r in apo_rsa]
            pred_rsa_vals = [pred_rsa[r] for r in cryptic_resseqs if r in pred_rsa]

            if apo_rsa_vals:
                res.cryptic_rsa_apo = float(np.mean(apo_rsa_vals))
            if pred_rsa_vals:
                res.cryptic_rsa_predicted = float(np.mean(pred_rsa_vals))
            if res.cryptic_rsa_apo is not None and res.cryptic_rsa_predicted is not None:
                res.delta_cryptic_rsa = res.cryptic_rsa_predicted - res.cryptic_rsa_apo

        except Exception as e:
            res.warnings_list.append(f"SASA/RSA failed: {e}")

    # RMSD (apo vs predicted, holo vs predicted)
    if target_only_pdb is not None and target_only_pdb.exists():
        bb, sc = compute_rmsd_cryptic(
            apo_pdb, target_only_pdb, apo_chain, target_chain, cryptic_resseqs,
        )
        res.cryptic_backbone_rmsd_vs_apo = bb
        res.cryptic_sidechain_rmsd_vs_apo = sc

        if holo_pdb is not None and holo_pdb.exists():
            bb_h, sc_h = compute_rmsd_cryptic(
                holo_pdb, target_only_pdb, holo_chain, target_chain, cryptic_resseqs,
            )
            res.cryptic_backbone_rmsd_vs_holo = bb_h
            res.cryptic_sidechain_rmsd_vs_holo = sc_h

    return res


# =========================
# Boltz 出力のスキャン
# =========================
def discover_predictions(
    boltz_output_root: Path,
) -> List[Tuple[str, Path, Path, Path]]:
    """
    boltz_output_root から (name, pdb, confidence_json, plddt_npz) を列挙。

    Boltz は入力ディレクトリ名から出力ディレクトリを決めるため、
    実際の出力構造は2パターンある:

    パターンA (yaml 1ファイル = 1結果ディレクトリ):
        boltz_results_<name>/predictions/<name>/<name>_model_0.pdb

    パターンB (yaml をまとめたディレクトリを入力した場合):
        boltz_results_<input_dir_name>/predictions/<name>/<name>_model_0.pdb
            ← 複数の name サブディレクトリが並ぶ

    両方に対応するため、boltz_results_* 配下の predictions/ を再帰的に探索する。
    """
    found = []

    for results_dir in sorted(boltz_output_root.glob("boltz_results_*")):
        if not results_dir.is_dir():
            continue
        predictions_root = results_dir / "predictions"
        if not predictions_root.is_dir():
            continue

        # predictions/ 直下のサブディレクトリが各ペプチドの結果
        for pred_dir in sorted(predictions_root.iterdir()):
            if not pred_dir.is_dir():
                continue
            name = pred_dir.name

            # 構造ファイル (.pdb 優先、なければ .cif)
            pdb = pred_dir / f"{name}_model_0.pdb"
            cif = pred_dir / f"{name}_model_0.cif"
            if pdb.exists():
                struct_path = pdb
            elif cif.exists():
                struct_path = cif
            else:
                # model_0 以外も探す
                pdbs = sorted(pred_dir.glob(f"{name}_model_*.pdb"))
                cifs = sorted(pred_dir.glob(f"{name}_model_*.cif"))
                if pdbs:
                    struct_path = pdbs[0]
                elif cifs:
                    struct_path = cifs[0]
                else:
                    print(f"  ⚠ no structure file in {pred_dir}, skipping")
                    continue

            conf = pred_dir / f"confidence_{name}_model_0.json"
            plddt = pred_dir / f"plddt_{name}_model_0.npz"
            found.append((name, struct_path, conf, plddt))

    return found


# =========================
# CSV writer
# =========================
def write_summary_csv(results: List[EvaluationResult], out_csv: Path) -> None:
    if not results:
        return

    # 出力フィールド (errors/warnings は末尾)
    fieldnames = [
        "name", "peptide_sequence",
        # (a)
        "confidence_score", "ptm", "iptm",
        "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
        "peptide_plddt_mean", "peptide_plddt_min", "interface_plddt_mean",
        # (b)
        "n_interface_residues", "interface_residues",
        "cryptic_recall", "cryptic_precision", "cryptic_jaccard", "cryptic_f1",
        "peptide_centroid_to_cryptic_centroid_dist", "min_dist_peptide_to_cryptic",
        "holo_ligand_recall", "holo_ligand_precision", "holo_ligand_jaccard",
        # (a+) receptor→peptide ipTM
        "receptor_to_peptide_iptm",
        # (c)
        "fpocket_druggability_apo", "fpocket_druggability_predicted", "delta_druggability",
        "fpocket_volume_apo", "fpocket_volume_predicted", "delta_volume",
        "cryptic_sasa_apo", "cryptic_sasa_predicted", "delta_cryptic_sasa",
        "cryptic_rsa_apo", "cryptic_rsa_predicted", "delta_cryptic_rsa",
        "cryptic_backbone_rmsd_vs_apo", "cryptic_sidechain_rmsd_vs_apo",
        "cryptic_backbone_rmsd_vs_holo", "cryptic_sidechain_rmsd_vs_holo",
        # meta
        "n_warnings", "n_errors",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            d = asdict(r)
            row = {k: d.get(k) for k in fieldnames if k in d}
            row["interface_residues"] = ";".join(
                str(x) for x in d["interface_residues"]
            )
            row["n_warnings"] = len(d.get("warnings_list", []))
            row["n_errors"] = len(d.get("errors", []))
            writer.writerow(row)


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Boltz-2 peptide-target complex predictions."
    )
    # 入力
    parser.add_argument("--boltz_output_root", required=True,
                        help="boltz_results_* を含むディレクトリ")
    parser.add_argument("--apo_pdb", required=True, help="apo 構造のファイル (.pdb または .cif)")
    parser.add_argument("--holo_pdb", default=None, help="holo 構造のファイル (.pdb または .cif、任意)")

    # cryptic 残基指定 (どちらか必須)
    parser.add_argument("--cryptic_residues", default=None,
                        help='明示指定 (例: "45,67,89-95")')
    parser.add_argument("--cryptic_scores_json", default=None,
                        help="cryptic スコア JSON")
    parser.add_argument("--cryptic_threshold", type=float, default=None,
                        help="cryptic_scores_json と併用するしきい値")

    # チェーン ID
    parser.add_argument("--target_chain", default="A",
                        help="予測複合体のレセプター鎖 (default: A)")
    parser.add_argument("--peptide_chain", default="B",
                        help="予測複合体のペプチド鎖 (default: B)")
    parser.add_argument("--apo_chain", default="A",
                        help="apo PDB のレセプター鎖 (default: A)")
    parser.add_argument("--holo_chain", default="A",
                        help="holo PDB のレセプター鎖 (default: A)")
    parser.add_argument("--holo_ligand_name", default=None,
                        help="holo の特定リガンド名 (例: ATP)")

    # その他
    parser.add_argument("--interface_cutoff", type=float, default=5.0,
                        help="界面残基判定の距離カットオフ Å (default: 5.0)")

    # 出力
    parser.add_argument("--output_dir", required=True,
                        help="評価結果の出力ディレクトリ")

    args = parser.parse_args()

    boltz_root = Path(args.boltz_output_root)
    apo_pdb = Path(args.apo_pdb)
    holo_pdb = Path(args.holo_pdb) if args.holo_pdb else None
    output_dir = Path(args.output_dir)

    if not boltz_root.is_dir():
        sys.exit(f"boltz_output_root not found: {boltz_root}")
    if not apo_pdb.exists():
        sys.exit(f"apo PDB not found: {apo_pdb}")
    if holo_pdb is not None and not holo_pdb.exists():
        print(f"⚠ holo PDB not found, ignoring: {holo_pdb}")
        holo_pdb = None

    cryptic_def = load_cryptic_definition(
        explicit_residues=args.cryptic_residues,
        scores_json=args.cryptic_scores_json,
        threshold=args.cryptic_threshold,
    )

    print(f"Cryptic residues ({cryptic_def.source}): "
          f"{sorted(cryptic_def.residues)} (n={len(cryptic_def.residues)})")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_peptide_dir = output_dir / "per_peptide"
    per_peptide_dir.mkdir(exist_ok=True)
    fpocket_cache_dir = output_dir / "fpocket_cache"
    fpocket_cache_dir.mkdir(exist_ok=True)

    # メタ情報を残す
    meta = {
        "boltz_output_root": str(boltz_root),
        "apo_pdb": str(apo_pdb),
        "holo_pdb": str(holo_pdb) if holo_pdb else None,
        "target_chain": args.target_chain,
        "peptide_chain": args.peptide_chain,
        "apo_chain": args.apo_chain,
        "holo_chain": args.holo_chain,
        "holo_ligand_name": args.holo_ligand_name,
        "interface_cutoff": args.interface_cutoff,
        "cryptic_definition": cryptic_def.to_dict(),
    }
    with open(output_dir / "evaluation_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 評価
    predictions = discover_predictions(boltz_root)
    if not predictions:
        sys.exit(f"No boltz_results_* found in {boltz_root}")
    print(f"Found {len(predictions)} predictions")

    all_results: List[EvaluationResult] = []
    for i, (name, pdb, conf, plddt) in enumerate(predictions, 1):
        print(f"[{i}/{len(predictions)}] {name}")
        res = evaluate_one(
            name=name,
            predicted_pdb=pdb,
            confidence_json=conf,
            plddt_npz=plddt,
            apo_pdb=apo_pdb,
            holo_pdb=holo_pdb,
            cryptic_def=cryptic_def,
            target_chain=args.target_chain,
            peptide_chain=args.peptide_chain,
            fpocket_cache_dir=fpocket_cache_dir,
            interface_cutoff=args.interface_cutoff,
            apo_chain=args.apo_chain,
            holo_chain=args.holo_chain,
            holo_ligand_name=args.holo_ligand_name,
        )
        all_results.append(res)

        # 個別 JSON
        with open(per_peptide_dir / f"{name}.json", "w") as f:
            json.dump(asdict(res), f, indent=2, default=str)

        if res.errors:
            for e in res.errors:
                print(f"    ❌ {e}")

    # CSV 集約
    summary_csv = output_dir / "evaluation_summary.csv"
    write_summary_csv(all_results, summary_csv)
    print(f"\n✅ Wrote summary: {summary_csv}")
    print(f"✅ Per-peptide JSONs: {per_peptide_dir}")


if __name__ == "__main__":
    main()
