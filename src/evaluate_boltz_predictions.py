#!/usr/bin/env python
# ============================================================
# Evaluate Boltz-2 predictions of peptide-target complexes
#
# 評価軸:
#   (a) 構造的信頼度
#       - confidence_score, iptm, receptor_to_peptide_iptm, peptide_plddt_mean
#   (b) 狙った位置に結合しているか
#       - cryptic recall / f1
#       - peptide-centroid vs cryptic-centroid 距離, min_dist_to_pocket
#       - ペプチド重心 vs holo リガンド重心 距離 (holo がある場合)
#   (c) クリプティックポケットになっているか (CryptoBank 流)
#       - fpocket pocket volume (predicted / apo / delta)
#         ※ predicted: ペプチド界面残基 と overlap する pocket を選択
#            apo:       predicted と同じ残基集合 (interface) で overlap する pocket を選択
#       - cryptic 残基の RSA 変化 (Tien 2013 正規化)
#       - cryptic 残基の backbone RMSD (apo vs predicted)
#       - CryptoBank crypticity score (apo vs predicted, 同心球シェルモデル)
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

    # (a) 信頼度 — 絞り込み済み
    confidence_score: Optional[float] = None
    iptm: Optional[float] = None
    receptor_to_peptide_iptm: Optional[float] = None
    peptide_plddt_mean: Optional[float] = None

    # Boltz JSON の追加フィールド（CSVには出さないがJSONには残す）
    complex_iplddt: Optional[float] = None
    complex_pde: Optional[float] = None
    complex_ipde: Optional[float] = None
    chains_ptm: Optional[dict] = None
    pair_chains_iptm: Optional[dict] = None

    # (b) 狙った場所か
    n_interface_residues: int = 0
    interface_residues: List[int] = field(default_factory=list)
    cryptic_recall: Optional[float] = None
    cryptic_f1: Optional[float] = None
    peptide_centroid_to_cryptic_centroid_dist: Optional[float] = None
    min_dist_peptide_to_cryptic: Optional[float] = None
    # 新規: ペプチド重心 ↔ holo リガンド重心
    peptide_centroid_to_holo_ligand_centroid_dist: Optional[float] = None

    # (c) クリプティックか
    fpocket_volume_predicted: Optional[float] = None
    fpocket_volume_apo: Optional[float] = None
    delta_volume: Optional[float] = None
    cryptic_rsa_apo: Optional[float] = None
    cryptic_rsa_predicted: Optional[float] = None
    delta_cryptic_rsa: Optional[float] = None
    cryptic_backbone_rmsd_vs_apo: Optional[float] = None
    # CryptoBank crypticity score (同心球シェルモデルによる連続値, ≥0.5 で cryptic)
    crypticity_score: Optional[float] = None
    is_cryptic: Optional[bool] = None
    crypticity_rms_apo_align: Optional[float] = None

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
    if explicit_residues:
        residues = parse_residue_list(explicit_residues)
        return CrypticDefinition(residues=residues, threshold=None, source="explicit_list")

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
        return CrypticDefinition(residues=residues, threshold=threshold, source="threshold")

    raise ValueError(
        "Cryptic residues unspecified: pass either --cryptic_residues or "
        "(--cryptic_scores_json + --cryptic_threshold)"
    )


def parse_residue_list(spec: str) -> Set[int]:
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
    ext = Path(path).suffix.lower()
    if ext in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(name, path)


def get_chain(structure: Structure, chain_id: str) -> Chain:
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
    model = next(structure.get_models())
    protein_chains = []
    for ch in model:
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
    out: Dict[int, Residue] = {}
    for res in chain:
        if res.id[0] != " ":
            continue
        if res.get_resname().strip() not in STANDARD_AAS:
            continue
        out[res.id[1]] = res
    return out


def _rmsd(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


# =========================
# (a) 信頼度: Boltz JSON
# =========================
def parse_boltz_confidence(json_path: Path, result: EvaluationResult) -> None:
    if not json_path.exists():
        result.errors.append(f"confidence JSON not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    def pick(*keys):
        for k in keys:
            if k in data and data[k] is not None:
                return data[k]
        return None

    result.confidence_score = pick("confidence_score", "aggregate_score")
    result.iptm = pick("iptm")
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
    peptide_chain: str,
    result: EvaluationResult,
) -> None:
    if not plddt_npz.exists():
        return _plddt_from_bfactor(structure, peptide_chain, result)

    try:
        arr = np.load(plddt_npz)
    except Exception as e:
        result.warnings_list.append(f"failed to load plddt npz: {e}")
        return _plddt_from_bfactor(structure, peptide_chain, result)

    if "plddt" in arr:
        plddt = arr["plddt"]
    else:
        plddt = arr[list(arr.keys())[0]]
    plddt = np.asarray(plddt).flatten()

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
        return _plddt_from_bfactor(structure, peptide_chain, result)

    pep_vals = []
    for i, (ch_id, _resseq) in enumerate(idx_to_chain_res):
        if ch_id == peptide_chain:
            pep_vals.append(plddt[i])

    if pep_vals:
        result.peptide_plddt_mean = float(np.mean(pep_vals))


def _plddt_from_bfactor(
    structure: Structure,
    peptide_chain: str,
    result: EvaluationResult,
) -> None:
    model = next(structure.get_models())
    pep_vals = []
    if peptide_chain in model:
        for res in model[peptide_chain]:
            if res.id[0] != " ":
                continue
            ca = res["CA"] if "CA" in res else None
            if ca is not None:
                pep_vals.append(ca.get_bfactor())
    if pep_vals:
        result.peptide_plddt_mean = float(np.mean(pep_vals))


# =========================
# (b) interface 残基
# =========================
def compute_interface_residues(
    structure: Structure,
    target_chain: str,
    peptide_chain: str,
    cutoff: float = 5.0,
) -> Set[int]:
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


def compute_recall_f1(
    pred_set: Set[int],
    ref_set: Set[int],
) -> Tuple[Optional[float], Optional[float]]:
    """recall と F1 のみ返す（precision/jaccard は削除）"""
    if not ref_set:
        return None, None
    inter = pred_set & ref_set
    recall = len(inter) / len(ref_set)
    precision = len(inter) / len(pred_set) if pred_set else 0.0
    f1 = (2 * recall * precision / (recall + precision)
          if (recall + precision) > 0 else 0.0)
    return recall, f1


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
    diffs = pep_coords[:, None, :] - poc_coords[None, :, :]
    min_dist = float(np.linalg.norm(diffs, axis=-1).min())

    return centroid_dist, min_dist


# =========================
# (b) holo リガンド原子座標を取得
# =========================
def collect_ligand_coords(
    holo_structure: Structure,
    ligand_name: Optional[str] = None,
) -> np.ndarray:
    """
    holo PDB から water/ion 以外の HETATM 原子座標を返す。
    ligand_name を指定すればその残基名のみ。
    """
    model = next(holo_structure.get_models())
    coords = []
    for ch in model:
        for res in ch:
            hetflag = res.id[0]
            resname = res.get_resname().strip()
            if hetflag == " ":
                continue
            if resname in NON_LIGAND_HET:
                continue
            if ligand_name is not None and resname != ligand_name:
                continue
            for atom in res:
                coords.append(atom.get_coord())
    return np.array(coords) if coords else np.empty((0, 3))


def compute_holo_ligand_centroid_distance(
    pred_structure: Structure,
    holo_structure: Structure,
    peptide_chain: str,
    receptor_chain_pred: str,
    receptor_chain_holo: str,
    ligand_name: Optional[str] = None,
) -> Optional[float]:
    """
    ペプチド重心 ↔ holo リガンド重心 の距離 (Å)。

    手順:
      1. predicted 構造のreceptor鎖 と holo 構造の receptor 鎖を CA でスーパーインポーズ
      2. スーパーインポーズ後の holo リガンド座標を取得
      3. ペプチド重心との距離を計算
    """
    # --- receptor 鎖の CA を取得 ---
    pred_model = next(pred_structure.get_models())
    holo_model = next(holo_structure.get_models())

    if receptor_chain_pred not in pred_model or receptor_chain_holo not in holo_model:
        return None

    pred_ca, holo_ca = [], []
    pred_res_dict: Dict[int, Residue] = {}
    holo_res_dict: Dict[int, Residue] = {}

    for res in pred_model[receptor_chain_pred]:
        if res.id[0] == " " and res.get_resname().strip() in STANDARD_AAS and "CA" in res:
            pred_res_dict[res.id[1]] = res
    for res in holo_model[receptor_chain_holo]:
        if res.id[0] == " " and res.get_resname().strip() in STANDARD_AAS and "CA" in res:
            holo_res_dict[res.id[1]] = res

    common_resseqs = sorted(set(pred_res_dict.keys()) & set(holo_res_dict.keys()))
    if len(common_resseqs) < 3:
        return None

    fixed_atoms = [pred_res_dict[r]["CA"] for r in common_resseqs]
    moving_atoms = [holo_res_dict[r]["CA"] for r in common_resseqs]

    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)
    # holo 構造の全原子に回転行列を適用
    sup.apply(list(holo_model.get_atoms()))

    # --- スーパーインポーズ後のリガンド座標 ---
    ligand_coords = collect_ligand_coords(holo_structure, ligand_name)
    if ligand_coords.shape[0] == 0:
        return None

    # --- ペプチド重心 ---
    if peptide_chain not in pred_model:
        return None
    pep_coords = []
    for res in pred_model[peptide_chain]:
        if res.id[0] != " ":
            continue
        for atom in res:
            pep_coords.append(atom.get_coord())
    if not pep_coords:
        return None

    pep_centroid = np.mean(pep_coords, axis=0)
    lig_centroid = ligand_coords.mean(axis=0)
    return float(np.linalg.norm(pep_centroid - lig_centroid))


# =========================
# (c) apo に holo ligand を重ね合わせて 3Å 接触判定
# =========================
def compute_cryptic_contact_with_apo_ligand(
    apo_structure: Structure,
    holo_structure: Structure,
    cryptic_resseqs: Set[int],
    apo_chain: str,
    holo_chain: str,
    ligand_name: Optional[str] = None,
    contact_cutoff: float = 3.0,
) -> Optional[bool]:
    """
    apo 構造に holo リガンドを重ね合わせたとき、
    cryptic 残基のいずれかの原子が ligand から contact_cutoff Å 以内にあるか判定。

    手順:
      1. apo receptor と holo receptor を CA でスーパーインポーズ
      2. スーパーインポーズ後の holo リガンド座標を得る
      3. apo の cryptic 残基原子との最短距離を計算
      4. contact_cutoff 以内の原子ペアが1つでもあれば True
    """
    if not cryptic_resseqs:
        return None

    apo_model = next(apo_structure.get_models())
    holo_model = next(holo_structure.get_models())

    if apo_chain not in apo_model or holo_chain not in holo_model:
        return None

    # --- CA 共通残基でアライン ---
    apo_res_dict: Dict[int, Residue] = {}
    holo_res_dict: Dict[int, Residue] = {}
    for res in apo_model[apo_chain]:
        if res.id[0] == " " and res.get_resname().strip() in STANDARD_AAS and "CA" in res:
            apo_res_dict[res.id[1]] = res
    for res in holo_model[holo_chain]:
        if res.id[0] == " " and res.get_resname().strip() in STANDARD_AAS and "CA" in res:
            holo_res_dict[res.id[1]] = res

    common = sorted(set(apo_res_dict.keys()) & set(holo_res_dict.keys()))
    if len(common) < 3:
        return None

    fixed_atoms  = [apo_res_dict[r]["CA"]  for r in common]
    moving_atoms = [holo_res_dict[r]["CA"] for r in common]

    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)
    sup.apply(list(holo_model.get_atoms()))

    # --- スーパーインポーズ後のリガンド座標 ---
    lig_coords = collect_ligand_coords(holo_structure, ligand_name)
    if lig_coords.shape[0] == 0:
        return None

    # --- apo の cryptic 残基原子座標 ---
    cryptic_coords = []
    for res in apo_model[apo_chain]:
        if res.id[0] == " " and res.id[1] in cryptic_resseqs:
            for atom in res:
                cryptic_coords.append(atom.get_coord())
    if not cryptic_coords:
        return None

    cryptic_coords = np.array(cryptic_coords)
    # 全ペアの最短距離
    diffs = cryptic_coords[:, None, :] - lig_coords[None, :, :]
    min_dist = float(np.linalg.norm(diffs, axis=-1).min())
    return min_dist <= contact_cutoff


# =========================
# (c) fpocket
# =========================
def run_fpocket(input_pdb: Path, output_dir: Path) -> Optional[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_pdb = output_dir / input_pdb.name
    if target_pdb.resolve() != input_pdb.resolve():
        shutil.copy(input_pdb, target_pdb)

    expected_out = output_dir / f"{target_pdb.stem}_out"
    if expected_out.exists():
        return expected_out

    try:
        subprocess.run(
            ["fpocket", "-f", str(target_pdb)],
            check=True, capture_output=True, text=True, cwd=str(output_dir),
        )
    except FileNotFoundError:
        warnings.warn("fpocket not found in PATH")
        return None
    except subprocess.CalledProcessError as e:
        warnings.warn(f"fpocket failed: {e.stderr}")
        return None

    return expected_out if expected_out.exists() else None


def parse_fpocket_info(fpocket_out_dir: Path) -> List[Dict]:
    info_files = list(fpocket_out_dir.glob("*_info.txt"))
    if not info_files:
        return []
    pockets = []
    current = None
    with open(info_files[0], "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("Pocket "):
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
                val = val.replace("\t", "").strip()
                try:
                    val_f = float(val)
                except ValueError:
                    continue
                key_map = {
                    "Druggability Score": "druggability",
                    "Score": "score",
                    "Volume": "volume",
                    "Real volume (approximation)": "volume",
                    "Pocket volume (Monte Carlo)": "volume_mc",
                    "Pocket volume (convex hull)": "volume_convex",
                    "Number of Alpha Spheres": "n_alpha_spheres",
                }
                current[key_map.get(key, key)] = val_f
    if current is not None:
        pockets.append(current)
    return pockets


def get_pocket_residue_sets(fpocket_out_dir: Path, target_chain: str) -> Dict[int, Set[int]]:
    pockets_dir = fpocket_out_dir / "pockets"
    if not pockets_dir.exists():
        return {}
    out: Dict[int, Set[int]] = {}
    for atm_pdb in pockets_dir.glob("pocket*_atm.pdb"):
        stem = atm_pdb.stem
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


def find_best_pocket_for_residues(
    pocket_residues: Dict[int, Set[int]],
    target_resseqs: Set[int],
    pockets_info: List[Dict],
) -> Optional[Dict]:
    """
    target_resseqs と最も overlap が大きい pocket を選び、その info を返す。
    overlap=0 のときは None。
    """
    if not pocket_residues or not target_resseqs:
        return None
    best_pid, best_overlap = None, 0
    for pid, residues in pocket_residues.items():
        overlap = len(residues & target_resseqs)
        if overlap > best_overlap:
            best_overlap = overlap
            best_pid = pid
    if best_pid is None:
        return None
    for info in pockets_info:
        if info.get("pocket_id") == best_pid:
            return info
    return None


# 後方互換エイリアス
find_best_pocket_for_cryptic = find_best_pocket_for_residues


def extract_target_chain_pdb(input_structure: Path, output_pdb: Path, target_chain: str) -> Path:
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
# RSA (Tien 2013 正規化)
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
    if not FREESASA_AVAILABLE:
        return {}
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
            rsa = min(asa / max_asa, 1.0) if max_asa and max_asa > 0 else asa
            out[resnum] = rsa
        except (ValueError, AttributeError):
            continue
    return out


# =========================
# backbone RMSD (apo vs predicted)
# =========================
def compute_backbone_rmsd_cryptic(
    apo_pdb: Path,
    predicted_pdb: Path,
    target_chain_apo: str,
    target_chain_pred: str,
    cryptic_resseqs: Set[int],
) -> Optional[float]:
    """backbone (N,CA,C,O) RMSD of cryptic residues after global CA superimpose"""
    if not cryptic_resseqs:
        return None

    s1 = load_structure(str(apo_pdb), name="apo")
    s2 = load_structure(str(predicted_pdb), name="pred")

    try:
        c1 = get_chain(s1, target_chain_apo)
        c2 = get_chain(s2, target_chain_pred)
    except ValueError:
        return None

    d1 = get_residue_dict(c1)
    d2 = get_residue_dict(c2)

    common = sorted(set(d1.keys()) & set(d2.keys()))
    fixed_atoms, moving_atoms = [], []
    for rsq in common:
        if "CA" in d1[rsq] and "CA" in d2[rsq]:
            fixed_atoms.append(d1[rsq]["CA"])
            moving_atoms.append(d2[rsq]["CA"])

    if len(fixed_atoms) < 3:
        return None

    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)
    sup.apply(s2.get_atoms())

    bb_names = {"N", "CA", "C", "O"}
    bb_d1, bb_d2 = [], []
    for rsq in sorted(cryptic_resseqs):
        if rsq not in d1 or rsq not in d2:
            continue
        r1, r2 = d1[rsq], d2[rsq]
        for atom_name in [a.name for a in r1]:
            if atom_name.startswith("H") or atom_name not in bb_names:
                continue
            if atom_name not in r2:
                continue
            bb_d1.append(r1[atom_name].get_coord())
            bb_d2.append(r2[atom_name].get_coord())

    return _rmsd(bb_d1, bb_d2) if bb_d1 else None


# =========================
# ペプチド配列の抽出
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
    apo_pdb: Path,
    holo_pdb: Optional[Path],
    cryptic_def: CrypticDefinition,
    target_chain: str,
    peptide_chain: str,
    fpocket_cache_dir: Path,
    interface_cutoff: float,
    apo_chain: str,
    holo_chain: str,
    holo_ligand_name: Optional[str],
    cryptobank_dir: Optional[Path] = None,
    cryptobank_n_lig_splits: int = 1,
) -> EvaluationResult:
    res = EvaluationResult(name=name)
    cryptic_resseqs = cryptic_def.residues

    if not predicted_pdb.exists():
        res.errors.append(f"predicted structure not found: {predicted_pdb}")
        return res

    pred_struct = load_structure(str(predicted_pdb), name=name)
    warn_multiple_chains(pred_struct, target_chain, "predicted", res)

    res.peptide_sequence = extract_peptide_sequence(pred_struct, peptide_chain)

    interface = compute_interface_residues(
        pred_struct, target_chain, peptide_chain, cutoff=interface_cutoff,
    )
    res.interface_residues = sorted(interface)
    res.n_interface_residues = len(interface)

    # (a) 信頼度
    parse_boltz_confidence(confidence_json, res)
    parse_per_residue_plddt(plddt_npz, pred_struct, peptide_chain, res)

    # (b) cryptic overlap — recall & f1 のみ
    if cryptic_resseqs:
        recall, f1 = compute_recall_f1(interface, cryptic_resseqs)
        res.cryptic_recall = recall
        res.cryptic_f1 = f1

        cd, mind = compute_centroid_distance(
            pred_struct, target_chain, peptide_chain, cryptic_resseqs,
        )
        res.peptide_centroid_to_cryptic_centroid_dist = cd
        res.min_dist_peptide_to_cryptic = mind

    # (b) ペプチド重心 ↔ holo リガンド重心
    if holo_pdb is not None and holo_pdb.exists():
        try:
            holo_struct = load_structure(str(holo_pdb), name="holo")
            warn_multiple_chains(holo_struct, holo_chain, "holo", res)
            dist = compute_holo_ligand_centroid_distance(
                pred_struct, holo_struct,
                peptide_chain, target_chain, holo_chain,
                holo_ligand_name,
            )
            res.peptide_centroid_to_holo_ligand_centroid_dist = dist
            if dist is None:
                res.warnings_list.append("holo ligand centroid distance: no ligand found or alignment failed")
        except Exception as e:
            res.warnings_list.append(f"holo ligand centroid distance failed: {e}")

    # (c) predicted 構造からターゲット鎖のみ抽出
    target_only_dir = fpocket_cache_dir / "predicted" / name
    target_only_dir.mkdir(parents=True, exist_ok=True)
    target_only_pdb = target_only_dir / f"{name}_targetonly.pdb"
    try:
        target_only_pdb = extract_target_chain_pdb(predicted_pdb, target_only_pdb, target_chain)
    except Exception as e:
        res.warnings_list.append(f"target chain extraction failed: {e}")
        target_only_pdb = None

    # fpocket on predicted: 「ペプチドが実際に結合した界面残基」ベースで pocket 選択
    pred_vol = None
    if target_only_pdb is not None and target_only_pdb.exists():
        pred_fp_dir = run_fpocket(target_only_pdb, target_only_dir)
        if pred_fp_dir is not None:
            pockets_info = parse_fpocket_info(pred_fp_dir)
            pocket_residues = get_pocket_residue_sets(pred_fp_dir, target_chain)
            # interface(=ペプチド結合残基) と最大 overlap する pocket
            best = find_best_pocket_for_residues(pocket_residues, interface, pockets_info)
            if best is not None:
                pred_vol = best.get("volume") or best.get("volume_mc")
            else:
                res.warnings_list.append(
                    "no fpocket overlaps with peptide interface residues in predicted"
                )

    res.fpocket_volume_predicted = pred_vol

    # fpocket on apo: 同じ「ペプチド界面残基」(predicted で得た) と overlap する pocket を選択
    # → apo 上で同じ場所のポケット体積を取り、predicted との変化を見る
    apo_vol = None
    apo_cache_dir = fpocket_cache_dir / "apo"
    apo_cache_dir.mkdir(parents=True, exist_ok=True)
    apo_targetonly = apo_cache_dir / f"{apo_pdb.stem}_targetonly.pdb"
    if not apo_targetonly.exists():
        try:
            apo_targetonly = extract_target_chain_pdb(apo_pdb, apo_targetonly, apo_chain)
        except Exception as e:
            res.warnings_list.append(f"apo extraction failed: {e}")
            apo_targetonly = None

    if apo_targetonly is not None and apo_targetonly.exists() and interface:
        apo_fp_dir = run_fpocket(apo_targetonly, apo_cache_dir)
        if apo_fp_dir is not None:
            pockets_info = parse_fpocket_info(apo_fp_dir)
            pocket_residues = get_pocket_residue_sets(apo_fp_dir, apo_chain)
            best = find_best_pocket_for_residues(pocket_residues, interface, pockets_info)
            if best is not None:
                apo_vol = best.get("volume") or best.get("volume_mc")
            else:
                res.warnings_list.append(
                    "no fpocket overlaps with peptide interface residues in apo"
                )

    res.fpocket_volume_apo = apo_vol

    if apo_vol is not None and pred_vol is not None:
        res.delta_volume = pred_vol - apo_vol

    # RSA (Tien2013 正規化)
    if FREESASA_AVAILABLE and cryptic_resseqs:
        try:
            apo_struct_for_resnames = load_structure(str(apo_pdb), name="apo_rsa")
            pred_pdb_for_sasa = (target_only_pdb
                                 if target_only_pdb is not None and target_only_pdb.exists()
                                 else predicted_pdb)
            apo_rsa = compute_residue_rsa(apo_pdb, apo_chain, apo_struct_for_resnames)
            pred_struct_for_resnames = load_structure(str(pred_pdb_for_sasa), name="pred_rsa")
            pred_rsa = compute_residue_rsa(pred_pdb_for_sasa, target_chain, pred_struct_for_resnames)

            apo_rsa_vals  = [apo_rsa[r]  for r in cryptic_resseqs if r in apo_rsa]
            pred_rsa_vals = [pred_rsa[r] for r in cryptic_resseqs if r in pred_rsa]

            if apo_rsa_vals:
                res.cryptic_rsa_apo = float(np.mean(apo_rsa_vals))
            if pred_rsa_vals:
                res.cryptic_rsa_predicted = float(np.mean(pred_rsa_vals))
            if res.cryptic_rsa_apo is not None and res.cryptic_rsa_predicted is not None:
                res.delta_cryptic_rsa = res.cryptic_rsa_predicted - res.cryptic_rsa_apo
        except Exception as e:
            res.warnings_list.append(f"RSA failed: {e}")

    # backbone RMSD (apo vs predicted)
    if target_only_pdb is not None and target_only_pdb.exists():
        bb = compute_backbone_rmsd_cryptic(
            apo_pdb, target_only_pdb, apo_chain, target_chain, cryptic_resseqs,
        )
        res.cryptic_backbone_rmsd_vs_apo = bb

    # (c) CryptoBank crypticity score (同心球シェルモデル)
    if cryptobank_dir is not None:
        try:
            from evaluate_boltz_predictions_patch import compute_cryptobank_crypticity
            cb = compute_cryptobank_crypticity(
                predicted_pdb              = str(predicted_pdb),
                apo_pdb                    = str(apo_pdb),
                cryptobank_dir             = str(cryptobank_dir),
                predicted_receptor_chain_id= target_chain,
                predicted_peptide_chain_id = peptide_chain,
                apo_receptor_chain_id      = apo_chain,
                n_lig_splits               = cryptobank_n_lig_splits,
            )
            res.crypticity_score         = cb["crypticity_score"]
            res.is_cryptic               = cb["is_cryptic"]
            res.crypticity_rms_apo_align = cb["rms_apo_align"]
        except Exception as e:
            res.warnings_list.append(f"CryptoBank crypticity score failed: {e}")

    return res


# =========================
# Boltz 出力のスキャン
# =========================
def discover_predictions(boltz_output_root: Path) -> List[Tuple[str, Path, Path, Path]]:
    found = []
    for results_dir in sorted(boltz_output_root.glob("boltz_results_*")):
        if not results_dir.is_dir():
            continue
        predictions_root = results_dir / "predictions"
        if not predictions_root.is_dir():
            continue
        for pred_dir in sorted(predictions_root.iterdir()):
            if not pred_dir.is_dir():
                continue
            name = pred_dir.name
            pdb = pred_dir / f"{name}_model_0.pdb"
            cif = pred_dir / f"{name}_model_0.cif"
            if pdb.exists():
                struct_path = pdb
            elif cif.exists():
                struct_path = cif
            else:
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

    fieldnames = [
        "name", "peptide_sequence",
        # (a) 信頼度
        "confidence_score", "iptm", "receptor_to_peptide_iptm", "peptide_plddt_mean",
        # (b) 結合位置
        "n_interface_residues", "interface_residues",
        "cryptic_recall", "cryptic_f1",
        "peptide_centroid_to_cryptic_centroid_dist", "min_dist_peptide_to_cryptic",
        "peptide_centroid_to_holo_ligand_centroid_dist",
        # (c) Crypticity
        "fpocket_volume_predicted", "fpocket_volume_apo", "delta_volume",
        "cryptic_rsa_apo", "cryptic_rsa_predicted", "delta_cryptic_rsa",
        "cryptic_backbone_rmsd_vs_apo",
        "crypticity_score",
        "is_cryptic",
        "crypticity_rms_apo_align",
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
            row["interface_residues"] = ";".join(str(x) for x in d["interface_residues"])
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
    parser.add_argument("--boltz_output_root", required=True)
    parser.add_argument("--apo_pdb", required=True)
    parser.add_argument("--holo_pdb", default=None)
    parser.add_argument("--cryptic_residues", default=None)
    parser.add_argument("--cryptic_scores_json", default=None)
    parser.add_argument("--cryptic_threshold", type=float, default=None)
    parser.add_argument("--target_chain", default="A")
    parser.add_argument("--peptide_chain", default="B")
    parser.add_argument("--apo_chain", default="A")
    parser.add_argument("--holo_chain", default="A")
    parser.add_argument("--holo_ligand_name", default=None)
    parser.add_argument("--interface_cutoff", type=float, default=5.0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cryptobank_dir", default=None,
                        help="CryptoBank リポジトリのルートパス（scoring_function.py がある場所）")
    parser.add_argument("--cryptobank_n_lig_splits", type=int, default=1,
                        help="CryptoBank のリガンド分割数 (1 or 3)。ペプチド ≤20残基なら 1 推奨")
    args = parser.parse_args()

    boltz_root = Path(args.boltz_output_root)
    apo_pdb    = Path(args.apo_pdb)
    holo_pdb   = Path(args.holo_pdb) if args.holo_pdb else None
    output_dir = Path(args.output_dir)
    cryptobank_dir = Path(args.cryptobank_dir) if args.cryptobank_dir else None
    if cryptobank_dir is not None and not cryptobank_dir.is_dir():
        print(f"⚠ CryptoBank dir not found, skipping crypticity score: {cryptobank_dir}")
        cryptobank_dir = None

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
    per_peptide_dir  = output_dir / "per_peptide"
    per_peptide_dir.mkdir(exist_ok=True)
    fpocket_cache_dir = output_dir / "fpocket_cache"
    fpocket_cache_dir.mkdir(exist_ok=True)

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
        "cryptobank_dir": str(cryptobank_dir) if cryptobank_dir else None,
        "cryptobank_n_lig_splits": args.cryptobank_n_lig_splits,
    }
    with open(output_dir / "evaluation_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    predictions = discover_predictions(boltz_root)
    if not predictions:
        sys.exit(f"No boltz_results_* found in {boltz_root}")
    print(f"Found {len(predictions)} predictions")

    all_results: List[EvaluationResult] = []
    for i, (name, pdb, conf, plddt) in enumerate(predictions, 1):
        print(f"[{i}/{len(predictions)}] {name}")
        res = evaluate_one(
            name=name, predicted_pdb=pdb, confidence_json=conf,
            plddt_npz=plddt, apo_pdb=apo_pdb, holo_pdb=holo_pdb,
            cryptic_def=cryptic_def, target_chain=args.target_chain,
            peptide_chain=args.peptide_chain, fpocket_cache_dir=fpocket_cache_dir,
            interface_cutoff=args.interface_cutoff,
            apo_chain=args.apo_chain, holo_chain=args.holo_chain,
            holo_ligand_name=args.holo_ligand_name,
            cryptobank_dir=cryptobank_dir,
            cryptobank_n_lig_splits=args.cryptobank_n_lig_splits,
        )
        all_results.append(res)
        with open(per_peptide_dir / f"{name}.json", "w") as f:
            json.dump(asdict(res), f, indent=2, default=str)
        for e in res.errors:
            print(f"    ❌ {e}")

    summary_csv = output_dir / "evaluation_summary.csv"
    write_summary_csv(all_results, summary_csv)
    print(f"\n✅ Wrote summary: {summary_csv}")
    print(f"✅ Per-peptide JSONs: {per_peptide_dir}")


if __name__ == "__main__":
    main()
