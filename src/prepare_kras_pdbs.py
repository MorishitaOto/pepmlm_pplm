#!/usr/bin/env python3
"""
prepare_kras_pdbs.py
====================
4LDJ (KRAS G12C, apo) と 5XCO (KRAS G12D + KRpep-2d, holo) を
RCSB からダウンロードし、パイプライン用にクリーニングして

    data/pdb/4LDJ_apo.pdb
    data/pdb/5XCO_holo.pdb

に保存するスクリプト。

使い方 (pepmlm_pplm/ ディレクトリで実行):
    python src/prepare_kras_pdbs.py

オプション:
    --out_dir   出力先ディレクトリ (default: data/pdb)
    --raw_dir   RAW PDB の保存先   (default: /tmp/kras_raw)
    --skip_download  既に RAW PDB がある場合はダウンロードをスキップ
    --check_only    前処理後の検証結果だけ出力して終了 (ファイルは生成済みが前提)
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

try:
    from Bio.PDB import PDBParser, PDBIO, Select
    from Bio.PDB.Structure import Structure
except ImportError:
    sys.exit("BioPython が必要です: pip install biopython")


# ============================================================
# RCSB ダウンロード
# ============================================================
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def download_pdb(pdb_id: str, dest: Path, skip: bool = False) -> Path:
    out = dest / f"{pdb_id}_raw.pdb"
    if skip and out.exists():
        print(f"  [skip] {out} already exists")
        return out
    url = RCSB_URL.format(pdb_id=pdb_id)
    print(f"  Downloading {url} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, out)
    print(f"done ({out.stat().st_size // 1024} KB)")
    return out


# ============================================================
# Selector クラス
# ============================================================
STANDARD_AAS = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}


class ApoSelect(Select):
    """
    4LDJ 用:
    - Chain A のみ
    - 標準アミノ酸 (HETATM / HOH / GDP / MG を除外)
    - 残基番号 >= 1 (Gly0 発現タグを除外)
    - 代替コンフォメーション A のみ採用
    """
    def accept_chain(self, chain):
        return chain.id == "A"

    def accept_residue(self, res):
        hetflag = res.id[0]
        resnum  = res.id[1]
        resname = res.get_resname().strip()
        return hetflag == " " and resnum >= 1 and resname in STANDARD_AAS

    def accept_atom(self, atom):
        altloc = atom.get_altloc()
        return altloc in (" ", "A")


class HoloSelect(Select):
    """
    5XCO 用:
    - Chain A (receptor) と Chain B (ペプチド) のみ
    - 標準アミノ酸のみ (ACE/NH2 末端修飾・GDP・MG・HOH を除外)
    - Chain A: 残基番号 >= 1 (Gly0 発現タグを除外)
    - 代替コンフォメーション A のみ採用
    """
    def accept_chain(self, chain):
        return chain.id in ("A", "B")

    def accept_residue(self, res):
        hetflag = res.id[0]
        resnum  = res.id[1]
        resname = res.get_resname().strip()
        chain_id = res.get_parent().id

        if hetflag != " ":          # HETATM (GDP, MG, ACE, NH2, HOH など)
            return False
        if resname not in STANDARD_AAS:
            return False
        if chain_id == "A" and resnum < 1:   # Gly0 除外
            return False
        return True

    def accept_atom(self, atom):
        altloc = atom.get_altloc()
        return altloc in (" ", "A")


# ============================================================
# 前処理本体
# ============================================================
def process_apo(raw_pdb: Path, out_pdb: Path) -> Structure:
    print(f"  Parsing {raw_pdb} ...")
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("4LDJ", str(raw_pdb))

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(struct)
    io.save(str(out_pdb), ApoSelect())
    print(f"  Saved -> {out_pdb}")
    return struct


def process_holo(raw_pdb: Path, out_pdb: Path) -> Structure:
    print(f"  Parsing {raw_pdb} ...")
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("5XCO", str(raw_pdb))

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(struct)
    io.save(str(out_pdb), HoloSelect())
    print(f"  Saved -> {out_pdb}")
    return struct


# ============================================================
# 検証
# ============================================================
def verify_pdb(pdb_path: Path, label: str) -> bool:
    """保存した PDB を BioPython で再読みして基本検証"""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(label, str(pdb_path))
    model  = next(struct.get_models())

    chains = list(model.get_chains())
    chain_ids = [c.id for c in chains]

    ok = True
    issues = []

    for chain in chains:
        residues = [r for r in chain if r.id[0] == " " and r.get_resname().strip() in STANDARD_AAS]
        resids   = [r.id[1] for r in residues]
        resnames = [r.get_resname().strip() for r in residues]

        if not residues:
            issues.append(f"Chain {chain.id}: 標準 AA 残基が 0 件")
            ok = False
            continue

        # 先頭残基が Met かどうか
        first_resname = resnames[0]
        first_resid   = resids[0]
        if first_resname != "MET":
            issues.append(f"Chain {chain.id}: 先頭残基が MET ではなく {first_resname} (res {first_resid})")
            ok = False

        # 残基番号が 1 から始まるか
        if first_resid != 1:
            issues.append(f"Chain {chain.id}: 先頭残基番号が {first_resid} (1 を期待)")
            ok = False

        # Gly0 が混入していないか
        if 0 in resids:
            issues.append(f"Chain {chain.id}: 残基番号 0 (Gly0 発現タグ) が残っている!")
            ok = False

        # HETATM 混入チェック
        hets = [r for r in chain if r.id[0] != " "]
        if hets:
            het_names = list({r.get_resname().strip() for r in hets})
            issues.append(f"Chain {chain.id}: HETATM が残っている: {het_names}")
            ok = False

        print(f"    Chain {chain.id}: {len(residues)} AA残基, "
              f"res {resids[0]}–{resids[-1]}, "
              f"先頭={first_resname}, 末端={resnames[-1]}")

    # HETATM が構造全体に残っていないか (ATOM/HETATM 行カウント)
    atom_count = sum(1 for ch in model for r in ch for _ in r)
    print(f"    Chains: {chain_ids}, total atoms: {atom_count}")

    if issues:
        for iss in issues:
            print(f"    ⚠ {iss}")
    else:
        print(f"    ✅ 検証 OK")

    return ok


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Prepare 4LDJ_apo.pdb and 5XCO_holo.pdb")
    parser.add_argument("--out_dir",        default="data/pdb",    help="出力ディレクトリ")
    parser.add_argument("--raw_dir",        default="/tmp/kras_raw", help="RAW PDB 保存先")
    parser.add_argument("--skip_download",  action="store_true",   help="ダウンロードをスキップ")
    parser.add_argument("--check_only",     action="store_true",   help="検証だけ実行")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    apo_out  = out_dir / "4LDJ_apo.pdb"
    holo_out = out_dir / "5XCO_holo.pdb"

    # -------- check_only モード --------
    if args.check_only:
        print("\n=== 検証モード (--check_only) ===")
        all_ok = True
        for path, label in [(apo_out, "4LDJ_apo"), (holo_out, "5XCO_holo")]:
            print(f"\n[{label}] {path}")
            if not path.exists():
                print(f"  ❌ ファイルが存在しません: {path}")
                all_ok = False
            else:
                ok = verify_pdb(path, label)
                if not ok:
                    all_ok = False
        sys.exit(0 if all_ok else 1)

    # -------- 4LDJ (apo) --------
    print("\n" + "=" * 50)
    print("4LDJ: KRAS G12C (apo)")
    print("=" * 50)
    raw_4ldj = download_pdb("4LDJ", raw_dir, skip=args.skip_download)
    process_apo(raw_4ldj, apo_out)
    print("\n  [検証]")
    apo_ok = verify_pdb(apo_out, "4LDJ_apo")

    # -------- 5XCO (holo) --------
    print("\n" + "=" * 50)
    print("5XCO: KRAS G12D + KRpep-2d (holo)")
    print("=" * 50)
    raw_5xco = download_pdb("5XCO", raw_dir, skip=args.skip_download)
    process_holo(raw_5xco, holo_out)
    print("\n  [検証]")
    holo_ok = verify_pdb(holo_out, "5XCO_holo")

    # -------- サマリ --------
    print("\n" + "=" * 50)
    print("完了サマリ")
    print("=" * 50)
    for path, ok, label in [
        (apo_out,  apo_ok,  "4LDJ_apo.pdb"),
        (holo_out, holo_ok, "5XCO_holo.pdb"),
    ]:
        status = "✅" if ok else "⚠ 要確認"
        size_kb = path.stat().st_size // 1024 if path.exists() else 0
        print(f"  {status}  {path}  ({size_kb} KB)")

    print()
    if not (apo_ok and holo_ok):
        print("⚠ 一部の検証で警告が出ました。上記メッセージを確認してください。")
        sys.exit(1)
    else:
        print("すべて正常に完了しました。")
        print()
        print("次のステップ:")
        print("  python src/prepare_kras_pdbs.py --check_only  # 再検証")
        print("  → 問題なければ cryptic_json の準備へ")


if __name__ == "__main__":
    main()
