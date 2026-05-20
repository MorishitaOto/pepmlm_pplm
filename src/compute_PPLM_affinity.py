from pathlib import Path
import subprocess
import tempfile
import re
import time
from PPLM.pplm_affinity_predictor import PPLMPredictor
#from PPLM.pplm_affinity_predictor_model_3 import PPLMPredictor

class PPLMAffinityPredictor:
    def __init__(
        self,
        pplm_script: str | Path = "/home/users/gds/src/PPLM/run_pplm-affinity.py",
        python_cmd: str = "/home/users/gds/.conda/envs/pplm/bin/python3",
        verbose: bool = False,
    ):
        self.pplm_script = Path(pplm_script)
        self.python_cmd = python_cmd
        self.verbose = verbose
        self.pplm_predictor = PPLMPredictor(gpu_id=0)


    # -----------------------------
    # 1. 一時 FASTA 作成
    # -----------------------------
    def _write_fasta(self, path: Path, header: str, seq: str):
        path.write_text(f">{header}\n{seq}\n")

    # -----------------------------
    # 2. PPLM 実行
    # -----------------------------
    def run_pplm(
        self,
        protein_seq: str,
        peptide_seq: str,
    ) -> float:
        """
        PPLM を実行し、標準出力から affinity を抽出する
        """

        # <start> トークンがある場合は除去
        if peptide_seq.startswith("<start>"):
            peptide_seq = peptide_seq[len("<start>"):]
        peptide_seq = peptide_seq.strip()
        peptide_seq = peptide_seq.replace("\n", "")


        print(f"peptide_seq:{peptide_seq}")


        affinity = self.pplm_predictor.predict(protein_seq, peptide_seq)
        return affinity







        """


        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            protein_fasta = tmpdir / "protein.fasta"
            peptide_fasta = tmpdir / "peptide.fasta"

            self._write_fasta(protein_fasta, "protein", protein_seq)
            self._write_fasta(peptide_fasta, "peptide", peptide_seq)

            cmd = [
                self.python_cmd,
                str(self.pplm_script),
                str(protein_fasta),
                str(peptide_fasta),
            ]

            if self.verbose:
                print("Running:", " ".join(cmd))
                
            try:
                time.sleep(10.0)
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return self._parse_affinity(result.stdout)

            except subprocess.CalledProcessError as e:
                print("PPLM failed!")
                print("STDOUT:", e.stdout)
                print("STDERR:", e.stderr)
                """ 

    # -----------------------------
    # 3. affinity 抽出
    # -----------------------------
    def _parse_affinity(self, stdout: str) -> float:
        """
        例:
        Predicted binding affinity: -7.602727
        """
        match = re.search(
            r"Predicted binding affinity:\s*([-+]?\d*\.\d+|\d+)",
            stdout,
        )
        if not match:
            raise ValueError(
                f"Affinity not found in PPLM output:\n{stdout}"
            )

        return float(match.group(1))

    # -----------------------------
    # 4. 外部から使うメインAPI
    # -----------------------------
    def predict_affinity(
        self,
        protein_seq: str,
        peptide_seq: str,
        step: int | None = None,
    ) -> float:
        """
        step は互換性のため残しているが、PPLMでは未使用
        """
        return self.run_pplm(protein_seq, peptide_seq)


import sys
import csv

def recursive_predict(predictor, data, idx=0, results=None):
    """
    data: [(protein, peptide), ...]
    """
    if results is None:
        results = []

    # 終了条件
    if idx >= len(data):
        return results

    protein_seq, peptide_seq = data[idx]

    print(f"\n--- Prediction {idx} ---")
    affinity = predictor.predict_affinity(protein_seq, peptide_seq)
    print(f"Affinity: {affinity}")

    results.append(affinity)

    # 再帰
    return recursive_predict(predictor, data, idx + 1, results)


if __name__ == "__main__":
    """
    使い方:
    python script.py input.csv
    """

    if len(sys.argv) < 2:
        print("Usage: python script.py input.csv")
        sys.exit(1)

    input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)

    data = []

    # CSV読み込み
    with open(input_file, newline="") as f:
        reader = csv.DictReader(f)

        if "protein" not in reader.fieldnames or "peptide" not in reader.fieldnames:
            raise ValueError("CSV must have 'protein' and 'peptide' columns")

        for row in reader:
            protein_seq = row["protein"].strip()
            peptide_seq = row["peptide"].strip()

            if protein_seq and peptide_seq:
                data.append((protein_seq, peptide_seq))

    predictor = PPLMAffinityPredictor(verbose=True)

    results = recursive_predict(predictor, data)

    print("\n=== All Results ===")
    for i, val in enumerate(results):
        print(f"{i}: {val}")
        print(f"average: {sum(results) / len(results)}")
