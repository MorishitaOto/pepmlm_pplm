import subprocess
import tempfile
import shutil
import os
import sys
import pathlib
import subprocess
from typing import List


def mmseqs_cluster_from_sequences(sequences: List[str], min_seq_id: float = 0.9, coverage: float = 0.8) -> int:
    """
    sequences: List[str]
    return: num_clusters
    """

    with tempfile.TemporaryDirectory() as work_dir:
        fasta_path = os.path.join(work_dir, "input.fasta")
        output_prefix = os.path.join(work_dir, "result")
        tmp_dir = os.path.join(work_dir, "tmp")

        # --- FASTA ---
        with open(fasta_path, "w") as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq_{i}\n{seq}\n")

        # --- mmseqs easy-cluster ---
        cmd = [
            "mmseqs", "easy-cluster",
            fasta_path,
            output_prefix,
            tmp_dir,
            "--min-seq-id", str(min_seq_id),
            "-c", str(coverage),
            "--cov-mode", "1",
            "--mask", "0"
        ]
        subprocess.run(cmd, check=True)

        # --- cluster.tsv ---
        cluster_tsv = f"{output_prefix}_cluster.tsv"

        cluster_map = {}
        with open(cluster_tsv) as f:
            for line in f:
                rep, member = line.strip().split("\t")
                if rep not in cluster_map:
                    cluster_map[rep] = []
                cluster_map[rep].append(member)

        return len(cluster_map)
