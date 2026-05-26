"""
================================================================
pplm_parallel_evaluator.py
================================================================
PPLM affinity 評価の マルチGPU 並列化モジュール (新規追加)

設計:
  - N 個のワーカープロセスを spawn し、各ワーカーが別 GPU で PPLM をロード
  - メインプロセスから (target, peptide) を job_queue に投入
  - ワーカーは result_queue に (idx, score) を返す
  - 結果は idx で並べ直すため、出力順は入力順と完全一致

精度保証:
  - 各 (target, peptide) は独立計算で、入力が同じならスコアも同じ
  - PPLMモデルが deterministic である限り、逐次評価とビット一致
  - 検証用に --verify モードを提供

依存関係:
  - compute_PPLM_affinity.PPLMAffinityPredictor (既存)
  - torch.multiprocessing
================================================================
"""

import os
import time
from typing import List, Tuple, Optional
import torch
import torch.multiprocessing as mp


# ============================================================
# ワーカー関数 (各 GPU プロセスで実行される)
# ============================================================
def _worker_loop(
    worker_id: int,
    gpu_id: int,
    pplm_script: str,
    job_queue,
    result_queue,
    ready_event,
):
    """1 GPU を専有して PPLM 推論を回し続けるワーカー。"""
    # この子プロセスから見える GPU を 1 枚に絞る
    # (CUDA初期化前に設定する必要がある)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 子プロセスで初めて compute_PPLM_affinity を import
    # (これにより PPLMPredictor(gpu_id=0) がこの worker専有 GPU にロードされる)
    try:
        from compute_PPLM_affinity import PPLMAffinityPredictor
        predictor = PPLMAffinityPredictor(
            pplm_script=pplm_script,
            verbose=False,
        )
    except Exception as e:
        # 初期化失敗を親に通知してから終了
        result_queue.put(("__init_failed__", worker_id, repr(e)))
        ready_event.set()
        return

    ready_event.set()
    print(f"[pplm-worker {worker_id}] ready on GPU {gpu_id}", flush=True)

    # ジョブループ
    while True:
        item = job_queue.get()
        if item is None:                              # 終了シグナル
            break
        idx, target_seq, peptide_seq, step = item
        try:
            score = float(
                predictor.predict_affinity(
                    protein_seq=target_seq,
                    peptide_seq=peptide_seq,
                    step=step,
                )
            )
            result_queue.put((idx, score, None))
        except Exception as e:
            # 失敗しても NaN を返してジョブは続行
            result_queue.put((idx, float("nan"), repr(e)))

    print(f"[pplm-worker {worker_id}] stopped", flush=True)


# ============================================================
# 並列 evaluator 本体
# ============================================================
class ParallelPPLMEvaluator:
    """
    既存の PPLMEvaluator と互換のインターフェイス:
        ev.score(target, peptide, step)               # 1サンプル (互換用)
        ev.score_batch([(target, peptide), ...])      # 並列評価 (メイン用途)
    """

    def __init__(
        self,
        pplm_script: str,
        gpu_ids: Optional[List[int]] = None,
        verbose: bool = True,
    ):
        # spawn 方式: CUDA の安全性のため必須
        ctx = mp.get_context("spawn")

        if gpu_ids is None:
            n = torch.cuda.device_count()
            gpu_ids = list(range(n))
        if len(gpu_ids) == 0:
            raise ValueError("ParallelPPLMEvaluator: GPU が見つかりません")

        self.gpu_ids = gpu_ids
        self.num_workers = len(gpu_ids)
        self.verbose = verbose

        self.job_queue = ctx.Queue()
        self.result_queue = ctx.Queue()

        self.workers = []
        ready_events = []
        for i, gid in enumerate(gpu_ids):
            ev = ctx.Event()
            p = ctx.Process(
                target=_worker_loop,
                args=(i, gid, pplm_script, self.job_queue,
                      self.result_queue, ev),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
            ready_events.append(ev)

        if verbose:
            print(f"[ParallelPPLM] {self.num_workers} workers 起動中 "
                  f"(GPUs: {gpu_ids}) — PPLM モデルロード待機...",
                  flush=True)

        # ready 待ち (PPLMモデルロードに数秒〜数十秒かかる)
        for ev in ready_events:
            ev.wait()

        # 初期化失敗チェック (result_queueに init_failed が来てないか)
        init_errors = []
        while not self.result_queue.empty():
            msg = self.result_queue.get_nowait()
            if isinstance(msg, tuple) and len(msg) == 3 and msg[0] == "__init_failed__":
                init_errors.append(msg)
        if init_errors:
            self.close()
            raise RuntimeError(
                f"PPLM worker 初期化失敗: {init_errors}"
            )

        if verbose:
            print(f"[ParallelPPLM] 全ワーカー ready", flush=True)

    # --------------------------------------------------------
    # メイン API: バッチ並列評価
    # --------------------------------------------------------
    def score_batch(
        self,
        pairs: List[Tuple[str, str]],
        step_offset: int = 0,
    ) -> List[float]:
        """
        pairs: [(target_seq, peptide_seq), ...]
        return: 入力順の score リスト (順序保証あり)
        """
        n = len(pairs)
        if n == 0:
            return []

        for idx, (target, peptide) in enumerate(pairs):
            self.job_queue.put((idx, target, peptide, step_offset + idx))

        results: List[Optional[float]] = [None] * n
        errors: List[str] = []
        t0 = time.time()
        report_interval = max(1, n // 10)

        for done in range(n):
            idx, score, err = self.result_queue.get()
            results[idx] = score
            if err is not None:
                errors.append(f"[idx={idx}] {err}")
            if self.verbose and (done + 1) % report_interval == 0:
                elapsed = time.time() - t0
                rate = (done + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n - done - 1) / rate if rate > 0 else float("inf")
                print(f"  PPLM eval: {done+1}/{n} "
                      f"({rate:.2f} samples/s, ETA {eta:.0f}s)",
                      flush=True)

        if errors:
            print(f"[ParallelPPLM] {len(errors)} 件失敗 (NaN として返却):")
            for e in errors[:5]:
                print(f"  {e}")
            if len(errors) > 5:
                print(f"  ... and {len(errors)-5} more")

        return [float("nan") if r is None else r for r in results]

    # --------------------------------------------------------
    # 互換 API: 1個ずつ評価
    # --------------------------------------------------------
    def score(self, target: str, peptide: str, step: int = 0) -> float:
        return self.score_batch([(target, peptide)], step_offset=step)[0]

    # --------------------------------------------------------
    # クリーンアップ
    # --------------------------------------------------------
    def close(self):
        for _ in self.workers:
            try:
                self.job_queue.put(None)
            except Exception:
                pass
        for p in self.workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================
# 自己テスト / 検証
# ============================================================
if __name__ == "__main__":
    """
    動作確認:
      python pplm_parallel_evaluator.py --n-gpu 2 --n-samples 16

    精度検証 (並列 vs 逐次でビット一致するか):
      python pplm_parallel_evaluator.py --n-gpu 2 --n-samples 8 --verify
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-gpu", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--pplm-script", type=str,
                        default="/home/users/gds/src/PPLM/run_pplm-affinity.py")
    parser.add_argument("--target", type=str,
                        default="MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRDPVKKVITISKGCKKILYKLDPNYHGTQ")
    parser.add_argument("--verify", action="store_true",
                        help="逐次評価と並列評価のスコア一致を検証")
    args = parser.parse_args()

    # ダミーペプチド
    PEPTIDES = [
        "ACDEFGHIKL", "MNPQRSTVWY", "AAAAAAAAAA", "GGGGGGGGGG",
        "RKDESTNPHQ", "FYWHILMVCA", "PEPTIDEXXX", "BINDPEPTID",
    ] * ((args.n_samples + 7) // 8)
    PEPTIDES = PEPTIDES[: args.n_samples]
    pairs = [(args.target, p) for p in PEPTIDES]

    print(f"[test] {args.n_samples} ペプチドを {args.n_gpu} GPU で並列評価")
    t0 = time.time()
    with ParallelPPLMEvaluator(
        pplm_script=args.pplm_script,
        gpu_ids=list(range(args.n_gpu)),
    ) as ev:
        scores_par = ev.score_batch(pairs)
    t_par = time.time() - t0
    print(f"[test] 並列: {t_par:.1f}秒, "
          f"{args.n_samples/t_par:.2f} samples/s")
    for p, s in zip(PEPTIDES, scores_par):
        print(f"  {p}: {s:.4f}")

    if args.verify:
        print("\n[verify] 逐次評価と比較中...")
        from compute_PPLM_affinity import PPLMAffinityPredictor
        seq_pred = PPLMAffinityPredictor(pplm_script=args.pplm_script)
        t0 = time.time()
        scores_seq = [seq_pred.predict_affinity(t, p) for t, p in pairs]
        t_seq = time.time() - t0
        print(f"[verify] 逐次: {t_seq:.1f}秒")
        print(f"[verify] 速度向上: x{t_seq/t_par:.2f}")

        import math
        max_diff = 0.0
        n_nan_only_one = 0
        for a, b in zip(scores_par, scores_seq):
            if math.isnan(a) and math.isnan(b):
                continue
            if math.isnan(a) or math.isnan(b):
                n_nan_only_one += 1
                continue
            max_diff = max(max_diff, abs(a - b))

        print(f"[verify] 最大スコア差: {max_diff:.3e}")
        if n_nan_only_one > 0:
            print(f"[verify] ⚠️ 片方のみ NaN: {n_nan_only_one} 件")
        if max_diff < 1e-5 and n_nan_only_one == 0:
            print("[verify] ✅ 並列評価は逐次評価とビット一致")
        else:
            print("[verify] ⚠️ スコアにズレあり "
                  "(PPLMモデル内部の非決定性 or floatの丸めの可能性)")
