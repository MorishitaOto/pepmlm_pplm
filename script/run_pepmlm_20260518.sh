#!/bin/sh
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:ngpus=1
#PBS -l walltime=24:00:00
cd ${PBS_O_WORKDIR}

# 環境設定（必要に応じて）
source ~/.bashrc
conda activate pplm  # or source activate myenv
LOGFILE=run_$(date +%Y%m%d_%H%M%S).txt

# 実行コマンド
python3 ../src/pepmlm_pplm_cryptic.py --config ../config/20260518_6XI7.json > ${LOGFILE} 2>&1
