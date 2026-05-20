#!/bin/sh
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:ngpus=1
#PBS -l walltime=24:00:00
cd ${PBS_O_WORKDIR}

# 環境設定（必要に応じて）
source ~/.bashrc
conda activate pplm  # or source activate myenv

# 実行コマンド
python3 ../src/pepmlm_pplm_cryptic.py --config ../config/20260502_6XI7_cryptic.json 

