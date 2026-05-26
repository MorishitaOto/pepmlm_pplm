#!/bin/sh
#PBS -l select=1:ncpus=4:mpiprocs=1:ompthreads=4:ngpus=1
#PBS -l walltime=24:00:00
cd ${PBS_O_WORKDIR}
source ~/.bashrc
conda activate pplm
LOGFILE=run_$(date +%Y%m%d_%H%M%S).txt
python ../src/run_train_and_evaluate.py \
    --pipeline_config ../config/pipeline_config/pipeline_20260522_4LDJ_cryptic_iter100.json \
    > ${LOGFILE} 2>&1
