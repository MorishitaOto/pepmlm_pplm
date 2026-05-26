#!/bin/sh
#PBS -l select=1:ncpus=4:mpiprocs=1:ompthreads=4:ngpus=1
#PBS -l walltime=24:00:00
cd ${PBS_O_WORKDIR}
source ~/.bashrc
conda activate pplm
LOGFILE=infer_$(date +%Y%m%d_%H%M%S).txt
python -u ../src/run_full_pipeline.py \
    --pipeline_config ../config/pipeline_config/pipeline_20260522_4LDJ_cryptic_iter100_infer.json \
    > ${LOGFILE} 2>&1
