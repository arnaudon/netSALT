#!/bin/bash
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

mkdir -p out
mkdir -p figures

python create_pump.py

cp ../buffon_uniform/out/qualities.h5  out/
cp ../buffon_uniform/out/passive_modes.h5 out/

luigi --module netsalt.tasks.workflow ComputeLasingModes --local-scheduler --log-level INFO # --rerun #--rerun-all

