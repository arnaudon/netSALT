#!/bin/bash
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

mkdir -p out
mkdir -p figures

luigi --module netsalt.tasks ComputeLasingModes --local-scheduler --log-level INFO # --rerun-all # --rerun 

