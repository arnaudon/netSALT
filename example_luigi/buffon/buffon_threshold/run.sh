#!/bin/bash
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

mkdir -p out
mkdir -p figures

# copy passive quantities from uniform run if possible
cp ../buffon_uniform/out/passive_modes.h5 out/passive_modes.h5
cp ../buffon_uniform/out/qualities.h5 out/qualities.h5
cp ../buffon_uniform/out/quantum_graph.gpickle out/quantum_graph.gpickle

luigi --module netsalt.tasks.workflow ComputePassiveModes --local-scheduler --log-level INFO
luigi --module netsalt.tasks.workflow ComputeLasingModes --local-scheduler --log-level INFO

