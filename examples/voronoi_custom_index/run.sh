#!/bin/bash
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

mkdir -p out
mkdir -p figures

cp ../voronoi/voronoi_graph.gpickle voronoi_graph.gpickle

python define_index_profile.py

#luigi --module netsalt.tasks.workflow ComputePassiveModes --local-scheduler --log-level INFO --rerun
luigi --module netsalt.tasks.workflow ComputeLasingModes --local-scheduler --log-level INFO  --rerun 

