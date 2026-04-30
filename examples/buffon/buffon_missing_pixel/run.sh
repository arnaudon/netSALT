#!/bin/bash
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

mkdir -p out figures

cp ../buffon_uniform/out/passive_modes.h5 out/passive_modes.h5
cp ../buffon_uniform/out/qualities.h5 out/qualities.h5
cp ../buffon_uniform/out/quantum_graph.json out/quantum_graph.json

python make_pump.py
python -m netsalt lasing config.yaml
