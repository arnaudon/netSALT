#!/bin/bash
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

mkdir -p out figures

python create_graph.py

# python -m netsalt passive config.yaml --force
python -m netsalt lasing config.yaml --force
