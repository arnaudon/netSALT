#!/bin/bash
# Compare the four modal-intensity solvers across graph topologies.
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

python compare_intensity_methods.py
