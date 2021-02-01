#!/bin/bash
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
#python create_graph.py
#rm -rf figures
#rm -rf out
#luigi --module netsalt.tasks ComputePassiveModes --local-scheduler --log-level INFO
#luigi --module netsalt.tasks ComputeLasingModes --local-scheduler --log-level INFO
#luigi --module netsalt.tasks ComputeLasingModesWithPumpOptimization --local-scheduler --log-level INFO
luigi --module netsalt.tasks ComputeControllability --local-scheduler --log-level INFO

