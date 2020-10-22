#!/bin/bash

#python create_graph.py
#rm -rf figures
#rm -rf out
#luigi --module netsalt.tasks ComputePassiveModes --local-scheduler --log-level INFO
luigi --module netsalt.tasks ComputeLasingModesWithPumpOptimization --local-scheduler --log-level INFO
#luigi --module netsalt.tasks.pump PlotOptimizedPump --local-scheduler --log-level INFO
