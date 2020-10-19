#!/bin/bash

#python create_graph.py
#rm -rf figures
#rm -rf out
#luigi --module netsalt.tasks ComputePassiveModes --local-scheduler --log-level INFO
luigi --module netsalt.tasks ComputeLasingModes --local-scheduler --log-level INFO
