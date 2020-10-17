#!/bin/bash

#python create_graph.py
#luigi --module netsalt.tasks ComputePassiveModes --local-scheduler --log-level INFO
luigi --module netsalt.tasks ComputeLasingModes --local-scheduler --log-level INFO
