#!/bin/bash

#python create_graph.py
luigi --module netsalt.tasks ComputePassiveModes --local-scheduler --log-level INFO
