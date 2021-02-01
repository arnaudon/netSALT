#!/bin/bash

luigi --module netsalt.tasks PlotThresholdModes --local-scheduler --log-level INFO

