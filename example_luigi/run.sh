#!/bin/bash

luigi --module netsalt.tasks ComputePassiveModes --local-scheduler --log-level INFO
