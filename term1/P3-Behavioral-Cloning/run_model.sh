#!/bin/bash
python ${BASH_SOURCE%/*}/drive.py $1 & ${BASH_SOURCE%/*}/simulator/linux_sim/linux_sim.x86_64
pkill -9 -f drive.py
