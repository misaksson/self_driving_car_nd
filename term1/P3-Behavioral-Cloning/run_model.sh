#!/bin/bash
python drive.py model.h5 & ./simulator/linux_sim/linux_sim.x86_64
pkill -9 -f drive.py
