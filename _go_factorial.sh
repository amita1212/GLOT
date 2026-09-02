#!/usr/bin/env bash
# Launch the 2x2 geometry factorial detached, logging to _factorial_geom.log.
cd /home/t-amitalfasi/glot || exit 1
mkdir -p results
nohup ~/glotenv/bin/python factorial_geom.py > _factorial_geom.log 2>&1 &
echo "launched pid $!"
