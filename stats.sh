#!/bin/bash

# setup script
set -e
cd "$(dirname "$0")"

# build and run tests
./build.sh lh_stats RELEASE
cmake-build-release/src/lh_stats $@

# export stats
python plot_stats.py stats/**/*.csv
