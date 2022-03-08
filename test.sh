#!/bin/bash

# setup script
set -e
cd "$(dirname "$0")"

# build and run tests
./build.sh lh_tests DEBUG
cmake-build-debug/src/lh_tests $@
