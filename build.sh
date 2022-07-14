#!/bin/bash

export CPATH=$CPATH:$CONDA_PREFIX/include
cd build-conda
cmake --build .
cd ../python
rm -rf dist
./make_sdist.sh
cd dist
MUJOCO_PATH=/home/nghia/git/mujoco pip install mujoco-2.2.0.tar.gz
