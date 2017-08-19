#!/usr/bin/env bash


export MKL_NUM_THREADS=8
export MKL_DYNAMIC=FALSE

sudo apt-get build-dep python-matplotlib

mkdir Output/
mkdir Output/quad
mkdir Output/logis

cd Resource/
bash LinuxDownloadData.sh
python txt2npz.py
python rfm.py
python toydataLogis.py
cd ..
