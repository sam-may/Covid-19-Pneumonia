#!/bin/bash


echo
echo

env | sort

echo
echo

nvidia-smi 2>&1

echo
echo

# so we get good exit codes 
set -e

# stashcache
echo "About to run stashcp"
ls -atlrh

# need to debug why these don't work
#module load stashcache
#stashcp /osgconnect/public/smay/covid_ct_data/features/14Jul2020_z_score_downsample256/features.hdf5 .
#stashcp /osgconnect/public/smay/covid_ct_data/features/14Jul2020_z_score_downsample256/features.json .
echo "Finished running stashcp"
ls -althr

# actual job
echo "[executable.sh] running: tar -xvf package.tar.gz"
tar -xvf package.tar.gz

cd zephyr

python3 train_unet.py
