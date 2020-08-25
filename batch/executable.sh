#!/bin/bash
unset PYTHONPATH

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

# actual job
echo "[executable.sh] running: tar -xvf package.tar.gz"
tar -xvf package.tar.gz

cd zephyr

# stashcache
echo "About to run stashcp"
ls -atlrh

python3 -m pip install --upgrade stashcp # included in dockerfile, but needs to be run again for some reason
stashcp /osgconnect/public/smay/covid_ct_data/features/14Jul2020_z_score_downsample256/features.hdf5 .
stashcp /osgconnect/public/smay/covid_ct_data/features/14Jul2020_z_score_downsample256/features.json .
echo "Finished running stashcp"
ls -althr

python3 train_unet.py --data_hdf5 "features.hdf5" --metadata_json "features.json" --max_epochs 10 --n_trainings 10 --tag "test_condor"

echo "Done training"
ls -altrh */*
