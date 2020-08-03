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
#module load stashcache
#stashcp /osgconnect/public/smay/covid_ct_data/features/14Jul2020_z_score_downsample256/features.hdf5 .
#stashcp /osgconnect/public/smay/covid_ct_data/features/14Jul2020_z_score_downsample256/features.json .

python3 test.py 2>&1

