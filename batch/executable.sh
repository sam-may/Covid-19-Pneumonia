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

# pip
pip3 install nibabel
pip3 install pydicom

# stashcache
echo "About to run stashcp"
ls -atlrh
#module load stashcache
#stashcp /osgconnect/public/smay/covid_ct_data/features/14Jul2020_z_score_downsample256/features.hdf5 .
#stashcp /osgconnect/public/smay/covid_ct_data/features/14Jul2020_z_score_downsample256/features.json .
echo "Finished running stashcp"
ls -althr

python3 test.py 2>&1

