import os, sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--make_tarball", help = "make tarball (overwriting if it already exists)", action="store_true")
args = parser.parse_args()

if args.make_tarball:
    os.system("XZ_OPT='-3e -T24' tar -Jc --exclude='*.log' --exclude='*.output' --exclude='*.error' --exclude='*.pdf' --exclude='*.hdf5' --exclude='.git' -f package.tar.gz ../../zephyr")

os.system("condor_submit job.sub")
