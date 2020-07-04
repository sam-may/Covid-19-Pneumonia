import os, sys
import argparse

# CLI
parser = argparse.ArgumentParser()
parser.add_argument(
    "--tag",
    help="tag to identify these trainings",
    type=str,
    default="test"
)
args = parser.parse_args()

os.chdir("../")

summary_jsons = []
labels = []
for n_slice in [0, 1, 2]:
    command = 'python train.py\
              --input "/xilinx/scratch/covid_ct_data/6Jun2020_full_z_score_downsample256/features.hdf5"\
              --input_metadata "/xilinx/scratch/covid_ct_data/6Jun2020_full_z_score_downsample256/features.json"\
              --tag "%s_M%d"\
              --n_extra_slices %d' % (args.tag, n_slice, n_slice)

    print(command) 
    #os.system(command)

    summary_jsons.append("results_%s_M%d.json"
                         % (args.tag, n_slice))
    if n_slice == 0:
        labels.append("2d")
    else:
        labels.append("2.5d (%d slices)"
                      % (n_slice))

command = 'python compare.py --inputs "%s" --labels "%s"' % (",".join(summary_jsons), ",".join(labels))
print(command)
os.system(command)

