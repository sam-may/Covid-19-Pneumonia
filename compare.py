import os, sys
import numpy
import json
import glob
import matplotlib.pyplot as plt
import argparse

# CLI
parser = argparse.ArgumentParser()
parser.add_argument(
    "--inputs",
    help="csv list of model summary jsons",
    type=str
)
parser.add_argument(
    "--labels",
    help="csv list of labels corresponding to each model",
    type=str
)
parser.add_argument(
    "--colors",
    help="csv list of plot colors",
    type=str,
    default="black,red,blue,green,orange"
)
args = parser.parse_args()

inputs = args.inputs.split(",")
labels = args.labels.split(",")
colors = args.colors.split(",")

if len(inputs) != len(labels):
    print("[compare.py] Length of inputs is %d while length of labels is %d! Exiting."
          % (len(inputs), len(labels)))
    sys.exit(1)

# ROC Curve
fig = plt.figure()
ax1 = fig.add_subplot(111)

for input, label, color in zip(inputs, labels, colors):
    full_inputs = glob.glob(input)
    print(full_inputs)

    tprs = []
    fprs = []
    aucs = []
    for individual_input in full_inputs:
        with open(individual_input, "r") as f_in:
            info = json.load(f_in)

        fprs.append(numpy.array(info["fpr_mean"]))
        tprs.append(numpy.array(info["tpr_mean"]))
        aucs.append(info["auc"])

    fpr = numpy.mean(fprs, axis=0)
    tpr = numpy.mean(tprs, axis=0)
    tpr_unc = numpy.std(tprs, axis=0)

    auc = numpy.mean(aucs)
    auc_unc = numpy.std(aucs)
    print(aucs)

    ax1.plot(
            fpr,
            tpr,
            label = label + " [AUC = %.3f +/- %.3f]" % (auc, auc_unc),
            color = color
    )

    ax1.fill_between(
            fpr,
            tpr - 1*(tpr_unc/2.),
            tpr + 1*(tpr_unc/2.),
            color = color,
            alpha = 0.25
    )

ax1.set_xscale("log")
plt.xlim(0.005, 1)
plt.ylim(0,1)
plt.xlabel('False Positive Rate (Background Efficiency)')
ax1.set_ylabel('True Positive Rate (Signal Efficiency)')
legend = ax1.legend(loc='lower right')
ax1.yaxis.set_ticks_position('both')
ax1.grid(True)
plt.savefig('auc_comparison.pdf')
plt.close(fig)

# TODO: add plots of accuracy vs. epoch, dice vs. epoch, bce vs. epoch
