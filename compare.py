import os, sys
import numpy
import json
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
    with open(input, "r") as f_in:
        info = json.load(f_in)

    fpr = numpy.array(info["fpr_mean"])
    tpr = numpy.array(info["tpr_mean"])
    tpr_unc = numpy.array(info["tpr_std"])
    auc = info["auc"]
    auc_unc = info["auc_std"]

    ax1.plot(
            fpr,
            tpr,
            label = label + " [AUC = %.3f]" % auc,
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
