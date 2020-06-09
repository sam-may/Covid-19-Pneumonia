import json
import numpy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--inputs", help = "csv list of which .npz files to consider", type=str)
parser.add_argument("--labels", help = "csv list of labels for each .npz file", type=str)
args = parser.parse_args()

inputs = args.inputs.split(",")
labels = args.labels.split(",")
colors = ["red", "cyan", "green", "navy"]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)

for input, label, color in zip(inputs, labels, colors[:len(inputs)]):
    with open(input, "r") as f_in:
        results = json.load(f_in)

    x = numpy.arange(len(results["metrics"]["dice_loss"]))
    accuracy = numpy.array(results["metrics"]["accuracy"])
    dice_loss = numpy.array(results["metrics"]["dice_loss"])

    ax1.plot(x, dice_loss, label = label, color = color)

plt.xlabel('Epoch')
plt.ylabel('1 - Dice Coefficient')
plt.legend()
plt.savefig('loss_comparison.pdf')
