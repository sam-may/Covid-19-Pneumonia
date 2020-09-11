import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def hist_1D(data, bins, name, title="", xlabel="", fig=None, save=True, 
            tag=None):
    # Plot
    if not fig:
        fig = plt.figure()
    else:
        plt.figure(fig.number)
    plt.hist(data, bins=bins, alpha=0.4, label=tag)
    # Formatting
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("Counts", fontsize=16)
    plt.legend()
    # Save and close
    if save:
        plt.savefig(name)
        plt.close(fig)

    return
