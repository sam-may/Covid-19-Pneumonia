import numpy
from sklearn.metrics import roc_curve, auc
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

def roc_plot(fpr_mean, fpr_std, tpr_mean, tpr_std, auc, auc_std, name, fig=None,
             save=True, tag=None):
    if not fig:
        fig, axes = plt.subplots()
    else:
        axes = fig.axes[0]
    axes.yaxis.set_ticks_position('both')
    axes.grid(True)
    axes.plot(
        fpr_mean, 
        tpr_mean,
        # color='blue', 
        label="%s [AUC: %.3f +/- %.3f]" % (tag, auc, auc_std)
    )
    axes.fill_between(
        fpr_mean,
        tpr_mean - (tpr_std/2.),
        tpr_mean + (tpr_std/2.),
        # color='blue',
        alpha=0.25, label=r'$\pm 1\sigma$'
    )
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    if save:
        plt.savefig(name)
        plt.close(fig)

    return
