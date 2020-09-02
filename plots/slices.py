import numpy
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from .utils import *

def image_truth_plot(image, truth, name, fs=6, fig=None, save=True):
    if not fig:
        fig = plt.figure()

    ax = fig.add_subplot(131)
    image_scaled = scale_image(image)
    plt.imshow(image_scaled, cmap='gray')
    plt.title("Original CT Scan", fontsize=fs)

    ax = fig.add_subplot(132)
    plt.imshow(truth, cmap='gray')
    plt.title("Radiologist Ground Truth", fontsize=fs)

    ax = fig.add_subplot(133)
    blend = image_scaled
    max_val = numpy.amax(blend)
    for i in range(len(truth)):
        for j in range(len(truth)):
            if truth[i][j] == 1:
                blend[i][j] = 1

    plt.imshow(blend, cmap='gray')
    plt.title("Original & Truth", fontsize=fs)
    if save:
        plt.savefig(name)
        plt.close(fig)

    return fig

def image_truth_pred_plot(image, truth, pred, name, title=None, fs=6, fig=None, save=True):
    if not fig:
        fig = plt.figure()
    ax = fig.add_subplot(231)

    image_scaled = scale_image(image)
    plt.imshow(image_scaled, cmap='gray')
    plt.title("Original", fontsize=fs)
    plt.axis('off')

    ax = fig.add_subplot(232)
    plt.imshow(truth, cmap='gray')
    plt.title("Ground Truth", fontsize=fs)
    plt.axis('off')

    ax = fig.add_subplot(233)
    plt.imshow(pred, cmap='gray')
    plt.title("U-Net Prediction", fontsize=fs)
    plt.axis('off')

    ax = fig.add_subplot(234)
    plt.imshow(image_scaled, cmap='gray')
    x, y, cmap, levels = make_heatmap(truth)
    heatmap = plt.contourf(x, y, truth.transpose(), cmap=cmap, levels=levels)
    plt.title('Original + Truth', fontsize=fs)
    plt.axis('off')

    ax = fig.add_subplot(235)
    plt.imshow(image_scaled, cmap='gray')
    x, y, cmap, levels = make_heatmap(pred)
    heatmap = plt.contourf(x, y, pred.transpose(), cmap=cmap, levels=levels)
    plt.title('Original + Prediction', fontsize=fs)
    plt.axis('off')

    ax = fig.add_subplot(236)
    plt.imshow(image_scaled, cmap='gray')
    x, y, cmap, levels = make_heatmap(truth - pred, True)
    corrmap = plt.contourf(
        x, 
        y, 
        (truth - pred).transpose(), 
        cmap=cmap, 
        levels=levels
    )
    plt.title('Truth - Prediction', fontsize=fs)
    plt.axis('off')


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.55, 0.05, 0.35])
    cbar = fig.colorbar(heatmap, cax=cbar_ax)
    cbar.ax.set_ylabel(
        'Pneumonia Score', 
        rotation=270, 
        labelpad=15, 
        fontsize=fs
    ) 
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.35])
    cbar = fig.colorbar(corrmap, cax=cbar_ax)
    cbar.ax.set_ylabel(
        'Truth - Prediction', 
        rotation=270, 
        labelpad=15, 
        fontsize=fs
    )
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    fig.suptitle(title)

    if save:
        plt.savefig(name, bbox_inches='tight')
        plt.close(fig)

    return fig
