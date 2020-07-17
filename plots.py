import numpy
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def hist_1D(data, bins, name, title="", xlabel=""):
    # Plot
    fig = plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title, fontsize=16)
    # Formatting
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("Counts", fontsize=16)
    # Save and close
    plt.savefig(name)
    plt.close(fig)
    return

def scale_image(image):
    """Normalize image pixel values to range from 0 to 1"""
    image = numpy.array(image)

    min_val = numpy.amin(image)
    image += -min_val

    max_val = numpy.amax(image)
    image *= 1./max_val
    return image

def image_truth_plot(image, truth, name, fs=6):
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
    plt.savefig(name)
    plt.close(fig)
    return fig

def transparent_cmap(cmap, N=255):
    """Copy colormap and set alpha values"""
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = numpy.linspace(0, 0.8, N+4)
    return mycmap

def make_heatmap(pred, divergent_colormap=False):
    w, h = pred.shape
    x, y = numpy.mgrid[0:w, 0:h]

    if not divergent_colormap:
        cmap = transparent_cmap(plt.cm.cool)
        levels = numpy.linspace(0, 1, 15)
    else:
        cmap = plt.cm.bwr
        levels = numpy.linspace(-1, 1, 15)

    return x, y, cmap, levels

def image_truth_pred_plot(image, truth, pred, name, fs=6):
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

    plt.savefig(name, bbox_inches='tight')
    plt.close(fig)

    return fig

def roc_plot(fpr_mean, fpr_std, tpr_mean, tpr_std, auc, auc_std, name):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_ticks_position('both')
    ax1.grid(True)
    ax1.plot(
        fpr_mean, 
        tpr_mean,
        color='blue', 
        label="U-Net [AUC: %.3f +/- %.3f]" % (auc, auc_std)
    )
    ax1.fill_between(
        fpr_mean,
        tpr_mean - (tpr_std/2.),
        tpr_mean + (tpr_std/2.),
        color='blue',
        alpha=0.25, label=r'$\pm 1\sigma$'
    )
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(name)
    plt.close(fig)

    return fig
   
def calc_auc(y, pred, interp=1000):
    fpr, tpr, thresh = roc_curve(y, pred)

    fpr_interp = numpy.linspace(0, 1, interp)
    tpr_interp = numpy.interp(fpr_interp, fpr, tpr)

    auc_ = auc(fpr, tpr)

    return fpr_interp, tpr_interp, auc_ 
