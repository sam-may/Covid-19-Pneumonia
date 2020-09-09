import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def scale_image(image):
    """Normalize image pixel values to range from 0 to 1"""
    image = numpy.array(image)

    min_val = numpy.amin(image)
    image += -min_val

    max_val = numpy.amax(image)
    image *= 1./max_val
    return image

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
   
def calc_auc(y, pred, interp=1000):
    fpr, tpr, thresh = roc_curve(y, pred)

    fpr_interp = numpy.linspace(0, 1, interp)
    tpr_interp = numpy.interp(fpr_interp, fpr, tpr)

    auc_ = auc(fpr, tpr)

    return fpr_interp, tpr_interp, auc_ 
