import cv2
import matplotlib.pyplot as plt
import numpy

def enforce_positive(img):
    min_val = numpy.min(img)
    if min_val < 0:
        img += -min_val
    return img

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = numpy.linspace(0, 0.8, N+4)
    return mycmap

def plot_pneumonia_heatmap(img, pred, name = "test"):
    img = enforce_positive(img)
    img = img[:,:,0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(img)

    w, h, z = pred.shape
    x, y = numpy.mgrid[0:w, 0:h]

    cmap = transparent_cmap(plt.cm.cool)

    pneumonia_score = plt.contourf(x, y, pred[:,:,0], 15, cmap = cmap)
    cbar = plt.colorbar(pneumonia_score)
    cbar.set_label('Pneumonia Score', rotation=270)

    plt.savefig('plots/pneumonia_score_%s.pdf' % name)
    plt.close(fig)

def overlay_images(img1, img2, alpha, name = "test"):
    img1 = enforce_positive(img)
    img2 = enforce_positive(img)

    overlay = numpy.zeros_like(img1)
    cv2.addWeighted(img1, 1, img2, alpha, 0, overlay)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(overlay)
    plt.savefig("plots/overlay_%s.pdf" % name)
    plt.close(fig)
