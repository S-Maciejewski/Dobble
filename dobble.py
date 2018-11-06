from pylab import *
import skimage as ski
from skimage import data, io, filters, exposure, measure, color, feature
from skimage.filters import rank
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import disk
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.filters.edges import convolve
from matplotlib import pylab as plt
import numpy as np
from numpy import array
from IPython.display import display
from ipywidgets import interact, interactive, fixed
from ipywidgets import *
from ipykernel.pylab.backend_inline import flush_figures
from matplotlib.backend_bases import NavigationToolbar2
from skimage.filters import threshold_otsu, threshold_local, threshold_adaptive

warnings.simplefilter("ignore")

imgs = [
    # dark bg, natural daylight
    "./img/dobble01.jpg", "./img/dobble02.jpg", "./img/dobble03.jpg", "./img/dobble04.jpg",
    # dark bg, natural daylight, perspective
    "./img/dobble05.jpg", "./img/dobble06.jpg", "./img/dobble07.jpg", "./img/dobble08.jpg", "./img/dobble09.jpg", "./img/dobble10.jpg",
    # dark bg, directional white light
    "./img/dobble11.jpg", "./img/dobble12.jpg",
    # white bg, natural daylight
    "./img/dobble13.jpg", "./img/dobble14.jpg", 
    # dark bg, directional colored light
    "./img/dobble15.jpg", "./img/dobble16.jpg", "./img/dobble17.jpg"
    ]

displayed = 0


def forward_click(event):
    global displayed
    displayed += 1 if displayed < len(imgs) - 1 else -(len(imgs) - 1)
    show_img(data.imread(imgs[displayed]))
    print("dobble" + "{:02d}".format(displayed) + ".jpg displayed")
    plt.show()


def back_click(event):
    global displayed
    displayed -= 1 if displayed > 0 else -(len(imgs) - 1)
    show_img(data.imread(imgs[displayed]))
    print("dobble" + "{:02d}".format(displayed) + ".jpg displayed")
    plt.show()

def gray_out(img):
    hsv = rgb2hsv(img)
    hsv[:, :, 1] = 0
    return rgb2gray(hsv2rgb(hsv))

def show_img(img):
    img = gray_out(img)
    plt.subplot(2, 2, 1)
    imshow(img)
    block_size = 35
    thresh = threshold_adaptive(img, block_size, offset=0.1)
    plt.subplot(2, 2, 2)
    imshow(thresh)

NavigationToolbar2.forward = forward_click
NavigationToolbar2.back = back_click

figure(figsize=(100, 100))
plt.gray()

show_img(data.imread(imgs[0]))

plt.show()