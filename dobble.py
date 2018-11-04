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

imgs = [
    # dark bg, natural daylight
    "./img/dobble01.jpg", "./img/dobble02.jpg", "./img/dobble03.jpg", "./img/dobble04.jpg",
    # dark bg, natural daylight, perspective
    "./img/dobble05.jpg", "./img/dobble06.jpg", "./img/dobble07.jpg", "./img/dobble08.jpg", "./img/dobble09.jpg", "./img/dobble10.jpg", "./img/dobble11.jpg",
    # dark bg, directional white light
    "./img/dobble12.jpg", "./img/dobble13.jpg", "./img/dobble14.jpg", "./img/dobble15.jpg",
    # dark bg, directional colored light
    "./img/dobble16.jpg", "./img/dobble17.jpg", "./img/dobble18.jpg", "./img/dobble19.jpg"]

displayed = 0


def forward_click(event):
    global displayed
    displayed += 1 if displayed < len(imgs) - 1 else -(len(imgs) - 1)
    imshow(data.imread(imgs[displayed]))
    print("dobble" + "{:02d}".format(displayed) + ".jpg displayed")
    plt.show()


def back_click(event):
    global displayed
    displayed -= 1 if displayed > 0 else -(len(imgs) - 1)
    imshow(data.imread(imgs[displayed]))
    print("dobble" + "{:02d}".format(displayed) + ".jpg displayed")
    plt.show()


NavigationToolbar2.forward = forward_click
NavigationToolbar2.back = back_click

imshow(data.imread(imgs[0]))

plt.show()