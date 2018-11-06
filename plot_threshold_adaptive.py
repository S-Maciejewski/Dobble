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


def grayOut(img):
    hsv = rgb2hsv(img)
    hsv[:, :, 1] = 0
    return rgb2gray(hsv2rgb(hsv))


image = grayOut(data.imread("./img/dobble01.jpg"))

global_thresh = threshold_otsu(image)
binary_global = image > global_thresh

block_size = 35
binary_adaptive = threshold_adaptive(image, block_size, offset=0.05)

# fig, axes = plt.subplots(nrows=1, figsize=(7, 8))
# ax0, ax1, ax2 = axes
plt.gray()

# ax0.imshow(image)
# ax0.set_title('Image')

# ax1.imshow(binary_global)
# ax1.set_title('Global thresholding')

imshow(binary_adaptive)
# ax2.set_title('Adaptive thresholding')

# for ax in axes:
#     ax.axis('off')

plt.show()
