import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter, ListedColormap, get_named_colors_mapping

from Rignak_ImageProcessing.miscellaneous_image_operations import inverse_fourier_transform

DEFAULT_VMIN = 0
DEFAULT_VMAX = 1
DEFAULT_COLORMAP = 'gray'
THRESHOLD = 0.5


def make_colormap(color):
    colors = [np.array(color) / 255 * i for i in range(256)]
    return ListedColormap(colors)


NAMED_COLORS = get_named_colors_mapping()
COLORS = [ColorConverter.to_rgb(NAMED_COLORS[color])
          for color in ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                        'lime', 'gold', 'indigo', 'w')]
COLORMAPS = [make_colormap(COLOR) for COLOR in COLORS]


def fuse_canals(im, colors=COLORS, threshold=THRESHOLD):
    new_im = np.zeros((im.shape[0], im.shape[1], 3))
    for x, line in enumerate(np.argmax(im, axis=-1)):
        for y, px in enumerate(line):
            if im[x, y, px] > threshold:
                new_im[x, y] = colors[px]
    return new_im


def imshow(im, cmap=DEFAULT_COLORMAP, vmin=DEFAULT_VMIN, vmax=DEFAULT_VMAX, fourier_additive_term=0.5,
           denormalizer=None):
    if denormalizer is not None:
        im = denormalizer(im)
    if len(im.shape) == 2:
        plt.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap)
    elif im.shape[2] == 1:
        plt.imshow(im[:, :, 0], vmin=vmin, vmax=vmax, cmap=cmap)
    elif im.shape[2] == 2:  # assume the image is a fourier transform
        # TODO plot as map dataset
        fourier_transform = im[:, :, 0] + im[:, :, 1] * 1j
        im = np.abs(fourier_transform)
        im /= np.max(im)

        plt.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap)

    elif im.shape[2] == 3:
        plt.imshow(im)
    else:
        im = fuse_canals(im)
        plt.imshow(im)
