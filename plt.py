import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter, ListedColormap, get_named_colors_mapping
import matplotlib.patches as mpatches

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


def fuse_canals(im, colors=COLORS, threshold=THRESHOLD, labels=()):
    new_im = np.zeros((im.shape[0], im.shape[1], 3))
    for x, line in enumerate(np.argmax(im, axis=-1)):
        for y, px in enumerate(line):
            if im[x, y, px] > threshold:
                new_im[x, y] = colors[px]

    plt.legend(handles=[mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)],
               loc='center left', bbox_to_anchor=(1, 0.5))

    return new_im


def imshow(im, cmap=DEFAULT_COLORMAP, vmin=DEFAULT_VMIN, vmax=DEFAULT_VMAX,
           denormalizer=None, interpolation="nearest", labels=None, threshold=THRESHOLD):
    if denormalizer is not None:
        im = denormalizer(im)
    if im.max() < 1 and vmax != 1:
        im = im * 255
        im = im.astype('uint8')
    if im.max() > 1 and vmax == 1:
        im = im / 255

    if len(im.shape) == 2:
        plt.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)
    elif im.shape[2] == 1:
        plt.imshow(im[:, :, 0], vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)
    elif im.shape[2] == 2:  # assume the image is a fourier transform
        # TODO plot as map dataset
        fourier_transform = im[:, :, 0] + im[:, :, 1] * 1j
        im = np.abs(fourier_transform)
        im /= np.max(im)

        plt.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)

    elif im.shape[2] == 3:
        plt.imshow(im, interpolation=interpolation)
    else:
        im = fuse_canals(im, labels=labels, threshold=threshold)
        plt.imshow(im, interpolation=interpolation)