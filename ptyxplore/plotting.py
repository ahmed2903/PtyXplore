# Generic array/list plotting utilities for ptychography data
#
# Author: Ahmed H. Mokhtar
# Email: ahmed.mokhtar@desy.de

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from PIL import Image
import ipywidgets as widgets

from .utils import time_it


def plot_roi_from_numpy(array, roi=None, name=None, vmin=None, vmax=None, save=False, **kwargs):
    """Plots a region of interest from a 2D (or 3D) numpy array.

    Args:
        array: 2D array, or 3D array with 'frame' kwarg.
        roi:   [row_start, row_end, col_start, col_end]; defaults to full array.
        name:  title string and save filename.
        vmin, vmax: colormap limits; defaults to mean ± 50%.
        save:  if True, saves the figure as <name>.
        frame: (kwarg) frame index for 3D arrays.
    """
    if roi is None:
        roi = [0, -1, 0, -1]
    if name is None:
        name = "Plot"

    if len(array.shape) == 3 and 'frame' in kwargs:
        data_roi = array[kwargs['frame'], roi[0]:roi[1], roi[2]:roi[3]]
    elif len(array.shape) == 2:
        data_roi = array[roi[0]:roi[1], roi[2]:roi[3]]
    else:
        raise ValueError("Pass a 2D array or a 3D array with 'frame' kwarg.")

    if vmin is None:
        me = np.mean(data_roi)
        vmin = me - 0.5 * me
    if vmax is None:
        me = np.mean(data_roi)
        vmax = me + 0.5 * me

    plt.figure()
    plt.imshow(data_roi, vmin=vmin, vmax=vmax, cmap='viridis')
    plt.title(name)
    plt.colorbar()
    plt.show()

    if save:
        plt.savefig(name)


def plot_map_on_detector(detec_image, k_map, vmin, vmax, title, cmap, crop=False, **kwargs):
    """Overlays coordinate markers (k_map) onto a detector image.

    Optional kwargs:
        roi:      [xs, xe, ys, ye] used for offset/crop.
        color:    marker colour (default 'white').
        rec_size: rectangle size in pixels (default 1).
        fname:    if provided, saves the figure.
    """
    sx, sy = 0, 0
    if 'roi' in kwargs:
        roi = kwargs['roi']
        sx, ex, sy, ey = roi

    if crop:
        detec_image = detec_image[sx:ex, sy:ey]

    color    = kwargs.get('color', 'white')
    rec_size = kwargs.get('rec_size', 1)

    fig, ax = plt.subplots()
    for idx in k_map:
        rect = patches.Rectangle(
            (idx[1] - 1 - sy, idx[0] - 1 - sx),
            rec_size, rec_size,
            linewidth=2, edgecolor=color, facecolor=color,
        )
        ax.add_patch(rect)

    im = ax.imshow(detec_image, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    fig.colorbar(im, orientation='vertical')
    plt.title(title)
    if 'fname' in kwargs:
        plt.savefig(kwargs['fname'])
    plt.show()


def plot_pixel_space(ordered_pixels, connection=False):
    """Scatter plot of pixel coordinates in image space.

    Args:
        ordered_pixels: array-like of (row, col) pairs.
        connection:     if True, also draws connecting lines.
    """
    ordered_pixels = np.array(ordered_pixels)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ordered_pixels[:, 1], ordered_pixels[:, 0], color='red', label="Pixels")
    if connection:
        ax.plot(ordered_pixels[:, 1], ordered_pixels[:, 0], color='red')
    ax.set_title("Scatter")
    ax.invert_yaxis()
    plt.show()


def two_lists_one_slider(img_list1, img_list2,
                         scale_fac=0.3,
                         vmin1=None, vmax1=None,
                         vmin2=None, vmax2=None,
                         cmap1='viridis', cmap2='viridis'):
    """Side-by-side interactive viewer for two image sequences."""
    num_images = len(img_list1)
    img_slider = widgets.IntSlider(min=0, max=num_images - 1, value=0, description="Image")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axes[0].imshow(img_list1[0], vmin=vmin1, vmax=vmax1, cmap=cmap1)
    axes[0].set_title(f"Image 1  0/{num_images - 1}")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(img_list2[0], vmin=vmin2, vmax=vmax2, cmap=cmap2)
    axes[1].set_title(f"Image 2  0/{num_images - 1}")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()

    def update_image(img_idx):
        im1.set_data(img_list1[img_idx])
        im1.set_clim(vmin1, vmax1)
        im2.set_data(img_list2[img_idx])
        im2.set_clim(vmin2, vmax2)
        axes[0].set_title(f"Image 1  {img_idx}/{num_images - 1}")
        axes[1].set_title(f"Image 2  {img_idx}/{num_images - 1}")
        fig.canvas.draw_idle()

    display(widgets.interactive(update_image, img_idx=img_slider))


def plot_list_slider(img_list, scale_fac=0.3, vmin1=None, vmax1=None):
    """Interactive slider viewer for a single image sequence."""
    num_images = len(img_list)
    img_slider = widgets.IntSlider(min=0, max=num_images - 1, value=0, description="Image")

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(img_list[0], cmap='viridis', vmin=vmin1, vmax=vmax1)
    ax.set_title(f"Image  0/{num_images - 1}")
    plt.colorbar(im, ax=ax, label="Intensity")
    plt.tight_layout()

    def update_image(img_idx):
        img = img_list[img_idx]
        img_mean = np.mean(img)
        vmin = max(img_mean - scale_fac * img_mean, 0)
        vmax = img_mean + scale_fac * img_mean
        im.set_data(img)
        im.set_clim(vmin, vmax)
        ax.set_title(f"Image  {img_idx}/{num_images - 1}")
        fig.canvas.draw_idle()

    display(widgets.interactive(update_image, img_idx=img_slider))


@time_it
def create_gif_from_arrays(array_list, gif_name, fps=10, cmap="viridis"):
    """Save a list of 2D arrays as an animated GIF.

    Args:
        array_list: list of 2D numpy arrays.
        gif_name:   output file path (e.g. 'output.gif').
        fps:        frames per second.
        cmap:       matplotlib colormap name.
    """
    frame_files = []
    for i, array in enumerate(array_list):
        plt.figure(figsize=(6, 6))
        plt.imshow(array, cmap=cmap)
        plt.colorbar()
        plt.title(f"Frame {i + 1}")
        frame_file = f"frame_{i}.png"
        plt.savefig(frame_file)
        frame_files.append(frame_file)
        plt.close()

    frames = [Image.open(f) for f in frame_files]
    frames[0].save(
        gif_name,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    for f in frame_files:
        os.remove(f)


def two_lists_one_gif(array_list1, array_list2, gif_name,
                      title1="Image 1", title2="Image 2",
                      fps=10, cmap="viridis"):
    """Save two image sequences side-by-side as an animated GIF.

    Args:
        array_list1, array_list2: lists of 2D numpy arrays (same length).
        gif_name: output file path.
        fps:      frames per second.
        cmap:     matplotlib colormap name.
    """
    num_images = len(array_list1)
    frame_files = []
    for i, (a1, a2) in enumerate(zip(array_list1, array_list2)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(a1, cmap=cmap)
        axes[0].set_title(title1)
        axes[1].imshow(a2, cmap=cmap)
        axes[1].set_title(title2)
        fig.suptitle(f"Frame {i}/{num_images}", fontsize=16, y=1.05)
        frame_file = f"frame_{i}.png"
        plt.savefig(frame_file)
        frame_files.append(frame_file)
        plt.close()

    frames = [Image.open(f) for f in frame_files]
    frames[0].save(
        gif_name,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    for f in frame_files:
        os.remove(f)
