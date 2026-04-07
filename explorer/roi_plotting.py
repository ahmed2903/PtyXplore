# ROI-based plotting utilities for ptychography data
#
# Author: Ahmed H. Mokhtar
# Email: ahmed.mokhtar@desy.de

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.patches import Rectangle
import ipywidgets as widgets

from .cbd_loader import ROI
from .data_io import stack_4d_data_old
from .data_proc import (
    compute_histograms,
    make_coherent_image,
    make_detector_image,
    mask_hot_pixels,
)
from .plotting import plot_map_on_detector, plot_roi_from_numpy


def plot_detector_roi(roi: ROI, file_no, frame_no,
                      title=None, vmin=None, vmax=None,
                      mask_hot=True, save=False):
    """Plot a single detector frame cropped to the ROI extent.

    Args:
        roi:      ROI object.
        file_no:  1-based file index within the scan directory.
        frame_no: frame index within the file.
    """
    direc = roi.data_dir
    if roi.beamtime == 'new':
        file_no_st = (6 - len(str(file_no))) * '0' + str(file_no)
        file_name = os.path.join(direc, f'Scan_{roi.scan_num}_data_{file_no_st}.h5')
        with h5py.File(file_name, 'r') as f:
            data = f['/entry/data/data'][frame_no, :, :]
    else:
        file_no_st = '0' * 5 + str(1)
        filename = os.path.join(direc, f'Scan_{roi.scan_num}_data_{file_no_st}.h5')
        data = stack_4d_data_old(filename, roi.coords, roi.fast_axis_steps, roi.slow_axis)[file_no, frame_no, :, :]

    if mask_hot:
        data = mask_hot_pixels(data)
    if title is None:
        title = f"Detector image — file {file_no}, frame {frame_no}"
    plot_roi_from_numpy(data, roi.coords, title, vmin=vmin, vmax=vmax, save=save)


def plot_full_detector(roi: ROI, file_no, frame_no,
                       vmin1=None, vmax1=None,
                       vmin2=None, vmax2=None):
    """Plot a full detector frame (linear + log scale side by side).

    Args:
        roi:      ROI object.
        file_no:  1-based file index.
        frame_no: frame index within the file.
    """
    direc = roi.data_dir
    try:
        file_no_st = (6 - len(str(file_no))) * '0' + str(file_no)
        file_name = os.path.join(direc, f'Scan_{roi.scan_num}_data_{file_no_st}.h5')
        with h5py.File(file_name, 'r') as f:
            data = f['/entry/data/data'][frame_no, :, :]
    except Exception:
        file_no_st = '0' * 5 + str(1)
        filename = os.path.join(direc, f'Scan_{roi.scan_num}_data_{file_no_st}.h5')
        data = stack_4d_data_old(filename, [0, -1, 0, -1], roi.fast_axis_steps, roi.slow_axis)[file_no, frame_no, :, :]

    data = mask_hot_pixels(data)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(data, cmap='jet', vmin=vmin1, vmax=vmax1)
    axes[0].set_title('Full Detector')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(np.log1p(data), cmap='jet', vmin=vmin2, vmax=vmax2)
    axes[1].set_title('Log Scale')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_4d_dataset(roi: ROI,
                    vmin1=None, vmax1=None,
                    vmin2=None, vmax2=None,
                    pixel_threshold=None):
    """Interactive viewer for a 4D dataset: coherent image ↔ detector image.

    Linked sliders let you pick a detector pixel to form the coherent image
    and a scan position to form the detector image.

    Args:
        roi: ROI with `data_4d` loaded (shape M×N×px×py).
    """
    coherent_shape = roi.data_4d.shape[:2]
    detector_shape = roi.data_4d.shape[2:]

    pcol_slider = widgets.IntSlider(min=0, max=detector_shape[1] - 1, value=detector_shape[1] // 2, description="px")
    prow_slider = widgets.IntSlider(min=0, max=detector_shape[0] - 1, value=detector_shape[0] // 2, description="py")
    lcol_slider = widgets.IntSlider(min=0, max=coherent_shape[1] - 1, value=coherent_shape[1] // 2, description="lx")
    lrow_slider = widgets.IntSlider(min=0, max=coherent_shape[0] - 1, value=coherent_shape[0] // 2, description="ly")

    rect_size_det = 4
    rect_size_coh = 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    coh_img = make_coherent_image(roi.data_4d, np.array([prow_slider.value, pcol_slider.value]))
    det_img = make_detector_image(roi.data_4d, np.array([lrow_slider.value, lcol_slider.value]))

    im0 = axes[0].imshow(coh_img, cmap='plasma', vmin=vmin1, vmax=vmax1)
    axes[0].set_title(f"Coherent Image (lx={lrow_slider.value}, ly={lcol_slider.value})")
    rect_coh = Rectangle(
        (lcol_slider.value - rect_size_coh / 2, lrow_slider.value - rect_size_coh / 2),
        rect_size_coh, rect_size_coh, edgecolor='white', facecolor='none', lw=2,
    )
    axes[0].add_patch(rect_coh)
    plt.colorbar(im0, ax=axes[0], label="Intensity")

    im1 = axes[1].imshow(det_img, cmap='viridis', vmin=vmin2, vmax=vmax2)
    axes[1].set_title(f"Detector Image (px={pcol_slider.value}, py={prow_slider.value})")
    rect_det = Rectangle(
        (pcol_slider.value - rect_size_det / 2, prow_slider.value - rect_size_det / 2),
        rect_size_det, rect_size_det, edgecolor='white', facecolor='none', lw=2,
    )
    axes[1].add_patch(rect_det)
    plt.colorbar(im1, ax=axes[1], label="Detector Intensity")
    plt.tight_layout()

    def update_plot(prow, pcol, lrow, lcol):
        coh = make_coherent_image(roi.data_4d, np.array([prow, pcol]))
        det = make_detector_image(roi.data_4d, np.array([lrow, lcol]))
        m = np.mean(coh)
        im0.set_data(coh)
        im0.set_clim(m - 0.05 * m, m + 0.05 * m)
        axes[0].set_title(f"Coherent Image from pixel ({pcol}, {prow})")
        rect_coh.set_xy((lcol - rect_size_coh / 2, lrow - rect_size_coh / 2))
        m = np.mean(det)
        im1.set_data(det)
        im1.set_clim(m - 0.3 * m, m + 0.2 * m)
        axes[1].set_title(f"Detector Image at location ({lcol}, {lrow})")
        rect_det.set_xy((pcol - rect_size_det / 2, prow - rect_size_det / 2))
        fig.canvas.draw_idle()

    display(widgets.interactive(update_plot,
                                prow=prow_slider, pcol=pcol_slider,
                                lrow=lrow_slider, lcol=lcol_slider))


def plot_coherent_sequence(roi: ROI, scale_factor=0.4, vmin=None, vmax=None):
    """Interactive slider viewer for `roi.coherent_imgs`."""
    img_list   = roi.coherent_imgs
    num_images = len(img_list)

    img_slider = widgets.IntSlider(min=0, max=num_images - 1, value=0, description="Image")
    vmin_box   = widgets.FloatText(value=0.1, description='Vmin:')
    vmax_box   = widgets.FloatText(value=1.0,  description='Vmax:')

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(img_list[0], cmap='viridis')
    ax.set_title(f"Coherent Image  0/{num_images - 1}")
    ax.set_xlabel('Scan-X')
    ax.set_ylabel('Scan-Y')
    plt.colorbar(im, ax=ax, label="Intensity")
    plt.tight_layout()

    def update_image(img_idx, vmin, vmax):
        im.set_data(img_list[img_idx])
        im.set_clim(vmin, vmax)
        ax.set_title(f"Coherent Image  {img_idx}/{num_images - 1}")
        fig.canvas.draw_idle()

    display(widgets.interactive(update_image, img_idx=img_slider, vmin=vmin_box, vmax=vmax_box))


def plot_averag_coh_imgs(roi: ROI, vmin=None, vmax=None, cmap='viridis', title=None):
    """Plot the average of `roi.coherent_imgs` with scan-axis labels."""
    avg = roi.averaged_coherent_images
    if title is None:
        title = f"Average Coherent Images: Scan {roi.scan_num}"

    if roi.slow_axis == 'scan_x':
        x_label, y_label = 'Scan-X: Slow Axis', 'Scan-Y: Fast Axis'
    else:
        x_label, y_label = 'Scan-X: Fast Axis', 'Scan-Y: Slow Axis'

    plt.figure()
    plt.imshow(avg, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar()
    plt.show()


def plot_detected_objects(roi: ROI):
    """Interactive slider viewer for `roi.detected_objects`."""
    img_list   = roi.detected_objects
    num_images = len(img_list)
    img_slider = widgets.IntSlider(min=0, max=num_images - 1, value=0, description="Image")

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(img_list[0], cmap='viridis')
    ax.set_title(f"Detected Object  0/{num_images - 1}")
    plt.colorbar(im, ax=ax, label="Intensity")
    plt.tight_layout()

    def update_image(img_idx):
        im.set_data(img_list[img_idx])
        ax.set_title(f"Detected Object  {img_idx}/{num_images - 1}")
        fig.canvas.draw_idle()

    display(widgets.interactive(update_image, img_idx=img_slider))


def plot_intensity_histograms(roi: ROI, bins=256):
    """Interactive slider viewer of per-image intensity histograms for `roi.coherent_imgs`."""
    histograms = compute_histograms(roi.coherent_imgs, bins=bins)
    num_images = len(histograms)
    img_slider = widgets.IntSlider(min=0, max=num_images - 1, value=0, description="Image")

    fig, ax = plt.subplots(figsize=(6, 6))
    line, = ax.plot(histograms[0])
    ax.set_yscale('log')
    ax.set_title(f"Intensity Histogram  0/{num_images - 1}")
    ax.set_ylabel("Log Frequency")
    plt.tight_layout()

    def update_image(img_idx):
        line.set_ydata(histograms[img_idx])
        ax.set_title(f"Intensity Histogram  {img_idx}/{num_images - 1}")
        fig.canvas.draw_idle()

    display(widgets.interactive(update_image, img_idx=img_slider))


def plot_average_roi(roi: ROI, vmin=None, vmax=None, title=None):
    """Plot the average of all detector frames in the ROI."""
    plot_roi_from_numpy(
        roi.averaged_det_images, [0, -1, 0, -1],
        title or "Averaged Detector Frames",
        vmin=vmin, vmax=vmax,
    )


def plot_calculated_kins(roi: ROI, pupil_roi: ROI,
                         vmin=None, vmax=None,
                         title="Mapped kins onto pupil",
                         cmap="viridis"):
    """Overlay `roi.kin_coords` onto the pupil averaged detector image."""
    plot_map_on_detector(
        pupil_roi.averaged_det_images, roi.kin_coords,
        vmin, vmax, title, cmap, crop=False, roi=pupil_roi.coords,
    )


def plot_kouts(roi: ROI, vmin=None, vmax=None,
               title="Mapped kouts", cmap="viridis"):
    """Overlay `roi.kout_coords` onto the ROI averaged detector image."""
    plot_map_on_detector(
        roi.averaged_det_images, roi.kout_coords,
        vmin, vmax, title, cmap, crop=False, roi=roi.coords,
    )
