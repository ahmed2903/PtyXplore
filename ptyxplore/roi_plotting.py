# ROI-based plotting utilities for ptychography data
#
# Author: Ahmed H. Mokhtar
# Email: ahmed.mokhtar@desy.de

import os
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.patches import Rectangle
import ipywidgets as widgets

from .loader import ROI
from .data_io import stack_4d_data_old
from .data_proc import (
    compute_histograms,
    make_coherent_image,
    make_detector_image,
    mask_hot_pixels,
)
from .plotting import plot_map_on_detector, plot_roi_from_numpy
from .roi_ops import tilt_correct

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
    im0 = axes[0].imshow(data, cmap='jet', vmin=vmin1, vmax=vmax1, origin='lower')
    axes[0].set_title('Full Detector')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(np.log1p(data), cmap='jet', vmin=vmin2, vmax=vmax2, origin='lower')
    axes[1].set_title('Log Scale')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def _scan_extent(roi):
    """Return (extent, xlabel, ylabel) for imshow of a coherent image.

    `extent` is [left, right, bottom, top] in nm, suitable for matplotlib's
    ``imshow(extent=...)``.  Returns ``None`` for extent when the scan-position
    attributes are not present on *roi*.

    The stacking functions always put scan_y on rows and scan_x on columns,
    whichever axis is the slow one.
    """
    _x = ("scan_x_start_point", "scan_x_end_point")
    _y = ("scan_y_start_point", "scan_y_end_point")
    if not (all(hasattr(roi, a) for a in _x) and all(hasattr(roi, a) for a in _y)):
        return None, "Scan X", "Scan Y"

    x0, x1 = roi.scan_x_start_point, roi.scan_x_end_point
    y0, y1 = roi.scan_y_start_point, roi.scan_y_end_point

    # rows = scan_y, cols = scan_x
    extent = [x0, x1, y0, y1]      # [left, right, bottom, top], origin='lower'
    xlabel, ylabel = "Scan X (nm)", "Scan Y (nm)"

    return extent, xlabel, ylabel


def plot_4d_dataset(roi: ROI,
                    vmin1=None, vmax1=None,
                    vmin2=None, vmax2=None,
                    pixel_threshold=None,
                    use_positions=False):
    """Interactive viewer for a 4D dataset: coherent image ↔ detector image.

    Linked sliders let you pick a detector pixel to form the coherent image
    and a scan position to form the detector image.

    Args:
        roi:           ROI with `data_4d` loaded (shape M×N×px×py).
        use_positions: If True, label the coherent-image axes with motor
                       positions in nm instead of pixel indices.
    """
    coherent_shape = roi.data_4d.shape[:2]
    detector_shape = roi.data_4d.shape[2:]

    pcol_slider = widgets.IntSlider(min=0, max=detector_shape[1] - 1, value=detector_shape[1] // 2, description="px")
    prow_slider = widgets.IntSlider(min=0, max=detector_shape[0] - 1, value=detector_shape[0] // 2, description="py")
    lcol_slider = widgets.IntSlider(min=0, max=coherent_shape[1] - 1, value=coherent_shape[1] // 2, description="lx")
    lrow_slider = widgets.IntSlider(min=0, max=coherent_shape[0] - 1, value=coherent_shape[0] // 2, description="ly")

    rect_size_det = 4
    rect_size_coh = 2

    # Position extent for coherent image
    extent, coh_xlabel, coh_ylabel = _scan_extent(roi) if use_positions else (None, "Scan X", "Scan Y")

    # Helpers to convert pixel index → nm for the rectangle overlay
    def _coh_col_to_nm(col):
        if extent is None:
            return col
        left, right = extent[0], extent[1]
        return left + col * (right - left) / max(coherent_shape[1] - 1, 1)

    def _coh_row_to_nm(row):
        if extent is None:
            return row
        top, bottom = extent[3], extent[2]
        return top + row * (bottom - top) / max(coherent_shape[0] - 1, 1)

    def _coh_size_col(n):
        if extent is None:
            return n
        return n * (extent[1] - extent[0]) / max(coherent_shape[1] - 1, 1)

    def _coh_size_row(n):
        if extent is None:
            return n
        return n * (extent[2] - extent[3]) / max(coherent_shape[0] - 1, 1)

    # Global data ranges for slider bounds
    coh_data_min, coh_data_max = 0, roi.data_4d.max()
    det_data_min, det_data_max = 0, roi.data_4d.max()
    coh_step = max((coh_data_max - coh_data_min) / 500, 1e-6)
    det_step = max((det_data_max - det_data_min) / 500, 1e-6)

    clim1_slider = widgets.FloatRangeSlider(
        value=[vmin1 if vmin1 is not None else coh_data_min,
               vmax1 if vmax1 is not None else coh_data_max],
        min=coh_data_min, max=coh_data_max, step=coh_step,
        description='Coh range', continuous_update=True,
        layout=widgets.Layout(width='60%'),
    )
    clim2_slider = widgets.FloatRangeSlider(
        value=[vmin2 if vmin2 is not None else det_data_min,
               vmax2 if vmax2 is not None else det_data_max],
        min=det_data_min, max=det_data_max, step=det_step,
        description='Det range', continuous_update=True,
        layout=widgets.Layout(width='60%'),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    coh_img = make_coherent_image(roi.data_4d, np.array([prow_slider.value, pcol_slider.value]))
    det_img = make_detector_image(roi.data_4d, np.array([lrow_slider.value, lcol_slider.value]))

    if extent is not None:
        im0 = axes[0].imshow(coh_img, cmap='plasma', vmin=clim1_slider.value[0], vmax=clim1_slider.value[1], extent=extent, origin='lower')
    else:
        im0 = axes[0].imshow(coh_img, cmap='plasma', vmin=clim1_slider.value[0], vmax=clim1_slider.value[1], origin='lower')
    axes[0].set_title(f"Coherent Image (lx={lrow_slider.value}, ly={lcol_slider.value})")
    axes[0].set_xlabel(coh_xlabel)
    axes[0].set_ylabel(coh_ylabel)
    rect_coh = Rectangle(
        (_coh_col_to_nm(lcol_slider.value) - _coh_size_col(rect_size_coh) / 2,
         _coh_row_to_nm(lrow_slider.value) - _coh_size_row(rect_size_coh) / 2),
        _coh_size_col(rect_size_coh), _coh_size_row(rect_size_coh),
        edgecolor='white', facecolor='none', lw=2,
    )
    axes[0].add_patch(rect_coh)
    plt.colorbar(im0, ax=axes[0], label="Intensity")

    im1 = axes[1].imshow(det_img, cmap='viridis', vmin=clim2_slider.value[0], vmax=clim2_slider.value[1], origin='lower')
    axes[1].set_title(f"Detector Image (px={pcol_slider.value}, py={prow_slider.value})")
    rect_det = Rectangle(
        (pcol_slider.value - rect_size_det / 2, prow_slider.value - rect_size_det / 2),
        rect_size_det, rect_size_det, edgecolor='white', facecolor='none', lw=2,
    )
    axes[1].add_patch(rect_det)
    plt.colorbar(im1, ax=axes[1], label="Detector Intensity")
    plt.tight_layout()

    def update_plot(prow, pcol, lrow, lcol, clim1, clim2):
        coh = make_coherent_image(roi.data_4d, np.array([prow, pcol]))
        det = make_detector_image(roi.data_4d, np.array([lrow, lcol]))
        im0.set_data(coh)
        im0.set_clim(*clim1)
        axes[0].set_title(f"Coherent Image from pixel ({pcol}, {prow})")
        rect_coh.set_xy((_coh_col_to_nm(lcol) - _coh_size_col(rect_size_coh) / 2,
                         _coh_row_to_nm(lrow) - _coh_size_row(rect_size_coh) / 2))
        im1.set_data(det)
        im1.set_clim(*clim2)
        axes[1].set_title(f"Detector Image at location ({lcol}, {lrow})")
        rect_det.set_xy((pcol - rect_size_det / 2, prow - rect_size_det / 2))
        fig.canvas.draw_idle()

    display(widgets.interactive(update_plot,
                                prow=prow_slider, pcol=pcol_slider,
                                lrow=lrow_slider, lcol=lcol_slider,
                                clim1=clim1_slider, clim2=clim2_slider))


def plot_coherent_sequence(roi: ROI, scale_factor=0.4, vmin=None, vmax=None,
                           use_positions=False):
    """Interactive slider viewer for `roi.coherent_imgs`.

    Handles both single-streak ROIs (coherent_imgs is an (N, H, W) array) and
    multi-streak ROIs (coherent_imgs is a list of (N_s, H, W) arrays).
    Multi-streak mode adds a Streak slider that updates the Image slider range.

    Args:
        use_positions: If True, label axes with motor positions in nm.
    """
    multi = isinstance(roi.coherent_imgs, list)
    extent, xlabel, ylabel = _scan_extent(roi) if use_positions else (None, "Scan X", "Scan Y")

    if multi:
        all_imgs  = roi.coherent_imgs
        n_streaks = len(all_imgs)

        streak_slider = widgets.IntSlider(min=0, max=n_streaks - 1, value=0, description="Streak")
        img_slider    = widgets.IntSlider(min=0, max=len(all_imgs[0]) - 1, value=0, description="Image")
        vmin_box      = widgets.FloatText(value=vmin if vmin is not None else 0.1, description='Vmin:')
        vmax_box      = widgets.FloatText(value=vmax if vmax is not None else 1.0,  description='Vmax:')

        fig, ax = plt.subplots(figsize=(6, 6))
        first = np.array(all_imgs[0][0])
        im = ax.imshow(first, cmap='viridis', **({"extent": extent} if extent else {}), origin='lower')
        ax.set_title(f"Streak 0 — Image 0/{len(all_imgs[0]) - 1}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, label="Intensity")
        plt.tight_layout()

        def _update_img_range(change):
            s = streak_slider.value
            img_slider.max   = len(all_imgs[s]) - 1
            img_slider.value = min(img_slider.value, img_slider.max)

        streak_slider.observe(_update_img_range, names='value')

        def update_image(streak_idx, img_idx, vmin, vmax):
            imgs    = all_imgs[streak_idx]
            img_idx = min(img_idx, len(imgs) - 1)
            im.set_data(np.array(imgs[img_idx]))
            im.set_clim(vmin, vmax)
            ax.set_title(f"Streak {streak_idx} — Image {img_idx}/{len(imgs) - 1}")
            fig.canvas.draw_idle()

        display(widgets.interactive(update_image,
                                    streak_idx=streak_slider, img_idx=img_slider,
                                    vmin=vmin_box, vmax=vmax_box))
    else:
        img_list   = roi.coherent_imgs
        num_images = len(img_list)

        img_slider = widgets.IntSlider(min=0, max=num_images - 1, value=0, description="Image")
        vmin_box   = widgets.FloatText(value=vmin if vmin is not None else 0.1, description='Vmin:')
        vmax_box   = widgets.FloatText(value=vmax if vmax is not None else 1.0,  description='Vmax:')

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(img_list[0], cmap='viridis', **({"extent": extent} if extent else {}), origin='lower')
        ax.set_title(f"Coherent Image  0/{num_images - 1}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, label="Intensity")
        plt.tight_layout()

        def update_image(img_idx, vmin, vmax):
            im.set_data(img_list[img_idx])
            im.set_clim(vmin, vmax)
            ax.set_title(f"Coherent Image  {img_idx}/{num_images - 1}")
            fig.canvas.draw_idle()

        display(widgets.interactive(update_image, img_idx=img_slider, vmin=vmin_box, vmax=vmax_box))


def _plot_coherent_image(roi: ROI, img, title: str, vmin: float | None = None,
                         vmax: float | None = None, cmap: str = 'viridis',
                         use_positions: bool = False):
    """Plot a 2D coherent (scan-space) image with scan-axis labels."""
    # Resolve extent and axis labels
    if use_positions:
        extent, xlabel, ylabel = _scan_extent(roi)
    else:
        extent = None

    if extent is None:
        if getattr(roi, 'slow_axis', None) == 'scan_x':
            xlabel, ylabel = 'Scan-X: Slow Axis', 'Scan-Y: Fast Axis'
        else:
            xlabel, ylabel = 'Scan-X: Fast Axis', 'Scan-Y: Slow Axis'

    # Build imshow kwargs to avoid passing None for typed parameters
    ikw: dict[str, Any] = {"cmap": cmap}
    if vmin is not None:
        ikw["vmin"] = vmin
    if vmax is not None:
        ikw["vmax"] = vmax
    if extent is not None:
        ikw["extent"] = extent

    _, ax = plt.subplots()
    im = ax.imshow(img, **ikw, origin='lower')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)
    plt.show()


def plot_averag_coh_imgs(roi: ROI, vmin: float | None = None, vmax: float | None = None,
                         cmap: str = 'viridis', title: str | None = None,
                         use_positions: bool = False):
    """Plot `roi.averaged_coherent_images` (data_4d averaged over the detector) with scan-axis labels.

    Args:
        use_positions: If True, label axes with motor positions in nm instead
                       of generic 'Scan-X / Scan-Y' labels.
    """
    if title is None:
        title = f"Averaged Coherent Image: Scan {roi.scan_num}"
    _plot_coherent_image(roi, roi.averaged_coherent_images, title,
                         vmin=vmin, vmax=vmax, cmap=cmap, use_positions=use_positions)


def plot_summed_coh_imgs(roi: ROI, vmin: float | None = None, vmax: float | None = None,
                         cmap: str = 'viridis', title: str | None = None,
                         use_positions: bool = False):
    """Plot `roi.summed_coherent_images` (data_4d summed over the detector) with scan-axis labels.

    Args:
        use_positions: If True, label axes with motor positions in nm instead
                       of generic 'Scan-X / Scan-Y' labels.
    """
    if title is None:
        title = f"Summed Coherent Image: Scan {roi.scan_num}"
    _plot_coherent_image(roi, roi.summed_coherent_images, title,
                         vmin=vmin, vmax=vmax, cmap=cmap, use_positions=use_positions)


def plot_tilt_corrected_image(roi: ROI, vmin: float | None = None, vmax: float | None = None,
                         cmap: str = 'viridis', title: str | None = None,
                         use_positions: bool = False):
    """Plot `roi.tilt_corrected_image` with scan-axis labels.

    Args:
        use_positions: If True, label axes with motor positions in nm instead
                       of generic 'Scan-X / Scan-Y' labels.
    """
    if roi.tilt_corrected_image is None: 
        tilt_correct(roi = roi, iters = 3)

    if title is None:
        title = f"Tilt-corrected Coherent Image: Scan {roi.scan_num}"
    _plot_coherent_image(roi, roi.tilt_corrected_image, title,
                         vmin=vmin, vmax=vmax, cmap=cmap, use_positions=use_positions)


def plot_detected_objects(roi: ROI):
    """Interactive slider viewer for `roi.detected_objects`."""
    img_list   = roi.detected_objects
    num_images = len(img_list)
    img_slider = widgets.IntSlider(min=0, max=num_images - 1, value=0, description="Image")

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(img_list[0], cmap='viridis', origin='lower')
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
        roi.averaged_det_images, None,
        title or "Averaged Detector Frames",
        vmin=vmin, vmax=vmax,
    )
def plot_sum_roi(roi: ROI, vmin=None, vmax=None, title=None):
    """Plot the sum of all detector frames in the ROI."""
    plot_roi_from_numpy(
        roi.summed_det_images, None,
        title or "Summed Detector Frames",
        vmin=vmin, vmax=vmax,
    )

def plot_calculated_kins(roi: ROI, pupil_roi: ROI,
                         vmin=None, vmax=None,
                         title="Mapped kins onto pupil",
                         cmap="viridis"):
    """Overlay `roi.kin_coords` onto the pupil averaged detector image.

    For multi-streak ROIs (kin_coords is a list) an interactive Streak slider
    is shown so each streak's kin positions can be inspected individually.
    """
    if isinstance(roi.kin_coords, list):
        n_streaks     = len(roi.kin_coords)
        streak_slider = widgets.IntSlider(min=0, max=n_streaks - 1, value=0, description="Streak")
        out           = widgets.Output()

        def _render(change=None):
            with out:
                out.clear_output(wait=True)
                s = streak_slider.value
                plot_map_on_detector(
                    pupil_roi.averaged_det_images, roi.kin_coords[s],
                    vmin, vmax, f"{title} — Streak {s}", cmap,
                    crop=False, roi=pupil_roi.coords,
                )

        streak_slider.observe(_render, names='value')
        _render()
        display(widgets.VBox([streak_slider, out]))
    else:
        plot_map_on_detector(
            pupil_roi.averaged_det_images, roi.kin_coords,
            vmin, vmax, title, cmap, crop=False, roi=pupil_roi.coords,
        )


def plot_kouts(roi: ROI, vmin=None, vmax=None,
               title="Mapped kouts", cmap="viridis"):
    """Overlay `roi.kout_coords` onto the ROI averaged detector image.

    For multi-streak ROIs (kout_coords is a list) an interactive Streak slider
    is shown so each streak can be inspected individually.
    """
    if isinstance(roi.kout_coords, list):
        n_streaks     = len(roi.kout_coords)
        streak_slider = widgets.IntSlider(min=0, max=n_streaks - 1, value=0, description="Streak")
        out           = widgets.Output()

        def _render(change=None):
            with out:
                out.clear_output(wait=True)
                s = streak_slider.value
                plot_map_on_detector(
                    roi.averaged_det_images, roi.kout_coords[s],
                    vmin, vmax, f"{title} — Streak {s}", cmap,
                    crop=False, roi=roi.coords,
                )

        streak_slider.observe(_render, names='value')
        _render()
        display(widgets.VBox([streak_slider, out]))
    else:
        plot_map_on_detector(
            roi.averaged_det_images, roi.kout_coords,
            vmin, vmax, title, cmap, crop=False, roi=roi.coords,
        )
