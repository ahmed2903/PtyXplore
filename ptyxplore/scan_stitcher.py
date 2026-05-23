# Scan stitcher — combine averaged coherent images from multiple ROIs
# into a single image aligned by physical motor positions.
#
# Author: Ahmed H. Mokhtar
# Email:  ahmed.mokhtar@desy.de

import numpy as np


# ─── helpers ──────────────────────────────────────────────────────────────────

def get_pixel_positions(roi) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_nm, Y_nm) 2-D arrays matching `roi.averaged_coherent_images`.

    ``X_nm[i, j]`` and ``Y_nm[i, j]`` are the physical X- and Y-motor positions
    (in nm) for pixel ``[i, j]`` of the averaged coherent image.

    Requires the ROI (or its parent Scan) to carry the attributes
    ``scan_x_start_point``, ``scan_x_end_point``,
    ``scan_y_start_point``, ``scan_y_end_point``, and ``slow_axis``.

    Raises
    ------
    AttributeError
        When any of the required scan-position attributes are missing.
    ValueError
        When ``roi.averaged_coherent_images`` is None.
    """
    img = roi.averaged_coherent_images
    if img is None:
        raise ValueError(f"ROI has no averaged_coherent_images — run Make coherent images first.")

    required = ("scan_x_start_point", "scan_x_end_point",
                "scan_y_start_point", "scan_y_end_point")
    missing = [a for a in required if not hasattr(roi, a)]
    if missing:
        raise AttributeError(f"ROI is missing scan-position attributes: {missing}")

    n_rows, n_cols = img.shape
    x0, x1 = roi.scan_x_start_point, roi.scan_x_end_point
    y0, y1 = roi.scan_y_start_point, roi.scan_y_end_point

    if getattr(roi, "slow_axis", None) == "scan_x":
        # rows → X motor, cols → Y motor
        x_row = np.linspace(x0, x1, n_rows)
        y_col = np.linspace(y0, y1, n_cols)
        X_nm, Y_nm = np.meshgrid(x_row, y_col, indexing="ij")   # (n_rows, n_cols)
    else:
        # rows → Y motor, cols → X motor
        y_row = np.linspace(y0, y1, n_rows)
        x_col = np.linspace(x0, x1, n_cols)
        Y_nm, X_nm = np.meshgrid(y_row, x_col, indexing="ij")   # (n_rows, n_cols)

    return X_nm, Y_nm


def _min_step(roi) -> float:
    """Smallest motor step size (nm) present in this ROI."""
    img = roi.averaged_coherent_images
    n_rows, n_cols = img.shape
    x0, x1 = roi.scan_x_start_point, roi.scan_x_end_point
    y0, y1 = roi.scan_y_start_point, roi.scan_y_end_point

    steps = []
    if n_rows > 1:
        axis0_span = abs(x1 - x0) if getattr(roi, "slow_axis", None) == "scan_x" else abs(y1 - y0)
        steps.append(axis0_span / (n_rows - 1))
    if n_cols > 1:
        axis1_span = abs(y1 - y0) if getattr(roi, "slow_axis", None) == "scan_x" else abs(x1 - x0)
        steps.append(axis1_span / (n_cols - 1))

    return min(steps) if steps else 1.0


# ─── main stitching function ───────────────────────────────────────────────────

def stitch(
    roi_list,
    pixel_size: float | None = None,
    mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stitch averaged coherent images from multiple ROIs by motor position.

    Each pixel of every ``roi.averaged_coherent_images`` is placed on a common
    Cartesian grid whose axes correspond to physical X- and Y-motor positions
    (in nm).  Overlapping pixels are combined according to *mode*.

    Parameters
    ----------
    roi_list:
        Iterable of ROI objects.  Each must have ``averaged_coherent_images``
        computed and carry scan-position attributes (see `get_pixel_positions`).
    pixel_size:
        Output grid spacing in nm.  Defaults to the smallest step size found
        across all input ROIs.
    mode:
        How to combine overlapping pixels.
        ``'mean'`` — average; ``'max'`` — element-wise maximum;
        ``'sum'`` — accumulate.

    Returns
    -------
    stitched : np.ndarray, shape (n_y, n_x), float32
        Stitched image.  Pixels with no data are ``np.nan``.
        Rows index the Y motor axis (row 0 = y_min),
        columns index the X motor axis (col 0 = x_min).
    x_axis : np.ndarray, shape (n_x,)
        X motor positions (nm) for each column of *stitched*.
    y_axis : np.ndarray, shape (n_y,)
        Y motor positions (nm) for each row of *stitched*.
    """
    if mode not in ("mean", "max", "sum"):
        raise ValueError(f"Unknown mode '{mode}'. Use 'mean', 'max', or 'sum'.")

    # ── collect per-ROI data ──────────────────────────────────────────────────
    items = []
    for roi in roi_list:
        X, Y = get_pixel_positions(roi)          # raises if not ready
        items.append({"img": roi.averaged_coherent_images, "X": X, "Y": Y, "roi": roi})

    if not items:
        raise ValueError("roi_list is empty.")

    # ── global extent ─────────────────────────────────────────────────────────
    x_min = min(d["X"].min() for d in items)
    x_max = max(d["X"].max() for d in items)
    y_min = min(d["Y"].min() for d in items)
    y_max = max(d["Y"].max() for d in items)

    # ── output grid ───────────────────────────────────────────────────────────
    if pixel_size is None:
        pixel_size = min(_min_step(d["roi"]) for d in items)

    n_x = max(2, int(round((x_max - x_min) / pixel_size)) + 1)
    n_y = max(2, int(round((y_max - y_min) / pixel_size)) + 1)

    x_axis = np.linspace(x_min, x_min + (n_x - 1) * pixel_size, n_x)
    y_axis = np.linspace(y_min, y_min + (n_y - 1) * pixel_size, n_y)

    # ── accumulate ────────────────────────────────────────────────────────────
    if mode in ("mean", "sum"):
        out_acc = np.zeros((n_y, n_x), dtype=np.float64)
        out_cnt = np.zeros((n_y, n_x), dtype=np.int32)
    else:  # max
        out_acc = np.full((n_y, n_x), np.nan, dtype=np.float64)

    for d in items:
        img = d["img"].astype(np.float64)
        ix = np.round((d["X"] - x_min) / pixel_size).astype(np.intp)
        iy = np.round((d["Y"] - y_min) / pixel_size).astype(np.intp)

        valid = (ix >= 0) & (ix < n_x) & (iy >= 0) & (iy < n_y)
        ix_v, iy_v, img_v = ix[valid], iy[valid], img[valid]

        if mode in ("mean", "sum"):
            np.add.at(out_acc, (iy_v, ix_v), img_v)
            np.add.at(out_cnt, (iy_v, ix_v), 1)
        else:  # max — use fmax so NaN is overwritten by any real value
            np.fmax.at(out_acc, (iy_v, ix_v), img_v)

    # ── finalise ──────────────────────────────────────────────────────────────
    if mode == "mean":
        with np.errstate(invalid="ignore"):
            stitched = np.where(out_cnt > 0, out_acc / out_cnt, np.nan)
    elif mode == "sum":
        stitched = np.where(out_cnt > 0, out_acc, np.nan)
    else:
        stitched = out_acc   # already NaN where no data

    return stitched.astype(np.float32), x_axis, y_axis
