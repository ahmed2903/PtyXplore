import numpy as np
from scipy.optimize import brentq
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import math



# ___________________ Global G (one reflection, one G for every pixel) ___________________
#
# For one Bragg reflection every dark-field pixel shares the same G. Pixels away from the
# exact Bragg condition are NOT errors: their deviation is q_z, the depth information.
# So G is fitted once for the whole ROI, and every pixel gets
#
#     k_in,perp = k_out,perp - G_perp,   k_in,z = sqrt(k^2 - |k_in,perp|^2)   (elastic)
#     q_z(0)    = k_out,z - k_in,z - G_z
#
# k_in is the incident direction paired with the pixel at scan frequency K = 0, i.e. the
# pupil position of its coherent image.

def kins_from_G(kouts, G, k):
    """Incident vectors paired with each k_out for a single G, put on the elastic sphere."""
    kins = np.empty_like(kouts, dtype=float)
    kins[:, :2] = kouts[:, :2] - G[:2]
    kins[:, 2] = np.sqrt(np.clip(k**2 - np.sum(kins[:, :2]**2, axis=1), 0.0, None))
    return kins


def qz_from_G(kouts, G, k):
    """Ewald q_z of every pixel at K = 0 (deviation from the Bragg condition), (N,)."""
    return kouts[:, 2] - kins_from_G(kouts, G, k)[:, 2] - G[2]


def _footprint_centre(ks_2d, direction):
    p = ks_2d @ direction
    return 0.5 * (p.max() + p.min())


def fit_global_G(kouts, pupil_kins, wavelength, weights=None, ttheta=None,
                 g_u_from_footprint=False):
    """
    One G for all dark-field pixels of a reflection.

    Directions: u = from the pupil centre towards the dark-field centroid (the 2θ direction,
    across the streaks); v = perpendicular to u (along the streaks).

      G_v : the dark-field footprint is a rigid copy of the pupil along v, so
            G_v = centre(DF along v) - centre(pupil along v).
      G_u : if g_u_from_footprint, the same footprint rule along u (only valid when the
            relrod lights up the whole pupil width, i.e. thin crystals). Otherwise fixed
            together with G_z by |G| = 2k sin(θ) and the condition below.
      G_z : intensity-weighted mean of q_z(0) = 0, i.e. the relrod centre is at q_z = 0.

    A global error in G is mostly harmless for the 3D reconstruction (G_z -> phase ramp
    along z; G_perp -> a global pupil registration shift). Errors that differ between
    pixels are what matter, and this construction has none.

    Args:
        kouts      : (N, 3) dark-field k_out (rad / length unit of wavelength).
        pupil_kins : (M, 3) bright-field (pupil) k vectors.
        wavelength : same length unit as used for the k vectors (k = 2π/λ).
        weights    : (N,) dark-field intensities; None -> uniform.
        ttheta     : 2θ in radians for |G| = 2k sin(θ); None -> angle between the
                     weighted dark-field centroid and the pupil centre.
        g_u_from_footprint : see above.

    Returns:
        G    : (3,)
        info : dict with u, v, ttheta, qz0 (per pixel), frac_outside_pupil.
    """
    k = 2.0 * np.pi / wavelength
    kouts = np.asarray(kouts, dtype=float)
    pupil_kins = np.asarray(pupil_kins, dtype=float)
    w = np.ones(len(kouts)) if weights is None else np.clip(np.asarray(weights, dtype=float), 0, None)
    w = w / w.sum()

    (px0, px1), (py0, py1) = compute_k_bounds(pupil_kins)
    c = np.array([0.5 * (px0 + px1), 0.5 * (py0 + py1)])
    kin_c = np.array([c[0], c[1], np.sqrt(k**2 - c @ c)])

    k_w = w @ kouts
    d = k_w[:2] - c
    u = d / np.linalg.norm(d)
    v = np.array([-u[1], u[0]])

    G_v = _footprint_centre(kouts[:, :2], v) - _footprint_centre(pupil_kins[:, :2], v)

    def _G(G_u, G_z):
        G = np.empty(3)
        G[:2] = G_u * u + G_v * v
        G[2] = G_z
        return G

    def _Gz_from_qz(G_u):
        # q_z is linear in G_z: choose G_z so the weighted mean of q_z(0) is zero
        kin_z = kins_from_G(kouts, _G(G_u, 0.0), k)[:, 2]
        return w @ (kouts[:, 2] - kin_z)

    if g_u_from_footprint:
        G_u = _footprint_centre(kouts[:, :2], u) - _footprint_centre(pupil_kins[:, :2], u)
        G = _G(G_u, _Gz_from_qz(G_u))
        tth_used = 2 * np.arcsin(np.linalg.norm(G) / (2 * k))
    else:
        if ttheta is None:
            ttheta = np.arccos(np.clip(k_w @ kin_c / (np.linalg.norm(k_w) * k), -1, 1))
        G_mag = 2.0 * k * np.sin(ttheta / 2.0)
        G_u_max = np.sqrt(max(G_mag**2 - G_v**2, 0.0))

        def resid(G_u):
            G_z = -np.sqrt(max(G_mag**2 - G_u**2 - G_v**2, 0.0))
            return w @ qz_from_G(kouts, _G(G_u, G_z), k)

        G_u0 = d @ u
        grid = np.clip(G_u0 * np.linspace(0.8, 1.2, 41), 0.0, G_u_max)
        r = np.array([resid(g) for g in grid])
        sign_change = np.nonzero(np.diff(np.sign(r)))[0]
        if len(sign_change):
            i = sign_change[np.argmin(np.abs(grid[sign_change] - G_u0))]
            G_u = brentq(resid, grid[i], grid[i + 1], xtol=1e-12)
        else:
            G_u = grid[np.argmin(np.abs(r))]
            print("fit_global_G: no exact root for the q_z condition; using the best grid value")
        G = _G(G_u, -np.sqrt(max(G_mag**2 - G_u**2 - G_v**2, 0.0)))
        tth_used = ttheta

    kins = kins_from_G(kouts, G, k)
    outside = ((kins[:, 0] < px0) | (kins[:, 0] > px1) | (kins[:, 1] < py0) | (kins[:, 1] > py1))
    info = dict(u=u, v=v, ttheta=tth_used, qz0=qz_from_G(kouts, G, k),
                frac_outside_pupil=float(outside.mean()))
    return G, info


def shift_G(G, dk_perp, k, kouts=None, weights=None, keep='qz0'):
    """
    Move the streak in the pupil by dk_perp = (dk_x, dk_y): G_perp -> G_perp - dk_perp, so
    every k_in,perp = k_out,perp - G_perp moves rigidly by +dk_perp. The kins stay on the
    elastic sphere (kins_from_G). Only G_z needs a rule:

      keep='G_mag' : |G| fixed, i.e. G is rotated (d-spacing / 2θ unchanged). The relrod
                     centre moves off q_z = 0, by about -cot(θ)·(u·dk_perp) for a shift
                     across the streaks and only ~NA·|dk| along them.
      keep='qz0'   : G_z chosen so the weighted mean of q_z(0) stays 0 (|G| changes),
                     same rule as fit_global_G. Needs kouts.

    The two differ only by a constant G_z, i.e. a uniform q_z offset (a z phase ramp); the
    relative q_z between pixels is the same.
    """
    G = np.asarray(G, dtype=float)
    G_new = G.copy()
    G_new[:2] = G[:2] - np.asarray(dk_perp, dtype=float)

    if keep == 'G_mag':
        gz2 = G @ G - G_new[:2] @ G_new[:2]
        if gz2 < 0:
            raise ValueError("shift_G: shift too large to keep |G| fixed")
        G_new[2] = np.sign(G[2]) * np.sqrt(gz2)
    elif keep == 'qz0':
        if kouts is None:
            raise ValueError("shift_G: keep='qz0' needs kouts")
        w = np.ones(len(kouts)) if weights is None else np.clip(np.asarray(weights, dtype=float), 0, None)
        w = w / w.sum()
        G_new[2] = 0.0
        G_new[2] = w @ qz_from_G(kouts, G_new, k)
    else:
        raise ValueError("keep must be 'G_mag' or 'qz0'")
    return G_new


def make_coordinates(array, mask_val, roi, crop=False):

    """
    Args:
        array (ndarray): The array from which to calculate the coordinates. 
        mask_val (float): Only pixels above this value will be considered
        roi (list or tuple): the region of interest in pixels (row_start, row_end, column_start, column_end)
    
    Returns:
        coords (ndarray): (N,2) array that is structured as (rows, columns) 
    """
    if crop:
        array = array[roi[0]:roi[1], roi[2]:roi[3]]
        
    indices = np.where(array > mask_val)
    
    coords = np.array([(int(i)+ roi[0], int(j)+roi[2]) for i, j in zip(indices[0], indices[1])])

    return coords

def reverse_kins_to_pixels(kins, pixel_size, detector_distance, central_pixel):
    """
    Reverse map k_out vectors to detector pixel indices.

    Args:
        kouts (np.ndarray): Array of k_out vectors (N, 3), normalized.
        pixel_size (float): Pixel size in micrometers.
        detector_distance (float): Distance from the crystal to the center of the detector in micro meters.
        detector_shape: Tuple (num_rows, num_cols) of the detector
    Returns:
        np.ndarray: Array of pixel indices for k_out vectors.
    """

    #cen_x,cen_y = np.array(detector_shape)/2

    cen_row, cen_col = np.array(central_pixel)
    
    # pixel size in meters
    pixel_size = np.array(pixel_size)
    kins = kins / np.linalg.norm(kins, axis=1)[:, np.newaxis]
    
    # Reverse mapping to pixel indices
    row_pixels = (kins[:, 0] / kins[:, 2]) * detector_distance / pixel_size + cen_row
    col_pixels = (kins[:, 1] / kins[:, 2]) * detector_distance / pixel_size + cen_col
    
    # Convert to the nearest integer pixel indices
    row_pixel_indices = np.rint(row_pixels).astype(int)
    col_pixel_indices = np.rint(col_pixels).astype(int)

    coord = np.vstack((row_pixel_indices, col_pixel_indices)).T

    return coord

def compute_k_bounds(ks):

    kx_min, kx_max = np.min(ks[:,0]), np.max(ks[:,0])
    ky_min, ky_max = np.min(ks[:,1]), np.max(ks[:,1])
    return (kx_min, kx_max), (ky_min, ky_max)
    
def compute_vectors(coordinates, detector_distance, pixel_size, central_pixel, wavelength):
    """
    Compute vectors from the origin to each pixel.

    Parameters:
    - coordinates: (N, 2) array of pixel indices (row, col).
    - detector_distance: Distance from origin to detector center.
    - pixel_size: Pixel size [units] # FIX ME!
    - central_pixel: Tuple (row, col) of the detector central pixel.

    Returns:
    - vectors: (N, 3) array of vectors from origin to each pixel.
    """

    center_row, center_col = np.array(central_pixel) * pixel_size
    
    vectors = []
    for coord in coordinates:
        i, j = coord
        row = i * pixel_size - center_row
        col = j * pixel_size - center_col
        
        vector = np.array([row, col, detector_distance])
        vectors.append(vector)

    pixel_vectors = np.array(vectors)
    
    unit_vectors = pixel_vectors/np.linalg.norm(pixel_vectors, axis = 1)[:,np.newaxis]
    
    k = 2.0*np.pi /wavelength

    unit_vectors = np.array(unit_vectors)
    
    ks = k*unit_vectors

    return ks

def extract_streak_region(kins, percentage=10, start_position='random', start_idx=None, seed=42):
    np.random.seed(seed)  # Seed for reproducibility
    kins = np.asarray(kins)

    if kins.ndim != 2 or kins.shape[1] < 2:
        raise ValueError(f"Expected kins to be (N, 2+) but got {kins.shape}")

    kins = kins[:, :2]
    num_pixels = len(kins)
    num_to_select = max(1, int(num_pixels * (percentage / 100)))

    # Determine starting index
    if start_idx is None:
        sorted_indices = np.argsort(kins[:, 0] + kins[:, 1])  # Sort by sum(x, y)

        if start_position == 'lowest':
            start_idx = sorted_indices[0]
        elif start_position == 'highest':
            start_idx = sorted_indices[-1]
        elif start_position == 'middle':
            start_idx = sorted_indices[len(sorted_indices) // 2]
        elif start_position == 'random':
            start_idx = np.random.choice(num_pixels)
        else:
            raise ValueError("start_position must be 'lowest', 'highest', 'middle', or 'random'")

    # Use KDTree for efficient nearest-neighbor search
    tree = KDTree(kins)
    selected_mask = np.zeros(num_pixels, dtype=bool)
    selected_mask[start_idx] = True

    # Initialize BFS queue
    selected_points = [start_idx]
    
    while len(selected_points) < num_to_select:
        new_candidates = []
        for idx in selected_points:
            # Find nearest neighbors (restrict search to small neighborhood)
            _, neighbors = tree.query(kins[idx], k=min(10, num_pixels))  
            for n_idx in neighbors:
                if not selected_mask[n_idx]:  # Only add unselected points
                    selected_mask[n_idx] = True
                    new_candidates.append(n_idx)
                    if len(selected_points) + len(new_candidates) >= num_to_select:
                        break
            if len(selected_points) + len(new_candidates) >= num_to_select:
                break

        selected_points.extend(new_candidates)

    return selected_mask

    
def get_line_direction(kins_or_kouts):
    """Compute principal component (streak direction)"""

    pca = PCA(n_components=2)
    pca.fit(kins_or_kouts)
    direction = pca.components_[0]  
    return direction


def calc_effective_pixel_size(kouts, det_psize):

    kouts_avg = np.mean(kouts, axis = 0)
    azimuth = np.arctan2(kouts_avg[1], kouts_avg[0])
    
    return det_psize/max(np.abs(np.sin(azimuth)), np.abs(np.cos(azimuth)))
    
    
def extract_parallel_line(kins, width=1, offset=0):
    """
    Extracts a parallel thin line (1-2 pixels wide) from a streak in the kin coordinates.

    Args:
        kins (np.ndarray): (N, 2) array of kins coordinates.
        width (int): Thickness of the extracted line (1 or 2 pixels).
        position (str): Where to extract the line from. Options: ['center', 'top', 'bottom'].
        offset (int): Custom offset from the center (in pixels).

    Returns:
        np.ndarray: A boolean mask for selected pixels.
    """

    kins = np.asarray(kins)
    if kins.ndim != 2 or kins.shape[1] != 2:
        raise ValueError(f"Expected kins to be (N, 2), but got {kins.shape}")

    # Compute principal component (streak direction)
    direction =  get_line_direction(kins)

    kins_centered = kins - np.mean(kins, axis=0)  
    projected_dist = np.dot(kins_centered, direction)  # Projection onto principal axis

    perpendicular_vec = np.array([-direction[1], direction[0]])  
    perp_dist = np.dot(kins_centered, perpendicular_vec) 

    lower_bound = offset
    upper_bound = offset + width

    selected_mask = (perp_dist >= lower_bound) & (perp_dist <= upper_bound)

    return selected_mask

def extract_qz_streaks(kouts, kins, dqz, extent=None, min_pixels=10):
    """Group pixels into streaks by their Ewald q_z.

    q_z = k_out,z - k_in,z - G_z is constant along a streak; k_out,z alone is not
    (it also contains -|k_in|^2 / 2k_zG). G_z is a constant offset, so it is left
    out here: it doesn't change the grouping, but an explicit ``extent`` must be
    given without it too. kouts, kins and dqz must be in the same units (all
    cycles/nm, or all rad/nm).

    Returns
    -------
    masks : list[np.ndarray]
        Boolean masks for each streak with at least ``min_pixels`` pixels.
    edges : ndarray
        Edges of all bins, including the dropped ones, so ``len(edges) - 1`` can
        exceed ``len(masks)``.
    """
    qz = kouts[:, 2] - kins[:, 2] #- G[2]

    if extent is None:
        qz_min, qz_max = qz.min(), qz.max()
    else:
        qz_min, qz_max = extent

    # bins centred on the streak values, so rounding noise can't split a streak
    n = int(np.round((qz_max - qz_min) / dqz)) + 1
    edges = qz_min - dqz / 2 + np.arange(n + 1) * dqz

    masks = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (qz >= lo) & (qz < hi)
        if mask.sum() >= min_pixels:
            masks.append(mask)

    return masks, edges

def calc_angle(vec1, vec2):
    """
    The angle between two vectors
    """

    angle = np.rad2deg(math.acos(np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
    return angle
    
def estimate_ttheta(kouts, pupil_kins):
    avg_kins = np.mean(pupil_kins, axis=0)
    avg_kout = np.mean(kouts, axis=0)
    angle = calc_angle(avg_kout, avg_kins)
    return np.deg2rad(angle)

def calculate_delta_qz(ttheta, det_distance, wavelength, det_psize):
    """
    ttheta (rad)
    det_distance (nm)
    wavlength (nm^-1)
    det_psize (nm)
    """
    return 2*np.pi * det_psize * np.cos(ttheta)**2 * np.sin(ttheta) / (wavelength * det_distance)

def calculate_delta_q(ttheta, det_distance, wavelength, det_psize):
    """
    ttheta (rad)
    det_distance (nm)
    wavlength (nm^-1)
    det_psize (nm)
    """
    return 2*np.pi * det_psize * np.cos(ttheta)**2 * np.cos(ttheta/2) / (wavelength * det_distance)