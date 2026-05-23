# Define the objective function

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import math 

def rotation_matrix(alpha, beta, gamma):
    # Construct rotation matrix using standard formulas
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    
    return np.dot(Rz, np.dot(Ry, Rx)) #Rz @ Ry @ Rx

def objective_function(params, kouts, G_initial, ttheta, wavelength ):
    # Rotate G using Euler angles
    alpha, beta, gamma = params
    R = rotation_matrix(alpha, beta, gamma)
    try:
        G = R @ G_initial
    except:
        G = R @ G_initial[0]
    
    C = G/2
    
    k_mag = 2*np.pi/wavelength 
    R_circle = k_mag * np.cos(ttheta)
    Gmag = 4*np.pi / wavelength * np.sin(ttheta/2)
        
    # Update k_in
    kin = kouts - G

    # Compute deviations for k_out (circle condition)
    f_circle = np.mean(np.abs(np.linalg.norm(kin - C, axis=1) - R_circle) / R_circle)

    # Compute error in the 2theta angle 
    dots = np.sum( (kin/np.linalg.norm(kin, axis=1)[:,np.newaxis] ) * (kouts/np.linalg.norm(kouts, axis=1)[:,np.newaxis]), axis = 1)
    f_angle = np.sum( np.abs( (dots - np.cos(ttheta)) / np.cos(ttheta) ) ) # np.std(dots)**2) #
    
    return f_angle

def optimise_kin(G_init, ttheta, kouts, wavelength, method, gtol):

    # Initial guess for optimization
    initial_guess = [0, 0, 0]

    # Perform optimization
    result = minimize(objective_function, 
                      initial_guess, 
                      args=(kouts, G_init, ttheta, wavelength),
                        method=method, 
                      options={'disp': True, 
                                "gtol" : gtol, 
                               #'return_all': True
                                })

    # Optimized orientation
    optimal_angles = result.x

    opt_mat = rotation_matrix(*optimal_angles)

    try:
        G_opt =  opt_mat @ G_init
    except:
        G_opt =  opt_mat @ G_init[0]
        
    kin_opt = (kouts-G_opt)
    #kin_opt /= np.linalg.norm(kin_opt, axis = 1)[:,np.newaxis]
    #kin_opt *= 2*np.pi/wavelength

    return kin_opt, optimal_angles
    

def objective_function_w_geometry(params, pixel_coords_out, pixel_coords_in, G_initial, wavelength, pixel_size, central_pixel):
    
    # Rotate G using Euler angles, Refine kouts using detector distance
    det_distance, alpha, beta, gamma  = params

    # Get new G esimate
    R = rotation_matrix(alpha, beta, gamma)
    try:
        G = R @ G_initial
    except:
        G = R @ G_initial[0]
    

    k_mag = 2*np.pi/wavelength 

    # Get new kouts
    kouts = compute_vectors(pixel_coords_out, det_distance, pixel_size, central_pixel, wavelength)

    # Get new kins
    kins_pupil = compute_vectors(pixel_coords_in, det_distance, pixel_size, central_pixel, wavelength)
    
    # Update k_in
    kins_calc = kouts - G
    

    # ---- elastic scattering + Bragg condition
    r_elastic = np.abs(np.linalg.norm(kins_calc, axis=1) - k_mag)
    r_kout = np.abs(np.linalg.norm(kouts, axis=1) - k_mag)

    
    # --- pupil support    
    kx_bds, ky_bds = compute_k_bounds(kins_pupil)

    kx_min, kx_max = kx_bds
    ky_min, ky_max = ky_bds
    
    r_pupil_x = np.maximum(0, kins_calc[:,0] - kx_max) + np.maximum(0, kx_min - kins_calc[:,0])
    r_pupil_y = np.maximum(0, kins_calc[:,1] - ky_max) + np.maximum(0, ky_min - kins_calc[:,1])
    r_pupil = r_pupil_x + r_pupil_y

    w_pupil = 1
    w_elastic = 1
    w_kout = 0
    
    return np.sum(
        w_kout * r_kout +
        w_elastic * r_elastic+
        w_pupil * r_pupil)/(w_pupil+w_elastic)
    
def optimise_kin_w_geometry(G_init, det_distance_init, kin_pixels, kout_pixels, wavelength, pixel_size, central_pixel, method, gtol):

    # Initial guess for optimization
    initial_guess = [det_distance_init, 0, 0, 0]

    # Perform optimization
    result = minimize(objective_function_w_geometry, 
                      initial_guess, 
                      args=(kout_pixels, kin_pixels, G_init, wavelength, pixel_size, central_pixel),
                        method=method,
                      options={'disp': True, 
                                "gtol" : gtol, 
                                })

    # Optimized orientation
    optimal_params = result.x

    optimal_angles = optimal_params[1:]
    optimal_distance = optimal_params[0]
    
    opt_mat = rotation_matrix(*optimal_angles)

    try:
        G_opt =  opt_mat @ G_init
    except:
        G_opt =  opt_mat @ G_init[0]

    kouts = compute_vectors(kout_pixels, optimal_distance, pixel_size, central_pixel, wavelength)
    kins = compute_vectors(kin_pixels, optimal_distance, pixel_size, central_pixel, wavelength)
    
    kin_opt = (kouts-G_opt)
    #kin_opt /= np.linalg.norm(kin_opt, axis = 1)[:,np.newaxis]
    #kin_opt *= 2*np.pi/wavelength

    return kin_opt, kouts, kins, optimal_params


def _extent_projection(kouts, pupil_kbounds):
    """
    Closed-form G_xy from the streak extent constraint.

    The streak in kout space is a rigid shift of the pupil by G, so the
    centre of the streak in x/y equals the centre of the pupil plus G_xy.
    """
    kx_min, kx_max, ky_min, ky_max = pupil_kbounds
    G_x = (kouts[:, 0].max() + kouts[:, 0].min()) / 2 - (kx_max + kx_min) / 2
    G_y = (kouts[:, 1].max() + kouts[:, 1].min()) / 2 - (ky_max + ky_min) / 2
    return np.array([G_x, G_y])


def _elastic_z(kouts, G_xy, k):
    """
    G_z from the elastic-scattering condition |kin| = k, evaluated per pixel.

    kin_z = sqrt(k^2 - |kin_xy|^2),  G_z = median(kout_z - kin_z).
    Median is used for robustness against outlier pixels.
    """
    kin_xy = kouts[:, :2] - G_xy
    kin_z = np.sqrt(np.maximum(0.0, k ** 2 - np.sum(kin_xy ** 2, axis=1)))
    return float(np.median(kouts[:, 2] - kin_z))


def _ttheta_z(G_xy, k, ttheta, G_z_sign):
    """
    G_z from the 2theta constraint: |G| = 2k sin(ttheta/2).

    Returns None when the current G_xy already exceeds the target |G|,
    in which case the caller falls back to the elastic estimate.
    """
    G_mag_target = 2.0 * k * np.sin(ttheta / 2.0)
    G_z_sq = G_mag_target ** 2 - float(np.dot(G_xy, G_xy))
    if G_z_sq < 0:
        return None
    return G_z_sign * np.sqrt(G_z_sq)


def _pupil_clamp_G(kouts, G, pupil_kbounds):
    """
    Projection onto the pupil-support constraint in G-space.

    For all kins = kout - G to lie inside the pupil box, G must satisfy:
        max(kout_x) - kx_max  <=  G_x  <=  min(kout_x) - kx_min
    (and similarly for y).  We clamp G_xy to this feasible interval.
    """
    kx_min, kx_max, ky_min, ky_max = pupil_kbounds
    G_clamped = G.copy()
    G_clamped[0] = np.clip(G[0],
                            kouts[:, 0].max() - kx_max,
                            kouts[:, 0].min() - kx_min)
    G_clamped[1] = np.clip(G[1],
                            kouts[:, 1].max() - ky_max,
                            kouts[:, 1].min() - ky_min)
    return G_clamped


def _ewald_circle_projection(kouts, G, k):
    """
    Project each kin_j = kout_j - G onto the Ewald circle for this streak.

    The Ewald circle is the intersection of:
      - Elastic sphere : |kin| = k
      - Bragg plane    : G · kin = -|G|^2 / 2
        (derived from |kin+G|^2 = k^2 and |kin|^2 = k^2)

    Algorithm per pixel j:
      1. Compute kin_j = kout_j - G.
      2. Remove the excess component along G so that G · kin = -|G|^2 / 2.
      3. Scale the result to lie on the sphere |kin| = k.

    After projection: |kin_j_proj| = k and G · kin_j_proj = -|G|^2 / 2 exactly.

    Returns
    -------
    kins_proj : (N, 3) — projected incident k-vectors with |kin| = k.
    """
    G_sq = np.dot(G, G)
    if G_sq < 1e-30:
        return kouts - G  # G ≈ 0 — degenerate, return raw difference

    kins = kouts - G  # (N, 3) current estimate

    # Step 1: project onto the Bragg plane  G · kin = -|G|^2 / 2
    dot_G_kin = kins @ G                        # (N,)
    excess = (dot_G_kin + G_sq / 2.0) / G_sq   # (N,)
    kins_plane = kins - excess[:, np.newaxis] * G  # (N, 3)

    # Step 2: scale to the elastic sphere radius k
    norms = np.linalg.norm(kins_plane, axis=1, keepdims=True)
    norms = np.where(norms < 1e-30, 1.0, norms)
    kins_proj = k * kins_plane / norms          # |kin_proj| = k

    return kins_proj


def _project_single_streak(kouts, pupil_kbounds, k, n_inner,
                            clamp_to_pupil=False, G_init=None):
    """
    Alternating projections for one streak.

    Each inner iteration:
      1. Ewald circle projection  → kins  (enforces |kin| = k AND G·kin = -|G|²/2)
      2. G update                 → G = mean(kouts - kins_proj)
      3. Pupil-support clamp      → G_xy  (optional; only when clamp_to_pupil=True)

    After the loop a final Ewald projection is applied so the returned kins
    satisfy |kin| = k exactly regardless of how many inner iterations were run.

    Parameters
    ----------
    G_init : (3,) array or None
        Starting G vector.  Pass roi.g_init (from a prior optimise_kin call)
        for best results.  If None, G is bootstrapped from the elastic-z
        estimate with G_xy = 0 — only G_z will be meaningful in that case.

    Returns
    -------
    G           : (3,) refined scattering vector for this streak
    kins        : (N, 3) incident k-vectors with |kin| = k
    width_ratio : (2,) streak_width / pupil_width in x and y  (should be ~1)
    """
    if G_init is not None:
        G = np.array(G_init, dtype=float).ravel()
    else:
        G = np.zeros(3)
        G[2] = _elastic_z(kouts, G[:2], k)
    for _ in range(n_inner):
        # 1. Project kins onto the Ewald circle for current G → |kin| = k
        kins = _ewald_circle_projection(kouts, G, k)
        # 2. Update G as the mean residual kout - kin_proj
        G = np.mean(kouts - kins, axis=0)
        # 3. Optionally clamp G_xy to keep all kins inside the pupil box
        if clamp_to_pupil:
            G = _pupil_clamp_G(kouts, G, pupil_kbounds)

    # Final Ewald projection: guarantees |kin| = k on exit
    kins = _ewald_circle_projection(kouts, G, k)
    # Recompute G consistently with the projected kins
    G = np.mean(kouts - kins, axis=0)

    kx_min, kx_max, ky_min, ky_max = pupil_kbounds
    width_ratio = np.array([
        (kouts[:, 0].max() - kouts[:, 0].min()) / max(kx_max - kx_min, 1e-30),
        (kouts[:, 1].max() - kouts[:, 1].min()) / max(ky_max - ky_min, 1e-30),
    ])

    return G, kins, width_ratio


def projection_kin_refinement(
    kouts_list,
    pupil_kbounds,
    wavelength,
    n_iterations=5,
    n_inner=3,
    clamp_to_pupil=False,
    G_init_list=None,
):
    """
    Projection-based solver for kin vectors across multiple streaks.

    Each streak has its own scattering vector G_s; all streaks share the same
    pupil aperture.  Per inner iteration the algorithm:

      1. Projects kins onto the Ewald circle for current G  (enforces |kin| = k)
      2. Updates G = mean(kouts - kins_proj)
      3. Optionally clamps G_xy to the pupil-support feasible interval

    The outer loop tracks convergence via dG and allows the caller to update
    kouts_list externally (e.g. after geometry refinement) and re-enter.

    Parameters
    ----------
    kouts_list     : list of (N_s, 3) arrays — k-out vectors per streak
    pupil_kbounds  : (kx_min, kx_max, ky_min, ky_max) — pupil aperture in k-space
    wavelength     : X-ray wavelength (same units as the k-vectors)
    n_iterations   : outer iteration count
    n_inner        : inner Ewald-projection steps per streak per outer iteration
    clamp_to_pupil : if True, clamp G_xy so all implied kins stay inside the
                     pupil box after each Ewald step (only reliable when
                     width_ratio ≈ 1; leave False to preserve delta_q)
    G_init_list    : list of (3,) initial G vectors, one per streak, or None.
                     Provide roi.g_init / roi.g_list from a prior optimise_kin
                     call for best convergence.  If None, G is bootstrapped
                     from the elastic-z estimate with G_xy = 0.

    Returns
    -------
    G_list       : list of (3,) G vectors, one per streak
    kins_list    : list of (N_s, 3) kin arrays with |kin| = k, one per streak
    width_ratios : (n_streaks, 2) streak_width / pupil_width — should be ~1.0
    history      : list of per-iteration dicts with keys
                   'iteration', 'G_list', 'width_ratios', 'dG'
    """
    k = 2.0 * np.pi / wavelength
    n_streaks = len(kouts_list)

    if G_init_list is not None and len(G_init_list) != n_streaks:
        raise ValueError(
            f"G_init_list has {len(G_init_list)} entries but kouts_list has {n_streaks}"
        )

    # Seed G_list from G_init_list so each outer iteration refines the G
    # produced by the previous one rather than restarting from scratch.
    if G_init_list is not None:
        G_list = [np.array(G_init, dtype=float).ravel() for G_init in G_init_list]
    else:
        G_list = [np.zeros(3) for _ in range(n_streaks)]

    kins_list = [None] * n_streaks
    width_ratios = np.zeros((n_streaks, 2))
    history = []
    print(n_iterations)
    for iteration in range(n_iterations):
        if iteration % 100 == 0:
            print(iteration)
        G_prev = [G.copy() for G in G_list]

        for s in range(n_streaks):
            # Pass the current best G so outer iterations genuinely refine it.
            G, kins, wr = _project_single_streak(
                kouts_list[s], pupil_kbounds, k, n_inner,
                clamp_to_pupil, G_list[s],
            )
            G_list[s] = G
            kins_list[s] = kins
            width_ratios[s] = wr

        dG = max(np.linalg.norm(G_list[s] - G_prev[s]) for s in range(n_streaks))

        history.append({
            'iteration': iteration + 1,
            'G_list': [G.copy() for G in G_list],
            'width_ratios': width_ratios.copy(),
            'dG': dG,
        })

        if dG < 1e-12:
            break

    return G_list, kins_list, width_ratios, history


def iterate_kin_refinement(
    G_init,
    det_distance_init,
    kin_pixels,
    kout_pixels,
    ttheta,
    wavelength,
    pixel_size,
    central_pixel,
    n_iterations=5,
    method='BFGS',
    gtol=1e-6,
    reestimate_ttheta=False,
):
    """
    Alternates between geometry refinement (detector distance + orientation)
    and 2theta/angle refinement until n_iterations is reached.

    Each cycle:
      1. optimise_kin_w_geometry — refines det_distance and G orientation
         using elastic / pupil-support constraints.
      2. optimise_kin — refines G orientation using the 2theta dot-product
         objective, working with the updated k-vectors from step 1.

    Args:
        G_init:            Initial G vector (3,).
        det_distance_init: Initial detector distance (same units as pixel_size).
        kin_pixels:        (N, 2) pixel coordinates of incident beam pixels.
        kout_pixels:       (M, 2) pixel coordinates of diffracted beam pixels.
        ttheta:            Two-theta angle in radians.
        wavelength:        X-ray wavelength (same units as pixel_size / distance).
        pixel_size:        Detector pixel size.
        central_pixel:     (row, col) of the beam centre on the detector.
        n_iterations:      Number of geometry ↔ angle cycles to run.
        method:            scipy.optimize.minimize method string.
        gtol:              Gradient tolerance for both optimisers.
        reestimate_ttheta: If True, re-derive ttheta from the current k-vectors
                           after each geometry step (uses estimate_ttheta).

    Returns:
        kin_opt:        Final refined k_in vectors (M, 3).
        kouts:          Final k_out vectors (M, 3).
        kins_pupil:     Final pupil k_in vectors (N, 3).
        G_opt:          Final refined G vector (3,).
        det_distance:   Final refined detector distance.
        history:        List of dicts with per-iteration diagnostics.
    """
    G_current = np.array(G_init, dtype=float)
    det_distance = float(det_distance_init)
    current_ttheta = float(ttheta)
    history = []

    for i in range(n_iterations):
        # --- Step 1: geometry refinement ---
        kin_opt, kouts, kins_pupil, geo_params = optimise_kin_w_geometry(
            G_current,
            det_distance,
            kin_pixels,
            kout_pixels,
            wavelength,
            pixel_size,
            central_pixel,
            method,
            gtol,
        )
        det_distance = float(geo_params[0])
        geo_angles = geo_params[1:]
        G_current = rotation_matrix(*geo_angles) @ G_current

        if reestimate_ttheta:
            current_ttheta = np.deg2rad(estimate_ttheta(kouts, kins_pupil))

        # --- Step 2: 2theta / angle refinement ---
        kin_opt, angle_angles = optimise_kin(
            G_current,
            current_ttheta,
            kouts,
            wavelength,
            method,
            gtol,
        )
        G_current = rotation_matrix(*angle_angles) @ G_current

        history.append({
            'iteration': i + 1,
            'det_distance': det_distance,
            'ttheta_rad': current_ttheta,
            'G': G_current.copy(),
            'geo_angles': geo_angles,
            'angle_angles': angle_angles,
        })

    # Recompute final k-vectors with the converged detector distance
    kouts = compute_vectors(kout_pixels, det_distance, pixel_size, central_pixel, wavelength)
    kins_pupil = compute_vectors(kin_pixels, det_distance, pixel_size, central_pixel, wavelength)
    kin_opt = kouts - G_current

    return kin_opt, kouts, kins_pupil, G_current, det_distance, history


def calc_qvec(kout, kin, **kwargs):
    """
    optional args:
        ttheta: the real two theta value
        wavelength : wavelength of the experiment 
    """

    if len(kout.shape)>1:
        kout = np.mean(kout, axis = 0, keepdims = True)
    
    G_0 = kout - kin
    
    return G_0

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
    
    # Convert to integer pixel indices
    row_pixel_indices = np.floor(row_pixels).astype(int)
    col_pixel_indices = np.floor(col_pixels).astype(int)

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
    pca = PCA(n_components=2)
    pca.fit(kins)
    direction = pca.components_[0]  

    kins_centered = kins - np.mean(kins, axis=0)  
    projected_dist = np.dot(kins_centered, direction)  # Projection onto principal axis

    perpendicular_vec = np.array([-direction[1], direction[0]])  
    perp_dist = np.dot(kins_centered, perpendicular_vec) 

    
    lower_bound = offset
    upper_bound = offset + width

    selected_mask = (perp_dist >= lower_bound) & (perp_dist <= upper_bound)

    return selected_mask

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
    return angle