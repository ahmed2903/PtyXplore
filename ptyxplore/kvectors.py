# Define the objective function

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import math 



# ___________________ Projection Algorithm ___________________

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


def _ttheta_projection(G, k, ttheta):
    """
    Project G onto the 2theta constraint sphere |G| = 2k sin(ttheta/2).

    Radially rescales G so its magnitude equals the target set by the
    scattering angle.  This enforces C3 exactly without changing the
    direction of G.

    Returns G unchanged when |G| ≈ 0 (degenerate).
    """
    G_mag = np.linalg.norm(G)
    if G_mag < 1e-30:
        return G.copy()
    G_mag_target = 2.0 * k * np.sin(ttheta / 2.0)
    return G * (G_mag_target / G_mag)


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


def _coplanar_projection(G_list):
    """
    Project all G vectors onto their best-fit plane through the origin.

    The plane is the one that minimises the sum of squared out-of-plane
    distances across all streaks — equivalently, its normal is the smallest
    right singular vector of the stacked G matrix (via SVD).

    This constraint enforces that all scattering vectors G_s lie in the same
    crystallographic plane (the scattering plane of the sample).

    Requires at least 3 streaks to be non-trivial; with 2 streaks the two G
    vectors already define a plane through the origin and are unchanged.

    Parameters
    ----------
    G_list : list of (3,) arrays

    Returns
    -------
    G_proj : list of (3,) projected G vectors lying in the best-fit plane
    n_hat  : (3,) unit normal of the best-fit plane
    """
    G_mat = np.array(G_list)          # (S, 3)
    _, _, Vt = np.linalg.svd(G_mat, full_matrices=False)
    n_hat = Vt[-1]                    # unit normal = smallest right singular vector
    G_proj = [G - np.dot(n_hat, G) * n_hat for G in G_list]
    return G_proj, n_hat


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
                            clamp_to_pupil=False, G_init=None, ttheta=None):
    """
    Alternating projections for one streak.

    Each inner iteration:
      1. Ewald circle projection  → kins  (enforces |kin| = k AND G·kin = -|G|²/2)
      2. G update                 → G = mean(kouts - kins_proj)
      3. 2theta projection        → G rescaled to |G| = 2k sin(ttheta/2)  (optional)
      4. Pupil-support clamp      → G_xy  (optional; only when clamp_to_pupil=True)

    After the loop a final Ewald projection is applied so the returned kins
    satisfy |kin| = k exactly regardless of how many inner iterations were run.

    Parameters
    ----------
    G_init : (3,) array or None
        Starting G vector.  Pass roi.g_init (from a prior optimise_kin call)
        for best results.  If None, G is bootstrapped from the elastic-z
        estimate with G_xy = 0 — only G_z will be meaningful in that case.
    ttheta : float or None
        Two-theta angle in radians.  When provided, G is radially rescaled
        after each G update to enforce |G| = 2k sin(ttheta/2) (C3).

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
        # 3. Optionally rescale G to enforce |G| = 2k sin(ttheta/2)
        if ttheta is not None:
            G = _ttheta_projection(G, k, ttheta)
        # 4. Optionally clamp G_xy to keep all kins inside the pupil box
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
    enforce_coplanar=False,
    ttheta_list=None,
):
    """
    Projection-based solver for kin vectors across multiple streaks.

    Each streak has its own scattering vector G_s; all streaks share the same
    pupil aperture.  Per outer iteration the algorithm:

      Per streak (independent):
        1. Projects kins onto the Ewald circle for current G  (enforces |kin| = k)
        2. Updates G = mean(kouts - kins_proj)
        3. Optionally rescales G to enforce |G| = 2k sin(ttheta/2)  (C3)
        4. Optionally clamps G_xy to the pupil-support feasible interval

      Across all streaks (coupling, optional):
        5. Coplanar projection — projects all G_s onto their best-fit plane
           through the origin, enforcing that every scattering vector lies in
           the same crystallographic (scattering) plane of the sample.

    After the main loop a final Ewald projection is applied to re-sync kins
    with any G change made by the coplanar step on the last iteration.

    Parameters
    ----------
    kouts_list       : list of (N_s, 3) arrays — k-out vectors per streak
    pupil_kbounds    : (kx_min, kx_max, ky_min, ky_max) — pupil aperture in k-space
    wavelength       : X-ray wavelength (same units as the k-vectors)
    n_iterations     : outer iteration count
    n_inner          : inner Ewald-projection steps per streak per outer iteration
    clamp_to_pupil   : if True, clamp G_xy so all implied kins stay inside the
                       pupil box after each Ewald step
    G_init_list      : list of (3,) initial G vectors, one per streak, or None
    enforce_coplanar : if True (and n_streaks >= 3), project all G_s onto their
                       best-fit plane after each outer iteration.  The plane
                       normal is the smallest right singular vector of the
                       stacked G matrix, i.e. the direction all G_s are most
                       orthogonal to.  Requires at least 3 streaks to be
                       non-trivial.
    ttheta_list      : list of two-theta angles in radians, one per streak, or
                       a single scalar shared across all streaks, or None.
                       When provided, G is radially rescaled after each G update
                       to enforce |G| = 2k sin(ttheta/2) (C3).

    Returns
    -------
    G_list       : list of (3,) G vectors, one per streak
    kins_list    : list of (N_s, 3) kin arrays with |kin| = k, one per streak
    width_ratios : (n_streaks, 2) streak_width / pupil_width — should be ~1.0
    history      : list of per-iteration dicts with keys
                   'iteration', 'G_list', 'width_ratios', 'dG', 'plane_normal'
    """
    k = 2.0 * np.pi / wavelength
    n_streaks = len(kouts_list)
    apply_coplanar = enforce_coplanar and n_streaks >= 3
    if G_init_list is not None and len(G_init_list) != n_streaks:
        raise ValueError(
            f"G_init_list has {len(G_init_list)} entries but kouts_list has {n_streaks}"
        )

    # Normalise ttheta_list to a per-streak list (or all-None).
    if ttheta_list is None:
        tthetas = [None] * n_streaks
    elif np.isscalar(ttheta_list):
        tthetas = [float(ttheta_list)] * n_streaks
    else:
        tthetas = list(ttheta_list)
        if len(tthetas) != n_streaks:
            raise ValueError(
                f"ttheta_list has {len(tthetas)} entries but kouts_list has {n_streaks}"
            )

    # Seed G_list from G_init_list so each outer iteration refines the G
    # produced by the previous one rather than restarting from scratch.
    if G_init_list is not None:
        G_list = [np.array(G_init, dtype=float).ravel() for G_init in G_init_list]
    else:
        G_list = []
        for kouts in kouts_list:
            G = np.zeros(3)
            G[2] = _elastic_z(kouts, G[:2], k)
            G_list.append(G)


    kins_list = [None] * n_streaks
    width_ratios = np.zeros((n_streaks, 2))
    history = []
    plane_normal = None

    for iteration in range(n_iterations):
        if iteration % 100 == 0:
            print(iteration)

        G_prev = [G.copy() for G in G_list]

        # --- per-streak Ewald projection ---
        for s in range(n_streaks):
            G, kins, wr = _project_single_streak(
                kouts_list[s], pupil_kbounds, k, n_inner,
                clamp_to_pupil, G_list[s], tthetas[s],
            )
            G_list[s] = G
            kins_list[s] = kins
            width_ratios[s] = wr

        if apply_coplanar:
            G_list, plane_normal = _coplanar_projection(G_list)
        
        dG = max(np.linalg.norm(G_list[s] - G_prev[s]) for s in range(n_streaks))

        history.append({
            'iteration': iteration + 1,
            'G_list': [G.copy() for G in G_list],
            'width_ratios': width_ratios.copy(),
            'dG': dG,
            'plane_normal': plane_normal.copy() if plane_normal is not None else None,
        })

        if dG < 1e-12:
            break

    # After the loop the coplanar step may have shifted G on the last iteration
    # without a matching Ewald projection. Re-sync kins so the returned vectors
    # satisfy |kin| = k with the final coplanar-corrected G.
    if apply_coplanar:
        for s in range(n_streaks):
            kins_list[s] = _ewald_circle_projection(kouts_list[s], G_list[s], k)
            G_list[s] = np.mean(kouts_list[s] - kins_list[s], axis=0)
        if history:
            history[-1]['plane_normal'] = plane_normal.copy() if plane_normal is not None else None

    return G_list, kins_list, width_ratios, history


# ___________________ GS Algorithm ___________________

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
    kouts = compute_vectors(coordinates=pixel_coords_out, detector_distance=det_distance, pixel_size=pixel_size,central_pixel=central_pixel, wavelength=wavelength)

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

# _________ Utils _________
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