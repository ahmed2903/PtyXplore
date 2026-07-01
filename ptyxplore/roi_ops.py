# Imports data from Ptychography experiments
# 
# Author: Ahmed H. Mokhtar 
# Email: ahmed.mokhtar@desy.de
# Date : Feb 2025

import numpy as np
import h5py
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
# import cv2

from .data_io import *
from .data_proc import *
from .kvectors import (make_coordinates, compute_vectors, optimise_kin, optimise_kin_w_geometry,
                       reverse_kins_to_pixels, rotation_matrix, calc_qvec, extract_parallel_line,
                       extract_streak_region, compute_k_bounds,
                       projection_kin_refinement, iterate_kin_refinement, calculate_delta_qz, calculate_delta_q, calc_effective_pixel_size, extract_qz_streaks)

from .utils import time_it, log_roi_params
from .geometry import *

from .loader import ROI, Scan

from .tilt_correction import tilt_correction

################### Loading data ###################

@time_it
def make_4d_dataset(roi: ROI, num_jobs = 64, mask = None):
    
    """Creates a 4D dataset from a region of interest (ROI) on the detector.

    Args:
        roi_coords (list): The region of interest (ROI) to process.

    Returns:
        The 4D dataset for the given ROI.
    """

    
    
        
    # try:
        
    direc = roi.data_dir
    print(direc)
    fnames = list_datafiles(direc)[:-2]     
    rs,re,cs,ce = roi.coords
    if roi.detector.lower() == 'eiger':
        data_4d = stack_4d_data(direc, fnames,  roi.coords,
                                                slow_axis = roi.slow_axis,
                                                conc=True,
                                                num_jobs=num_jobs)
        if mask is not None:
            mask_roi = mask[rs:re, cs:ce]
            data_4d = data_4d * mask_roi[None, None, :,:]
        if getattr(roi, 'detector_mask', None) is not None:
            data_4d = data_4d * roi.detector_mask[None, None, :, :]

        # data_4d = np.flip(data_4d, axis = 2)

    elif roi.detector.lower() == 'lambda':

        data_4d = stack_4d_data_lambda(direc, fnames,
                                                roi.coords,
                                                slow_axis = roi.slow_axis,
                                                conc=True,
                                                num_jobs=num_jobs)

        if mask is not None:
            mask_roi = mask[rs:re, cs:ce]
            data_4d = data_4d * mask_roi[None, None, :,:]
        if getattr(roi, 'detector_mask', None) is not None:
            data_4d = data_4d * roi.detector_mask[None, None, :, :]

        data_4d = np.flip(data_4d, axis = 3)
    

    # except:
    #     direc = os.path.join(roi.data_dir , f"Scan_{roi.scan_num}_data_000001.h5")
        
    #     if roi.fast_axis_steps is None:
    #         raise ValueError("fast_axis_steps is required")            
        
    #     data_4d = stack_4d_data_old(direc, 
    #                                 roi.coords, 
    #                                 roi.fast_axis_steps, 
    #                                 roi.slow_axis)
        
    roi.data_4d = mask_hot_pixels(data_4d, 
                                mask_max_coh = False, 
                                mask_min_coh = False)

        

def apply_detector_mask(roi:ROI, mask):

    rs,re,cs,ce = roi.coords

    mask_roi = mask[rs:re, cs:ce]

    roi.data_4d = roi.data_4d * mask_roi[None, None, :,:]


@log_roi_params
def subtract_white_field(roi: ROI, white_field: np.ndarray):
    """Subtract a white-field image from every frame in the ROI 4D dataset.

    Parameters
    ----------
    roi : ROI
        Must have data_4d already computed (shape slow × fast × dH × dW).
    white_field : np.ndarray
        2-D array of shape (H_detector, W_detector) — full-detector size.
        It is cropped to the ROI bounding box before subtraction.
        Pixel values are clipped to zero after subtraction so that no
        negative intensities are introduced.
    """
    if roi.data_4d is None:
        raise ValueError("Run 'Make 4D dataset' before subtracting the white field.")
    rs, re, cs, ce = roi.coords
    wf_crop = white_field[rs:re, cs:ce].astype(np.float32)
    roi.data_4d = np.clip(
        roi.data_4d.astype(np.float32) - wf_crop[None, None, :, :],
        0.0, None,
    )
    
def estimate_pupil_size(pupil_roi:ROI, mask_val):
    """
    Estimates the pupil size from the averaged pupil data and updates the instance attribute.

    Args:
        mask_val (float or int): A threshold value used to mask or filter the pupil data
            before estimating its size.
    
    Updates:
        self.pupil_size: The estimated pupil size, computed using the averaged pupil data
        and the given mask threshold.
    """
    # pupil_size = estimate_pupil_size(pupil_roi.averaged_det_images, 
    #                    mask_val=mask_val,
    #                    pixel_size=pupil_roi.det_psize)

    array = pupil_roi.averaged_det_images
    pixel_size = pupil_roi.det_psize

    min_lx = array.shape[0] /3
    min_ly = array.shape[1] /3
    
    nx , ny = np.where(array>mask_val)

    x_lengths = []
    for x in np.unique(nx):
        y_indices = np.where(array[x,:] > mask_val)[0]
        if len(y_indices) > min_lx:
            x_lengths.append(y_indices[-1] - y_indices[0] + 1)

    
    y_lengths = []
    for y in np.unique(ny):
        x_indices = np.where(array[:,y] > mask_val)[0]
        if len(x_indices) > min_ly:
            y_lengths.append(x_indices[-1] - x_indices[0] + 1)

    x_lengths = np.array(x_lengths) 
    y_lengths = np.array(y_lengths)

    avg_x = np.max(x_lengths) * pixel_size
    avg_y = np.max(y_lengths) * pixel_size

    return avg_x, avg_y
    
    # return pupil_size

def add_lens(focal_length:float, height:float):
    """
    Adds a lens with given parameters to the internal lens parameter dictionary.

    Args:
        lens_name (str): The name/key to identify the lens.
        focal_length (float): The focal length of the lens in microns.
        height (float): The physical aperture height of the lens in microns.
    """
    lens_na = calculate_NA(focal_length, height)

    return lens_na

def estimate_detector_distance(na1, na2, pupil_size):
    """
    Estimates the average detector distance based on the range of numerical apertures (NA)
    from the added lenses and current pupil size.

    Updates:
        self.exp_params['det_distance']: Estimated detector distance in microns based on optical geometry.
    """
    largest_na = 0
    smallest_na = 1e4
    
    for na in [na1, na2]:

        largest_na = max(largest_na, na)
        smallest_na = min(smallest_na, na)    
    
    distance1 = estimate_detector_distance_from_NA(largest_na, pupil_size=max(pupil_size))
    distance2 = estimate_detector_distance_from_NA(smallest_na, pupil_size=min(pupil_size))

    print(f"First detector distance is {max(distance1, distance2)} nm")
    print(f"Second detector distance is {min(distance1,distance2)} nm")
    
    return max(distance1,distance2), min(distance1,distance2)


################### prepare detector roi ###################

@log_roi_params
def pool_detector(roi:ROI, kernel_size, stride = None, padding=0):
    """Performs pooling on the detector space for a given region of interest (ROI).
    
    Args:
        roi_name (str): The name of the region of interest (ROI) to be processed.
        kernel_size (int): The size of the pooling kernel, determining the window 
                            over which the sum pooling is applied.
        stride (int, optional): The stride of the pooling operation, i.e., the 
                                 step size between each pooling operation. If 
                                 not provided, defaults to `kernel_size`.
        padding (int, optional): The amount of zero padding to add around the 
                                  edges of the image before pooling. Defaults to 0.
    """
    print("Pooling detector ...")

    if stride is None: 
        stride = kernel_size
    
    
    roi.data_4d = sum_pool2d_array(roi.data_4d, kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=0)
            
    
    roi.det_psize *= stride

    if roi.kout_coords is not None:
        roi.kout_coords = np.unique(roi.kout_coords // stride)
    
    roi.coords = list(np.array(roi.coords) // stride)
    roi.averaged_det_images = np.mean(roi.data_4d, axis = (0,1))
    
    print("Done.")

# @log_roi_params
# def pool_detector(roi:ROI, kernel_size, stride = 1, padding=0):
#     """Performs pooling on the detector space for a given region of interest (ROI).
    
#     Args:
#         roi_name (str): The name of the region of interest (ROI) to be processed.
#         kernel_size (int): The size of the pooling kernel, determining the window 
#                             over which the sum pooling is applied.
#         stride (int, optional): The stride of the pooling operation, i.e., the 
#                                  step size between each pooling operation. If 
#                                  not provided, defaults to `kernel_size`.
#         padding (int, optional): The amount of zero padding to add around the 
#                                   edges of the image before pooling. Defaults to 0.
#     """
#     print("Pooling detector ...")
    
#     roi.data_4d = sum_pool2d_array(roi.data_4d, kernel_size=kernel_size,
#                                                    stride=1,
#                                                    padding=0)
            
    
#     roi.det_psize *= stride

#     if roi.kout_coords is not None:
#         roi.kout_coords = np.unique(roi.kout_coords // stride)
    
#     roi.coords = list(np.array(roi.coords) // stride)
#     roi.averaged_det_images = np.mean(roi.data_4d, axis = (0,1))
    
#     print("Done.")
    
    
@log_roi_params    
def normalise_detector(roi:ROI, reference_roi:ROI):
    """Normalizes the detector data based on the reference ROI's peak intensity.

    Args:
        roi_name_ref (str): The name of the reference region of interest (ROI) used 
                             for calculating the peak intensity.
        roi_name (str): The name of the operating region of interest (ROI) to be normalized.

    """
    try:
        peak_intensity = np.sum(reference_roi.data_4d, axis=(-2,-1))
        
    except: 
        make_4d_dataset(reference_roi)
        peak_intensity = np.sum(reference_roi.data_4d, axis=(-2,-1))
        print(peak_intensity.shape)
    
    avg_intensity = np.mean(peak_intensity)
    roi.data_4d = roi.data_4d/peak_intensity[...,np.newaxis, np.newaxis] * avg_intensity


@log_roi_params
def mask_region_detector(roi:ROI, region, mode = 'median'):
    """Masks a specific region of the detector data within the given ROI.

    Args:
        roi_name (str): The name of the region of interest (ROI) whose detector data 
                         will be modified.
        region (tuple): A tuple (sx, ex, sy, ey) specifying the start and end indices 
                        for the region to be masked:
                        - sx, ex: x-axis start and end indices
                        - sy, ey: y-axis start and end indices
        mode (str, optional): A string specifying how to replace the masked region. 
                        - 'zeros'. Default.
                        - 'median'
                        - 'ones'
    """
    sx,ex,sy,ey = region
    if mode == 'zeros':
        roi.data_4d[:,:,sx:ex,sy:ey] = 0.0
        roi.update_averaged_det_images()
    elif mode == 'median':
        roi.data_4d[:,:,sx:ex,sy:ey] = np.median(roi.data_4d, axis = (2,3))[:,:,np.newaxis, np.newaxis]
        roi.update_averaged_det_images()
    elif mode == 'ones':
        roi.data_4d[:,:,sx:ex,sy:ey] = 1.0
        roi.update_averaged_det_images()


################### Compute k_in vectors ###################

@log_roi_params
def make_kouts( roi:ROI, mask_val):
    """Computes the k-space vectors for a given region of interest (ROI).

    Args:
        roi_name (str): The name of the region of interest (ROI) for which the 
                         k-space vectors are to be computed.
        mask_val (float): The value used for masking pixels during coordinate 
                          computation.

    """
    roi.kout_coords = make_coordinates(roi.averaged_det_images, 
                                                            mask_val, 
                                                            roi.coords, 
                                                            crop=False)



    
    if roi.centre_pixel is None and roi.kind == 'pupil':
        roi.centre_pixel = np.array([(roi.kout_coords[:,0].max() + roi.kout_coords[:,0].min() )/2, (roi.kout_coords[:,1].max() + roi.kout_coords[:,1].min() )/2])

    
    roi.kouts = compute_vectors(roi.kout_coords, 
                                            roi.det_distance, 
                                            roi.det_psize, 
                                            roi.centre_pixel, 
                                            roi.wavelength)

    if roi.kind == 'pupil':
        
        roi.kins = roi.kouts
        roi.kin_coords = roi.kout_coords
        roi.kins_avg = np.mean(roi.kouts, axis = 0, keepdims = True )


def estimate_ttheta(roi:ROI, pupil_roi:ROI):
    """Estiamtes the two theta value for the signal in the roi, assuming there is one incident wavevector which is specified by the average of all k_in vectors in the pupil. 
    Args:
        roi_name (str): The name of the region of interest (ROI) for which the two_theta value is to be estimated.

    Updates:
        self.est_ttheta[roi_name]: Estimated 2 theta angle for that signal
    
    """
    try:
        pupil_roi.kins_avg = np.mean(pupil_roi.kouts, axis = 0, keepdims = True )
    except:
        raise ValueError("Must compute pupil kouts first")

    kouts_avg = np.mean(roi.kouts, axis = 0, keepdims = True )
    kouts_avg /= np.linalg.norm(kouts_avg)
    
    kins_avg = pupil_roi.kins_avg/np.linalg.norm(pupil_roi.kins_avg)

    
    angle = np.arccos(np.sum(kins_avg* kouts_avg))
    angle = np.rad2deg(angle)
    
    roi.est_ttheta = angle
    
    print(f"the initial 2theta angle is: {angle:.2f}")
    

    
@log_roi_params
def compute_kins(roi: ROI, pupil_roi: ROI, est_ttheta, method="BFGS", gtol=1e-6):
    """
    Compute incident wave-vectors for a single- or multi-streak ROI.

    For multi-streak ROIs (roi.kouts is a list), each streak is optimised
    independently with its own G vector.  est_ttheta may be a scalar
    (shared across all streaks) or a list with one value per streak.

    Updates
    -------
    Single-streak: roi.g_init, roi.kins, roi.optimal_angles, roi.kin_coords
    Multi-streak : all of the above as parallel lists (one entry per streak)
    """
    multi = isinstance(roi.kouts, list)
    kouts_list = roi.kouts if multi else [roi.kouts]

    if np.isscalar(est_ttheta):
        tthetas = [est_ttheta] * len(kouts_list)
    else:
        tthetas = list(est_ttheta)

    g_inits, kins_list, angles_list = [], [], []
    for kouts_s, ttheta_s in zip(kouts_list, tthetas):
        g_init_s = calc_qvec(kouts_s, pupil_roi.kins_avg)
        kins_s, angles_s = optimise_kin(g_init_s, ttheta_s, kouts_s,
                                        roi.wavelength, method, gtol)
        g_inits.append(g_init_s)
        kins_list.append(kins_s)
        angles_list.append(angles_s)

    if multi:
        roi.g_vectors = g_inits
        roi.kins = kins_list
        roi.optimal_angles = angles_list
        roi.kin_coords = [
            reverse_kins_to_pixels(kins_s, roi.det_psize, roi.det_distance, roi.centre_pixel)
            for kins_s in kins_list
        ]
    else:
        roi.g_vectors = g_inits[0]
        roi.kins = kins_list[0]
        roi.optimal_angles = angles_list[0]
        roi.kin_coords = reverse_kins_to_pixels(
            roi.kins, roi.det_psize, roi.det_distance, roi.centre_pixel
        )

@log_roi_params
def compute_kins_w_geometry(roi: ROI, pupil_roi: ROI, method="BFGS", gtol=1e-6):
    """
    Compute incident wave-vectors jointly with detector-geometry refinement,
    for a single- or multi-streak ROI.

    For multi-streak ROIs (roi.kouts is a list of (N_s, 3) arrays and
    roi.kout_coords is a matching list), each streak is optimised independently
    with its own G vector.  The detector distance — a global calibration
    parameter — is taken as the median across all per-streak estimates and
    written back to both roi and pupil_roi.

    Updates
    -------
    Single-streak: roi.g_init, roi.kins, roi.kouts, roi.optimal_angles,
                   roi.det_distance, roi.kin_coords, pupil_roi.kins,
                   pupil_roi.det_distance
    Multi-streak : all roi attributes as parallel lists; shared det_distance
                   and pupil_roi.kins recomputed at consensus distance
    """
    multi = isinstance(roi.kouts, list)
    kouts_list = roi.kouts if multi else [roi.kouts]
    kout_coords_list = roi.kout_coords if multi else [roi.kout_coords]

    g_inits, kins_list, kouts_refined, angles_list, distances = [], [], [], [], []

    for kouts_s, kout_coords_s in zip(kouts_list, kout_coords_list):
        g_init_s = calc_qvec(kouts_s, pupil_roi.kins_avg)
        
        kins_s, kouts_s_new, kins_pupil_s, params_s = optimise_kin_w_geometry(
            G_init =g_init_s, det_distance_init = roi.det_distance,
            kin_pixels = pupil_roi.kin_coords, kout_pixels = kout_coords_s,
            wavelength = roi.wavelength, pixel_size = roi.det_psize, central_pixel = roi.centre_pixel,
            method=method, gtol= gtol,
        )
        g_inits.append(g_init_s)
        kins_list.append(kins_s)
        kouts_refined.append(kouts_s_new)
        angles_list.append(params_s[1:])
        distances.append(float(params_s[0]))

    det_distance = float(np.median(distances))
    roi.det_distance = det_distance
    pupil_roi.det_distance = det_distance
    pupil_roi.kins = compute_vectors(
        pupil_roi.kin_coords, det_distance, roi.det_psize, roi.centre_pixel, roi.wavelength
    )

    if multi:
        roi.g_init = g_inits
        roi.kins = kins_list
        roi.kouts = kouts_refined
        roi.optimal_angles = angles_list
        roi.kin_coords = [
            reverse_kins_to_pixels(kins_s, roi.det_psize, det_distance, roi.centre_pixel)
            for kins_s in kins_list
        ]
    else:
        roi.g_init = g_inits[0]
        roi.kins = kins_list[0]
        roi.kouts = kouts_refined[0]
        roi.optimal_angles = angles_list[0]
        roi.kin_coords = reverse_kins_to_pixels(
            roi.kins, roi.det_psize, det_distance, roi.centre_pixel
        )
    
    
@log_roi_params
@time_it
def compute_kins_projection(roi: ROI, pupil_roi: ROI,
                            n_iterations=5, n_inner=3, clamp_to_pupil=False,
                            enforce_coplanar=False, ttheta=None):
    """
    Refine kins for one or multiple streaks using Ewald-circle projection.

    Per inner iteration:
      1. Project kins onto the Ewald circle for current G  (enforces |kin| = k)
      2. Update G = mean(kouts - kins_proj)
      3. Optionally rescale G to enforce |G| = 2k sin(ttheta/2)  (C3)
      4. Optionally clamp G_xy to the pupil-support feasible interval

    After all per-streak steps, optionally apply a coplanar projection across
    all streaks (see enforce_coplanar).

    Handles both single-streak ROIs (roi.kouts is an (N,3) array) and
    multi-streak ROIs produced by make_parallel_streaks (roi.kouts is a list
    of (N_s,3) arrays).

    If roi.g_init or roi.g_list exists (set by a prior compute_kins /
    compute_kins_w_geometry call), it is used as the starting G for each streak.

    Parameters
    ----------
    roi              : ROI whose kouts will be used as the streak k-vectors.
    pupil_roi        : ROI whose kouts define the pupil aperture bounds.
    n_iterations     : outer alternating-projection iterations.
    n_inner          : inner Ewald-projection steps per streak per outer iteration.
    clamp_to_pupil   : if True, additionally clamp G_xy so all implied kins lie
                       inside the pupil box (only reliable when width_ratio ≈ 1).
    enforce_coplanar : if True, project all G_s onto their best-fit plane after
                       each outer iteration, enforcing that all scattering vectors
                       lie in the same crystallographic plane.  Requires at least
                       3 streaks to be non-trivial; silently ignored otherwise.
    ttheta           : two-theta angle(s) in radians.  A scalar is shared across
                       all streaks; a list provides one value per streak.  When
                       given, G is radially rescaled after each G update so that
                       |G| = 2k sin(ttheta/2) is enforced (C3).  If None (default)
                       the magnitude of G floats freely.  Use roi.est_ttheta
                       (converted to radians) as a convenient source.

    Updates
    -------
    Single-streak: roi.kins (N,3), roi.g_init (3,), roi.kin_coords (N,2)
    Multi-streak : roi.kins (list), roi.g_list (list), roi.kin_coords (list)
    Always       : roi.width_ratios, roi.projection_history
    Multi-streak + coplanar: roi.scattering_plane_normal (3,)
    """
    multi = isinstance(roi.kouts, list)
    kouts_list = roi.kouts if multi else [roi.kouts]

    (kx_min, kx_max), (ky_min, ky_max) = compute_k_bounds(pupil_roi.kouts)
    pupil_kbounds = (kx_min, kx_max, ky_min, ky_max)

    # Forward G initialisation from a prior compute_kins / compute_kins_w_geometry call.
    g_init_attr = getattr(roi, 'g_vectors', None)
    
    if g_init_attr is None:
        g_init_attr = getattr(roi, 'g_vectors', None)
    if g_init_attr is not None:
        G_init_list = g_init_attr if isinstance(g_init_attr, list) else [g_init_attr]
        if len(G_init_list) != len(kouts_list):
            G_init_list = None  # shape mismatch — fall back to bootstrap
    else:
        G_init_list = None

    G_list, kins_list, width_ratios, history = projection_kin_refinement(
        kouts_list, pupil_kbounds, roi.wavelength,
        n_iterations, n_inner, clamp_to_pupil, G_init_list, enforce_coplanar,
        ttheta_list=ttheta,
    )

    # Store the final scattering plane normal if coplanar was applied
    last = history[-1] if history else {}
    if last.get('plane_normal') is not None:
        roi.scattering_plane_normal = last['plane_normal']

    roi.width_ratios = width_ratios
    roi.projection_history = history

    if multi:
        roi.kins = kins_list
        roi.g_vectors = G_list
        roi.kin_coords = [
            reverse_kins_to_pixels(kins, roi.det_psize, roi.det_distance, roi.centre_pixel)
            for kins in kins_list
        ]
    else:
        roi.kins = kins_list[0]
        roi.g_vectors = G_list[0]
        roi.kin_coords = reverse_kins_to_pixels(
            roi.kins, roi.det_psize, roi.det_distance, roi.centre_pixel
        )


@log_roi_params
def compute_kins_iterate(roi: ROI, pupil_roi: ROI, method='BFGS', gtol=1e-6,
                         n_iterations=5, reestimate_ttheta=False):
    """
    Refine kins for a single-streak ROI by alternating between geometry
    refinement and 2theta/angle refinement.

    Each cycle:
      1. optimise_kin_w_geometry — jointly refines the detector distance and G
         orientation using elastic-scattering and pupil-support constraints.
      2. optimise_kin — refines G orientation using the 2theta dot-product
         objective on the updated k-vectors.

    Parameters
    ----------
    roi               : single-streak Bragg ROI.
    pupil_roi         : ROI whose kin_coords define the pupil pixel positions.
    method            : scipy.optimize.minimize method string.
    gtol              : gradient tolerance for both optimisers.
    n_iterations      : number of geometry <-> angle cycles.
    reestimate_ttheta : if True, re-derive 2theta from refined k-vectors after
                        each geometry step instead of holding it fixed.

    Updates
    -------
    roi.kins, roi.kouts, roi.g_init, roi.det_distance,
    roi.kin_coords, roi.refinement_history
    pupil_roi.kins, pupil_roi.det_distance
    """
    if isinstance(roi.kouts, list):
        raise ValueError(
            "compute_kins_iterate works on a single-streak ROI. "
            "Run it per-streak or use compute_kins_projection for multi-streak ROIs."
        )

    roi.g_init = calc_qvec(roi.kouts, pupil_roi.kins_avg)
    ttheta = np.deg2rad(roi.est_ttheta)

    kin_opt, kouts, kins_pupil, G_opt, det_distance, history = iterate_kin_refinement(
        roi.g_init,
        roi.det_distance,
        pupil_roi.kin_coords,
        roi.kout_coords,
        ttheta,
        roi.wavelength,
        roi.det_psize,
        roi.centre_pixel,
        n_iterations=n_iterations,
        method=method,
        gtol=gtol,
        reestimate_ttheta=reestimate_ttheta,
    )

    roi.kins = kin_opt
    roi.kouts = kouts
    roi.g_init = G_opt
    roi.det_distance = det_distance
    roi.refinement_history = history
    roi.kin_coords = reverse_kins_to_pixels(
        roi.kins, roi.det_psize, roi.det_distance, roi.centre_pixel
    )

    pupil_roi.kins = kins_pupil
    pupil_roi.det_distance = det_distance


@log_roi_params
def refine_kins(roi:ROI, shifts):
    """
    Refine the estimated incident wavevectors (kins) by adjusting the optimal angles.

    Args:
        roi_name (str): The region of interest name.
        shifts (tuple): A tuple of (alpha_shift, beta_shift, gamma_shift) to refine angles.
    """
    alpha,beta,gamma = (roi.optimal_angles[0] + shifts[0],
                        roi.optimal_angles[1] + shifts[1],
                        roi.optimal_angles[2] + shifts[2] )
    
    R = rotation_matrix(alpha, beta, gamma)
    try:
        g_update =  R @ roi.g_init
    except:
        g_update =  R @ roi.g_init[0]
    kin_opt = (roi.kouts-g_update)
    kin_opt /= np.linalg.norm(kin_opt, axis = 1)[:,np.newaxis]
    kin_opt *= 2*np.pi/roi.wavelength
    roi.kins = kin_opt
    roi.kin_coords = reverse_kins_to_pixels(roi.kins,   roi.det_psize, 
                                                        roi.det_distance, 
                                                        roi.centre_pixel)



def select_single_pixel_streak(roi:ROI, width=1, offset=0, inplace = False):

    """
    Selects a single-pixel-wide streak along a specified direction in the region of interest (ROI).

    Args:
        roi_name (str): The name of the region of interest.
        width (int, optional): The width of the streak to extract. Default is 1.
        position (str, optional): The position of the streak relative to the main region.
            Options: 'center', 'top', 'bottom'. Default is 'center'.
        offset (int, optional): An offset value to shift the streak selection up or down. Default is 0.
        inplace (bool): if True, it changes the kin_coords, kins, kouts, and kout_coords dict for that roi, if false, if returns the mask. (default = False)

    """

    # mask = extract_parallel_line(roi.kouts[:,:2], width=width, offset=offset)
    mask = extract_parallel_line(roi.kouts[:,:2], width=width, offset=offset)
    if inplace:
        roi.kins = roi.kins[mask]
        roi.kin_coords = roi.kin_coords[mask]
        roi.kouts = roi.kouts[mask]
        roi.kout_coords = roi.kout_coords[mask]
        roi.g_vectors = roi.g_vectors[mask]
        
        try:
            roi.coherent_imgs = roi.coherent_imgs[mask]
        except:
            print("coherent images are not calculated yet")
        
        return  
    else:
        return mask

def make_parallel_streaks(roi:ROI, width= None, extent= None, mask_thr = 0):

    if width is None: 
        width = roi.delta_qz
        extent = roi.delta_qz * 20 # 20 is a bit arbitrary 
        
    offsets = np.arange(-extent,extent,width)
    masks = [select_single_pixel_streak(roi, width = width, offset = off, inplace = False) for off in offsets]
    masks_clean= [mask for mask in masks if np.sum(mask)>mask_thr]
    
    roi.coherent_imgs = [roi.coherent_imgs[mask] for mask in masks_clean]
    roi.kins = [roi.kins[mask] for mask in masks_clean]
    roi.kouts = [roi.kouts[mask] for mask in masks_clean]
    roi.kin_coords = [roi.kin_coords[mask] for mask in masks_clean]
    roi.kout_coords = [roi.kout_coords[mask] for mask in masks_clean]

def make_qz_streaks(roi, dqz=None, extent=None, min_pixels=5):
    """
    Split an ROI into constant-qz streaks.
    """

    if dqz is None:
        dqz = roi.delta_qz

    masks, edges = extract_qz_streaks(
        roi.kouts,
        dqz=dqz,
        extent=extent,
        min_pixels=min_pixels,
    )

    roi.coherent_imgs = [roi.coherent_imgs[m] for m in masks]
    roi.kins          = [roi.kins[m] for m in masks]
    roi.kouts         = [roi.kouts[m] for m in masks]
    roi.kin_coords    = [roi.kin_coords[m] for m in masks]
    roi.kout_coords   = [roi.kout_coords[m] for m in masks]

    # if hasattr(roi, "g_vectors"):
    #     roi.g_vectors = [roi.g_vectors[m] for m in masks]

    roi.qz_edges = edges
    roi.qz_masks = masks

    Nz = len(masks)
    roi.dz = None if Nz == 0 else 2 * np.pi / (dqz * Nz)
    
def make_q_params(roi:ROI):

    roi.det_psize_eff = calc_effective_pixel_size(roi.kouts, roi.det_psize)
    roi.delta_q = calculate_delta_q(roi.ttheta, roi.det_distance, roi.wavelength, roi.det_psize_eff)
    roi.delta_qz = calculate_delta_qz(roi.ttheta, roi.det_distance, roi.wavelength, roi.det_psize_eff)
    

def select_streak_region(roi:ROI, percentage=10, start_position='lowest', start_idx=None, inplace = False):
    """
    Selects a region of the streak based on a percentage of its total size.

    Args:
        roi_name (str): The name of the region of interest.
        percentage (float, optional): The percentage of the streak to retain. Default is 10%.
        start_position (str, optional): The starting position for selection.
            Options: 'lowest' (bottom), 'highest' (top), or 'middle'. Default is 'lowest'.
        start_idx (int, optional): The specific index to start selection from. Overrides `start_position` if provided.
    """
    
    mask = extract_streak_region(roi.kin_coords, percentage=percentage, start_position=start_position, 
                                 start_idx=start_idx, seed=42)
    masks_clean= [mask for mask in masks if np.sum(mask)!=0]

    if inplace:
        roi.kins = roi.kins[mask]
        roi.kin_coords = roi.kin_coords[mask]
        roi.kouts = roi.kouts[mask]
        roi.kout_coords = roi.kout_coords[mask]
        roi.coherent_imgs = roi.coherent_imgs[mask]
    else:
        return mask


    
def bin_along_streak(roi:ROI, bin_size=4):
    """
    before running: coherent images have to be made into a list of images, each element corresponding to single pixel wide streak. 

    assumes the images are ordered along the streak
    """

    binned = []
    for img_arr in roi.coherent_imgs:

        img_binned = sliding_bin_1d(img_arr, k=bin_size)
        binned.append(img_binned)

    roi.coherent_imgs = binned
    
    
def order_pixels(roi:ROI):
    """Reorders the pixels of coherent images and corresponding k-vectors based on their distance from the center.

    Args:
        roi_name (str): The name of the region of interest (ROI) for which the pixels 
                         and k-vectors will be reordered.
    """
    sorted_indices = reorder_pixels_from_center(roi.kouts, connected_array=np.array(roi.coherent_imgs))
    
    # Debugging: Print types and shapes
    print(f"sorted_indices dtype: {sorted_indices.dtype}, shape: {sorted_indices.shape}")
    print(f"kouts shape before indexing: {roi.kouts.shape}")

    roi.kouts = roi.kouts[sorted_indices]
    roi.kout_coords = roi.kout_coords[sorted_indices]
    roi.coherent_imgs = roi.coherent_imgs[sorted_indices]

    try:
        roi.kins = roi.kins[sorted_indices]
        roi.kin_coords = roi.kin_coords[sorted_indices]
    except:
        print("Did not order kins and kin coords.")

    try:
        roi.detected_objects = roi.detected_objects[sorted_indices]
    except:
        print("Did not order detected objects.")   
        
################### preprocessing coherent images ###################
@time_it
def make_coherent_images(roi:ROI):
    
    """
    Generates a list of coherent images for a given region of interest (ROI).
    Args:
        roi_name (str): The name of the region of interest (ROI) for which the coherent 
                         images will be generated. The coordinates for the ROI are 
                         retrieved from `self.kout_coords[roi_name]`, and the corresponding 
                         ptychographic data is accessed from `self.ptychographs[roi_name]`.

    """
    
    coherent_imgs = []
    
    for i, coord in enumerate(roi.kout_coords):
        
        
        xp =  coord[0] - roi.coords[0]
        yp =  coord[1] - roi.coords[2]

        coherent_imgs.append(make_coherent_image(roi.data_4d, np.array([xp,yp])))

    roi.coherent_imgs = np.array(coherent_imgs)


@log_roi_params  
def filter_coherent_images(roi:ROI, variance_threshold):
    """Filters coherent images based on variance threshold.

    Args:
        roi_name (str): The name of the region of interest (ROI) for which the 
                         coherent images will be filtered.
        variance_threshold (float): The threshold value for variance. Images with 
                                    variance below this threshold will be filtered out.

    """
    N = roi.coherent_imgs.shape[0]
    
    cleaned_coh_images, cleaned_kins, cleaned_kxky = filter_images(images = roi.coherent_imgs, 
                                                                     coords = roi.kins,
                                                                     kin_coords=roi.kouts, 
                                                                     variance_threshold = variance_threshold)


    roi.coherent_imgs = cleaned_coh_images.copy()
    roi.kouts = cleaned_kxky.copy()
    roi.kins = cleaned_kins.copy()

    N2 =  roi.coherent_imgs.shape[0]

    print(f"Filtered out {N - N2} images")
    print(f"Number of images: {N2}")
    
@log_roi_params
def remove_coh_background(roi:ROI, sigma, num_jobs = -1):
    """Removes the background noise from coherent images for a given region of interest (ROI).

    Args:
        roi_name (str): The name of the region of interest (ROI) whose coherent 
                         images will be processed.
        sigma (float): The standard deviation of the Gaussian filter used for 
                       background removal. A higher value will smooth the image 
                       more, removing more background noise but potentially 
                       blurring fine details.
    """
    
    roi.coherent_imgs = remove_background_parallel(roi.coherent_imgs, sigma=sigma, n_jobs=num_jobs)
    
def even_dims_cohimages(roi:ROI, mode = 'edge', num_jobs = -1, **kwargs):
    """Adjusts the dimensions of coherent images to be even.

    This method takes the coherent images for a given region of interest (ROI) 
    and ensures that their dimensions are even by calling the `make_2dimensions_even` function.

    Args:
        roi_name (str): The name of the region of interest (ROI) for which the coherent 
                         images will have their dimensions adjusted. The coherent images 
                         are accessed from `self.coherent_imgs[roi_name]`.

    """
    
    roi.coherent_imgs = make_2dimensions_even(roi.coherent_imgs, mode=mode, num_jobs=num_jobs, **kwargs)
    roi.averaged_coherent_images = np.mean(roi.coherent_imgs, axis = 0)


@log_roi_params
def apply_median_filter(roi:ROI, kernel_size, stride, threshold, num_jobs = -1):
    """Applies a median filter to coherent images for a given region of interest (ROI).

    Args:
        roi_name (str): The name of the region of interest (ROI) whose coherent 
                         images will be processed.
        kernel_size (int): The size of the square kernel to be used for the 
                            median filter. A larger kernel size will result in 
                            stronger smoothing.
        stride (int): The step size for applying the kernel across the image. 
                       A higher stride will apply the filter less frequently.
        threshold (float): The threshold value used to filter out small intensity 
                           variations, helping to remove noise and improve the 
                           quality of the coherent images.
    """
    roi.coherent_imgs = median_filter_parallel(roi.coherent_imgs, 
                                                          kernel_size = kernel_size, 
                                                          stride = stride, 
                                                          threshold=threshold, 
                                                          n_jobs=num_jobs)
    
@log_roi_params
def apply_bilateral_filter(roi:ROI, sigma_spatial, sigma_range, kernel_size, num_jobs = -1):
    """Applies a bilateral filter to the coherent images for a given region of interest (ROI).

    Args:
        roi_name (str): The name of the region of interest (ROI) whose coherent 
                         images will be processed.
        sigma_spatial (float): The standard deviation of the spatial Gaussian filter. 
                               Larger values result in greater spatial smoothing.
        sigma_range (float): The standard deviation of the range (intensity) Gaussian 
                             filter. Larger values preserve more intensity variation.
        kernel_size (int): The size of the square kernel to be used for the bilateral 
                        filter. A larger kernel size will result in stronger filtering.
"""
    roi.coherent_imgs = bilateral_filter(roi.coherent_imgs, sigma_spatial, 
                                                                  sigma_range, 
                                                                  kernel_size, n_jobs=num_jobs)
    
@log_roi_params
def detect_object(roi:ROI, threshold, min_val, max_val, num_jobs = -1):
    """Detects objects within the specified region of interest (ROI) and filters based on intensity.
    Args:
        roi_name (str): The name of the region of interest (ROI) to process.
        threshold (float): The intensity threshold for object detection. Objects with intensity 
                           below this threshold will not be detected.
        min_val (float): The minimum intensity threshold for filtering detected objects.
        max_val (float): The maximum intensity threshold for filtering detected objects.

    """
    roi.detected_objects = detect_obj_parallel(roi.coherent_imgs, threshold=threshold, n_jobs=num_jobs)
    
    
    mask = [np.sum(im) > min_val and np.sum(im) < max_val for im in roi.detected_objects]
    print(f'length of the mask is {np.sum(mask)}')
    
    roi.detected_objects = np.array(roi.detected_objects)[mask].copy()
    roi.coherent_imgs = np.array(roi.coherent_imgs)[mask].copy()

    print(f'length of detected objects {roi.detected_objects.shape}')
    print(f'length of coherent images {roi.coherent_imgs.shape}')
    
    roi.kouts = np.array(roi.kouts)[mask].copy()
    roi.kins = np.array(roi.kins)[mask].copy()
    
@log_roi_params    
def blur_detected_objs(roi:ROI, sigma):
    """Applies Gaussian blur to the detected objects in the specified region of interest (ROI).

    Args:
        roi_name (str): The name of the region of interest (ROI) whose detected object 
                         images will be blurred.
        sigma (float): The standard deviation of the Gaussian filter. Larger values 
                       result in more blurring.

    """
    roi.detected_objects = apply_gaussian_blur(roi.detected_objects, sigma=sigma)
    
@log_roi_params
def mask_cohimgs_threshold(roi:ROI, threshold_value):
    """Applies a threshold mask to the coherent images in the given ROI.
    
    Args:
        roi_name (str): The name of the region of interest (ROI) whose coherent images 
                         will be thresholded.
        threshold_value (float): The threshold value used for masking the coherent images. 
                                  Values below this threshold will be masked.

    """
    roi.coherent_imgs = threshold_data(roi.coherent_imgs, threshold_value)
    
@log_roi_params
def mask_region_cohimgs(roi:ROI, region, mode = 'zeros'):
    """Masks a specific region of coherent images within the given ROI.

    Args:
        roi_name (str): The name of the region of interest (ROI) whose coherent images 
                         will be modified.
        region (tuple): A tuple (sx, ex, sy, ey) specifying the start and end indices 
                        for the region to be masked:
                        - sx, ex: x-axis start and end indices
                        - sy, ey: y-axis start and end indices
        mode (str, optional): A string specifying how to replace the masked region. 
                        - 'zeros'. Default.
                        - 'median'
                        - 'ones'
    """
    sx,ex,sy,ey = region
    roi.coherent_imgs = np.array(roi.coherent_imgs)
    if mode == 'zeros':
        roi.coherent_imgs[:,sx:ex,sy:ey] = 0.0
        roi.update_averaged_coherent_images()
    elif mode == 'median':
        roi.coherent_imgs[:,sx:ex,sy:ey] = np.median(roi.coherent_imgs, 
                                                                    axis = (1,2))[:,np.newaxis, np.newaxis]
        roi.update_averaged_coherent_images()
    elif mode == 'ones':
        roi.coherent_imgs[:,sx:ex,sy:ey] = 1.0
        roi.update_averaged_coherent_images()
        
def align_coherent_images(roi:ROI, ref_idx = None):
    """Aligns a list of coherent images for a given region of interest (ROI).

    Args:
        roi_name (str): The name of the region of interest (ROI) whose coherent images
                         will be aligned.

    """
    print('aligned coherent images')
    roi.coherent_imgs = align_images(roi.coherent_imgs, ref_idx = ref_idx)

def flip_images(roi:ROI, flip_mode):
    """
    Flips images based on the selected mode.
    
    Args:
        images (np.ndarray): A batch of images with shape (N, H, W).
        flip_mode (str): The flipping mode, one of:
                        - "xy"  (original)
                        - "x_neg_y" (flip y-axis)
                        - "neg_x_y" (flip x-axis)
                        - "neg_x_neg_y" (flip both axes)

    Returns:
        np.ndarray: The transformed batch of images.
    """
    if flip_mode == "xy":
        return roi.coherent_imgs
    elif flip_mode == "x_neg_y":
        return [np.flip(img, axis=0) for img in roi.coherent_imgs]
    elif flip_mode == "neg_x_y":
        return [np.flip(img, axis=1) for img in roi.coherent_imgs]
    elif flip_mode == "neg_x_neg_y":
        return [np.flip(img, axis=(0, 1)) for img in roi.coherent_imgs]
    else:
        raise ValueError(f"Invalid flip_mode: {flip_mode}")

def tilt_correct(roi:ROI, iters = 3):
    
    ref = None
    if roi.tilt_corrected_image is not None:
        ref = roi.tilt_corrected_image

    roi.tilt_corrected_image = tilt_correction(roi.coherent_imgs, ref, iters=iters)[0]

def sum_line(args):
    file, dataset_key = args
    with h5py.File(file, "r") as h5:
            
        dset = h5[dataset_key] 

        accumulator = np.sum(dset, axis = 0)

    return accumulator

def sum_line(args):
    file, dataset_key = args
    with h5py.File(file, "r") as h5:
            
        dset = h5[dataset_key] 

        accumulator = np.sum(dset, axis = 0)

    return accumulator
    
@time_it
def sum_detector_frames(scan: Scan, detector = 'Eiger', plot=False, num_jobs = 32):

    if detector == 'Eiger':
        dataset_key="entry/data/data"

    elif detector == "Lambda":
        dataset_key = "/entry/instrument/detector/data"
        
    args_list = [(file, dataset_key) for file in scan.file_list[:-1]]

    print("got list")
    
    with Pool(processes=num_jobs) as pool:    
        
        sums = list(tqdm(
            pool.imap(sum_line, args_list),
            total=len(args_list),
            desc="Loading frames"
        ))

    print("Sum Done")
    stacked_data = np.array(sums)
    accumulator = np.sum(stacked_data, axis=0)

    accumulator = mask_hot_pixels(accumulator)

    print("Masking done")
    
    if plot:
        plt.figure()
        plt.imshow(accumulator)
        plt.title(f"Summed Detector Frames; Scan no. {scan.scan_num}")
        plt.show()

    else:
        return accumulator

def std_line(args):
    file, dataset_key = args
    with h5py.File(file, "r") as h5:
            
        dset = h5[dataset_key] 

        accumulator = np.std(dset, axis = 0)

    return accumulator
    
@time_it
def std_detector_frames(scan: Scan, detector = 'Eiger', plot=False, num_jobs = 32):

    detector = scan.detector
    if detector.lower() == 'eiger':
        dataset_key="entry/data/data"

    elif detector.lower() == "lambda":
        dataset_key = "/entry/instrument/detector/data"
        
    args_list = [(file, dataset_key) for file in scan.file_list[:-1]]

    print("got list")
    
    with Pool(processes=num_jobs) as pool:    
        
        sums = list(tqdm(
            pool.imap(std_line, args_list),
            total=len(args_list),
            desc="Loading frames"
        ))

    print("Sum Done")
    stacked_data = np.array(sums)
    accumulator = np.std(stacked_data, axis=0)

    accumulator = mask_hot_pixels(accumulator)

    print("Masking done")
    
    if plot:
        plt.figure()
        plt.imshow(accumulator)
        plt.title(f"Summed Detector Frames; Scan no. {scan.scan_num}")
        plt.show()

    else:
        return accumulator
        
        