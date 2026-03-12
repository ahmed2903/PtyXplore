# Image processing utilities for ptychography data at PETRA 3
#
# Author: Ahmed H. Mokhtar
# Email: ahmed.mokhtar@desy.de

import numpy as np
import scipy
import scipy.fft
import scipy.ndimage as ndimage
from scipy.ndimage import zoom, gaussian_filter, fourier_shift, binary_dilation, uniform_filter

from joblib import Parallel, delayed
from skimage.measure import shannon_entropy

# import cv2
import torch
import torch.nn.functional as F

from .utils import time_it


def sum_pool2d_array(input_array, kernel_size, stride=None, padding=0):
    """
    Perform sum pooling on a 4D NumPy array or PyTorch tensor.

    Parameters:
    - input_array: 4D NumPy array or PyTorch tensor
    - kernel_size: Size of the pooling kernel.
    - stride: Stride of the pooling operation. Default is kernel_size.
    - padding: Amount of zero-padding added to both sides of the input. Default is 0.

    Returns:
    - output: Pooled array as a NumPy array.
    """
    if stride is None:
        stride = kernel_size

    is_numpy = isinstance(input_array, np.ndarray)

    if is_numpy:
        input_tensor = torch.tensor(input_array, dtype=torch.float32)
    else:
        input_tensor = input_array

    if len(input_tensor.size()) < 3:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    pooled = F.avg_pool2d(input_tensor, kernel_size, stride, padding)
    pooled = pooled.squeeze(0).squeeze(0)

    if is_numpy:
        return pooled.numpy()

    return pooled

def make_coherent_image(data: np.ndarray, pixel_idx: np.ndarray, slow_axis=0):
    """
    Makes a single coherent image from a 4D data set that is size (M,N,x,y) using one pixel

    M is the slow scan
    N is the fast scan
    x,y are detector size

    pixel_idx: the index of the pixel in the detector to be used in the coherent image
    """
    px = pixel_idx[0]
    py = pixel_idx[1]

    return data[:,:,px,py]

def make_detector_image(data: np.ndarray, position_idx: np.ndarray):
    """
    Makes a single detector image from a 4D data set that is size (M,N,x,y) using one location

    M is the slow scan
    N is the fast scan
    x,y are detector size

    position_idx: the index of the location in the detector to be used in the detector image
    """
    px = position_idx[0]
    py = position_idx[1]

    return data[px,py,:,:]

@time_it
def mask_hot_pixels(array, mask_max_coh=False, mask_min_coh=False):
    """
    Masks hot pixels (maximum values in the last two dimensions) in a 4D array.

    Parameters:
    - array: 4D numpy array with shape (N, M, x, y) where the last two dimensions
      represent the detector image (x, y).

    Returns:
    - masked_array: The array with hot pixels masked (replaced with NaN or 0).
    """
    if len(array.shape) > 3:
        max_values = np.max(array, axis=(2, 3), keepdims=True)
        if mask_max_coh:
            max_values_coh = np.max(array, axis=(0, 1), keepdims=True)
        if mask_min_coh:
            min_values_coh = np.min(array, axis=(0, 1), keepdims=True)

    if len(array.shape) == 2:
        max_values = np.max(array, axis=(0, 1), keepdims=True)
    else:
        max_values = np.max(array, axis=(1, 2), keepdims=True)

    masked_array = np.where(array == max_values, 0.0, array)

    if mask_max_coh or mask_min_coh:
        med = np.median(masked_array, axis=(0,1))

    if mask_max_coh:
        masked_array = np.where(masked_array == max_values_coh, med, masked_array)
    if mask_min_coh:
        masked_array = np.where(masked_array == min_values_coh, med, masked_array)

    return masked_array

def estimate_pupil_size(array, mask_val, pixel_size, pupil_roi=None, crop=True):
    """
    Computes the average length of a box with jagged edges in x and y directions.

    Args:
        array (2D ndarray): A 2D array where the box is represented by non-zero values.
        mask_val (float): value of the mask
        pixel_size (float): size of the detector pixel
        pupil_roi (list): a 4 element list containing the roi of the pupil [xs, xe, ys, ye]

    Returns:
        tuple: (average_x_length, average_y_length)
    """
    if pupil_roi is not None:
        array = array[pupil_roi[0]:pupil_roi[1], pupil_roi[2]:pupil_roi[3]]

    min_lx = array.shape[0] / 2
    min_ly = array.shape[1] / 2

    nx, ny = np.where(array > mask_val)

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

    avg_x = np.mean(np.array(x_lengths)) * pixel_size
    avg_y = np.mean(np.array(y_lengths)) * pixel_size

    return avg_x, avg_y

def downsample_array(arr, new_shape):
    """
    Downsample a 2D array by taking mean of non-overlapping rectangular blocks

    Args:
        arr (np.ndarray): Input 2D array
        new_shape (tuple): Desired output shape

    Returns:
        np.ndarray: Downsampled array
    """
    orig_rows, orig_cols = arr.shape
    new_rows, new_cols = new_shape

    row_factor = orig_rows // new_rows
    col_factor = orig_cols // new_cols

    trimmed_arr = arr[:new_rows*row_factor, :new_cols*col_factor]

    return trimmed_arr.reshape(
        new_rows, row_factor,
        new_cols, col_factor
    ).max(axis=(1, 3))

def make_2dimensions_even(array_list, mode='constant', num_jobs=-1, **kwargs):
    """
    Takes a list of NumPy arrays and ensures that all arrays have even dimensions.
    If either dimension is odd, the array is padded at the end to make it even.

    Parameters:
        array_list (list of numpy.ndarray): List of arrays to process.

    Returns:
        list of numpy.ndarray: List of arrays with even dimensions.
    """
    ref_shp = array_list[0].shape
    target_shape = list(ref_shp)

    for i in range(len(target_shape)):
        if target_shape[i] % 2 != 0:
            target_shape[i] += 1

    padding = ((0, target_shape[0] - ref_shp[0]), (0, target_shape[1] - ref_shp[1]))

    def process_array(array):
        return np.pad(array, padding, mode=mode, **kwargs)

    return np.array(Parallel(n_jobs=num_jobs)(delayed(process_array)(arr) for arr in array_list))

@time_it
def filter_images(images, coords, variance_threshold, kin_coords=None, **kwargs):
    """
    Filters images based on variance, with an optional entropy threshold.
    Can also filter additional coordinates (kinematic coordinates) if provided.

    Parameters:
    -----------
    images : List of 2D images to filter.
    coords : List of coordinates associated with the images.
    variance_threshold : Minimum variance required to keep an image.
    kin_coords : Additional coordinates to filter along with `coords`.
    entropy_threshold : Minimum Shannon entropy required to keep an image.

    Returns:
    --------
    filtered_images : The filtered images.
    filtered_coords : The filtered coordinates.
    filtered_kin_coords : The filtered kinematic coordinates (if `kin_coords` is provided).
    """
    entropy_threshold = kwargs.get('entropy_threshold', None)
    filtered_images = []
    filtered_coords = []
    filtered_kin_coords = [] if kin_coords is not None else None
    entropies = [] if entropy_threshold is not None else None

    for i, (img, coord) in enumerate(zip(images, coords)):
        variance = np.var(img)
        keep = variance >= variance_threshold

        if entropy_threshold is not None:
            entropy = shannon_entropy(img)
            if keep and entropy >= entropy_threshold:
                if entropies is not None:
                    entropies.append(entropy)
            else:
                keep = False

        if keep:
            filtered_images.append(img)
            filtered_coords.append(coord)
            if kin_coords is not None:
                filtered_kin_coords.append(kin_coords[i])

    if kin_coords is not None:
        return np.array(filtered_images), np.array(filtered_coords), np.array(filtered_kin_coords)
    else:
        return np.array(filtered_images), np.array(filtered_coords)

def phase_correlation(ref_img, shifted_img):
    """Compute shift between two images using phase correlation."""
    ref_fft = scipy.fft.fft2(ref_img)
    shifted_fft = scipy.fft.fft2(shifted_img)

    cross_power = (ref_fft * np.conj(shifted_fft)) / np.abs(ref_fft * np.conj(shifted_fft))

    correlation = scipy.fft.ifft2(cross_power)
    correlation = np.abs(scipy.fft.fftshift(correlation))

    max_loc = np.unravel_index(np.argmax(correlation), correlation.shape)
    shift = np.array(max_loc) - np.array(ref_img.shape) // 2
    return shift

def align_images(image_list, ref_idx=None):
    """Align a list of shifted images based on the first image.

    This function aligns the images by calculating the shift between the reference image
    and each subsequent image using phase correlation.

    Args:
        image_list (list of np.ndarray): A list of images (numpy arrays) to be aligned.
        ref_idx: Index of the reference image. If None, the mean image is used.

    Returns:
        np.ndarray: Array of aligned images.
    """
    if ref_idx is None:
        ref_img = np.mean(image_list, axis=0)
    else:
        ref_img = image_list[ref_idx]

    aligned_images = []
    for img in image_list:
        shift = phase_correlation(ref_img, img)
        aligned_img = scipy.ndimage.shift(img, shift, mode='constant')
        aligned_images.append(aligned_img)

    return np.array(aligned_images)

def remove_background(image, sigma=20):
    """Removes smooth background by subtracting a Gaussian-blurred version of the image."""
    background = ndimage.gaussian_filter(image, sigma=sigma)
    tmp = image - background
    inverted = np.max(tmp) - tmp
    return inverted

def remove_background_parallel(image_list, sigma=20, n_jobs=8):
    """Applies background removal to all images in a list using multiprocessing."""
    result = Parallel(n_jobs=n_jobs)(delayed(remove_background)(im, sigma) for im in image_list)
    return np.array(result)

def upsample_image(im, zoom_factor):
    """Upsample a single image using zoom."""
    return zoom(im, zoom_factor, order=3).astype(complex)

def upsample_images(images, zoom_factor, n_jobs=4):
    """
    Upsample a list of images in parallel.

    Parameters:
        images (list of numpy.ndarray): List of input images.
        zoom_factor (float or tuple): Zoom factor for upsampling.
        n_jobs (int): Number of CPU cores to use.

    Returns:
        numpy.ndarray: Array of upsampled images.
    """
    return np.array(Parallel(n_jobs=n_jobs)(delayed(upsample_image)(im, zoom_factor) for im in images))

def detect_object(image, threshold_factor=0.5):
    """Detects the object by thresholding and finding the largest connected component."""
    image_norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-10)
    threshold = threshold_factor * np.max(image_norm)
    binary_mask = image_norm > threshold

    labeled, num_features = ndimage.label(binary_mask)

    if num_features == 0:
        return None

    object_sizes = ndimage.sum(binary_mask, labeled, range(num_features + 1))
    largest_object_label = np.argmax(object_sizes[1:]) + 1

    return labeled == largest_object_label

@time_it
def detect_obj_parallel(image_list, threshold=.1, n_jobs=8):
    """Detects objects in all images in a list using multiprocessing."""
    result = Parallel(n_jobs=n_jobs)(delayed(detect_object)(im, threshold) for im in image_list)
    return np.array(result)

def median_filter(image, kernel_size, stride, threshold):
    """
    Applies a median-based filter to an image to replace bad pixels.

    Parameters:
        image (numpy.ndarray): Input 2D image.
        kernel_size (int): Size of the sliding window (kernel) for computing the median.
        stride (int): Step size for moving the kernel across the image.
        threshold (float): Values outside `median * threshold` are replaced with the median.

    Returns:
        numpy.ndarray: Filtered image.
    """
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    filtered_image = np.copy(image)

    for i in range(0, image.shape[0], stride):
        for j in range(0, image.shape[1], stride):
            roi = padded_image[i:i + kernel_size, j:j + kernel_size]
            median_value = np.median(roi)

            lower_bound = median_value / threshold
            upper_bound = median_value * threshold

            roi_original = filtered_image[i:i + kernel_size, j:j + kernel_size]
            outliers = (roi_original < lower_bound) | (roi_original > upper_bound)
            roi_original[outliers] = median_value

    return filtered_image

@time_it
def median_filter_parallel(images, kernel_size, stride, threshold, n_jobs=32):
    """
    Applies the median filter to a list of images in parallel.

    Parameters:
        images (list of numpy.ndarray): List of input 2D images.
        kernel_size (int): Size of the sliding window.
        stride (int): Step size for moving the kernel.
        threshold (float): Outlier replacement threshold.
        n_jobs (int): Number of parallel jobs.

    Returns:
        numpy.ndarray: Array of filtered images.
    """
    filtered_images = Parallel(n_jobs=n_jobs)(
        delayed(median_filter)(image, kernel_size, stride, threshold) for image in images
    )
    return np.array(filtered_images)

def pad_to_double(img):
    """
    Pads a 2D numpy array to double its size, centering the original.

    Args:
        img: A 2D numpy array

    Returns:
        Padded 2D numpy array with double the dimensions
    """
    h, w = img.shape
    pad_h = h // 2
    pad_w = w // 2
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

@time_it
def pad_to_double_parallel(image_list, n_jobs=8):
    """Pads each image in a list to double its size, in parallel."""
    return Parallel(n_jobs=n_jobs)(delayed(pad_to_double)(image) for image in image_list)

def extract_centre(img):
    """
    Slices the center of a 2D numpy array, keeping only half the size in each dimension.

    Args:
        img: A 2D numpy array

    Returns:
        Center-sliced 2D numpy array with half the dimensions
    """
    h, w = img.shape
    h_start = h // 4
    h_end = h - h_start
    w_start = w // 4
    w_end = w - w_start
    return img[h_start:h_end, w_start:w_end]

def shrink_wrap_2d_numpy(input_data, threshold=0.2, sigma=4):
    """
    Applies the shrink-wrap method to create a mask around an object in a 2D NumPy array.

    Args:
        input_data (np.ndarray): 2D input array (height, width).
        threshold (float): Threshold for masking (relative to max value).
        sigma (float): Standard deviation for Gaussian blur.

    Returns:
        np.ndarray: Binary mask of the same shape as input_data.
    """
    if input_data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")

    abs_data = np.abs(input_data)
    max_val = np.max(abs_data)

    mask = np.where(abs_data < (threshold * max_val), 0, 1).astype(np.float32)
    mask = gaussian_filter(mask, sigma=sigma)
    mask = np.where(mask < threshold, 0, 1).astype(np.uint8)

    return mask

@time_it
def shrink_wrap_parallel(image_list, threshold=0.2, sigma=4, n_jobs=8):
    """Applies shrink-wrap masking to a list of images in parallel."""
    return Parallel(n_jobs=n_jobs)(
        delayed(shrink_wrap_2d_numpy)(image, threshold, sigma) for image in image_list
    )

def apply_gaussian_blur(arrays, sigma=1):
    """
    Apply Gaussian blur to a list of arrays.

    Parameters:
        arrays (list of numpy arrays): The list of arrays to blur.
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        numpy.ndarray: Array of blurred results.
    """
    return np.array([gaussian_filter(arr, sigma=sigma) for arr in arrays])

def compute_histograms(image_list, bins=256):
    """
    Computes the intensity histogram for each image in a list.

    Args:
        image_list (list of np.ndarray): List of grayscale images (2D NumPy arrays).
        bins (int): Number of bins for the histogram.

    Returns:
        list of np.ndarray: A list where each element is a histogram array.
    """
    histograms = []
    for img in image_list:
        maxima = img.max()
        hist, _ = np.histogram(img.ravel(), bins=bins, range=[0, maxima])
        histograms.append(hist)
    return histograms

def threshold_data(image_list, threshold_value):
    """
    Threshold the intensity of each image in a list.

    Args:
        image_list (list of np.ndarray): List of grayscale images (2D NumPy arrays).
        threshold_value (float): Values below this are set to zero.

    Returns:
        np.ndarray: Thresholded images.
    """
    return np.array([np.where(img < threshold_value, 0, img) for img in image_list])

@time_it
def bilateral_filter(images, sigma_spatial=3, sigma_range=50, kernel_size=7, n_jobs=8):

    def _bilateral_single(image):
        return cv2.bilateralFilter(image, d=kernel_size, sigmaColor=sigma_range, sigmaSpace=sigma_spatial)

    if images.ndim == 2:
        return _bilateral_single(images)

    elif images.ndim == 3:
        if n_jobs > 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_bilateral_single)(img) for img in images
            )
        else:
            results = [_bilateral_single(img) for img in images]

        return np.stack(results, axis=0)

@time_it
def reorder_pixels_from_center(pixel_coords, connected_array=None):

    pixel_coords = np.array(pixel_coords)
    centroid = np.mean(pixel_coords, axis=0)

    distances = np.linalg.norm(pixel_coords - centroid, axis=1)
    center_pixel = pixel_coords[np.argmin(distances)]

    distances_from_center = np.linalg.norm(pixel_coords - center_pixel, axis=1)
    sorted_indices = np.argsort(distances_from_center)

    return np.array(sorted_indices, dtype=int)

def sliding_bin_1d(images, k):
    """
    Bins data along the streak by sliding average.
    Assumes the images are ordered.
    """
    N = images.shape[0]
    out = np.empty_like(images)
    k2 = k // 2
    for i in range(N):
        start = max(0, i - k2)
        end = min(N, i + k2 + 1)
        out[i] = images[start:end].mean(axis=0)
    return out

def flip_images(images, flip_mode):
    """
    Flips images based on the selected mode.

    Args:
        images (np.ndarray): A batch of images with shape (N, H, W).
        flip_mode (str): One of "xy", "x_neg_y", "neg_x_y", "neg_x_neg_y".

    Returns:
        np.ndarray: The transformed batch of images.
    """
    if flip_mode == "xy":
        return images
    elif flip_mode == "x_neg_y":
        return np.flip(images, axis=2)
    elif flip_mode == "neg_x_y":
        return np.flip(images, axis=1)
    elif flip_mode == "neg_x_neg_y":
        return np.flip(images, axis=(1, 2))
    else:
        raise ValueError(f"Invalid flip_mode: {flip_mode}")
