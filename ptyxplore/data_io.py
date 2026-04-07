# HDF5 / NXS data loading and stacking for ptychography scans at PETRA 3
#
# Author: Ahmed H. Mokhtar
# Email: ahmed.mokhtar@desy.de

import os
import concurrent.futures

import numpy as np
import h5py
import hdf5plugin  # registers HDF5 plugin path on import
from tqdm.notebook import tqdm

from .utils import time_it


def load_hdf_roi(args):
    """
    loads the data from a h5 file into a numpy array for a region of interest
    """
    data_folder, f_name, roi = args
    file_path = os.path.join(data_folder, f_name)

    with h5py.File(file_path, 'r') as f:
        data = f['/entry/data/data'][1:-1, roi[0]:roi[1], roi[2]:roi[3]]

    return data

def load_hdf(data_folder, f_name):
    """
    loads the data from a h5 file into a numpy array for a region of interest
    """
    file_path = os.path.join(data_folder, f_name)

    with h5py.File(file_path,'r') as f:
        data = f['/entry/data/data']

    return data

def list_datafiles(data_folder):
    """
    lists all h5 data files in a folder
    """
    f_names = []

    for filename in sorted(os.listdir(data_folder)):
        if filename.endswith(".h5") or filename.endswith('.nxs'):
            f_names.append(filename)

    return f_names

def print_hdf5_keys(file_path):
    """
    Prints all keys (groups and datasets) in an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
    """
    def print_keys(name, obj):
        print(name)

    with h5py.File(file_path, "r") as f:
        print(f"Keys in {file_path}:")
        f.visititems(print_keys)

@time_it
def average_data_old(file_path, roi):

    with h5py.File(file_path,'r') as f:
        data = np.mean(f['/entry/data/data'][:,roi[0]:roi[1], roi[2]:roi[3]], axis=0)

    return data

@time_it
def average_data(data_folder, names_array, roi, conc=False, num_jobs=16):

    """
    averages all the data in a NxM scan
    first averages the data vertically in a roi
    then horizontally
    """

    args_list = [(data_folder, name, roi) for name in names_array]

    if conc:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_jobs) as executor:
            all_data = list(executor.map(load_hdf_roi, args_list))
    else:
        all_data = []
        for i in tqdm(range(len(names_array))):
            args = args_list[i]
            try:
                all_data.append(load_hdf_roi(args))
            except:
                print(f"failed to load frame no. {i}")

    stacked_data = np.concatenate(all_data, axis=0)
    average_data = np.mean(stacked_data, axis=0)

    return average_data


def _load_hdf_roi_into(args):
    """Load ROI from an HDF5 file directly into a pre-allocated array slot."""
    out, idx, data_folder, f_name, roi = args
    file_path = os.path.join(data_folder, f_name)
    with h5py.File(file_path, 'r') as f:
        out[idx] = f['/entry/data/data'][1:-1, roi[0]:roi[1], roi[2]:roi[3]]


def _load_nxs_roi_into(args):
    """Load ROI from an NXS file directly into a pre-allocated array slot."""
    out, idx, data_folder, f_name, roi = args
    x0, x1, y0, y1 = roi
    file_path = os.path.join(data_folder, f_name)
    with h5py.File(file_path, 'r') as f:
        out[idx] = f['/entry/instrument/detector/data'][1:-1, x0:x1, y0:y1]


def _get_frame_shape_hdf(data_folder, f_name, roi):
    file_path = os.path.join(data_folder, f_name)
    with h5py.File(file_path, 'r') as f:
        ds = f['/entry/data/data']
        n_frames = ds.shape[0] - 2  # discard first and last
        roi_h = roi[1] - roi[0]
        roi_w = roi[3] - roi[2]
    return n_frames, roi_h, roi_w


def _get_frame_shape_nxs(data_folder, f_name, roi):
    x0, x1, y0, y1 = roi
    file_path = os.path.join(data_folder, f_name)
    with h5py.File(file_path, 'r') as f:
        ds = f['/entry/instrument/detector/data']
        n_frames = ds.shape[0] - 2
    return n_frames, x1 - x0, y1 - y0


@time_it
def stack_4d_data(data_folder, names_array, roi, slow_axis: str = 'scan_x', conc=False, num_jobs=32):
    n_files = len(names_array)

    # Probe one file to get frame shape, then pre-allocate
    n_frames, roi_h, roi_w = _get_frame_shape_hdf(data_folder, names_array[0], roi)
    out = np.empty((n_files, n_frames, roi_h, roi_w), dtype=np.float32)

    args_list = [(out, i, data_folder, name, roi) for i, name in enumerate(names_array)]

    if conc:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_jobs) as executor:
            list(executor.map(_load_hdf_roi_into, args_list))
    else:
        for args in args_list:
            _load_hdf_roi_into(args)

    if slow_axis == 'scan_x':
        out = np.transpose(out, (1, 0, 2, 3))
    elif slow_axis == 'scan_y':
        pass
    else:
        raise ValueError("Scan axis should be scan_x or scan_y")

    return out


def load_nxs_roi(args):
    data_folder, filename, roi = args
    x0, x1, y0, y1 = roi
    filepath = f"{data_folder}/{filename}"

    with h5py.File(filepath, "r") as f:
        dataset = f["/entry/instrument/detector/data"]
        frame = dataset[1:-1, x0:x1, y0:y1]

    return frame

@time_it
def stack_4d_data_lambda(data_folder, names_array, roi, slow_axis='scan_x', conc=False, num_jobs=32):
    n_files = len(names_array)

    n_frames, roi_h, roi_w = _get_frame_shape_nxs(data_folder, names_array[0], roi)
    out = np.empty((n_files, n_frames, roi_h, roi_w), dtype=np.float32)

    args_list = [(out, i, data_folder, name, roi) for i, name in enumerate(names_array)]

    if conc:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_jobs) as executor:
            list(executor.map(_load_nxs_roi_into, args_list))
    else:
        for args in args_list:
            _load_nxs_roi_into(args)

    if slow_axis == 'scan_x':
        out = np.transpose(out, (1, 0, 2, 3))
    elif slow_axis == 'scan_y':
        pass
    else:
        raise ValueError("Scan axis should be scan_x or scan_y")

    return out

@time_it
def stack_4d_data_old(file_path, roi, fast_axis_steps, slow_axis=0):

    print(file_path)

    with h5py.File(file_path, "r") as f:
        if "entry/data/data" not in f:
            raise ValueError("The input file does not contain a dataset named 'data'.")

        data = f["entry/data/data"][:,roi[0]:roi[1],roi[2]:roi[3]]
        dims = data.shape
        slow_axis_steps = dims[0] // fast_axis_steps
        data = data.reshape(slow_axis_steps, fast_axis_steps, dims[1], dims[2])

    if slow_axis == 0:
        data = np.transpose(data, (1,0,2,3))

    print(f"The shape of the data is {data.shape}")
    return data
