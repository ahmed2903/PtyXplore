# HDF5 / NXS data loading and stacking for ptychography scans at PETRA 3
#
# Author: Ahmed H. Mokhtar
# Email: ahmed.mokhtar@desy.de

import os
import concurrent.futures
from multiprocessing import Pool

import numpy as np
import h5py
from tqdm.notebook import tqdm

from .utils import time_it


def load_hdf_roi(args):
    """
    loads the data from a h5 file into a numpy array for a region of interest
    """
    data_folder, f_name, roi = args
    file_path = os.path.join(data_folder, f_name)

    with h5py.File(file_path,'r') as f:
        data = f['/entry/data/data'][1:-1,roi[0]:roi[1], roi[2]:roi[3]] # Loading ROI only, and discarding first and last frames

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
        with Pool(processes=num_jobs) as pool:
            all_data = list(pool.imap(load_hdf_roi, args_list))

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

@time_it
def stack_data(data_folder, names_array, roi, conc=True):
    """
    Stacks all the data along the first dimension
    such that output has shape (NxM, x, y)
    where N is the number of files, and M is the number of frames per file
    """
    if conc:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            all_data = list(executor.map(load_hdf_roi, [data_folder]*len(names_array), names_array, [roi]*len(names_array)))
    else:
        all_data = []
        for i in range(len(names_array)):
            all_data.append(load_hdf_roi(data_folder, names_array[i], roi))

    return np.concatenate(all_data, axis=0)

@time_it
def stack_4d_data(data_folder, names_array, roi, slow_axis: str = 'scan_x', conc=False, num_jobs=4):

    args_list = [(data_folder, name, roi) for name in names_array]

    if conc:
        with Pool(processes=num_jobs) as pool:
            all_data = list(pool.imap(load_hdf_roi, args_list))
    else:
        all_data = [load_hdf_roi(args) for args in args_list]
        all_data = np.array(all_data)

    stacked_data = np.stack(all_data, axis=0)

    if slow_axis == 'scan_x':
        stacked_data = np.transpose(stacked_data, (1,0,2,3))
    elif slow_axis == 'scan_y':
        pass
    else:
        raise ValueError("Scan axis should be scan_x or scan_y")

    return stacked_data


def load_nxs_roi(args):
    data_folder, filename, roi = args
    x0, x1, y0, y1 = roi

    filepath = f"{data_folder}/{filename}"

    with h5py.File(filepath, "r") as f:
        dataset = f["/entry/instrument/detector/data"]
        frame = dataset[1:-1, x0:x1, y0:y1]  # Discard rubbish frames

    return frame

@time_it
def stack_4d_data_lambda(data_folder, names_array, roi, slow_axis='scan_x', conc=False, num_jobs=4):

    args_list = [(data_folder, name, roi) for name in names_array]

    if conc:
        with Pool(processes=num_jobs) as pool:
            all_data = list(pool.imap(load_nxs_roi, args_list))
    else:
        all_data = [load_nxs_roi(args) for args in args_list]

    stacked_data = np.stack(np.array(all_data), axis=0)

    if slow_axis == 'scan_x':
        stacked_data = np.transpose(stacked_data, (1,0,2,3))
    elif slow_axis == 'scan_y':
        pass
    else:
        raise ValueError("Scan axis should be scan_x or scan_y")

    return stacked_data

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
