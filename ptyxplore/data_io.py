# HDF5 / NXS data loading and stacking for ptychography scans at PETRA 3
#
# Author: Ahmed H. Mokhtar
# Email: ahmed.mokhtar@desy.de

import os
import concurrent.futures
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

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
def average_data(data_folder, names_array, roi, conc=True, num_jobs=64):

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


# ---------------------------------------------------------------------------
# HPC-optimised loaders: workers write directly into shared memory (zero-copy)
# and each worker processes a batch of files to amortise IPC overhead.
# On Linux the 'fork' start method is used so workers inherit the parent's
# address space (hdf5plugin already registered, no re-import needed).
# ---------------------------------------------------------------------------

def _hdf_batch_shm(args):
    """
    Load a batch of HDF files directly into a shared memory buffer.
    Each worker owns a slice of the pre-allocated output array.
    """
    shm_name, out_shape, out_dtype_str, indices, data_folder, f_names, roi = args
    shm = SharedMemory(name=shm_name)
    out = np.ndarray(out_shape, dtype=np.dtype(out_dtype_str), buffer=shm.buf)
    try:
        for idx, f_name in zip(indices, f_names):
            file_path = os.path.join(data_folder, f_name)
            with h5py.File(file_path, 'r', rdcc_nbytes=256 * 1024**2, rdcc_nslots=1021) as f:
                out[idx] = f['/entry/data/data'][1:-1, roi[0]:roi[1], roi[2]:roi[3]]
    finally:
        shm.close()


def _nxs_batch_shm(args):
    """
    Load a batch of NXS files directly into a shared memory buffer.
    Each worker owns a slice of the pre-allocated output array.
    """
    shm_name, out_shape, out_dtype_str, indices, data_folder, f_names, roi = args
    x0, x1, y0, y1 = roi
    shm = SharedMemory(name=shm_name)
    out = np.ndarray(out_shape, dtype=np.dtype(out_dtype_str), buffer=shm.buf)
    try:
        for idx, f_name in zip(indices, f_names):
            file_path = os.path.join(data_folder, f_name)
            with h5py.File(file_path, 'r', rdcc_nbytes=256 * 1024**2, rdcc_nslots=1021) as f:
                out[idx] = f['/entry/instrument/detector/data'][1:-1, x0:x1, y0:y1]
    finally:
        shm.close()


def _load_hdf_roi_into(args):
    """Load ROI from an HDF5 file directly into a pre-allocated array slot."""
    out, idx, data_folder, f_name, roi = args
    file_path = os.path.join(data_folder, f_name)
    with h5py.File(file_path, 'r', rdcc_nbytes=256 * 1024**2, rdcc_nslots=1021) as f:
        out[idx] = f['/entry/data/data'][1:-1, roi[0]:roi[1], roi[2]:roi[3]]


def _load_nxs_roi_into(args):
    """Load ROI from an NXS file directly into a pre-allocated array slot."""
    out, idx, data_folder, f_name, roi = args
    x0, x1, y0, y1 = roi
    file_path = os.path.join(data_folder, f_name)
    with h5py.File(file_path, 'r', rdcc_nbytes=256 * 1024**2, rdcc_nslots=1021) as f:
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


def _run_parallel(worker_fn, args_list, n_workers, use_processes):
    """Dispatch work to either a process pool or thread pool."""
    if use_processes:
        ctx = mp.get_context('fork')
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            list(ex.map(worker_fn, args_list))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(worker_fn, args_list))


@time_it
def stack_4d_data(data_folder, names_array, roi, slow_axis: str = 'scan_x',
                  conc=True, num_jobs=None, use_processes=True):
    """
    Stack HDF5 scan files into a 4-D array (scan_y, scan_x, det_y, det_x).

    Parameters
    ----------
    num_jobs : int or None
        Number of parallel workers. Defaults to os.cpu_count() (all cores on
        the dedicated HPC node).
    use_processes : bool
        True  → ProcessPoolExecutor with shared memory (HPC default, bypasses
                 GIL for CPU-bound HDF5 decompression, zero data copy).
        False → ThreadPoolExecutor (useful for uncompressed files or debugging).
    """
    n_files = len(names_array)
    n_workers = num_jobs or os.cpu_count()

    n_frames, roi_h, roi_w = _get_frame_shape_hdf(data_folder, names_array[0], roi)
    out = np.empty((n_files, n_frames, roi_h, roi_w), dtype=np.float32)

    if conc:
        if use_processes:
            # Shared memory: workers write directly into 'out', no pickle/IPC copy
            shm = SharedMemory(create=True, size=out.nbytes)
            shm_arr = np.ndarray(out.shape, dtype=out.dtype, buffer=shm.buf)
            try:
                # Batch files across workers so each process handles multiple files
                batches = [list(range(n_files))[i::n_workers] for i in range(n_workers)]
                args_list = [
                    (shm.name, out.shape, str(out.dtype),
                     batch, data_folder, [names_array[i] for i in batch], roi)
                    for batch in batches if batch
                ]
                _run_parallel(_hdf_batch_shm, args_list, n_workers, use_processes=True)
                out[:] = shm_arr
            finally:
                shm.close()
                shm.unlink()
        else:
            args_list = [(out, i, data_folder, name, roi) for i, name in enumerate(names_array)]
            _run_parallel(_load_hdf_roi_into, args_list, n_workers, use_processes=False)
    else:
        for i, name in enumerate(names_array):
            _load_hdf_roi_into((out, i, data_folder, name, roi))

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
def stack_4d_data_lambda(data_folder, names_array, roi, slow_axis='scan_x',
                         conc=True, num_jobs=None, use_processes=True):
    """
    Stack NXS (Lambda detector) scan files into a 4-D array.

    Parameters
    ----------
    num_jobs : int or None
        Number of parallel workers. Defaults to os.cpu_count().
    use_processes : bool
        True  → ProcessPoolExecutor with shared memory (HPC default).
        False → ThreadPoolExecutor.
    """
    n_files = len(names_array)
    n_workers = num_jobs or os.cpu_count()

    n_frames, roi_h, roi_w = _get_frame_shape_nxs(data_folder, names_array[0], roi)
    out = np.empty((n_files, n_frames, roi_h, roi_w), dtype=np.float32)

    if conc:
        if use_processes:
            shm = SharedMemory(create=True, size=out.nbytes)
            shm_arr = np.ndarray(out.shape, dtype=out.dtype, buffer=shm.buf)
            try:
                batches = [list(range(n_files))[i::n_workers] for i in range(n_workers)]
                args_list = [
                    (shm.name, out.shape, str(out.dtype),
                     batch, data_folder, [names_array[i] for i in batch], roi)
                    for batch in batches if batch
                ]
                _run_parallel(_nxs_batch_shm, args_list, n_workers, use_processes=True)
                out[:] = shm_arr
            finally:
                shm.close()
                shm.unlink()
        else:
            args_list = [(out, i, data_folder, name, roi) for i, name in enumerate(names_array)]
            _run_parallel(_load_nxs_roi_into, args_list, n_workers, use_processes=False)
    else:
        for i, name in enumerate(names_array):
            _load_nxs_roi_into((out, i, data_folder, name, roi))

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
