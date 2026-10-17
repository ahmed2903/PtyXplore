# Imports data from Ptychography experiments at CID, petra 3
# 
# Author: Ahmed H. Mokhtar 
# Email: ahmed.mokhtar@desy.de
# Date : Nov 2025

import numpy as np
import h5py
import copy
import json
from dataclasses import dataclass, field, fields
from functools import wraps
import re

import glob
import os
from .utils import time_it, parse_log, log_roi_params, inject_attrs
from . import utils as ut


def _save_streakable(group, name, data):
    """Save data that may be a list (multi-streak) or a single ndarray.

    Multi-streak: creates a sub-group with datasets streak_0, streak_1, …
    Single array: creates a dataset directly.
    Chunking is always enabled because gzip compression requires it.
    """
    if isinstance(data, list):
        sub = group.create_group(name)
        sub.attrs['n_streaks'] = len(data)
        for i, arr in enumerate(data):
            sub.create_dataset(f'streak_{i}', data=np.array(arr),
                               compression='gzip', chunks=True)
    elif data is not None:
        group.create_dataset(name, data=data, compression='gzip', chunks=True)


def _load_streakable(group, name):
    """Load a dataset that was saved by _save_streakable.

    Returns a list of arrays for multi-streak data, or a single array otherwise.
    """
    item = group[name]
    if isinstance(item, h5py.Group):
        n = int(item.attrs['n_streaks'])
        return [item[f'streak_{i}'][...] for i in range(n)]
    return item[...]

@dataclass
class Exp:
    year: int
    beamtime_id: str
    beamline: str = "p11"

    def __post_init__(self):
        base = f"/asap3/petra3/gpfs/{self.beamline}/{self.year}/data/{self.beamtime_id}/raw"
        self.scan_frames_dir = os.path.join(base, "scan_frames")
        self.logs_dir = os.path.join(base, "server_log", "Scan_logs")
            

@dataclass
class Scan:
    """
    A scan consists of:
      - metadata      (general info from the log header)
      - scan_defs     (per-scan info: points, step counts, start/end)
      - attributes    (per-point arrays: motor positions, exposure time, etc.)
      - file_list     (list of h5/nxs detector files)
    """

    exp: Exp
    scan_num: int
    file_list: list = None
    
    detector: str = field(default = 'Eiger', repr = True)
    sample_name: str = field(default = None, repr = True)
    
    det_distance: float = field(default = None, repr = True)

    def __post_init__(self):

        # ----------------------------
        self.detector = self.detector.title()
        # detector pixel size 
        if self.detector == 'Eiger':
            self.det_psize = 75 * ut.UNIT_MAP['um']
        elif self.detector == "Lambda":
            self.det_psize = 55 * ut.UNIT_MAP['um']
        
        
        
        # ----------------------------
        # Parse Log File
        self.log_path = self.log_file(self.scan_num) 
        with open(self.log_path, 'r') as log_file:
            log_str = ''
            for line in log_file:
                if line.startswith('# '):
                    log_str += line.strip('# ')
                    
        _, scan_defs, _ = parse_log(log_str)

        self.slow_axis = ut.clean_key(list(scan_defs.keys())[0])
                
        for scan_name, scan_dict in scan_defs.items():
            inject_attrs(self, scan_dict, scan_name)
            
        # ______________ Data Dir ______________
        self.data_dir = self.scan_dir(self.scan_num)
        
        print(f"Data Directory is : {self.data_dir}")
        
        files_h5  = sorted(glob.glob(os.path.join(self.data_dir, "*.h5")))
        files_nxs = sorted(glob.glob(os.path.join(self.data_dir, "*.nxs")))

        self.file_list = files_h5 if len(files_h5) else files_nxs

        if len(self.file_list) == 0:
            raise FileNotFoundError(f"No .h5 or .nxs files found in {self.data_dir}")

        if all(hasattr(self, x) for x in ["scan_x_start_point", "scan_x_end_point", "scan_x_steps_count"]):
            self.step_size_x = (self.scan_x_end_point - self.scan_x_start_point) / self.scan_x_steps_count

        if all(hasattr(self, x) for x in ["scan_y_start_point", "scan_y_end_point", "scan_y_steps_count"]):
            self.step_size_y = (self.scan_y_end_point - self.scan_y_start_point) / self.scan_y_steps_count
        
        # _________  Master File ________
        master_file = self.master_file(self.scan_num)

        with h5py.File(master_file, "r") as f:
            self.wavelength = f["/entry/instrument/beam/incident_wavelength"][...] * ut.UNIT_MAP['Angs']
            self.energy = ut.wavelength2energy(self.wavelength)

        
    
    def __repr__(self):
        class_name = type(self).__name__
        sample_name = self.sample_name if self.sample_name is not None else ''
        
        return f"scan {self.scan_num} | sample: {sample_name} | slow axis: {self.slow_axis} | step_size={self.step_size_x, self.step_size_y} nm, step_points = {self.scan_x_steps_count, self.scan_y_steps_count} | exposure time: {self.exposure_time}"
        
    def scan_dir(self, scan_num):
        """Returns full directory for a particular scan number."""
        if self.detector == "Eiger":
            return os.path.join(self.exp.scan_frames_dir, f"Scan_{scan_num}")
        else:
            return os.path.join(self.exp.scan_frames_dir, f"Scan_{scan_num}_{self.detector}")

    def log_file(self, scan_num):
        return os.path.join(self.exp.logs_dir, f"Scan_{scan_num}.log")

    def master_file(self, scan_num):
        return os.path.join(self.exp.scan_frames_dir, f"Scan_{scan_num}/Scan_{scan_num}_master.h5")
        
        
@dataclass
class ROI:

    scan: Scan 
    kind: str # Bragg or Pupil
    
    coords: np.ndarray = field(default=None, repr=False)

    # Optional pixel mask, shape == (re-rs, ce-cs) matching coords bounding box.
    # Non-zero pixels are kept; zero pixels are zeroed out in the 4D dataset.
    detector_mask: np.ndarray = field(default=None, repr=False)

    # acquired later
    data_4d: np.ndarray = field(default=None, repr=False) # 4D Data set (scan_y, scan_x, det_row, det_col)
    centre_pixel: np.ndarray = field(default=(0,0), repr=False)
    
    # to be set automatically
    _averaged_det_images: np.ndarray = field(default=None, repr=False)  # Average of all detector frames 
    _averaged_coherent_images: np.ndarray = field(default=None, repr=False)
    _summed_det_images: np.ndarray = field(default=None, repr=False)
    _summed_coherent_images: np.ndarray = field(default=None, repr=False)
    _delta_qz : float = field(default = 0.0, repr = True)
    _delta_q : float = field(default = 0.0, repr = True)
    _ttheta : float = field(default = 0.0, repr = True)
    _det_psize_eff : float = field(default = None, repr = True)
    _dz  : float = field(default = 0.0, repr = True)
    
    tilt_corrected_image : np.ndarray = field(default=None, repr = False)
    
    # Data
    # needs to be set manually
    detected_objects: np.ndarray = field(default=None, repr=False) # Detected Objects
    coherent_imgs: np.ndarray = field(default=None, repr=False)
        
    # k-space params
    kout_coords: np.ndarray = field(default=None, repr=False)  # Coords
    kouts: np.ndarray = field(default=None, repr=False)  # Kout vectors
    kin_coords: np.ndarray = field(default=None, repr=False)  # Coords
    kins: np.ndarray = field(default=None, repr=False)  # Kin vectors
    
    kins_avg: np.ndarray = field(default=None, repr=False)  # average Kins 
    # For computing kins 
    optimal_angles: list = field(default=None, repr=False)  # Euler angles that rotate Kout to Kin
    est_ttheta: float = field(default=None, repr=False)
    g_init: np.ndarray = field(default=None, repr=False)  # Initial G vector for a Signal
    
    log_params: list = field(default_factory=list, repr = False)
    
    checkpoint_stack:list = field(default_factory=list, repr = False)


    # _______________ functionality with Exp ___________
    def __repr__(self):
        class_name = type(self).__name__
        sample_name = self.sample_name if self.sample_name is not None else ''
        
        return f"scan {self.scan_num} | sample: {sample_name} | slow axis: {self.slow_axis} | step_size={self.step_size_x, self.step_size_y} nm, step_points = {self.scan_x_steps_count, self.scan_y_steps_count} | exposure time: {self.exposure_time}"
        
    def __post_init__(self):
        # Copy attributes from Scan into ROI
        for f in vars(self.scan):
            setattr(self, f, getattr(self.scan, f))

    _AGGREGATE_CACHES = ("_averaged_det_images", "_summed_det_images",
                         "_averaged_coherent_images", "_summed_coherent_images")

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        # the averaged/summed images are cached from data_4d, so drop them when it changes
        if name == "data_4d":
            self.invalidate_aggregates()

    def invalidate_aggregates(self):
        """Clear the cached averaged/summed images so they are recomputed from data_4d.

        Reassigning ``data_4d`` does this automatically; call it after editing
        ``data_4d`` in place.
        """
        for attr in self._AGGREGATE_CACHES:
            object.__setattr__(self, attr, None)

    # _______________ Properties ________________
    @property
    def averaged_det_images(self):
        
        if self._averaged_det_images is None:
            self._averaged_det_images = np.mean(self.data_4d, axis=(0,1))
            
        return self._averaged_det_images

    def update_averaged_det_images(self):
        self._averaged_det_images = np.mean(self.data_4d, axis=(0,1))

    @averaged_det_images.setter
    def averaged_det_images(self, value):
        self._averaged_det_images = value

    @property
    def summed_det_images(self):
        
        if self._summed_det_images is None:
            self._summed_det_images = np.sum(self.data_4d, axis=(0,1))
            
        return self._summed_det_images

    def update_summed_det_images(self):
        self._summed_det_images = np.sum(self.data_4d, axis=(0,1))

    @summed_det_images.setter
    def summed_det_images(self, value):
        self._summed_det_images = value
    
    @property
    def averaged_coherent_images(self):
        
        if self._averaged_coherent_images is None:
            self._averaged_coherent_images = np.mean(self.data_4d, axis=(2,3))
        
        return self._averaged_coherent_images
    
    @averaged_coherent_images.setter
    def averaged_coherent_images(self, value):
        self._averaged_coherent_images = value
        
    def update_averaged_coherent_images(self):
        self._averaged_coherent_images = np.mean(self.data_4d, axis=(2,3))
    
    @property
    def summed_coherent_images(self):
        
        if self._summed_coherent_images is None:
            self._summed_coherent_images = np.sum(self.data_4d, axis=(2,3))
        
        return self._summed_coherent_images
    
    @summed_coherent_images.setter
    def summed_coherent_images(self, value):
        self._summed_coherent_images = value
        
    def update_summed_coherent_images(self):
        self._summed_coherent_images = np.sum(self.data_4d, axis=(2,3))

    @property
    def ttheta(self):
        return self._ttheta
    @ttheta.setter
    def ttheta(self, value):
        self._ttheta = value
        
    @property
    def delta_qz(self):
        return self._delta_qz
        
    @delta_qz.setter
    def delta_qz(self, value):
        self._delta_qz = value

    @property
    def delta_q(self):
        return self._delta_q
        
    @delta_q.setter
    def delta_q(self, value):
        self._delta_q = value

    @property
    def det_psize_eff(self):
        return self._det_psize_eff
            
    @det_psize_eff.setter
    def det_psize_eff(self, value):
        self._det_psize_eff = value

    @property
    def dz(self):
        return self._dz
            
    @dz.setter
    def dz(self, value):
        self._dz = value
    
    # _______________ Arithmetic _______________

    def __add__(self, other):
        if not isinstance(other, ROI):
            print('here')
            return NotImplemented

        if self.data_4d is None or other.data_4d is None:
            raise ValueError("Both ROIs must have data_4d loaded before adding.")

        if self.data_4d.shape != other.data_4d.shape:
            raise ValueError(
                f"data_4d shape mismatch: {self.data_4d.shape} vs {other.data_4d.shape}"
            )

        if not np.isclose(self.step_size_x, other.step_size_x):
            raise ValueError(
                f"step_size_x mismatch: {self.step_size_x} vs {other.step_size_x}"
            )

        if not np.isclose(self.step_size_y, other.step_size_y):
            raise ValueError(
                f"step_size_y mismatch: {self.step_size_y} vs {other.step_size_y}"
            )

        new = copy.deepcopy(self)
        new.data_4d = self.data_4d + other.data_4d
        new.exposure_time = self.exposure_time + other.exposure_time

        new.checkpoint_stack = []
        new.log_params = []

        return new

    def __itruediv__(self, other):
        """Normalise self in-place by other's per-position detector intensity.

        Equivalent to normalise_detector(self, other): divides each scan
        position's detector frame by the total counts in the corresponding
        frame of *other*, then rescales by the mean so the overall intensity
        level is preserved.
        """
        if not isinstance(other, ROI):
            return NotImplemented

        if self.data_4d is None or other.data_4d is None:
            raise ValueError("Both ROIs must have data_4d loaded before dividing.")

        if self.data_4d.shape[:2] != other.data_4d.shape[:2]:
            raise ValueError(
                f"data_4d shape mismatch: {self.data_4d.shape[:2]} vs {other.data_4d.shape[:2]}"
            )

        if not np.isclose(self.step_size_x, other.step_size_x):
            raise ValueError(
                f"step_size_x mismatch: {self.step_size_x} vs {other.step_size_x}"
            )

        if not np.isclose(self.step_size_y, other.step_size_y):
            raise ValueError(
                f"step_size_y mismatch: {self.step_size_y} vs {other.step_size_y}"
            )

        peak_intensity = np.sum(other.data_4d, axis=(-2, -1))
        avg_intensity = np.mean(peak_intensity)
        self.data_4d = self.data_4d / peak_intensity[..., np.newaxis, np.newaxis] * avg_intensity

        return self

    # _______________ Saving ___________________
    
            
    @time_it
    def save(self, file_path):
    
        """
        Save the processed data and metadata to an HDF5 file.
    
        Args:

            file_path (str): The path where the HDF5 file will be saved.
        """
        
        with h5py.File(file_path, "w") as h5f:

            exp_params = h5f.create_group("experimental_params")

            # exp_params.attrs['year'] = self.year
            # exp_params.attrs['id'] = self.beamtime_id
            exp_params.attrs['scan_no'] = self.scan_num
            
            exp_params.attrs['step_size_x'] = self.step_size_x
            exp_params.attrs['step_size_y'] = self.step_size_y
            exp_params.attrs['slow_axis'] = self.slow_axis
            exp_params.attrs['detector'] = self.detector
            exp_params.attrs['delta_q'] = self.delta_q
            exp_params.attrs['delta_qz'] = self.delta_qz
            exp_params.attrs['ttheta'] = self.ttheta
            
            
            process_params = h5f.create_group("processing_params")
            
            process_params.attrs['roi'] = str(self.coords)
            process_params.attrs['det_distance'] = self.det_distance
            process_params.attrs['delta_q'] = self.delta_q
            process_params.attrs['delta_qz'] = self.delta_qz
            process_params.attrs['dz'] = self.dz
            
            images = h5f.create_group("processed_images")
            _save_streakable(images, "coherent_images", self.coherent_imgs)

            kvectors = h5f.create_group("kvectors")
            _save_streakable(kvectors, "kins",        self.kins)
            _save_streakable(kvectors, "kouts",       self.kouts)
            _save_streakable(kvectors, "kin_coords",  self.kin_coords)
            _save_streakable(kvectors, "kout_coords", self.kout_coords)
                
            print(f"Data saved at {file_path}")

        log_path = file_path.rstrip('.h5')+'.log'
        
        with open(log_path, "w") as f:
            
            f.write(f"# ROI processing log\n")
            f.write(f"# ROI name: {getattr(self, 'sample_name', None)}\n")
            f.write(f"# Centre pixel: {getattr(self, 'centre_pixel', None)}\n")
            f.write(f"# Total operations: {len(self.log_params)}\n\n")
    
            for i, entry in enumerate(self.log_params):
                func = entry["function"]
                args = entry["args"]
                kwargs = entry["kwargs"]
    
                f.write(f"--- Operation {i+1} ---\n")
                f.write(f"Function: {func}\n")
    
                if args:
                    f.write(f"Args:\n")
                    for a_i, a in enumerate(args):
                        f.write(f"  {a_i}: {a}\n")
    
                if kwargs:
                    f.write(f"Kwargs:\n")
                    for k, v in kwargs.items():
                        f.write(f"  {k}: {v}\n")
    
                f.write("\n")
                
    @classmethod
    def load_data(cls, file_path):
        """
        Load processed data and metadata for a specific ROI from an HDF5 file.
    
        Args:
            roi_name (str): The ROI name for which the data should be loaded.
            file_path (str): Path to the HDF5 file containing saved data.
        """
        
        self = cls.__new__(cls)

        with h5py.File(file_path, "r") as h5f:
                                        
            # Load Processed Images 
            params = h5f["experimental_params"]
            self.scan_num = params.attrs['scan_no']
            self.step_size_x = params.attrs['step_size_x']
            self.step_size_y = params.attrs['step_size_y']
            self.slow_axis = params.attrs['slow_axis']

            
            params = h5f["processing_params"]
            self.det_distance = params.attrs.get('det_distance')
            
            self.dz = params.attrs.get("dz")
            self.delta_qz = params.attrs.get("delta_qz")
            self.delta_q = params.attrs.get("delta_q")
            
            images = h5f["processed_images"]
            self.coherent_imgs = _load_streakable(images, "coherent_images")

            kvectors = h5f["kvectors"]
            self.kins       = _load_streakable(kvectors, "kins")
            self.kouts      = _load_streakable(kvectors, "kouts")
            self.kin_coords = _load_streakable(kvectors, "kin_coords")
            self.kout_coords = _load_streakable(kvectors, "kout_coords")
                
        return self
    
    
    @time_it
    def save_4d(self, file_path):
    
        """
        Save the 4D data set for an roi to an HDF5 file.
    
        Args:
            roi_name (str): The ROI name for which the data should be saved.

            file_path (str): The path where the HDF5 file will be saved.
        """

        with h5py.File(file_path, "w") as h5f:
            data = h5f.create_group("data")
            # Save metadata as attributes in the root group
            data.create_dataset("data", data = self.data_4d, compression="gzip",chunks=True)

        print(f"Data saved at {file_path}")
        
    @time_it
    def load_4d(self, file_path):
    
        """
        Load the 4D data set for an roi from an HDF5 file.
    
        Args:
            roi_name (str): The ROI name for which the data should be loaded.

            file_path (str): The path where the HDF5 is stored.
        """
        print("Loading data ...")
        with h5py.File(file_path, "r") as h5f:
            data = h5f["data"]
            # Save metadata as attributes in the root group
            self.data_4d = np.array(data["data"])

        print("Data loaded")

    # _____________ Checkpointing _____________
    
    def checkpoint_state(self):
        """Save a deep copy of the current state for potential rollback."""
        state = copy.deepcopy({
            key: getattr(self, key)
            for key in self.__dataclass_fields__.keys()
            if key not in ("checkpoint_stack", "log_params")  # don't save logs or stack itself
        })
        self.checkpoint_stack.append(state)
        print(f"[ROI] Checkpoint #{len(self.checkpoint_stack)} created.")

    def restore_checkpoint(self):
        """Restore the most recent checkpoint."""
        if not self.checkpoint_stack:
            print("[ROI] No checkpoints to restore.")
            return

        last_state = self.checkpoint_stack.pop()
        for key, value in last_state.items():
            setattr(self, key, value)
    
    def auto_checkpoint(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.checkpoint_state()
            return func(self, *args, **kwargs)
        return wrapper
    



        