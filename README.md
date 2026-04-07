# ptyxplore

Ptychography data exploration and processing toolkit for PETRA III experiments at DESY (Beamline P11 · CID).

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ahmed2903/PtyXplore.git
cd PtyXplore
```

### Option 1 — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate ptyxplore
pip install -e .
```

### Option 2 — pip

```bash
pip install -r requirements.txt
pip install -e .
```

> The `-e` flag installs the package in editable mode so local changes take effect immediately without reinstalling.

---

## Running the Dashboard

After installation, start the web app from any terminal with:

```bash
ptyxplore
```

Then open **http://localhost:5006/ptycho_dashboard** in your browser.

> On Maxwell, open an SSH tunnel first:
> ```bash
> ssh -L 5006:localhost:5006 <user>@max-display.desy.de
> ```
> then run `ptyxplore` in the remote terminal and open the URL locally.

---

## Package Structure

```
ptyxplore/
├── ptyxplore/
│   ├── __init__.py
│   ├── cbd_loader.py     # Exp, Scan, ROI dataclasses
│   ├── cli.py            # Command-line entry point
│   ├── data_io.py        # HDF5 / NXS file loading and stacking
│   ├── data_proc.py      # Image processing (filtering, pooling, alignment)
│   ├── roi_ops.py        # ROI-level operations and k-vector pipeline
│   ├── roi_plotting.py   # ROI-based plotting utilities
│   ├── kvectors.py       # k-space vector computation and optimisation
│   ├── geometry.py       # Geometric optics utilities (NA, detector distance)
│   ├── plotting.py       # Matplotlib / ipywidgets visualisation
│   └── utils.py          # Unit conversions, decorators, log parsing
├── ptycho_dashboard.py   # Panel-based interactive dashboard
├── environment.yml
└── setup.py
```

---

## Author

Ahmed H. Mokhtar — ahmed.mokhtar@desy.de
