from setuptools import setup, find_packages

setup(
    name="ptyxplore",
    version="0.1.0",
    author="Ahmed H. Mokhtar",
    author_email="ahmed.mokhtar@desy.de",
    description="Ptychography data exploration and processing toolkit for PETRA III experiments",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "ptyxplore=ptyxplore.cli:run",
        ],
    },
    install_requires=[
        "numpy>=1.26.4",
        "scipy>=1.15.2",
        "h5py>=3.15.1",
        "tqdm>=4.67.3",
        "torch>=2.7.1",
        "scikit-learn>=1.6.1",
        "scikit-image>=0.25.2",
        "joblib>=1.4.2",
        "matplotlib>=3.8.2",
        "Pillow>=11.1.0",
        "ipywidgets>=8.1.5",
        "IPython>=8.33.0",
        "panel>=1.8.8",
        "param>=2.3.2",
        "pandas>=2.2.3",
    ],
)
