[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "lunarice-pds4"
version = "0.1.0"
description = "PDS4-native Chandrayaan-2 DFSAR and OHRC processing for ISRO PS-8"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "scipy>=1.10",
  "pandas>=2.0",
  "lxml>=4.9",
  "rasterio>=1.3",
  "scikit-image>=0.21",
  "shapely>=2.0",
  "pyproj>=3.5",
  "matplotlib>=3.7",
  "opencv-python>=4.8",
  "pyyaml>=6.0",
  "tqdm>=4.66",
]

[tool.setuptools.packages.find]
where = ["src"]

