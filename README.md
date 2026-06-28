# LunarIce-Net

VS Code-ready starter code for ISRO Hackathon PS 8: lunar subsurface ice detection using Chandrayaan-2 DFSAR and supporting lunar datasets.

## Folder Structure

```text
lunarice_net_code/
  config.yaml
  requirements.txt
  README.md
  src/
    lunarice_net/
      __init__.py
      config.py
      preprocess.py
      dataset.py
      model.py
      losses.py
      train.py
      infer.py
      utils.py
```

## Where To Add Your PRADAN/ISRO Data

Edit `config.yaml`. Replace every path under `data.raw` and `data.labels` with your downloaded files.

Expected raster inputs:

- DFSAR S-band HH, HV, VH, VV SLC GeoTIFFs
- DFSAR L-band HH, HV, VH, VV SLC GeoTIFFs
- Diviner temperature raster, already coregistered or coregisterable
- LOLA slope raster
- Optional PSR mask raster
- Training label mask raster, where ice pixels are `1` and background is `0`

## Run

```bash
pip install -r requirements.txt
pip install -e .
python -m lunarice_net.train --config config.yaml
python -m lunarice_net.infer --config config.yaml --checkpoint outputs/checkpoints/best_model.pt
```

If your rasters are not already aligned to the same grid, coregister them first in QGIS/SNAP/GDAL, or extend `preprocess.py` using the `coregister_to_reference` helper.
