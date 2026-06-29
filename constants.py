# Paste PRADAN Folders Here

Put your downloaded PRADAN folders in this directory.

Example:

```text
data/pradan/
  CH2_DFSAR_2026_ORBIT_XXXXX/
    data/
      *.xml
      *.dat
    geometry/
      *.csv
    browse/
      *.png
    miscellaneous/
      *CALIBRATION*.txt

  CH2_OHRC_2026_ORBIT_XXXXX/
    data/
      *.xml
      *.img
    geometry/
      *.csv
    browse/
      *.png
    miscellaneous/
      *CALIBRATION*.txt
```

After pasting, edit `config.yaml`:

```yaml
data:
  pradan:
    dfsar_root: "data/pradan/CH2_DFSAR_2026_ORBIT_XXXXX"
    ohrc_root: "data/pradan/CH2_OHRC_2026_ORBIT_XXXXX"
```

