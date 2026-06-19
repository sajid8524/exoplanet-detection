# Data I Need From You

To train the real hackathon model, please collect these items.

## Required

1. The official hackathon dataset or download instructions.
2. A folder of light curves, preferably CSV.
3. A metadata CSV with at least:

```csv
target_id,path,label
```

Allowed labels:

- `planet`
- `eclipsing_binary`
- `background_blend`
- `noise`

If the official labels use different names, send the label dictionary and I
will map them.

## Strongly Recommended

Add these columns to metadata if available:

```csv
period,epoch,duration,stellar_radius,stellar_mass,teff,logg,tess_mag,sector,camera,ccd
```

Why they matter:

- `period`, `epoch`, `duration`: improves supervised training on known TCEs.
- `stellar_radius`: converts transit depth into planet radius.
- `teff`, `stellar_radius`: helps estimate habitable-zone relevance.
- `sector`, `camera`, `ccd`: helps track instrument/systematic patterns.

## Optional But High-Impact

If you can provide Target Pixel File or centroid columns:

```csv
centroid_col,centroid_row
```

This enables a background-blend check, which is a strong judging differentiator.

## Questions For You

Please tell me:

1. Do you already have the official curated training data, or should the pipeline
   download public TESS data from MAST?
2. What file format did they provide: CSV, FITS, HDF5, or something else?
3. What are the exact classes/dispositions in the labels?
4. Is there an official test set and scoring metric?
5. Are we allowed to use external confirmed planet/false-positive catalogs for
   extra training data?

