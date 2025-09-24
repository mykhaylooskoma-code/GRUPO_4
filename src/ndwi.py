import numpy as np

def compute_ndwi(green, nir, mask=None, eps=1e-12, clip=True):
    green = green.astype("float64")
    nir   = nir.astype("float64")
    denom = green + nir
    valid = np.isfinite(denom) & (np.abs(denom) > eps)
    ndwi = np.full_like(green, np.nan, dtype=np.float64)
    ndwi[valid] = (green[valid] - nir[valid]) / denom[valid]
    if mask is not None:
        ndwi = np.where(mask, ndwi, np.nan)
    if clip:
        ndwi = np.clip(ndwi, -1.0, 1.0)
    return ndwi.astype("float32")
