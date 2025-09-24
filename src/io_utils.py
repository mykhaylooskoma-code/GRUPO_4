from pathlib import Path
import numpy as np

def load_image(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path); return arr, {"shape": arr.shape}
    if path.suffix.lower() in {".tif",".tiff"}:
        try:
            import rasterio
            with rasterio.open(path) as ds:
                arr = ds.read();  # (Bands,H,W)
                return arr, {"shape": arr.shape, "crs": ds.crs}
        except Exception: pass
    import matplotlib.image as mpimg
    arr = mpimg.imread(path)  # (H,W,C)
    return arr, {"shape": arr.shape}
