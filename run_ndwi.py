from pathlib import Path
import numpy as np
from src.io_utils import load_image
from src.ndwi import compute_ndwi
from src.plot_utils import save_gray, save_heat

IMG_PATH = "data/imagen_profesor.tif"  # cambia si tu archivo es otro
GREEN_IDX = 1   # <- AJUSTA según tus bandas
NIR_IDX   = 3   # <- AJUSTA según tus bandas

def pick_band(arr, idx):
    if arr.ndim == 3 and arr.shape[0] in (3,4,5,6,7,8) and arr.shape[1] > 32:
        return arr[idx]              # (Bands,H,W)
    elif arr.ndim == 3:
        return arr[..., idx]         # (H,W,C)
    raise ValueError(f"Forma no soportada: {arr.shape}")

def main():
    arr, meta = load_image(IMG_PATH)
    print("shape:", meta.get("shape"))
    green = pick_band(arr, GREEN_IDX)
    nir   = pick_band(arr, NIR_IDX)
    mask = np.isfinite(green) & np.isfinite(nir)
    ndwi = compute_ndwi(green, nir, mask=mask)
    Path("outputs").mkdir(exist_ok=True)
    save_gray(green, "outputs/green.png", "Green")
    save_gray(nir,   "outputs/nir.png",   "NIR")
    save_heat(ndwi,  "outputs/ndwi.png",  "NDWI")
    print("Listo. Mira outputs/: green.png, nir.png, ndwi.png")
if __name__ == "__main__": main()
