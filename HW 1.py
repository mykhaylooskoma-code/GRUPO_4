import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Archivos en la misma carpeta que tu script
nir_path = "S2C_31TDF_20250630_0_L2A_nir.tif"
green_path = "S2C_31TDF_20250630_0_L2A_green.tif"

# Leer NIR
with rasterio.open(nir_path) as src:
    nir = src.read(1).astype(float)

# Leer Green
with rasterio.open(green_path) as src:
    green = src.read(1).astype(float)

# --- Cálculo NDWI ---
# Evitar divisiones por cero
ndwi = np.where(
    (green + nir) == 0,
    0,
    (green - nir) / (green + nir)
)

# Visualización del NDWI
plt.figure(figsize=(8, 8))
plt.imshow(ndwi, cmap="RdYlGn")
plt.colorbar(label="NDWI")
plt.title("NDWI (Normalized Difference Water Index)")
plt.axis("off")
plt.show()
