import rasterio
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("QtAgg")

# Ruta a tu archivo .tif
tif_file = "outputs_session6/coastline_4/costa_ex4.tif"

# Abrir y leer
with rasterio.open(tif_file) as src:
    img = src.read(1)

# Mostrar
plt.imshow(img, cmap="gray")
plt.colorbar()
plt.title("Vista rápida del GeoTIFF")
plt.show()
