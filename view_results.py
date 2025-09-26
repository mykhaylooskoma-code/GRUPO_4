import matplotlib.pyplot as plt
import rasterio
import os

# === Ver PNGs generados ===
figs_dir = "outputs_session5/figs/"

for img in ["rgb_cloudless.png", "rgb_cloudy.png", "ndwi_example.png"]:
    path = figs_dir + img
    try:
        img_data = plt.imread(path)
        plt.figure(figsize=(6,6))
        plt.imshow(img_data)
        plt.title(img)
        plt.axis("off")
        out_path = figs_dir + "VIEW_" + img
        plt.savefig(out_path, dpi=150)
        print(f"✅ Guardado: {out_path}")
    except Exception as e:
        print(f"No se pudo abrir {img}: {e}")

# === Ver un GeoTIFF de NDWI ===
ndwi_example = "outputs_session5/ndwi/"
files = [f for f in os.listdir(ndwi_example) if f.endswith(".tif")]

if files:
    ndwi_path = os.path.join(ndwi_example, files[0])  # el primero
    with rasterio.open(ndwi_path) as src:
        ndwi = src.read(1)
    plt.figure(figsize=(6,6))
    plt.imshow(ndwi, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(label="NDWI")
    plt.title(f"NDWI: {files[0]}")
    plt.axis("off")
    out_path = figs_dir + "VIEW_ndwi_example.png"
    plt.savefig(out_path, dpi=150)
    print(f"✅ Guardado: {out_path}")
else:
    print("No se encontraron ficheros NDWI en outputs_session5/ndwi/")
