import matplotlib.pyplot as plt
import matplotlib
import rasterio
import numpy as np
import os
import json
from rasterio import features, mask
from shapely.geometry import shape
import geopandas as gpd
from scipy.ndimage import convolve

matplotlib.use("QtAgg")

# === Directorios ===
figs_dir = "outputs_session5/figs/"
ndwi_example = "outputs_session5/ndwi/"

# === Cargar AOI desde archivo GeoJSON ===
aoi_path = "aoi.geojson"  # ⚠️ guarda tu polígono aquí
gdf = gpd.read_file(aoi_path)

# === Procesar el primer GeoTIFF de NDWI ===
files = [f for f in os.listdir(ndwi_example) if f.endswith(".tif")]

if files:
    ndwi_path = os.path.join(ndwi_example, files[0])  # el primero
    
    with rasterio.open(ndwi_path) as src:
        # --- Reproyectar AOI al CRS del raster ---
        if gdf.crs != src.crs:
            print(f"♻️ Reproyectando AOI de {gdf.crs} a {src.crs}")
            gdf = gdf.to_crs(src.crs)

        # Extraer geometría del AOI reproyectado
        geoms = [json.loads(gdf.to_json())["features"][0]["geometry"]]

        # --- Recortar al polígono AOI ---
        ndwi_crop, transform = mask.mask(src, geoms, crop=True)
        ndwi_crop = ndwi_crop[0]  # primera banda
        profile = src.profile
        profile.update({
            "height": ndwi_crop.shape[0],
            "width": ndwi_crop.shape[1],
            "transform": transform
        })

    # === Mostrar NDWI recortado ===
    plt.figure(figsize=(6,6))
    plt.imshow(ndwi_crop, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(label="NDWI")
    plt.title(f"NDWI (recortado AOI)")
    plt.axis("off")
    out_path = os.path.join(figs_dir, "VIEW_ndwi_crop.png")
    plt.savefig(out_path, dpi=150)
    print(f"✅ Guardado: {out_path}")

    # === 1. Detectar cuerpos de agua (recortado) ===
    water = (ndwi_crop > 0).astype(np.uint8)

    # Guardar GeoTIFF binario
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open("outputs_session5/waterbody_crop.tif", "w", **profile) as dst:
        dst.write(water, 1)
    print("✅ Guardado: outputs_session5/waterbody_crop.tif")

    # === 2. Estimar línea de costa (recortado) ===
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]])
    neighbor_sum = convolve(water, kernel, mode="constant", cval=0)
    coastline = np.logical_and(water==1, neighbor_sum < 8).astype(np.uint8)

    with rasterio.open("outputs_session5/coastline_crop.tif", "w", **profile) as dst:
        dst.write(coastline, 1)
    print("✅ Guardado: outputs_session5/coastline_crop.tif")

    # === 3. Exportar línea de costa a GeoJSON ===
    shapes_gen = features.shapes(coastline, transform=transform)
    geoms = [shape(geom) for geom, val in shapes_gen if val == 1]
    gdf_coast = gpd.GeoDataFrame(geometry=geoms, crs=src.crs)
    # Exportamos a EPSG:4326 para visualizar en geojson.io o Google Earth
    gdf_coast.to_crs("EPSG:4326").to_file("outputs_session5/coastline_crop.geojson", driver="GeoJSON")
    print("✅ Guardado: outputs_session5/coastline_crop.geojson")

    # === Visualización rápida ===
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title("Cuerpos de agua (recorte AOI)")
    plt.imshow(water, cmap="Blues")
    plt.subplot(1,2,2)
    plt.title("Línea de costa (recorte AOI)")
    plt.imshow(coastline, cmap="Reds")
    plt.show()

else:
    print("No se encontraron ficheros NDWI en outputs_session5/ndwi/")
