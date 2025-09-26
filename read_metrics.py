import pickle
import pandas as pd

# Ruta al archivo generado por tu pipeline
metrics_path = "outputs_session5/metrics.pkl"

# Cargar métricas
with open(metrics_path, "rb") as f:
    metrics = pickle.load(f)

print("\n=== Métricas cargadas ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Crear tabla de comparación descargas
data = {
    "Modo": ["1 hilo", "4 hilos"],
    "Tiempo (s)": [
        metrics["download_metrics_threads_1"]["elapsed_s"],
        metrics["download_metrics_threads_4"]["elapsed_s"],
    ],
    "Tamaño total (GB)": [
        metrics["download_metrics_threads_1"]["total_bytes"] / (1024**3),
        metrics["download_metrics_threads_4"]["total_bytes"] / (1024**3),
    ],
    "Ancho de banda (MB/s)": [
        metrics["download_metrics_threads_1"]["bandwidth_Bps"] / (1024**2),
        metrics["download_metrics_threads_4"]["bandwidth_Bps"] / (1024**2),
    ],
}

df = pd.DataFrame(data)
print("\n=== Comparación descargas ===")
print(df.to_string(index=False))

# Extra info
print("\n=== Extra ===")
print(f"Nº imágenes totales encontradas: {metrics['items_total']}")
print(f"Nº imágenes tras filtro nubes: {metrics['items_filtered_cloud']}")
print(f"Tiempo total NDWI (s): {metrics['time_total_seconds_range']}")
print(f"Tamaño total datos: {metrics['size_total_bytes_range'] / (1024**3):.2f} GB")
