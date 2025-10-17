import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. Cargar los datos ===
file_path = "shoreline_distances_castefa_gava_prat_2017_2024.csv.zip"

df = pd.read_csv(file_path, compression='zip')
print(df.head())
print(df.info())
# === 2. Crear tabla pivote ===
df['date'] = pd.to_datetime(df['date'])
pivot_df = df.pivot(index='date', columns='transect_id', values='distance_m')

# Ordenar por fecha
pivot_df = pivot_df.sort_index()
pivot_df.head()
# === 3.1 Estadísticas básicas ===
print(pivot_df.describe())

# === 3.2 Porcentaje de NaNs por transecto ===
nans_perc = pivot_df.isna().mean() * 100
plt.figure(figsize=(10, 5))
plt.bar(nans_perc.index, nans_perc.values)
plt.title('Porcentaje de NaNs por transecto')
plt.xlabel('Transecto ID')
plt.ylabel('% de NaNs')
plt.show()

# === 3.3 Filtrar transectos con menos del 20% de NaNs ===
filtered_df = pivot_df.loc[:, nans_perc < 20]
print(f"Transectos seleccionados: {filtered_df.shape[1]} / {pivot_df.shape[1]}")
# === 4. Crear serie media ===
mean_series = filtered_df.mean(axis=1)

plt.figure(figsize=(12, 6))
plt.plot(mean_series.index, mean_series.values, label='Serie media', linewidth=1.5)
plt.title('Evolución media de la línea de costa (2017–2024)')
plt.xlabel('Fecha')
plt.ylabel('Distancia media (m)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# === 5.1 Detectar outliers (IQR method) ===
Q1 = mean_series.quantile(0.25)
Q3 = mean_series.quantile(0.75)
IQR = Q3 - Q1
outliers_mask = (mean_series < (Q1 - 1.5 * IQR)) | (mean_series > (Q3 + 1.5 * IQR))
print(f"Outliers detectados: {outliers_mask.sum()}")

# Eliminarlos o mantenerlos (según criterio)
mean_series_clean = mean_series[~outliers_mask]

# === 5.2 Resamplear a mensual ===
mean_series_monthly = mean_series_clean.resample('M').mean().interpolate()

plt.figure(figsize=(12, 6))
plt.plot(mean_series_monthly.index, mean_series_monthly.values, label='Serie mensual (limpia)', linewidth=2)
plt.title('Serie mensual suavizada (2017–2024)')
plt.xlabel('Fecha')
plt.ylabel('Distancia media (m)')
plt.legend()
plt.show()
from sklearn.linear_model import LinearRegression

# === 6.1 Distribución ===
plt.hist(mean_series_monthly.values, bins=30, edgecolor='black')
plt.title("Distribución de distancias")
plt.xlabel("Distancia (m)")
plt.ylabel("Frecuencia")
plt.show()

# === 6.2 Tendencia (regresión lineal) ===
X = np.arange(len(mean_series_monthly)).reshape(-1, 1)
y = mean_series_monthly.values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(12, 6))
plt.plot(mean_series_monthly.index, y, label='Datos mensuales', alpha=0.7)
plt.plot(mean_series_monthly.index, y_pred, label='Tendencia (regresión lineal)', color='red', linewidth=2)
plt.title('Tendencia temporal de la línea de costa')
plt.xlabel('Fecha')
plt.ylabel('Distancia media (m)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Métricas básicas
slope = model.coef_[0]
intercept = model.intercept_
print(f"Pendiente: {slope:.4f} m/mes  → {slope*12:.4f} m/año")
# Guardar la serie y resultados
mean_series_monthly.to_csv("mean_series_monthly.csv")
print("Serie mensual guardada como mean_series_monthly.csv")
