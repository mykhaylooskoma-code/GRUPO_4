import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carguem i llegim el fitxer
file_path = "shoreline_distances_castefa_gava_prat_2017_2024.csv.zip"
df = pd.read_csv(file_path, compression='zip')
print(df.head())
print(df.info())
# Creem la taula "matriu"
df['date'] = pd.to_datetime(df['date'])
pivot_df = df.pivot(index='date', columns='transect_id', values='distance_m')

# Ordenem per fecha
pivot_df = pivot_df.sort_index() 
pivot_df.head()
print(pivot_df.describe())

# NaNs per cada tram (%, gràfica de barrres)
nans_perc = pivot_df.isna().mean() * 100
plt.figure(figsize=(10, 5))
plt.bar(nans_perc.index, nans_perc.values)
plt.title('Porcentaje de NaNs por transecto')
plt.xlabel('Transecto ID')
plt.ylabel('% de NaNs')
plt.show()

# Filtrem amb <20% dels NaNs
filtered_df = pivot_df.loc[:, nans_perc < 20]
print(f"Tramos seleccionados: {filtered_df.shape[1]} / {pivot_df.shape[1]}") #ens mostra els que han quedat
# Mitjana d distàncies per data
mean_series = filtered_df.mean(axis=1)
# Fem la gràfica de l’evolució de la línia de costa amb el temps
plt.figure(figsize=(12, 6))
plt.plot(mean_series.index, mean_series.values, label='Serie media', linewidth=1.5)
plt.title('Evolución media de la línea de costa (2017–2024)')
plt.xlabel('Fecha')
plt.ylabel('Distancia media (m)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# Valors extrems ho fem amb IQR
Q1 = mean_series.quantile(0.25)
Q3 = mean_series.quantile(0.75)
IQR = Q3 - Q1 # Rang
outliers_mask = (mean_series < (Q1 - 1.5 * IQR)) | (mean_series > (Q3 + 1.5 * IQR))
print(f"Outliers detectados: {outliers_mask.sum()}")

# Eliminem i calculem la mitjana al mes
mean_series_clean = mean_series[~outliers_mask]
mean_series_monthly = mean_series_clean.resample('M').mean().interpolate()
# Representem ara la serie mensual neta
plt.figure(figsize=(12, 6))
plt.plot(mean_series_monthly.index, mean_series_monthly.values, label='Serie mensual (limpia)', linewidth=2)
plt.title('Serie mensual suavizada (2017–2024)')
plt.xlabel('Fecha')
plt.ylabel('Distancia media (m)')
plt.legend()
plt.show()
from sklearn.linear_model import LinearRegression

# Gràfic per la distribució de les mitjanes mensuals
plt.hist(mean_series_monthly.values, bins=30, edgecolor='black')
plt.title("Distribución de distancias")
plt.xlabel("Distancia (m)")
plt.ylabel("Frecuencia")
plt.show()

# Calculem la tendència amb regressió lineal
X = np.arange(len(mean_series_monthly)).reshape(-1, 1)
y = mean_series_monthly.values # Valors de distància
model = LinearRegression()
model.fit(X, y) # Per tenir més accuracy de la recta als punts
y_pred = model.predict(X)
# Dibuixem les dades reals i la recta de tendència
plt.figure(figsize=(12, 6))
plt.plot(mean_series_monthly.index, y, label='Datos mensuales', alpha=0.7)
plt.plot(mean_series_monthly.index, y_pred, label='Tendencia (regresión lineal)', color='red', linewidth=2)
plt.title('Tendencia temporal de la línea de costa')
plt.xlabel('Fecha')
plt.ylabel('Distancia media (m)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Pendent per saber si és erosió o acreció
slope = model.coef_[0]
intercept = model.intercept_ #Valor inicial
print(f"Pendiente: {slope:.4f} m/mes  → {slope*12:.4f} m/año")
# Guardar la serie i resultats
mean_series_monthly.to_csv("mean_series_monthly.csv")
print("Serie mensual guardada como mean_series_monthly.csv")
