# Structural break modelling + reconstrucció FFT en el Glòria

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.fft import fft, ifft, fftfreq
from sklearn.metrics import mean_squared_error, r2_score

# Carguem sèries mensuals del primer codi
file = "mean_series_monthly.csv"
ts = pd.read_csv(file, index_col=0, parse_dates=True).squeeze()
ts = ts.asfreq("ME").sort_index() # f mensual
y = ts.values
n = len(ts)
t = np.arange(n).reshape(-1, 1)
print(f"Loaded {n} monthly points ({ts.index.min().date()} → {ts.index.max().date()})")
# Evaluem tendència
GLORIA_DATE = pd.to_datetime("2020-02-01")
t_jump = ts.index.get_indexer([GLORIA_DATE], method="nearest")[0]
step = (t.flatten() >= t_jump).astype(int).reshape(-1, 1)
X = np.hstack([t, step])
model = LinearRegression().fit(X, y)
trend_step = model.predict(X)
slope = model.coef_[0]
beta = model.coef_[1]
r2 = model.score(X, y)
rmse = np.sqrt(mean_squared_error(y, trend_step))
print(f"Trend slope: {slope:.6f} m/mes ({slope*12:.3f} m/año)")
print(f"Step (Gloria) magnitude: {beta:.3f} m")
print(f"R²={r2:.3f}  RMSE={rmse:.3f}")
# Ara apliquem la FFT
residuals = y - trend_step
fft_values = fft(residuals)
freqs = fftfreq(n, d=1)
magnitude = np.abs(fft_values)

# Deixem les freüències dominants
pos_mask = freqs > 0
freqs_pos = freqs[pos_mask]
mag_pos = magnitude[pos_mask]
sorted_idx = np.argsort(mag_pos)[::-1]
top_freqs = freqs_pos[sorted_idx][:3]
threshold = mag_pos[sorted_idx[2]]
fft_filtered = fft_values.copy()
fft_filtered[magnitude < threshold] = 0
cyclical = np.real(ifft(fft_filtered))
reconstructed = trend_step + cyclical
rmse_recon = np.sqrt(mean_squared_error(y, reconstructed))
r2_recon = r2_score(y, reconstructed)
print(f"[Step+FFT] RMSE={rmse_recon:.3f}, R²={r2_recon:.3f}")
# Gràfiques
plt.figure(figsize=(12, 6))
plt.plot(ts.index, y, color="steelblue", alpha=0.6, label="Original")
plt.plot(ts.index, trend_step, color="purple", lw=2, label="Trend + Step")
plt.plot(ts.index, cyclical, color="orange", lw=1.5, label="Cyclical (FFT)")
plt.plot(ts.index, reconstructed, color="red", lw=2, label="Reconstructed")
plt.axvline(ts.index[t_jump], color="gray", ls="--", alpha=0.7, label="Storm Gloria")
plt.title("Trend + Step + FFT Reconstruction")
plt.xlabel("Date")
plt.ylabel("Mean shoreline distance (m)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
residuals_final = y - reconstructed
plt.figure(figsize=(8, 4))
plt.hist(residuals_final, bins=25, color="cornflowerblue", edgecolor="black", alpha=0.8)
plt.axvline(residuals_final.mean(), color="red", ls="--", label=f"Mean = {residuals_final.mean():.2f} m")
plt.title("Residuals distribution")
plt.xlabel("Distance (m)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#Mostrem els períodes
print("\FFT periods:")
for f in top_freqs:
    period = 1 / f
    print(f"  Period ≈ {period:.1f} months (freq={f:.4f})")

print("\n✅ Model completed successfully — Step + FFT + Residuals")
