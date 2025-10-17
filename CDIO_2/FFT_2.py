"""
CDIO1 - Session 11
Advanced Time Series Analysis - Coastal Erosion Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.fft import fft, ifft, fftfreq
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("ADVANCED TIME SERIES ANALYSIS - COASTAL DATASET")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING CLEAN MONTHLY SERIES...")

# Usa la serie generada en la sesión 10
file = "mean_series_monthly.csv"
ts = pd.read_csv(file, index_col=0, parse_dates=True)
ts = ts.squeeze()  # convertir dataframe a serie
ts = ts.sort_index()

print(f"Loaded {len(ts)} monthly points ({ts.index.min()} → {ts.index.max()})")
plt.figure(figsize=(12, 5))
plt.plot(ts, label="Mean shoreline distance")
plt.title("Mean shoreline distance (2017–2024)")
plt.xlabel("Date")
plt.ylabel("Distance (m)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ============================================================================
# 2. TREND ANALYSIS (LINEAR REGRESSION)
# ============================================================================
print("\n" + "=" * 70)
print("[2] TREND ANALYSIS (LINEAR REGRESSION)")
print("=" * 70)

n = len(ts)
X = np.arange(n).reshape(-1, 1)
y = ts.values

model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)

slope = model.coef_[0]
intercept = model.intercept_
r2 = model.score(X, y)
rmse = np.sqrt(mean_squared_error(y, trend))

print(f"Trend slope: {slope:.6f} m/month  ({slope*12:.3f} m/year)")
print(f"R²: {r2:.3f}, RMSE: {rmse:.3f} m")

plt.figure(figsize=(12, 5))
plt.plot(ts.index, y, label="Observed")
plt.plot(ts.index, trend, label="Linear trend", color="red")
plt.title("Linear Trend - Mean Shoreline Distance")
plt.xlabel("Date")
plt.ylabel("Distance (m)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ============================================================================
# 3. REMOVE TREND & APPLY FFT
# ============================================================================
print("\n" + "=" * 70)
print("[3] FREQUENCY ANALYSIS (FFT)")
print("=" * 70)

# Remove trend to analyze periodic components
ts_detrended = y - trend

# Apply FFT
fft_values = fft(ts_detrended)
fft_freq = fftfreq(n, d=1)  # d=1 → monthly sampling
magnitude = np.abs(fft_values)

# Keep only positive frequencies
positive_idx = fft_freq >= 0
freqs = fft_freq[positive_idx]
magnitudes = magnitude[positive_idx]

plt.figure(figsize=(10, 5))
plt.plot(freqs, magnitudes)
plt.title("FFT Spectrum (Magnitude vs Frequency)")
plt.xlabel("Frequency [cycles/month]")
plt.ylabel("Magnitude")
plt.grid(alpha=0.3)
plt.show()

# Find top dominant frequencies
sorted_idx = np.argsort(magnitudes)[::-1]
top_freqs = freqs[sorted_idx][:5]
top_periods = 1 / top_freqs
print("\nTop 5 detected cycles:")
for i, (f, p) in enumerate(zip(top_freqs, top_periods)):
    if np.isfinite(p):
        print(f"  {i+1}. Period: {p:.2f} months  (freq={f:.4f})")

# ============================================================================
# 4. FILTER & RECONSTRUCT SIGNAL
# ============================================================================
print("\n" + "=" * 70)
print("[4] SIGNAL RECONSTRUCTION")
print("=" * 70)

# Keep only the strongest frequencies
threshold = np.sort(magnitude)[-3]  # top 3 frequencies
fft_filtered = fft_values.copy()
fft_filtered[magnitude < threshold] = 0

cyclical = np.real(ifft(fft_filtered))
reconstructed = trend + cyclical

plt.figure(figsize=(12, 5))
plt.plot(ts.index, y, label="Observed", alpha=0.6)
plt.plot(ts.index, reconstructed, label="Trend + Cyclic (FFT)", color="orange")
plt.title("Reconstructed Signal: Trend + Cyclic Components")
plt.xlabel("Date")
plt.ylabel("Distance (m)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Evaluate reconstruction
rmse_reconstructed = np.sqrt(mean_squared_error(y, reconstructed))
r2_reconstructed = r2_score(y, reconstructed)
print(f"Reconstructed model → RMSE: {rmse_reconstructed:.3f}, R²: {r2_reconstructed:.3f}")

# ============================================================================
# 5. FORECASTING 2025
# ============================================================================
print("\n" + "=" * 70)
print("[5] FORECASTING (2025)")
print("=" * 70)

n_future = 6  # 6 months ahead (Jan–Jun 2025)
t_future = np.arange(n, n + n_future)

# Predict future trend
trend_future = model.predict(t_future.reshape(-1, 1))

# Reconstruct future cyclic pattern
mask = magnitude >= threshold
significant_freq = fft_freq[mask]
significant_fft = fft_values[mask]
cyclical_future = np.zeros(n_future)

for freq, fft_val in zip(significant_freq, significant_fft):
    amplitude = np.abs(fft_val) / n
    phase = np.angle(fft_val)
    cyclical_future += amplitude * np.cos(2 * np.pi * freq * t_future + phase)

forecast = trend_future + cyclical_future
future_dates = pd.date_range(ts.index[-1] + pd.offsets.MonthEnd(1), periods=n_future, freq='ME')
forecast_series = pd.Series(forecast, index=future_dates)

# Combine observed and forecast
combined = pd.concat([ts, forecast_series])

plt.figure(figsize=(12, 5))
plt.plot(ts, label="Observed")
plt.plot(forecast_series, label="Forecast (2025)", color="red", linewidth=2)
plt.title("Forecasted Shoreline Distance (Jan–Jun 2025)")
plt.xlabel("Date")
plt.ylabel("Distance (m)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Save forecast
forecast_series.to_csv("forecast_2025.csv")
print("\nForecast saved as 'forecast_2025.csv'")

# ============================================================================
# 6. OPTIONAL: STRUCTURAL BREAK (STORM GLORIA)
# ============================================================================
print("\n" + "=" * 70)
print("[6] STRUCTURAL BREAK (OPTIONAL)")
print("=" * 70)

# Example: sudden change near Feb 2020 (Storm Gloria)
try:
    t = np.arange(len(ts)).reshape(-1, 1)
    t_jump = ts.index.get_loc(pd.to_datetime("2020-02-01"))
    step = (t.flatten() >= t_jump).astype(int).reshape(-1, 1)
    X_break = np.hstack([t, step])
    y = ts.values

    model_break = LinearRegression()
    model_break.fit(X_break, y)
    y_pred_break = model_break.predict(X_break)

    plt.figure(figsize=(12, 5))
    plt.plot(ts.index, y, label="Observed")
    plt.plot(ts.index, y_pred_break, color="green", label="Trend + Step (Gloria)")
    plt.title("Structural Break: Storm Gloria (Feb 2020)")
    plt.xlabel("Date")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print("Structural break model applied successfully.")
except Exception as e:
    print("Gloria modeling skipped:", e)

print()
print("=" * 70)
print("ADVANCED PIPELINE COMPLETED SUCCESSFULLY ✅")
print("=" * 70)
