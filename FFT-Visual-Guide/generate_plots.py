import os
import numpy as np
import scipy.fft
from scipy.signal import chirp
import matplotlib.pyplot as plt

# Setup directories
assets_dir = r"c:\Users\sallu\Desktop\Estudios\Estudios\CICLO 9\IA\UNIDAD 3\FFT-Visual-Guide\assets"
os.makedirs(assets_dir, exist_ok=True)

# Apply modern dark-theme style for visual premium look
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['grid.color'] = '#222222'

# 1. TIME VS FREQUENCY PLOT (FFT Denoising)
fs = 1000  # Sampling freq
t = np.linspace(0, 1, fs, endpoint=False)
# Composite signal: 50Hz (amplitude 1.0) + 120Hz (amplitude 0.5)
clean_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
noise = np.random.normal(0, 0.6, len(t))
noisy_signal = clean_signal + noise

# FFT Computation
N = len(t)
fft_vals = scipy.fft.fft(noisy_signal)
fft_freqs = scipy.fft.fftfreq(N, 1/fs)

# Keep positive frequencies
pos_mask = fft_freqs >= 0
freqs_plot = fft_freqs[pos_mask]
amplitude_plot = (2.0/N) * np.abs(fft_vals[pos_mask])

# Generate figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=150)
fig.suptitle("Análisis Espectral: Dominio del Tiempo vs. Frecuencia", fontsize=14, color='#00adb5', fontweight='bold')

# Top plot: Time domain
ax1.plot(t, noisy_signal, color='#ff2e63', alpha=0.5, label='Señal Ruidosa (Medida)')
ax1.plot(t, clean_signal, color='#00adb5', linewidth=2, label='Señal Real (Frecuencias Puras)')
ax1.set_title("Señal de Vibración en el Dominio del Tiempo", fontsize=11, color='#ffffff')
ax1.set_xlabel("Tiempo (s)", fontsize=9, color='#aaaaaa')
ax1.set_ylabel("Amplitud", fontsize=9, color='#aaaaaa')
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='upper right')

# Bottom plot: Frequency domain
ax2.plot(freqs_plot, amplitude_plot, color='#393e46', linewidth=1.5, alpha=0.7)
ax2.fill_between(freqs_plot, amplitude_plot, color='#08d9d6', alpha=0.3)
# Highlight the peaks
ax2.axvline(x=50, color='#ff2e63', linestyle=':', label='Falla Rodamiento (50 Hz)')
ax2.axvline(x=120, color='#eaeaea', linestyle=':', label='Vibración Eje (120 Hz)')
ax2.set_xlim(0, 200)
ax2.set_title("Espectro de Frecuencia (FFT)", fontsize=11, color='#ffffff')
ax2.set_xlabel("Frecuencia (Hz)", fontsize=9, color='#aaaaaa')
ax2.set_ylabel("Magnitud (Normalizada)", fontsize=9, color='#aaaaaa')
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend(loc='upper right')

plt.tight_layout()
fig_path_1 = os.path.join(assets_dir, "time_vs_frequency.png")
plt.savefig(fig_path_1, bbox_inches='tight', transparent=True)
plt.close()

# 2. SPECTROGRAM (STFT) OF A CHIRP (NON-STATIONARY ANOMALY)
# Generate a sweep from 10Hz to 250Hz simulating a machine accelerating
t_chirp = np.linspace(0, 5, 5000)
w = chirp(t_chirp, f0=10, f1=250, t1=5, method='linear')
noise_c = np.random.normal(0, 0.3, len(w))
noisy_w = w + noise_c

fig2, ax = plt.subplots(figsize=(10, 5), dpi=150)
Pxx, freqs, bins, im = ax.specgram(noisy_w, NFFT=256, Fs=1000, noverlap=128, cmap='viridis')
ax.set_title("Espectrograma STFT: Aceleración de Motor (Falla Progresiva)", fontsize=13, color='#00adb5', fontweight='bold')
ax.set_xlabel("Tiempo (s)", fontsize=10, color='#aaaaaa')
ax.set_ylabel("Frecuencia (Hz)", fontsize=10, color='#aaaaaa')
ax.set_ylim(0, 300)
fig2.colorbar(im, ax=ax, label="Intensidad de Potencia (dB)")

fig_path_2 = os.path.join(assets_dir, "spectrogram.png")
plt.savefig(fig_path_2, bbox_inches='tight', transparent=True)
plt.close()

print(f"Assets creados exitosamente en {assets_dir}")
