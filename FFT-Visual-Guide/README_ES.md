# 🚀 Transformada Rápida de Fourier (FFT) y Transformada Discreta de Fourier (DFT) para Mantenimiento Predictivo

Una guía matemática y visual sobre procesamiento de señales, transformaciones tiempo-frecuencia y su aplicación en la **Industria 4.0** y el **Diagnóstico de Fallas basado en Deep Learning**.

---

## 📖 Introducción al Análisis de Fourier

En los sistemas industriales físicos, los sensores (acelerómetros, micrófonos de emisión acústica, sensores de corriente) capturan datos en el **Dominio del Tiempo**. Sin embargo, la maquinaria compleja contiene múltiples componentes que rotan a diferentes velocidades. La señal de vibración bruta es una mezcla ruidosa de todas estas fuentes.

La **Transformada de Fourier** actúa como un prisma matemático. Así como un prisma de vidrio descompone la luz blanca en sus diferentes longitudes de onda (colores), la Transformada de Fourier descompone una señal temporal compleja en sus ondas sinusoidales constituyentes a diferentes frecuencias.

```text
Señal en el Dominio del Tiempo ──▶ [ Transformada de Fourier ] ──▶ Espectro de Frecuencia (Velocidades)
```

---

## 🧮 1. La Transformada Discreta de Fourier (DFT)

Para analizar señales digitales muestreadas a intervalos discretos de tiempo, utilizamos la **Transformada Discreta de Fourier (DFT)**.

### Formulación Matemática
Dada una secuencia de $N$ números complejos $x_n$ que representan la señal muestreada en los pasos de tiempo $n = 0, \dots, N-1$, la DFT los convierte en una secuencia de $N$ componentes de frecuencia $X_k$:

$$X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-i 2\pi \frac{k}{N} n}, \quad k = 0, \dots, N-1$$

Donde:
*   $x_n$: La amplitud de la señal de entrada digital en el índice $n$.
*   $X_k$: El coeficiente complejo que representa la amplitud y la fase en el índice de frecuencia $k$.
*   $e^{-i 2\pi \frac{k}{N} n}$: La función de base exponencial compleja.

### Relación de Euler y Correlación Compleja
Para entender cómo la DFT extrae las frecuencias, expandimos la exponencial compleja utilizando la **Fórmula de Euler**:

$$e^{-i\theta} = \cos(\theta) - i\sin(\theta)$$

Sustituyendo esto en la ecuación de la DFT:

$$X_k = \sum_{n=0}^{N-1} x_n \left[ \cos\left(2\pi \frac{k}{N} n\right) - i\sin\left(2\pi \frac{k}{N} n\right) \right]$$

De este modo, cada coeficiente de frecuencia $X_k$ consiste en:
*   **Parte Real ($\text{Re}\{X_k\}$):** La correlación de la señal con una onda de coseno a la frecuencia $k$ (captura simetrías pares).
*   **Parte Imaginaria ($\text{Im}\{X_k\}$):** La correlación de la señal con una onda de seno a la frecuencia $k$ (captura simetrías impares).

---

## ⚡ 2. La Transformada Rápida de Fourier (FFT)

Calcular la DFT directamente requiere $N$ multiplicaciones para cada una de las $N$ salidas, lo que conlleva una complejidad computacional de **$\mathcal{O}(N^2)$**. Para grandes conjuntos de datos (ej. $10^6$ puntos de vibración), esto es demasiado lento para el monitoreo en tiempo real.

La **Transformada Rápida de Fourier (FFT)** es un algoritmo optimizado (desarrollado por Cooley y Tukey en 1965) que calcula la misma DFT exacta en solo **$\mathcal{O}(N \log_2 N)$** operaciones.

### Deducción de Cooley-Tukey Radix-2
El algoritmo utiliza una estrategia de "divide y vencerás", dividiendo una DFT de tamaño $N$ (donde $N$ es una potencia de 2) en dos DFTs más pequeñas de tamaño $N/2$: una para los puntos de índice par y otra para los de índice impar.

Utilizando la notación del **Twiddle Factor** $W_N = e^{-i \frac{2\pi}{N}}$, la DFT es:

$$X_k = \sum_{n=0}^{N-1} x_n W_N^{nk}$$

Dividiendo la sumatoria en índices pares ($n = 2r$) e impares ($n = 2r+1$):

$$X_k = \sum_{r=0}^{N/2-1} x_{2r} W_N^{2rk} + \sum_{r=0}^{N/2-1} x_{2r+1} W_N^{(2r+1)k}$$

Dado que $W_N^{2rk} = e^{-i \frac{2\pi}{N} (2rk)} = e^{-i \frac{2\pi}{N/2} rk} = W_{N/2}^{rk}$, podemos reescribir esto como:

$$X_k = \sum_{r=0}^{N/2-1} x_{2r} W_{N/2}^{rk} + W_N^k \sum_{r=0}^{N/2-1} x_{2r+1} W_{N/2}^{rk}$$

$$X_k = E_k + W_N^k O_k$$

Donde:
*   $E_k$: La DFT de los términos con índices pares de $x_n$.
*   $O_k$: La DFT de los términos con índices impares de $x_n$.

### Simetría y Periodicidad (Operación Mariposa)
Debido a que $E_k$ y $O_k$ son periódicos con periodo $N/2$, solo necesitamos calcularlos para $k = 0, \dots, N/2 - 1$. Para los índices $k \ge N/2$:

$$X_{k + N/2} = E_k - W_N^k O_k$$

Esta simetría matemática forma el **Diagrama de Mariposa**, permitiendo calcular ambas salidas de manera simultánea y evitando cálculos redundantes.

---

## 📊 3. Análisis Visual (Generado con Python)

A continuación se presentan los gráficos espectrales reales generados en Python que muestran cómo se analiza una señal.

### A. Dominio del Tiempo vs. Espectro de Frecuencia
El gráfico superior muestra una señal de vibración ruidosa (frecuencias mezcladas $+$ ruido gaussiano). El gráfico inferior muestra el espectro FFT calculado. La FFT aísla claramente las dos frecuencias físicas ocultas ($50\text{ Hz}$ y $120\text{ Hz}$), filtrando el ruido de fondo.

![Tiempo vs Espectro de Frecuencia](assets/time_vs_frequency.png)

### B. Espectrograma de la Transformada de Fourier de Tiempo Corto (STFT)
Para señales no estacionarias (donde la velocidad de rotación de la máquina cambia con el tiempo, ej. durante el arranque o la desaceleración), una FFT estándar falla porque promedia las frecuencias a lo largo de toda la señal.

Aplicamos la **STFT**, dividiendo la señal en ventanas deslizantes superpuestas y calculando la FFT para cada ventana. Esto genera un **Espectrograma** (Tiempo vs. Frecuencia vs. Potencia/Decibelios).

![Espectrograma STFT](assets/spectrogram.png)

---

## ⚙️ 4. Aplicación en Mantenimiento Predictivo y Deep Learning

En sistemas mecánicos rotativos, los defectos específicos generan impactos periódicos que se alinean con frecuencias de falla calculables:

$$\text{BPFI} = \frac{N_{bolas}}{2} f_r \left( 1 + \frac{d}{D} \cos\alpha \right)$$
$$\text{BPFO} = \frac{N_{bolas}}{2} f_r \left( 1 - \frac{d}{D} \cos\alpha \right)$$

*   **BPFI/BPFO:** Frecuencia de paso de bola por pista interna / externa.
*   $f_r$: Frecuencia de rotación del eje.
*   $d, D$: Diámetro de la bola y del paso (*pitch*).

### Transformación de Señal a Imagen para CNNs
En el diagnóstico moderno con IA:
1.  Las **señales de vibración 1D** son registradas por acelerómetros.
2.  Las señales se transforman en **imágenes 2D** (como espectrogramas STFT o campos angulares de Gramian - GAF).
3.  Una **Red Neuronal Convolucional (CNN)** 2D (ej. ResNet, MobileNet o YOLO) se entrena con estas imágenes para clasificar la salud de la maquinaria en tiempo real.

---

## 💻 Código Python para la Generación de Gráficos

Este es el script de Python utilizado para generar los gráficos anteriores utilizando `scipy` y `matplotlib`:

```python
import os
import numpy as np
import scipy.fft
from scipy.signal import chirp
import matplotlib.pyplot as plt

# Estilo visual oscuro premium
plt.style.use('dark_background')

# 1. Generación de señal compuesta con ruido
fs = 1000  # Frecuencia de muestreo
t = np.linspace(0, 1, fs, endpoint=False)
clean_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
noise = np.random.normal(0, 0.6, len(t))
noisy_signal = clean_signal + noise

# 2. Cálculo de la FFT
fft_vals = scipy.fft.fft(noisy_signal)
fft_freqs = scipy.fft.fftfreq(len(t), 1/fs)

# Filtrar frecuencias positivas
pos_mask = fft_freqs >= 0
freqs_plot = fft_freqs[pos_mask]
amplitude_plot = (2.0/len(t)) * np.abs(fft_vals[pos_mask])

# Graficación
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=150)
ax1.plot(t, noisy_signal, color='#ff2e63', alpha=0.5, label='Señal Ruidosa')
ax1.plot(t, clean_signal, color='#00adb5', linewidth=2, label='Señal Limpia')
ax1.set_title("Señal de Vibración en el Dominio del Tiempo")
ax1.set_xlabel("Tiempo (s)")
ax1.set_ylabel("Amplitud")
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend()

ax2.plot(freqs_plot, amplitude_plot, color='#08d9d6')
ax2.set_xlim(0, 200)
ax2.set_title("Espectro de Frecuencia (FFT)")
ax2.set_xlabel("Frecuencia (Hz)")
ax2.set_ylabel("Magnitud Normalizada")
ax2.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
```

---
*Desarrollado para el portafolio de proyectos de Inteligencia Artificial e Internet de las Cosas Industrial (IIoT).*
