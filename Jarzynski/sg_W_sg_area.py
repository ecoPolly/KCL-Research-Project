import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import json

# === Carica dati dal JSON ===
file_path = "MT3Data_JSON 2.Json"  # <-- cambia percorso se serve

with open(file_path, 'r') as f:
    d = json.load(f)

xs, forces, times, current = [], [], [], []

# === Carica i dati ===
for i in range(0, len(d) - 1):
    xs_array = np.array(d["Pulse_Number_" + str(i)]["z"])
    forces_array = np.array(d["Pulse_Number_" + str(i)]["force"])
    times_array = np.array(d["Pulse_Number_" + str(i)]["time"]) - d["Pulse_Number_" + str(i)]["time"][0]
    current_array = np.array(d["Pulse_Number_" + str(i)]["current"])
    
    I = current_array
    F = (1.504e-5) * I**2 - 0.0133 * I

    # Applica smoothing pesante sulla posizione
    # (finestra ampia, polinomio basso per eliminare oscillazioni)
    window_length = max(51, len(xs_array)//30 * 2 + 1)  # finestra dinamica (minimo 51, dispari)
    poly_order = 2  # basso per evitare overfitting
    xs_smooth = savgol_filter(xs_array, window_length, poly_order)

    xs.append(xs_smooth)
    forces.append(F)
    times.append(times_array)
    current.append(current_array)

# === Parametri ===
lower_limit = 7
upper_limit = 18

# === Calcolo lavoro per ogni pulse ===
work_list = []

for idx in range(len(forces)):
    try:
        xmin_idx = next(i for i in range(1, len(forces[idx]))
                        if forces[idx][i] > lower_limit and forces[idx][i - 1] <= lower_limit)
        xmax_idx = next(i for i in range(xmin_idx + 1, len(forces[idx]))
                        if forces[idx][i] > upper_limit and forces[idx][i - 1] <= upper_limit)
    except StopIteration:
        print(f"⚠️ Pulse {idx} skipped: thresholds not found.")
        continue

    integrate_force = forces[idx][xmin_idx:xmax_idx + 1]
    integrate_x = xs[idx][xmin_idx:xmax_idx + 1]

    # Calcolo del lavoro (trapezoidale)
    W = np.trapz(integrate_force, integrate_x)
    work_list.append(W)

    # === Plot singolo pulse con smoothing ===
    plt.figure(figsize=(10, 6))
    plt.plot(xs[idx], forces[idx], 'b-', linewidth=0.6, alpha=0.4, label='Smoothed Force curve')
    plt.fill_between(integrate_x, 0, integrate_force, alpha=0.5, color='orange', label=f'W={W:.2f} pN·nm')
    plt.axvline(xs[idx][xmin_idx], color='red', linestyle='--', label=f'Start (>{lower_limit} pN)')
    plt.axvline(xs[idx][xmax_idx], color='green', linestyle='--', label=f'Stop (>{upper_limit} pN)')
    plt.axhline(lower_limit, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
    plt.axhline(upper_limit, color='green', linestyle=':', linewidth=0.8, alpha=0.5)
    plt.xlabel('Position [nm]')
    plt.ylabel('Force [pN]')
    plt.title(f'Pulse {idx + 1} (Smoothed X)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

# === Plot complessivo del lavoro ===
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(work_list)), work_list, 'o-', linewidth=1.5)
plt.xlabel('Pulse number')
plt.ylabel('Work [pN·nm]')
plt.title('Work per Pulse (using smoothed X)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Mean W: {np.mean(work_list):.2f} pN·nm | Median: {np.median(work_list):.2f} pN·nm")
