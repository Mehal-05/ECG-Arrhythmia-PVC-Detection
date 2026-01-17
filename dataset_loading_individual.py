import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import csv
import os

# ---------------------------
# LOAD ECG
# ---------------------------
record = wfdb.rdrecord(r"data\raw\100")   # change record number each time
signal = record.p_signal[:, 0]
fs = record.fs

print("Sampling Frequency:", fs)
print("Signal Length:", len(signal))

# ---------------------------
# BANDPASS FILTER
# ---------------------------
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

filtered_signal = bandpass_filter(signal, 0.5, 40, fs)

# ---------------------------
# R‑PEAK DETECTION
# ---------------------------
threshold = np.mean(filtered_signal) + 0.5 * np.std(filtered_signal)

peaks, properties = find_peaks(
    filtered_signal,
    distance=0.6 * fs,
    height=threshold,
    prominence=0.6
)

print("Number of R-peaks detected:", len(peaks))

# ---------------------------
# RR INTERVAL & BPM
# ---------------------------
rr_intervals = np.diff(peaks) / fs
bpm = 60 / rr_intervals

print("First 10 RR intervals (s):", rr_intervals[:10])
print("First 10 BPM values:", bpm[:10])

avg_bpm = np.mean(bpm)
print("Average Heart Rate (BPM):", avg_bpm)

# ---------------------------
# HEART CONDITION
# ---------------------------
def classify_heart_rate(bpm_value):
    if bpm_value < 60:
        return "Bradycardia"
    elif 60 <= bpm_value <= 100:
        return "Normal"
    else:
        return "Tachycardia"

condition = classify_heart_rate(avg_bpm)
print("Heart Condition:", condition)

# ---------------------------
# FEATURE EXTRACTION
# ---------------------------
mean_rr = np.mean(rr_intervals)
std_rr = np.std(rr_intervals)
min_rr = np.min(rr_intervals)
max_rr = np.max(rr_intervals)

mean_bpm = np.mean(bpm)
std_bpm = np.std(bpm)
min_bpm = np.min(bpm)
max_bpm = np.max(bpm)

print("\n--- Features ---")
print("Mean RR:", mean_rr)
print("STD RR:", std_rr)
print("Min RR:", min_rr)
print("Max RR:", max_rr)
print("Mean BPM:", mean_bpm)
print("STD BPM:", std_bpm)
print("Min BPM:", min_bpm)
print("Max BPM:", max_bpm)

# ---------------------------
# LABEL TO NUMBER
# ---------------------------
if condition == "Normal":
    label = 0
elif condition == "Bradycardia":
    label = 1
else:
    label = 2

# ---------------------------
# SAVE TO CSV (DATASET)
# ---------------------------
data_row = [
    mean_rr, std_rr, min_rr, max_rr,
    mean_bpm, std_bpm, min_bpm, max_bpm,
    label
]

file_name = "ecg_features.csv"

if not os.path.exists(file_name):
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Mean_RR", "Std_RR", "Min_RR", "Max_RR",
            "Mean_BPM", "Std_BPM", "Min_BPM", "Max_BPM",
            "Label"
        ])
        writer.writerow(data_row)
else:
    with open(file_name, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data_row)

print("One row added to ecg_features.csv")

# ---------------------------
# FULL ECG PLOTS
# ---------------------------
plt.figure(figsize=(15,10))

plt.subplot(3,1,1)
plt.plot(signal)
plt.title("Raw ECG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.subplot(3,1,2)
plt.plot(filtered_signal)
plt.title("Filtered ECG Signal (0.5–40 Hz)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.subplot(3,1,3)
plt.plot(filtered_signal, label="Filtered ECG")
plt.plot(peaks, filtered_signal[peaks], "ro", markersize=4, label="R-peaks")
plt.title("Full ECG with R-peaks (Compressed View)")
plt.legend()

plt.tight_layout()
plt.show()

# ---------------------------
# 10‑SECOND ZOOM VIEW
# ---------------------------
start = 0
duration = 10 * fs
end = start + duration

zoom_peaks = peaks[(peaks >= start) & (peaks <= end)]

plt.figure(figsize=(15,5))
plt.plot(filtered_signal[start:end], label="Filtered ECG")
plt.plot(zoom_peaks - start, filtered_signal[zoom_peaks],
         "ro", markersize=6, label="R-peaks")
plt.title("Zoomed ECG (10 seconds) with R-peaks")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
