import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
import csv
import os

# ==============================
# CSV FILE SETUP
# ==============================
file_name = "ecg_features.csv"

if not os.path.exists(file_name):
    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Mean_RR", "Std_RR", "Min_RR", "Max_RR",
            "Mean_BPM", "Std_BPM", "Min_BPM", "Max_BPM",
            "Label"
        ])

# ==============================
# BANDPASS FILTER FUNCTION
# ==============================
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# ==============================
# LOOP THROUGH RECORDS
# ==============================
for record_no in range(108, 235):   # 108 to 234 inclusive
    try:
        print(f"\nProcessing record {record_no}...")

        # Load ECG
        record_path = f"data/raw/{record_no}"
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0]
        fs = record.fs

        # Filter ECG
        filtered_signal = bandpass_filter(signal, 0.5, 40, fs)

        # R-peak detection
        threshold = np.mean(filtered_signal) + 0.5 * np.std(filtered_signal)
        peaks, _ = find_peaks(
            filtered_signal,
            distance=0.6 * fs,
            height=threshold,
            prominence=0.6
        )

        # RR Intervals & BPM
        rr_intervals = np.diff(peaks) / fs
        bpm = 60 / rr_intervals
        avg_bpm = np.mean(bpm)

        # Classification
        if avg_bpm < 60:
            condition = "Bradycardia"
            label = 1
        elif 60 <= avg_bpm <= 100:
            condition = "Normal"
            label = 0
        else:
            condition = "Tachycardia"
            label = 2

        # Feature Extraction
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        min_rr = np.min(rr_intervals)
        max_rr = np.max(rr_intervals)

        mean_bpm = np.mean(bpm)
        std_bpm = np.std(bpm)
        min_bpm = np.min(bpm)
        max_bpm = np.max(bpm)

        # Save to CSV
        data_row = [
            mean_rr, std_rr, min_rr, max_rr,
            mean_bpm, std_bpm, min_bpm, max_bpm,
            label
        ]

        with open(file_name, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(data_row)

        print(f"Record {record_no} saved - Label: {condition}")

    except Exception as e:
        print(f"Skipping record {record_no} due to error: {e}")

print("\nALL RECORDS FROM 108 TO 234 HAVE BEEN PROCESSED.")
print("Your dataset is ready in ecg_features.csv")
