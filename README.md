# ECG-Arrhythmia-PVC-Detection

## Dataset

ECG data is taken from the MIT‑BIH Arrhythmia Database.
Download from:
https://physionet.org/content/mitdb/
After downloading, place the files inside:
data/raw/

## Project Workflow

1. **ECG Signal Input**
   - Raw ECG data from MIT‑BIH Arrhythmia Database  

2. **Preprocessing**
   - Noise filtering  
   - Signal normalization  

3. **R‑Peak Detection**
   - Detect heart beats  

4. **RR Interval & BPM Calculation**
   - Compute heart rate  

5. **Feature Extraction**
   - RR statistics  
   - BPM statistics  
   - PVC features  

6. **Dataset Creation**
   - Features saved in `ecg_features.csv`  
   - One ECG record = one row  

7. **Machine Learning**
   - Train multi‑class model  
   - Random Forest classifier  

8. **Disease Classification**
   - Normal  
   - Bradycardia  
   - Tachycardia  
   - PVC  
   - Arrhythmia  
