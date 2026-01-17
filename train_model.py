print("Program started")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Libraries imported")

# Load dataset
data = pd.read_csv("ecg_features.csv")
print("CSV loaded successfully")

print("\nFirst 5 rows:")
print(data.head())

print("\nTotal rows:", len(data))

# Split into features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("\nFeatures shape:", X.shape)
print("Labels shape:", y.shape)

# Train‑test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\nModel training completed")

# Predict
y_pred = model.predict(X_test)
print("\nPredictions:", y_pred)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)

# Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nProgram finished")
