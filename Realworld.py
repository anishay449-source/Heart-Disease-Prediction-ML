# ==========================================
# HEART DISEASE PREDICTION PROJECT
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("Heart_Disease_Prediction.csv")

print("="*50)
print("FIRST 5 RECORDS")
print("="*50)
print(df.head())

# ==========================================
# DATASET OVERVIEW
# ==========================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# REMOVE DUPLICATES
# ==========================================

duplicates = df.duplicated().sum()

print("\nDuplicate Records:", duplicates)

df = df.drop_duplicates()

# ==========================================
# AGE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8,5))

sns.histplot(df["Age"], bins=20)

plt.title("Age Distribution of Patients")

plt.xlabel("Age")

plt.ylabel("Number of Patients")

plt.savefig("age_distribution.png")

plt.show()

# ==========================================
# HEART DISEASE COUNT
# ==========================================

plt.figure(figsize=(6,4))

sns.countplot(x="Heart Disease", data=df)

plt.title("Heart Disease Cases")

plt.savefig("disease_count.png")

plt.show()

# ==========================================
# CHOLESTEROL DISTRIBUTION
# ==========================================

plt.figure(figsize=(8,5))

sns.histplot(df["Cholesterol"], bins=20)

plt.title("Cholesterol Distribution")

plt.savefig("cholesterol_distribution.png")

plt.show()

# ==========================================
# CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("heatmap.png")

plt.show()

# ==========================================
# MACHINE LEARNING SECTION
# ==========================================

X = df.drop("Heart Disease", axis=1)

y = df["Heart Disease"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# MODEL TRAINING
# ==========================================

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report")

print(classification_report(y_test, y_pred))

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")

plt.show()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance = importance.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nMost Important Features")

print(importance)

# ==========================================
# CONCLUSION
# ==========================================

print("\nPROJECT COMPLETED SUCCESSFULLY")