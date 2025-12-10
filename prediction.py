#!/usr/bin/env python
# coding: utf-8

"""
🚦 Accident Severity Prediction – Dual Model Trainer + Confusion Matrix Visualizer
✅ test1.pkl → Full model (all features)
✅ test2.pkl → Compact 8-field model (for Flask)
✅ Balanced data, tuned KNN, and visual performance reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import CondensedNearestNeighbour
from xgboost import XGBClassifier
from collections import Counter
import pickle, warnings
warnings.filterwarnings("ignore")

print("\n🚦 Accident Severity Prediction — Full Report with Graphs\n")

# ==============================
# 1️⃣ LOAD & CLEAN DATA
# ==============================
df = pd.read_csv("accidents_india.csv")
print(f"✅ Dataset loaded successfully — Shape: {df.shape}")

# Basic cleaning
df = df.applymap(lambda x: str(x).strip().lower() if isinstance(x, str) else x)
df.replace(['', ' ', 'nan', 'none', 'null'], np.nan, inplace=True)

# Convert labels
if df['Accident_Severity'].dtype == 'object':
    df['Accident_Severity'] = df['Accident_Severity'].map({'slight': 0, 'serious': 1})
else:
    df['Accident_Severity'] = df['Accident_Severity'].astype(float).astype(int)

df.dropna(subset=['Accident_Severity'], inplace=True)

# Fill missing values
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

df.drop_duplicates(inplace=True)
print("✅ Data cleaned and normalized")

# Label encoding
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# ==============================
# 2️⃣ FULL MODEL (ALL FEATURES)
# ==============================
X = df.drop('Accident_Severity', axis=1)
y = df['Accident_Severity']

# Feature transformations
poly = PolynomialFeatures(degree=2, include_bias=False)
X = poly.fit_transform(X)
pt = PowerTransformer()
X = pt.fit_transform(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA(n_components=0.9, random_state=42)
X = pca.fit_transform(X)

# Balance classes
class_counts = Counter(y)
if max(class_counts.values()) / min(class_counts.values()) > 1.2:
    sm = SMOTE(sampling_strategy=0.9, random_state=42)
    X, y = sm.fit_resample(X, y)
    print(f"✅ SMOTE applied — Balanced classes: {Counter(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42, stratify=y)

cnn = CondensedNearestNeighbour(random_state=42)
X_train, y_train = cnn.fit_resample(X_train, y_train)

# ==============================
# 3️⃣ DEFINE MODELS
# ==============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
knn_params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['distance'],
              'metric': ['manhattan', 'euclidean']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params,
                        scoring='f1', cv=cv, n_jobs=-1)
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42),
    "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "XGBoost": XGBClassifier(eval_metric='logloss', learning_rate=0.1, max_depth=5, random_state=42),
    "Tuned KNN": best_knn
}

# ==============================
# 4️⃣ EVALUATE & VISUALIZE
# ==============================
results = {}
print("\n📊 Model Performance (Accuracy | F1 Score):\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results[name] = {"Accuracy": acc, "F1": f1}
    print(f"{name:20s} → Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Save best full model
best_model = max(results, key=lambda m: results[m]["Accuracy"])
with open("test1.pkl", "wb") as f:
    pickle.dump(models[best_model], f)
print(f"\n🏆 Best Full Model: {best_model} | Saved as test1.pkl")

# ==============================
# 5️⃣ COMPACT MODEL (8 FIELDS) — FIXED
# ==============================
print("\n🚗 Training Compact Model (8 Flask Inputs) — Fixed Balance & Encoding\n")

compact_cols = [
    "Sex_Of_Driver", "Vehicle_Type", "Speed_limit", "Road_Type",
    "Number_of_Pasengers", "Day_of_Week", "Light_Conditions", "Weather", "Accident_Severity"
]

df_small = df[compact_cols].copy()

# Ensure labels are binary (0=Minor, 1=Major)
if df_small['Accident_Severity'].max() > 1:
    df_small['Accident_Severity'] = np.where(df_small['Accident_Severity'] > 1, 1, 0)

X_small = df_small.drop('Accident_Severity', axis=1)
y_small = df_small['Accident_Severity']

scaler_small = StandardScaler()
X_small = scaler_small.fit_transform(X_small)

X_train, X_test, y_train, y_test = train_test_split(
    X_small, y_small, test_size=0.25, random_state=42, stratify=y_small
)

# ✅ Apply SMOTE for balance
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print("✅ Applied SMOTE on compact model — Class balance:", Counter(y_train))

# ✅ Train tuned KNN
knn_params_small = {'n_neighbors': [3, 5, 7, 9], 'weights': ['distance'], 'metric': ['manhattan', 'euclidean']}
knn_grid_small = GridSearchCV(KNeighborsClassifier(), knn_params_small, scoring='f1', cv=5, n_jobs=-1)
knn_grid_small.fit(X_train, y_train)
best_knn_small = knn_grid_small.best_estimator_

# Evaluate compact model
y_pred_small = best_knn_small.predict(X_test)
acc_small = accuracy_score(y_test, y_pred_small)
f1_small = f1_score(y_test, y_pred_small)

print(f"✅ Compact Model Accuracy: {acc_small:.4f} | F1: {f1_small:.4f}")
print("Predicted distribution on test set:", Counter(y_pred_small))

# Confusion Matrix
cm_small = confusion_matrix(y_test, y_pred_small)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_small, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix — Compact KNN (test2.pkl)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ==============================
# 6️⃣ SAVE MODEL + SCALER TOGETHER
# ==============================
compact_pkg = {
    "model": best_knn_small,
    "scaler": scaler_small
}

with open("test2.pkl", "wb") as f:
    pickle.dump(compact_pkg, f)

print("💾 Compact 8-field KNN with scaler saved as test2.pkl")
print("\n🎯 Training Completed — All Confusion Matrices Displayed!\n")
