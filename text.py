import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, warnings
warnings.filterwarnings("ignore")

print("\n🚦 Accident Severity Prediction — KNN Dominant Optimized Model\n")

# ===========================
# 1️⃣ LOAD & CLEAN DATA
# ===========================
df = pd.read_csv("accidents_india.csv")
print(f"✅ Dataset loaded successfully — Shape: {df.shape}")

df = df.applymap(lambda x: str(x).strip().lower() if isinstance(x, str) else x)
df.replace(['', ' ', 'nan', 'none', 'null'], np.nan, inplace=True)

# Handle target
if df['Accident_Severity'].dtype == 'object':
    df['Accident_Severity'] = df['Accident_Severity'].map({'slight': 0, 'serious': 1, 'fatal': 2})
df['Accident_Severity'] = df['Accident_Severity'].fillna(0).astype(int)

# Fill missing numeric/categorical
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
df.drop_duplicates(inplace=True)
print("✅ Data cleaned and normalized")

# ===========================
# 2️⃣ ENCODE CATEGORICAL
# ===========================
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# ===========================
# 3️⃣ SCALE, PCA & BALANCE
# ===========================
X = df.drop('Accident_Severity', axis=1)
y = df['Accident_Severity']

# Dual scaling for perfect distance uniformity
X_scaled = StandardScaler().fit_transform(X)
X_scaled = MinMaxScaler().fit_transform(X_scaled)

# Dimensional noise reduction
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Feature selection (top 15 features)
selector = SelectKBest(score_func=f_classif, k=min(15, X_pca.shape[1]))
X_sel = selector.fit_transform(X_pca, y)

# Balanced resampling
sm = SMOTE(sampling_strategy=0.85, random_state=42, k_neighbors=7)
X_res, y_res = sm.fit_resample(X_sel, y)
print(f"✅ SMOTE applied — Balanced classes: {Counter(y_res)}")

# Add tiny Gaussian noise (helps KNN generalize)
X_res += np.random.normal(0, 0.01, X_res.shape)

# ===========================
# 4️⃣ TRAIN-TEST SPLIT
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.25, random_state=42, stratify=y_res
)
print(f"✅ Train-test split done — Train: {len(X_train)}, Test: {len(X_test)}")

# ===========================
# 5️⃣ GRID SEARCH FOR KNN
# ===========================
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

knn_params = {
    'n_neighbors': [5, 7, 9, 11, 13, 15, 17, 19, 21],
    'weights': ['distance'],
    'metric': ['manhattan', 'minkowski', 'chebyshev'],
    'p': [1, 2]
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=knn_params,
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1
)
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_
print(f"🏆 Best Tuned KNN: {best_knn}")

# ===========================
# 6️⃣ OTHER MODELS
# ===========================
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42),
    "SVM": SVC(kernel='rbf', C=1.2, gamma='scale', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=400),
    "XGBoost": XGBClassifier(eval_metric='logloss', max_depth=7, learning_rate=0.1, random_state=42),
    "Tuned KNN": best_knn
}

# ===========================
# 7️⃣ MODEL EVALUATION
# ===========================
print("\n📊 Model Performance (Accuracy | F1 Score):\n")
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    results[name] = {"Accuracy": acc, "F1": f1}
    print(f"{name:20s} → Accuracy: {acc:.4f} | F1: {f1:.4f}")

# ===========================
# 8️⃣ CROSS-VALIDATED KNN
# ===========================
knn_cv_acc = np.mean(cross_val_score(best_knn, X_res, y_res, cv=10, scoring='accuracy'))
print(f"\n🔁 Cross-validated KNN Accuracy: {knn_cv_acc:.4f}")

# ===========================
# 9️⃣ BEST MODEL
# ===========================
best_model = max(results, key=lambda m: results[m]["Accuracy"])
best_acc = results[best_model]["Accuracy"]
best_f1 = results[best_model]["F1"]
print(f"\n🏆 Best Model: {best_model} | Accuracy: {best_acc:.4f} | F1: {best_f1:.4f}")

# ===========================
# 🔟 SAVE MODEL
# ===========================
with open("test1.pkl", "wb") as f:
    pickle.dump(models[best_model], f)
print("\n✅ Best model saved as test1.pkl\n")

# ===========================
# 🔍 FEATURE IMPORTANCE VISUALIZATION
# ===========================
try:
    df_corr = pd.DataFrame(X_res, columns=df.drop('Accident_Severity', axis=1).columns[:X_res.shape[1]])
    df_corr['Severity'] = y_res.values
    corr = df_corr.corr()['Severity'].sort_values(ascending=False).head(10)

    plt.figure(figsize=(10,6))
    sns.barplot(x=corr.values, y=corr.index, palette="viridis")
    plt.title("🔥 Top 10 Features Correlated with Accident Severity")
    plt.xlabel("Correlation with Severity")
    plt.show()
except Exception as e:
    print(f"⚠️ Visualization skipped: {e}")
