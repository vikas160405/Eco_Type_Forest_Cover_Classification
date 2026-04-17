# =============================================================================
# EcoType: Forest Cover Type Prediction - PIPELINE
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import os
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = r"C:\Users\relan\Downloads\echo\cover_type.csv"  
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Fix column spacing
df.columns = df.columns.str.strip()

print("Columns:", df.columns)

# =============================================================================
# DATA CLEANING
# =============================================================================
df.drop_duplicates(inplace=True)

# Numeric & categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# SAFE REMOVE TARGET
numeric_cols = [col for col in numeric_cols if col != 'Cover_Type']

cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Missing values
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# =============================================================================
# OUTLIER HANDLING (IQR)
# =============================================================================
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

print("Outliers handled")

# =============================================================================
# SKEWNESS FIX
# =============================================================================
skewed_features = []

for col in numeric_cols:
    if abs(df[col].skew()) > 1:
        df[col] = np.log1p(df[col] - df[col].min())
        skewed_features.append(col)

# Save skewed features
with open(f"{MODEL_DIR}/skewed_features.pkl", "wb") as f:
    pickle.dump(skewed_features, f)

print("Skewness fixed")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
df['Distance_To_Hydrology'] = np.sqrt(
    df['Horizontal_Distance_To_Hydrology']**2 +
    df['Vertical_Distance_To_Hydrology']**2
)

df['Mean_Hillshade'] = (
    df['Hillshade_9am'] +
    df['Hillshade_Noon'] +
    df['Hillshade_3pm']
) / 3

df['Hydro_Road_Distance_Diff'] = (
    df['Horizontal_Distance_To_Hydrology'] -
    df['Horizontal_Distance_To_Roadways']
)

print("Feature engineering done")

# =============================================================================
# ENCODING
# =============================================================================
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

with open(f"{MODEL_DIR}/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Encoding done")

# =============================================================================
# SPLIT FEATURES
# =============================================================================
X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

# =============================================================================
# HANDLE IMBALANCE
# =============================================================================
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

print("Class imbalance handled")

# =============================================================================
# FEATURE SELECTION
# =============================================================================
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances[importances > 0.01].index.tolist()

X = X[top_features]

with open(f"{MODEL_DIR}/top_features.pkl", "wb") as f:
    pickle.dump(top_features, f)

print("Feature selection done")

# =============================================================================
# SCALING
# =============================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Scaling done")

# =============================================================================
# TRAIN TEST SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# MODEL TRAINING
# =============================================================================
models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss')
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# =============================================================================
# BEST MODEL
# =============================================================================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")

# =============================================================================
# CROSS VALIDATION
# =============================================================================
cv_score = cross_val_score(best_model, X_scaled, y, cv=5).mean()
print(f"Cross Validation Accuracy: {cv_score:.4f}")

# =============================================================================
# SAVE MODEL
# =============================================================================
with open(f"{MODEL_DIR}/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n Pipeline Completed Successfully!")
