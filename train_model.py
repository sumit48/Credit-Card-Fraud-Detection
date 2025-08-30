import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from joblib import dump
import json
import os

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Features and labels
X = df.drop("Class", axis=1)
y = df["Class"]

# Preprocess Amount and Time
scaler = StandardScaler()
X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train RandomForest
model = RandomForestClassifier(
    n_estimators=100,  # reduced for faster training
    random_state=42,
    class_weight="balanced_subsample"
)
print("Training RandomForest...")
model.fit(X_train_res, y_train_res)
print("Model training completed!")

# Evaluate
y_scores = model.predict_proba(X_test)[:, 1]
print("Classification Report:\n", classification_report(y_test, (y_scores >= 0.5).astype(int)))
print("ROC AUC:", roc_auc_score(y_test, y_scores))
print("PR AUC:", average_precision_score(y_test, y_scores))

# Save model + scaler
dump(model, "models/model.joblib")
dump(scaler, "models/preprocess.joblib")

# Save metadata (threshold + metrics)
metadata = {
    "threshold": 0.5,
    "metrics": {
        "roc_auc": roc_auc_score(y_test, y_scores),
        "pr_auc": average_precision_score(y_test, y_scores),
    },
}
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f)

print("Artifacts saved in 'models/' folder ✅")
