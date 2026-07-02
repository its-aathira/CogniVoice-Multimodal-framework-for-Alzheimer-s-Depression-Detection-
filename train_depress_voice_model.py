"""
train_depress_voice_model_V2.py

Retrains depression voice model on 19 real clinical features
extracted from actual audio files (not just Jitter+Shimmer from CSV).

Labels: 0=Healthy, 2=Depression  (consistent with EEG 3-class model)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    print("⚠️  pip install imbalanced-learn")
    HAS_SMOTE = False

# ── Feature columns — must match voice_feature_extractor_V2.py exactly ────────
FEATURE_COLS = (
    [f'mfcc_{i}' for i in range(1, 14)] +
    ['speech_rate', 'pitch_mean', 'pitch_std', 'jitter', 'shimmer', 'hnr']
)  # 19 features

# ── 1. Load ───────────────────────────────────────────────────────────────────
csv_file = "depression_audio_features.csv"
if not os.path.exists(csv_file):
    print(f"❌ {csv_file} not found. Run extract_depression_audio_features.py first.")
    exit()

df = pd.read_csv(csv_file)
print(f"📊 Dataset: {len(df)} rows")
print(f"📊 Label distribution: {dict(df['label'].value_counts().sort_index())}")
print(f"\n📈 Feature means per class:")
print(df.groupby('label')[['mfcc_1','jitter','shimmer','hnr','pitch_mean','speech_rate']].mean().round(4))

X = df[FEATURE_COLS].values
y = df['label'].astype(int).values

# ── 2. Scale ──────────────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. SMOTE ──────────────────────────────────────────────────────────────────
if HAS_SMOTE:
    min_count = pd.Series(y).value_counts().min()
    k = min(5, min_count - 1)
    print(f"\n🔄 Applying SMOTE (k={k})...")
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    print(f"📊 After SMOTE: {dict(zip(*np.unique(y_res, return_counts=True)))}")
else:
    X_res, y_res = X_scaled, y

# ── 4. Split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ── 5. Train ──────────────────────────────────────────────────────────────────
print("\n🧠 Training models...")

rf = RandomForestClassifier(
    n_estimators=500, max_depth=10,
    min_samples_leaf=2, class_weight='balanced',
    random_state=42, n_jobs=-1
)
gb = GradientBoostingClassifier(
    n_estimators=300, max_depth=4,
    learning_rate=0.05, random_state=42
)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

rf_acc = accuracy_score(y_test, rf.predict(X_test))
gb_acc = accuracy_score(y_test, gb.predict(X_test))
print(f"  Random Forest:     {rf_acc*100:.1f}%")
print(f"  Gradient Boosting: {gb_acc*100:.1f}%")

model      = rf if rf_acc >= gb_acc else gb
model_name = "Random Forest" if rf_acc >= gb_acc else "Gradient Boosting"
print(f"✅ Using: {model_name}")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
present = sorted(set(y_test))
names   = {0: 'Healthy', 2: 'Depression'}
tnames  = [names[l] for l in present]

print(f"\n🏆 Test Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, labels=present, target_names=tnames))

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_res, y_res, cv=cv, scoring='f1_macro')
print(f"5-Fold CV Macro-F1: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

if hasattr(model, 'feature_importances_'):
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\n📌 Top 8 features:\n{fi.head(8).round(4)}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
joblib.dump(model,        'depress_voice_model.pkl')
joblib.dump(scaler,       'scaler_depress_voice.pkl')
joblib.dump(FEATURE_COLS, 'depress_voice_feature_cols.pkl')

print("\n💾 Saved: depress_voice_model.pkl")
print("💾 Saved: scaler_depress_voice.pkl")
print(f"💾 Saved: depress_voice_feature_cols.pkl  ({len(FEATURE_COLS)} features)")

# ── 8. Confusion matrix ───────────────────────────────────────────────────────
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred, labels=present)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=tnames, yticklabels=tnames)
plt.title(f"Depression Voice V2 ({model_name})")
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("confusion_matrix_dep_voice_v2.png")
print("📊 Saved: confusion_matrix_dep_voice_v2.png")
plt.show()