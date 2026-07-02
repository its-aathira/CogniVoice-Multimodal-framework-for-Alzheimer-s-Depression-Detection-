import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# --- ROBUST PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

data_path = os.path.join(project_root, "data", "master_3class_eeg.csv")
model_path = os.path.join(project_root, "backend", "model_3class_eeg.pkl")
image_path = os.path.join(project_root, "frontend", "assets", "confusion_matrix_3class.png")

# 1. Load Data
if not os.path.exists(data_path):
    print("❌Error: Master dataset not found. Run the merge script first!")
    exit()

df = pd.read_csv(data_path)

# 2. Prepare Features (X) and Labels (y)
X = df[['theta_power', 'alpha_power', 'beta_power', 'theta_alpha_ratio']]
y = df['label']

# 3. Split into Train (80%) and Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the 3-Class Model
print("🧠 Training 3-Class Classifier (Healthy vs AD vs Depression)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🏆 FINAL MODEL ACCURACY: {acc*100:.2f}%")
print("-" * 30)

# Define class names for the report
target_names = ['Healthy', 'Alzheimers', 'Depression']
# Handle case where test set might miss a class
unique_labels = sorted(y_test.unique())
present_names = [target_names[i] for i in unique_labels]

print(classification_report(y_test, y_pred, target_names=present_names))

# 6. Save the Model
joblib.dump(model, model_path)
print(f"✅ Model saved to: {model_path}")

# 7. Generate & Save Confusion Matrix
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=present_names, yticklabels=present_names)
plt.title('Diagnostic Accuracy Matrix')
plt.xlabel('AI Prediction')
plt.ylabel('Actual Condition')
plt.tight_layout()

# Ensure assets folder exists
os.makedirs(os.path.dirname(image_path), exist_ok=True)
plt.savefig(image_path)
print(f"📊 Confusion Matrix saved to: {image_path}")
