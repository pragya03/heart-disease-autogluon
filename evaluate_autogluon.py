from autogluon.tabular import TabularPredictor
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load test data
df = pd.read_csv("heart.csv")
label = 'HeartDisease'

# Load model
predictor = TabularPredictor.load("AutogluonModels/ag-20250719_090238")  # Replace with actual path

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Predict
y_true = test_data[label]
y_pred = predictor.predict(test_data.drop(columns=[label]))

# Accuracy
print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Leaderboard
print("\nAutoGluon Model Leaderboard:")
print(predictor.leaderboard(test_data, silent=True))
