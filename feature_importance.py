from autogluon.tabular import TabularPredictor
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset and trained model
df = pd.read_csv("heart.csv")
label = 'HeartDisease'

predictor = TabularPredictor.load("AutogluonModels/ag-20250719_083456")  # Replace with your actual folder

# Compute feature importance
importance_df = predictor.feature_importance(df)

# Show the DataFrame
print("\n📊 Feature Importance:\n", importance_df)

# Plot the top features
importance_df = importance_df.sort_values("importance", ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(importance_df.index, importance_df["importance"], color='teal')
plt.xlabel("Importance Score")
plt.title("🔍 Feature Importance for Heart Disease Prediction")
plt.tight_layout()
plt.show()
