from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your dataset
df = pd.read_csv("heart.csv")

# Define label column
label = 'HeartDisease'

# Split into train and test
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Train AutoGluon model
predictor = TabularPredictor(label=label, eval_metric='accuracy').fit(
    train_data, 
    presets='best_quality',   # Use best quality models (e.g., ensembles)
    time_limit=300            # Train for up to 5 minutes
)

# Save for later use
predictor_path = predictor.path
print(f"\nModel saved to: {predictor_path}")
