import pandas as pd

# Load the dataset (make sure heart.csv is in your working directory)
df = pd.read_csv("heart.csv")

# Preview the data
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
