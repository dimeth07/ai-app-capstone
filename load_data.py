# load_data.py
# Proves the data loads. 
# Run: python load_data.py

import pandas as pd
import numpy as np
import os

# ==========================================
# STEP 1: Create fake data (so you can test)
# ==========================================
print("Generating 1,200 fake samples for testing...")
np.random.seed(42) # Keeps the random numbers the same every time

n_samples = 1200

# Create a dictionary with fake data
data = {
    'id': range(1, n_samples + 1),
    'feature_1': np.random.normal(0, 1, n_samples), # Random numbers
    'feature_2': np.random.randint(0, 100, n_samples), # Random integers
    'category': np.random.choice(['A', 'B', 'C'], n_samples), # Random categories
    'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]) # 0 or 1
}

df = pd.DataFrame(data)

# ==========================================
# STEP 2: Save it to a CSV file
# ==========================================
# Create the folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Save the fake data to a file
df.to_csv("data/raw/sample_data.csv", index=False)
print("Fake data saved to: data/raw/sample_data.csv\n")

# ==========================================
# STEP 3: "Load" the data (this is the real proof)
# ==========================================
# In your real project, you will change this line to point to your real data
df = pd.read_csv("data/raw/sample_data.csv")

# ==========================================
# STEP 4: Print proof that it loaded
# ==========================================
print("--- PROOF DATA LOADED ---")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print()

print("--- DATA INFO ---")
print(df.info())
print()

print("--- DATA STATS ---")
print(df.describe())
print()

print("--- MISSING VALUES ---")
print(df.isna().sum())
print()

print("--- FIRST ROW ---")
print(df.iloc[0])
