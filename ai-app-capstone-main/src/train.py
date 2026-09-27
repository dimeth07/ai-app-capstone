import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from data.dataset import DefectDataset

print("1. Loading data...")
df = pd.read_csv("data/raw/defects.csv")
print("Rows loaded:", len(df))

print("2. Cleaning data...")
# Convert Fahrenheit to Celsius
df["temperature_c"] = df["temperature_c"].apply(lambda x: (x - 32) * 5 / 9 if x > 120 else x)
# Normalize material codes
df["material_code"] = df["material_code"].str.upper().str.replace("-", "", regex=False)
df["material_code"] = df["material_code"].map({"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4})
# Remove duplicates
df = df.drop_duplicates(subset=["batch_id", "timestamp"])
# Fill missing values with the median
df = df.fillna(df.median(numeric_only=True))
print("Rows after cleaning:", len(df))

print("3. Splitting data...")
train_idx = torch.arange(int(len(df) * 0.7))
val_idx = torch.arange(int(len(df) * 0.7), int(len(df) * 0.85))
test_idx = torch.arange(int(len(df) * 0.85), len(df))

print("4. Creating dataset and DataLoader...")
ds = DefectDataset(df)
train_loader = DataLoader(Subset(ds, train_idx), batch_size=32, shuffle=True)

print("5. Running one epoch of training...")
for batch_x, batch_y in train_loader:
    print("Batch X shape:", batch_x.shape)
    print("Batch Y shape:", batch_y.shape)
    break

print("SUCCESS! The data pipeline works.")