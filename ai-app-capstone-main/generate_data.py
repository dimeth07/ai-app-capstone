import pandas as pd
import numpy as np
import os

print("Generating 12,480 fake rows...")
np.random.seed(42)

n = 12480

df = pd.DataFrame({
    "batch_id": np.random.randint(1, 1000, n),
    "timestamp": np.random.randint(1, 500, n),
    "temperature_c": np.random.normal(45, 5, n),
    "material_code": np.random.choice(["A1", "A2", "B1", "B2", "C1"], n),
    "age": np.random.normal(20, 3, n),
    "pressure_kpa": np.random.normal(101, 2, n),
    "humidity_pct": np.random.normal(50, 10, n),
    "thickness_mm": np.random.normal(5, 1, n),
    "label": np.random.choice([0, 1, 2, 3, 4], n, p=[0.4, 0.3, 0.2, 0.05, 0.05])
})

# Plant some defects (missing values, Fahrenheit temperatures)
df.loc[df.sample(frac=0.03).index, "age"] = np.nan
df.loc[df.sample(frac=0.01).index, "temperature_c"] = 200

# Create the folder and save
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/defects.csv", index=False)
print("Done! Saved to data/raw/defects.csv")
print("Rows:", len(df))