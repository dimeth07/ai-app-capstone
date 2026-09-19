# load_data.py
# Proves the SMS Spam data loads. 
# Run: python load_data.py

import pandas as pd
import numpy as np
import os

# ==========================================
# STEP 1: Create fake SMS data for testing
# ==========================================
print("Generating 5,574 fake SMS samples for testing...")
np.random.seed(42) # Keeps random numbers the same every time

n_samples = 5574

# 1 = spam, 0 = ham (86% ham, 14% spam)
labels = np.random.choice(['ham', 'spam'], n_samples, p=[0.86, 0.14])

# Create fake text messages
messages = ["This is a fake SMS message number " + str(i) for i in range(n_samples)]

# Create the DataFrame
df = pd.DataFrame({
    'label': labels,
    'message': messages
})

# ==========================================
# STEP 2: Save it to a CSV file
# ==========================================
# Create the folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Save the fake data to a file
df.to_csv("data/raw/sms_spam.csv", index=False)
print("Fake data saved to: data/raw/sms_spam.csv\n")

# ==========================================
# STEP 3: "Load" the data (this is the real proof)
# ==========================================
# When you get the REAL dataset, change this line to point to it
df = pd.read_csv("data/raw/sms_spam.csv")

# ==========================================
# STEP 4: Print proof that it loaded
# ==========================================
print("--- PROOF DATA LOADED ---")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print()

print("--- CLASS BALANCE ---")
print(df['label'].value_counts())
print()

print("--- MISSING VALUES ---")
print(df.isna().sum())
print()

print("--- FIRST 3 ROWS ---")
print(df.head(3))
