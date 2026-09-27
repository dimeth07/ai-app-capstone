# Data Card - defects-v1

## 1. Provenance and compliance
- Source: Synthetic dataset generated for the AI Application Development course (Week 3 lab).
- Licence: Educational / public use.

## 2. Scale and structure
- Rows: 12,480 (before cleaning) → 11,668 (after cleaning).
- Features: 6 (temperature_c, material_code, age, pressure_kpa, humidity_pct, thickness_mm).
- Target classes: 5 (labels 0, 1, 2, 3, 4).
- Split: 70% train, 15% validation, 15% test.

## 3. Quality and known issues
- Missing values: Approximately 3% of the `age` column is missing. Filled with the training median.
- Outliers: About 1% of `temperature_c` values are recorded in Fahrenheit (>120). Converted to Celsius.
- Duplicates: Duplicate rows on the natural key (`batch_id` + `timestamp`) were removed.
- Category coding: `material_code` was normalized (uppercase, removed hyphens) and mapped to numeric codes (A1=0, A2=1, B1=2, B2=3, C1=4).

## 4. Augmentation and imbalance
- Class imbalance: The dataset is imbalanced, with classes 3 and 4 appearing only 5% of the time.
- Handling: A WeightedRandomSampler is used during training to ensure the model sees minority classes equally.
- Augmentation: Gaussian noise is added to the training features only. Validation and test sets are not augmented.

## 5. Bias and limits of use
- This is a synthetic dataset generated for practice. It does not represent real production line data.
- Any model trained on this data cannot be used to make real-world claims about manufacturing defects.
- The class imbalance means accuracy is a misleading metric; macro-F1 should be used instead.