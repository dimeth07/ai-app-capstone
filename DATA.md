# DATA

## Source
- Where: UCI Machine Learning Repository - SMS Spam Collection Dataset
- Licence: CC BY 4.0 (Free for research and commercial use)
- Downloaded on: September 20, 2026

## Size
- Total rows: 5,574
- Rows per class:

| Class | Count |
|---|---|
| Ham (legitimate) | 4,825 |
| Spam | 749 |

## Known flaws
- The dataset is highly imbalanced: only 13% of the messages are spam.
- The text contains many typos, slang, and special characters that need cleaning.
- There are some duplicate rows in the dataset.

## Split strategy
- Will use: Stratified split (80% train, 20% test)
- Why: Because the classes are imbalanced, I need to keep the same ratio of spam to ham in both the training and testing sets so the model is evaluated fairly.
