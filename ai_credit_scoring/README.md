# AI Credit Scoring

Simple machine-learning project that predicts loan default probability and converts it into a 300-850 credit score.

## Setup
```
pip install -r requirements.txt
```

## Run
```
python credit_scoring.py
```

## Output
- Model metrics (accuracy, ROC-AUC, confusion matrix) in the terminal
- roc_curve.png and feature_importance.png
- credit_model.joblib (saved best model)
- Credit score and risk band for a sample applicant

## Using real data
Replace `load_data()` in credit_scoring.py with `pd.read_csv("your.csv")`.
The file needs a binary `default` column (1 = defaulted), and the FEATURES list must match your column names.
