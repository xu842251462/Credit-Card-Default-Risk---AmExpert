# Credit-Card-Default-Risk---AmExpert

This repository contains a machine-learning notebook for predicting credit card default risk (AmExpert challenge style workflow).

## Project structure

- `Credit-Card-Default-Risk---AmExpert-CodeLab-main/Credit Card Default Risk - Prediction.ipynb`: main end-to-end modeling notebook.
- `Credit-Card-Default-Risk---AmExpert-CodeLab-main/dataset/train.csv`: training data with target column `credit_card_default`.
- `Credit-Card-Default-Risk---AmExpert-CodeLab-main/dataset/test.csv`: test data used for inference.
- `Credit-Card-Default-Risk---AmExpert-CodeLab-main/dataset/sample_submission.csv`: sample submission template.
- `scripts/project_check.py`: lightweight local health-check script.

## Quick project check

Run:

```bash
python scripts/project_check.py
```

The check validates:

1. expected files are present,
2. dataset row/column counts are readable,
3. target column location (`credit_card_default`) is consistent,
4. key Python ML dependencies are installed,
5. notebook import statements can be parsed.

## Environment notes

The notebook uses libraries such as `pandas`, `scikit-learn`, `xgboost`, `imbalanced-learn`, and `tensorflow`. Install these before trying to execute the notebook end-to-end.
