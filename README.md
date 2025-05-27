# PollPredictAI Hyperparameter Tuning

This repository contains data and scripts to predict voter party affiliation.
The `model_tuning.py` script demonstrates how to optimize classifiers using
`RandomizedSearchCV`.

## Requirements

- Python 3.8+
- pandas
- scikit-learn

Make sure `Voter_Data.csv` is located in the project root. Install the required
packages, then run:

```bash
python model_tuning.py
```

The script prints the best parameters and cross validated accuracy for Random
Forest and Gradient Boosting classifiers.
