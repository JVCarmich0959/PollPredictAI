"""Hyperparameter tuning script for party affiliation prediction.

This script demonstrates how to train and tune ``RandomForestClassifier`` and
``GradientBoostingClassifier`` models on the voter demographics dataset.  The
focus is predicting the ``party_cd`` label with maximum accuracy.

Usage::

    python model_tuning.py

The script expects ``Voter_Data.csv`` in the same directory and prints the best
parameters and cross validated accuracy for each model.
"""

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

def load_data(csv_path: str):
    """Load the voter data from a CSV file."""
    data = pd.read_csv(csv_path)
    return data


def build_preprocessing_pipeline(df: pd.DataFrame):
    """Construct preprocessing transformers for numeric and categorical features."""
    target = 'party_cd'
    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    return X, y, preprocessor


def tune_model(model, param_dist, X, y, preprocessor, n_iter=20, cv=5, random_state=42):
    """Run RandomizedSearchCV for a given model and parameter distribution."""
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    search = RandomizedSearchCV(pipe, param_dist, n_iter=n_iter, cv=cv, scoring='accuracy',
                               random_state=random_state, n_jobs=-1)
    search.fit(X, y)
    return search


def main():
    df = load_data('Voter_Data.csv')
    X, y, preprocessor = build_preprocessing_pipeline(df)

    rf_params = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 5, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    gb_params = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7]
    }

    print("Tuning RandomForestClassifier...")
    rf_search = tune_model(RandomForestClassifier(random_state=42), rf_params, X, y, preprocessor)
    print(f"Best RF accuracy: {rf_search.best_score_:.4f}")
    print("Best RF params:", rf_search.best_params_)

    print("\nTuning GradientBoostingClassifier...")
    gb_search = tune_model(GradientBoostingClassifier(random_state=42), gb_params, X, y, preprocessor)
    print(f"Best GB accuracy: {gb_search.best_score_:.4f}")
    print("Best GB params:", gb_search.best_params_)

if __name__ == '__main__':
    main()
