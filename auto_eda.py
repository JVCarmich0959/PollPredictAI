import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def perform_eda(df: pd.DataFrame) -> None:
    """Perform a basic exploratory data analysis (EDA) on a DataFrame.

    The analysis includes:
    - Summary information about the dataframe
    - Missing value heatmap
    - Distribution plots for numerical and categorical features
    - Correlation heatmap for numerical variables

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to analyze.
    """

    # Basic dataset overview
    print("Data Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())

    # Set seaborn style
    sns.set(style="whitegrid")

    # Plot missing value heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.show()

    # Distribution plots for numerical features
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        df[num_cols].hist(bins=15, figsize=(15, 10), layout=(-1, 3))
        plt.suptitle("Distribution of Numerical Features")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # Distribution plots for categorical features
    cat_cols = df.select_dtypes(exclude=['number']).columns
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        df[col].value_counts().plot(kind='bar')
        plt.title(f"Distribution of {col}")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # Correlation heatmap for numerical features
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
