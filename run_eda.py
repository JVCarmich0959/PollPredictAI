import pandas as pd
from auto_eda import perform_eda


def main():
    df = pd.read_csv('Voter_Data.csv')
    perform_eda(df)


if __name__ == '__main__':
    main()
