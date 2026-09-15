import pandas as pd


def numerical_summary(df :pd.DataFrame):
    """
    Returnd the summary of the numerical columns in the given dataframe includes count mean, std,min,max ,25%,50%,75%
    """
    return df.describe()


