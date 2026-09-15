import pandas as pd 
 

def Load_csv(file_path: str) -> pd.DataFrame:
    """"
    Load data from a csv file and returns a pandas DataFrame
    """
    return pd.read_csv(file_path)