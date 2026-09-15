import pandas as pd


def get_dtype(df):
    """
    returns the data type of each columns in the given dataframe
    """
    return df.dtypes

def get_shape(df):
    """
    Returns the shape of the given dataframe
    """
    return df.shape

def get_columns(df):
    """
    Returns the columns of the given dataframe
    """
    return df.columns

def get_head(df,n=10):
    """
    Returns the first n rows of the given dataframe default n =10
    """
    return df.head(n)

def get_unique_values(df):
    """
    Returns the unique values of object or category columns in the given dataframe
    """
    unique_values = {}
    for col in df.select_dtypes(include=['object','category','str','bool']).columns:
        unique_values[col] = df[col].unique()
    return unique_values

def get_tail(df,n=10):
    """
    Returns the last n rows of the given dataframe default n =10
    """
    return df.tail(n)

def get_missing_values(df):
    """
    Returns the number of mising values in each coulmn with the percentage of the missing values 
    """
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values/len(df)) *100
    missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage':missing_percentage})
    return missing_df

def get_overview(df,n=10):
    """
    Returns the overview of the given dataframe includes shape,columns,missing values,data types,unique values,head and tail of dataframe
    """
    return{
        "Shape" : get_shape(df),
        "Columns" : get_columns(df),
        "Data Types" : get_dtype(df),
        "Data Frame Head" : get_head(df,n),
        "Data Frame Tail": get_tail(df,n),
        "Unique Values" : get_unique_values(df),
        "Missing Values" : get_missing_values(df),
        
        
    }    


# def get_summary(df):
#     """"
#     Return the mean mode median and standard deviation of the given dataframe only numeric columns   
#     """
#     return df.describe(include='numeric')
    
