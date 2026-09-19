import pandas as pd


def get_dtype(df):
    """
    Returns the data type of each column.
    """
    return df.dtypes.astype(str).to_dict()


def get_shape(df):
    """
    Returns the number of rows and columns.
    """
    rows, columns = df.shape

    return {
        "rows": rows,
        "columns": columns
    }


def get_columns(df):
    """
    Returns the column names.
    """
    return list(df.columns)


def get_head(df, n=10):
    """
    Returns the first n rows.
    Used for the detailed report, not necessarily for the LLM.
    """
    return df.head(n).to_dict(orient="records")


def get_tail(df, n=10):
    """
    Returns the last n rows.
    Used for the detailed report.
    """
    return df.tail(n).to_dict(orient="records")


def get_unique_values(df):
    """
    Returns unique-value information for categorical columns.

    For columns with a small number of unique values,
    actual values are returned.

    For high-cardinality columns, only the count is returned.
    """

    unique_values = {}

    for col in df.select_dtypes(
        include=["object", "category", "str", "bool"]
    ).columns:

        values = df[col].dropna().unique()
        count = len(values)

        if count <= 20:
            unique_values[col] = {
                "unique_count": count,
                "values": values.tolist()
            }
        else:
            unique_values[col] = {
                "unique_count": count
            }

    return unique_values


def get_missing_values(df):
    """
    Returns missing-value count and percentage for each column.
    """

    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    result = {}

    for column in df.columns:

        if missing_values[column] > 0:

            result[column] = {
                "missing_count": int(missing_values[column]),
                "missing_percentage": round(
                    float(missing_percentage[column]), 2
                )
            }

    return result


def get_overview(df, n=10):
    """
    Returns a compact dataset overview.

    The result is designed to be passed to the LLM.
    Large DataFrame objects are not returned.
    """

    shape = get_shape(df)
    dtypes = get_dtype(df)
    columns = get_columns(df)
    missing = get_missing_values(df)
    unique_values = get_unique_values(df)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "str", "bool"]
    ).columns.tolist()

    total_missing = int(df.isnull().sum().sum())

    return {
        "dataset": {
            "rows": shape["rows"],
            "columns": shape["columns"]
        },

        "columns": columns,

        "data_types": dtypes,

        "column_types": {
            "numeric": numeric_columns,
            "categorical": categorical_columns
        },

        "missing_values": {
            "total": total_missing,
            "columns_affected": missing
        },

        "unique_values": unique_values
    }