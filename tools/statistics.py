import pandas as pd


def numerical_summary(df: pd.DataFrame):
    """
    Calculate numerical statistics.

    Returns:
        report:
            Full statistics for report generation.
        observation:
            Compact information passed to the LLM.
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {
            "report": {},
            "observation": {
                "columns_analyzed": 0,
                "message": "No numerical columns found."
            }
        }

    report = {}

    for column in numeric_df.columns:

        series = numeric_df[column].dropna()

        report[column] = {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "q25": round(float(series.quantile(0.25)), 4),
            "q75": round(float(series.quantile(0.75)), 4)
        }

    # Compact information for the LLM
    observation = {
        "columns_analyzed": len(numeric_df.columns),
        "columns": list(numeric_df.columns),
        "metrics": {
            column: {
                "mean": stats["mean"],
                "median": stats["median"],
                "min": stats["min"],
                "max": stats["max"]
            }
            for column, stats in report.items()
        }
    }

    return {
        "report": report,
        "observation": observation
    }