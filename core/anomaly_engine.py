import pandas as pd
import numpy as np


def detect_anomalies(df):

    result = df.copy()
    anomaly_columns = []

    numeric_columns = result.select_dtypes(
        include="number"
    ).columns.tolist()

    for column in numeric_columns:

        values = result[column].dropna()

        if len(values) < 5:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower_limit = q1 - (1.5 * iqr)
        upper_limit = q3 + (1.5 * iqr)

        flag_column = f"{column}_anomaly"

        result[flag_column] = (
            (result[column] < lower_limit) |
            (result[column] > upper_limit)
        )

        anomaly_columns.append(flag_column)

    if anomaly_columns:

        result["Anomaly"] = result[anomaly_columns].any(axis=1)

    else:

        result["Anomaly"] = False

    return result