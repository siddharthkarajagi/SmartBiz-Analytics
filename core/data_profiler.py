import pandas as pd


def profile_data(df):

    profile = {}

    profile["rows"] = df.shape[0]
    profile["columns"] = df.shape[1]

    profile["missing_values"] = int(df.isna().sum().sum())

    profile["duplicate_rows"] = int(df.duplicated().sum())

    profile["column_info"] = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Unique Values": df.nunique().values
    })

    profile["numeric_columns"] = df.select_dtypes(
        include="number"
    ).columns.tolist()

    profile["categorical_columns"] = df.select_dtypes(
        include="object"
    ).columns.tolist()

    return profile