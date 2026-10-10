import pandas as pd


def analyze_root_causes(df, target_column):

    results = []

    if target_column not in df.columns:
        return results

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        return results

    overall_average = df[target_column].mean()

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns.tolist()

    for column in categorical_columns:

        if df[column].nunique() > 30:
            continue

        grouped = (
            df.groupby(column, dropna=False)[target_column]
            .agg(["mean", "count"])
            .reset_index()
        )

        grouped = grouped.dropna(
            subset=["mean"]
        )

        if grouped.empty:
            continue

        grouped["difference"] = (
            grouped["mean"] - overall_average
        )

        grouped["impact"] = (
            grouped["difference"].abs()
        )

        grouped["Driver"] = column

        grouped["Overall Average"] = overall_average

        results.append(
            grouped
        )

    if not results:
        return pd.DataFrame()

    result = pd.concat(
        results,
        ignore_index=True
    )

    result = result.sort_values(
        "impact",
        ascending=False
    )

    return result.head(15)