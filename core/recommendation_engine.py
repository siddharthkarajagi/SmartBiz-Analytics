
def generate_recommendations(df, analytics, anomaly_count, business_mode=None):

    recommendations = []

    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    if missing:
        recommendations.append(
            f"Review {missing:,} missing values before analysis."
        )
    else:
        recommendations.append(
            "No missing values found in the analyzed dataset."
        )

    if duplicates:
        recommendations.append(
            f"Review {duplicates:,} duplicate rows."
        )

    if anomaly_count:
        recommendations.append(
            f"Investigate {anomaly_count:,} potentially unusual records."
        )

    mode = "General Business"

    if isinstance(business_mode, dict):
        mode = business_mode.get("mode", mode)
    elif isinstance(business_mode, str):
        mode = business_mode

    mode_recommendations = {
        "Sales": "Review sales trends and identify high-performing products or categories.",
        "Marketing": "Compare campaign performance and investigate low-conversion campaigns.",
        "Finance": "Review expenses, profit trends, and unusually high financial values.",
        "HR": "Analyze workforce trends and investigate differences across departments.",
        "Inventory": "Review stock levels and identify potential overstock or stockout risks.",
        "Customer": "Analyze customer segments and investigate retention or churn patterns."
    }

    recommendations.append(
        f"Business Mode: {mode}."
    )

    recommendations.append(
        mode_recommendations.get(
            mode,
            "Review key metrics and trends to identify business improvement opportunities."
        )
    )

    numeric_columns = analytics.get("numeric_columns", [])

    for column in numeric_columns:
        series = df[column].dropna()

        if not series.empty and series.mean() != 0:
            if series.max() > series.mean() * 2:
                recommendations.append(
                    f"Review unusually high values in '{column}' "
                    "to understand their business impact."
                )

    return list(dict.fromkeys(recommendations))