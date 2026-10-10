import pandas as pd


def generate_insights(df, analytics, anomaly_count):

    insights = []

    # ==================================================
    # DATASET OVERVIEW
    # ==================================================

    insights.append(
        f"The dataset contains {len(df):,} records "
        f"across {len(df.columns)} columns."
    )

    # ==================================================
    # NUMERIC INSIGHTS
    # ==================================================

    numeric_columns = analytics["numeric_columns"]

    for column in numeric_columns:

        average = df[column].mean()
        maximum = df[column].max()

        insights.append(
            f"{column}: average is {average:,.2f} "
            f"and maximum is {maximum:,.2f}."
        )

    # ==================================================
    # ANOMALY INSIGHT
    # ==================================================

    if anomaly_count > 0:

        insights.append(
            f"{anomaly_count:,} potentially unusual "
            "record(s) were detected and should be reviewed."
        )

    else:

        insights.append(
            "No potentially unusual records were detected."
        )

    # ==================================================
    # MISSING VALUE INSIGHT
    # ==================================================

    missing_values = int(
        df.isna().sum().sum()
    )

    if missing_values > 0:

        insights.append(
            f"The dataset still contains "
            f"{missing_values:,} missing value(s)."
        )

    else:

        insights.append(
            "No missing values remain in the analyzed dataset."
        )

    return insights


# ==================================================
# EXECUTIVE SUMMARY
# ==================================================

def generate_executive_summary(
    df,
    analytics,
    anomaly_count,
    business_mode
):

    summary = []

    mode = business_mode.get(
        "mode",
        "General Business"
    )

    summary.append(
        f"SmartBiz identified this dataset as "
        f"{mode} data containing "
        f"{len(df):,} records."
    )

    detected_kpis = analytics.get(
        "detected_kpis",
        {}
    )

    if detected_kpis:

        kpi_names = ", ".join(
            detected_kpis.keys()
        )

        summary.append(
            f"Key business metrics detected: "
            f"{kpi_names}."
        )

    if anomaly_count > 0:

        summary.append(
            f"{anomaly_count:,} potentially unusual "
            "records require further review."
        )

    else:

        summary.append(
            "No potentially unusual records were detected."
        )

    return " ".join(summary)