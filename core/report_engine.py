import pandas as pd


def create_business_report(
    df,
    profile,
    analytics,
    business_mode,
    anomaly_count,
    insights,
    recommendations
):

    report = []

    report.append(
        "SMARTBIZ ANALYTICS — BUSINESS REPORT"
    )

    report.append(
        "=" * 50
    )

    report.append(
        f"Business Mode: {business_mode.get('mode', 'General Business')}"
    )

    report.append(
        f"Total Records: {len(df):,}"
    )

    report.append(
        f"Total Columns: {len(df.columns):,}"
    )

    report.append(
        f"Missing Values: {profile['missing_values']:,}"
    )

    report.append(
        f"Duplicate Rows: {profile['duplicate_rows']:,}"
    )

    report.append(
        f"Potential Anomalies: {anomaly_count:,}"
    )

    report.append(
        ""
    )

    report.append(
        "KEY BUSINESS METRICS"
    )

    report.append(
        "-" * 30
    )

    detected_kpis = analytics.get(
        "detected_kpis",
        {}
    )

    for kpi, values in detected_kpis.items():

        report.append(
            f"{kpi}: "
            f"Total = {values['total']:,.2f}, "
            f"Average = {values['average']:,.2f}"
        )

    report.append("")

    report.append(
        "BUSINESS INSIGHTS"
    )

    report.append(
        "-" * 30
    )

    for insight in insights:

        report.append(
            f"- {insight}"
        )

    report.append("")

    report.append(
        "BUSINESS RECOMMENDATIONS"
    )

    report.append(
        "-" * 30
    )

    for recommendation in recommendations:

        report.append(
            f"- {recommendation}"
        )

    return "\n".join(report)