import pandas as pd


def calculate_business_metrics(df):

    metrics = {}

    metrics["total_rows"] = len(df)

    metrics["numeric_columns"] = df.select_dtypes(
        include="number"
    ).columns.tolist()

    metrics["categorical_columns"] = df.select_dtypes(
        include="object"
    ).columns.tolist()

    # Numeric summaries
    numeric_summary = {}

    for column in metrics["numeric_columns"]:

        numeric_summary[column] = {
            "sum": float(df[column].sum()),
            "average": float(df[column].mean()),
            "minimum": float(df[column].min()),
            "maximum": float(df[column].max())
        }

    metrics["numeric_summary"] = numeric_summary

    # ----------------------------------------------
    # BUSINESS KPI DETECTION
    # ----------------------------------------------

    kpi_keywords = {

        "Revenue": [
            "sales",
            "revenue",
            "amount",
            "turnover"
        ],

        "Profit": [
            "profit",
            "net_profit",
            "gross_profit"
        ],

        "Quantity": [
            "quantity",
            "qty",
            "units"
        ],

        "Cost": [
            "cost",
            "expense",
            "spend"
        ]
    }

    detected_kpis = {}

    normalized_columns = {
        column: str(column).lower().replace(" ", "_")
        for column in df.columns
    }

    for kpi, keywords in kpi_keywords.items():

        for column, normalized_column in normalized_columns.items():

            if any(
                keyword in normalized_column
                for keyword in keywords
            ):

                detected_kpis[kpi] = {
                    "column": column,
                    "total": float(df[column].sum()),
                    "average": float(df[column].mean())
                }

                break

    metrics["detected_kpis"] = detected_kpis

    return metrics