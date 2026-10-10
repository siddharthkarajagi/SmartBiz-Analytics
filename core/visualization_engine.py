import pandas as pd
import plotly.express as px


def create_visualizations(df):

    charts = []

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns.tolist()

    # Category + Numeric → Bar Chart
    if categorical_columns and numeric_columns:

        category = categorical_columns[0]
        numeric = numeric_columns[0]

        chart_data = (
            df.groupby(category, dropna=False)[numeric]
            .sum()
            .reset_index()
            .sort_values(numeric, ascending=False)
            .head(15)
        )

        fig = px.bar(
            chart_data,
            x=category,
            y=numeric,
            title=f"{numeric} by {category}"
        )

        charts.append(fig)

    # Numeric → Histogram
    if numeric_columns:

        numeric = numeric_columns[0]

        fig = px.histogram(
            df,
            x=numeric,
            title=f"Distribution of {numeric}"
        )

        charts.append(fig)

    # Two Numeric Columns → Scatter
    if len(numeric_columns) >= 2:

        x_column = numeric_columns[0]
        y_column = numeric_columns[1]

        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            title=f"{y_column} vs {x_column}"
        )

        charts.append(fig)

    return charts