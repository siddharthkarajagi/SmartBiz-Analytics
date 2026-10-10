
import streamlit as st
import pandas as pd
import numpy as np

from core.data_profiler import profile_data
from core.data_cleaner import clean_data
from core.analytics_engine import calculate_business_metrics
from core.visualization_engine import create_visualizations
from core.anomaly_engine import detect_anomalies
from core.prediction_engine import build_prediction_model
from core.insight_engine import (
    generate_insights,
    generate_executive_summary
)
from core.recommendation_engine import generate_recommendations
from core.data_quality_engine import calculate_data_quality_score
from core.business_mode_engine import detect_business_mode
from core.forecast_engine import create_forecast
from core.segmentation_engine import create_segments
from core.qa_engine import answer_question
from core.root_cause_engine import analyze_root_causes
from core.report_engine import create_business_report


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SmartBiz Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        margin-bottom: 0;
    }

    .subtitle {
        color: #808495;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 15px;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<p class="main-title">📊 SmartBiz Analytics</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Automated Business Intelligence & Prediction Platform'
    '</p>',
    unsafe_allow_html=True
)

st.caption(
    "Upload business data to explore data quality, "
    "analytics, anomalies, predictions and business insights."
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_number(value, default=0):
    """Convert a value to a safe numeric value."""

    try:
        result = float(value)

        if np.isfinite(result):
            return result

        return default

    except (TypeError, ValueError):
        return default


def format_number(value):
    """Format numeric values for display."""

    try:
        return f"{float(value):,.2f}"

    except (TypeError, ValueError):
        return str(value)


def get_quality_status(score):

    if score >= 90:
        return "Excellent"

    elif score >= 75:
        return "Good"

    elif score >= 50:
        return "Needs Improvement"

    return "Poor"


def get_forecast_date_columns(df):

    date_columns = []

    for column in df.columns:

        column_name = str(column).lower()

        if any(
            keyword in column_name
            for keyword in [
                "date",
                "time",
                "day",
                "month",
                "year"
            ]
        ):

            parsed = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            if parsed.notna().sum() >= 10:
                date_columns.append(column)

    return date_columns


def safe_download_name(filename):

    return (
        str(filename)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ SmartBiz Control Center")

st.sidebar.markdown("---")

st.sidebar.subheader("📁 Upload Business Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("🔄 Analytics Pipeline")

pipeline_steps = [
    "1. Upload Data",
    "2. Data Cleaning",
    "3. Data Understanding",
    "4. Business Analytics",
    "5. Visualization",
    "6. Anomaly Detection",
    "7. Prediction",
    "8. Insights",
    "9. Recommendations",
    "10. Reporting"
]

for step in pipeline_steps:
    st.sidebar.write(f"✓ {step}")

st.sidebar.markdown("---")

st.sidebar.caption(
    "SmartBiz Analytics | Business Intelligence Platform"
)


# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_file is None:

    st.info(
        "👈 Upload a CSV or Excel file from the sidebar "
        "to start your business analysis."
    )

    st.subheader("🚀 What SmartBiz Can Do")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🧹 Data Quality")
        st.write(
            "Inspect missing values, duplicate rows, "
            "column types and data quality."
        )

    with col2:
        st.markdown("### 📈 Business Analytics")
        st.write(
            "Calculate business metrics and generate "
            "charts automatically."
        )

    with col3:
        st.markdown("### 🤖 Intelligence")
        st.write(
            "Explore anomalies, machine learning predictions, "
            "forecasts and recommendations."
        )

    st.markdown("---")

    st.subheader("📂 Supported Files")

    st.write(
        "CSV (.csv) and Excel (.xlsx)"
    )

    st.warning(
        "For reliable predictions and forecasts, use "
        "a dataset with enough valid records and suitable columns."
    )

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

try:

    filename = uploaded_file.name

    if filename.lower().endswith(".csv"):

        uploaded_df = pd.read_csv(uploaded_file)

    else:

        uploaded_df = pd.read_excel(uploaded_file)

    if uploaded_df.empty:

        st.error(
            "The uploaded file contains no data rows. "
            "Please upload another dataset."
        )

        st.stop()

    if len(uploaded_df.columns) == 0:

        st.error(
            "No columns were found in the uploaded file."
        )

        st.stop()

    if uploaded_df.columns.duplicated().any():

        st.warning(
            "Duplicate column names were found. "
            "Some analyses may need corrected column names."
        )

    st.sidebar.success(f"Loaded: {filename}")

except Exception as error:

    st.error(
        f"Could not read the uploaded file: {error}"
    )

    st.stop()


# =========================================================
# DATA CLEANING
# =========================================================

try:

    df = clean_data(uploaded_df.copy())

    if df is None or df.empty:

        st.error(
            "Data cleaning did not produce a usable dataset."
        )

        st.stop()

    df = df.copy()

    # Remove entirely empty columns.
    df = df.dropna(
        axis=1,
        how="all"
    )

    if df.empty or len(df.columns) == 0:

        st.error(
            "The dataset contains no usable data after cleaning."
        )

        st.stop()

    # Convert date-like text columns when enough values parse.
    for column in df.select_dtypes(
        include=["object", "string"]
    ).columns:

        column_name = str(column).lower()

        if any(
            keyword in column_name
            for keyword in [
                "date",
                "datetime",
                "timestamp"
            ]
        ):

            parsed_dates = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            if parsed_dates.notna().sum() >= max(
                3,
                int(len(df) * 0.5)
            ):

                df[column] = parsed_dates

except Exception as error:

    st.error(
        f"Data cleaning failed: {error}"
    )

    st.stop()


# =========================================================
# PROFILE AND BUSINESS METRICS
# =========================================================

try:

    profile = profile_data(df)

    analytics = calculate_business_metrics(df)

    business_mode = detect_business_mode(df)

    quality_score = calculate_data_quality_score(df)

    quality_status = get_quality_status(
        quality_score
    )

except Exception as error:

    st.error(
        f"Could not analyze the dataset: {error}"
    )

    st.stop()


# =========================================================
# ANOMALY DETECTION
# =========================================================

try:

    anomaly_df = detect_anomalies(df)

    if "Anomaly" in anomaly_df.columns:

        anomaly_count = int(
            anomaly_df["Anomaly"].fillna(False).sum()
        )

    else:

        anomaly_count = 0

except Exception as error:

    anomaly_df = df.copy()

    anomaly_df["Anomaly"] = False

    anomaly_count = 0

    st.warning(
        f"Anomaly detection could not complete: {error}"
    )


# =========================================================
# INSIGHTS AND RECOMMENDATIONS
# =========================================================

try:

    insights = generate_insights(
        df,
        analytics,
        anomaly_count
    )

except Exception as error:

    insights = [
        f"Automatic insight generation encountered an issue: {error}"
    ]


try:

    recommendations = generate_recommendations(
        df,
        analytics,
        anomaly_count,
        business_mode
    )

except Exception as error:

    recommendations = [
        f"Recommendation generation encountered an issue: {error}"
    ]


try:

    executive_summary = generate_executive_summary(
        df,
        analytics,
        anomaly_count,
        business_mode
    )

except Exception:

    executive_summary = (
        f"SmartBiz analyzed {len(df):,} records "
        f"across {len(df.columns)} columns. "
        f"Detected business mode: "
        f"{business_mode.get('mode', 'General Business')}."
    )


# =========================================================
# SIDEBAR DATA SUMMARY
# =========================================================

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Dataset Summary")

st.sidebar.metric(
    "Rows",
    f"{len(df):,}"
)

st.sidebar.metric(
    "Columns",
    f"{len(df.columns):,}"
)

st.sidebar.metric(
    "Data Quality",
    f"{quality_score:.2f}%"
)

st.sidebar.metric(
    "Potential Anomalies",
    f"{anomaly_count:,}"
)

st.sidebar.info(
    f"Business Mode: {business_mode.get('mode', 'General Business')}"
)


# =========================================================
# TABS
# =========================================================

tabs = st.tabs([
    "🏠 Overview",
    "🧹 Data Quality",
    "📈 Analytics",
    "🚨 Anomalies",
    "💡 Insights",
    "🤖 Prediction",
    "🔮 Forecast",
    "👥 Segmentation",
    "💬 Ask Your Data",
    "📤 Export"
])


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================

with tabs[0]:

    st.header("Executive Overview")

    st.write(executive_summary)

    st.markdown("---")

    kpis = analytics.get(
        "detected_kpis",
        {}
    )

    if kpis:

        st.subheader("Key Business Metrics")

        kpi_items = list(kpis.items())

        for start in range(0, len(kpi_items), 4):

            current_items = kpi_items[
                start:start + 4
            ]

            columns = st.columns(
                len(current_items)
            )

            for col, (kpi_name, values) in zip(
                columns,
                current_items
            ):

                with col:

                    st.metric(
                        kpi_name,
                        format_number(values.get("total", 0)),
                        help=(
                            f"Source column: "
                            f"{values.get('column', 'Unknown')}"
                        )
                    )

    else:

        st.info(
            "No standard Revenue, Profit, Quantity or Cost KPI "
            "was detected. Review the Analytics tab for numeric metrics."
        )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Total Columns",
            f"{len(df.columns):,}"
        )

    with col3:

        st.metric(
            "Data Quality Score",
            f"{quality_score:.2f}%"
        )

    with col4:

        st.metric(
            "Potential Anomalies",
            f"{anomaly_count:,}"
        )

    st.markdown("---")

    st.subheader("Business Profile")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"**Detected Business Mode:** "
            f"{business_mode.get('mode', 'General Business')}"
        )

        st.write(
            f"**Numeric Columns:** "
            f"{len(profile.get('numeric_columns', []))}"
        )

        st.write(
            f"**Categorical Columns:** "
            f"{len(profile.get('categorical_columns', []))}"
        )

    with col2:

        st.write(
            f"**Missing Values:** "
            f"{int(df.isna().sum().sum()):,}"
        )

        st.write(
            f"**Duplicate Rows:** "
            f"{int(df.duplicated().sum()):,}"
        )

        st.write(
            f"**Quality Status:** {quality_status}"
        )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# =========================================================
# TAB 2 — DATA QUALITY
# =========================================================

with tabs[1]:

    st.header("Data Quality & Cleaning")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Quality Score",
            f"{quality_score:.2f}%"
        )

    with col2:

        st.metric(
            "Missing Values",
            f"{int(df.isna().sum().sum()):,}"
        )

    with col3:

        st.metric(
            "Duplicate Rows",
            f"{int(df.duplicated().sum()):,}"
        )

    if quality_status == "Excellent":

        st.success(
            f"Data quality status: {quality_status}"
        )

    elif quality_status == "Good":

        st.info(
            f"Data quality status: {quality_status}"
        )

    else:

        st.warning(
            f"Data quality status: {quality_status}"
        )

    st.caption(
        "The score is a simple indicator based on missing cells "
        "and duplicate rows. It does not guarantee overall data accuracy."
    )

    st.markdown("---")

    st.subheader("Column-Level Profile")

    st.dataframe(
        profile["column_info"],
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Missing Values by Column")

    missing_by_column = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    missing_by_column.columns = [
        "Column",
        "Missing Values"
    ]

    st.dataframe(
        missing_by_column,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Cleaned Dataset Preview")

    st.dataframe(
        df.head(50),
        use_container_width=True
    )

    st.download_button(
        label="⬇️ Download Cleaned Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="smartbiz_cleaned_data.csv",
        mime="text/csv"
    )


# =========================================================
# TAB 3 — ANALYTICS
# =========================================================

with tabs[2]:

    st.header("Business Analytics")

    numeric_columns = analytics.get(
        "numeric_columns",
        []
    )

    categorical_columns = analytics.get(
        "categorical_columns",
        []
    )

    st.subheader("Numeric Summary")

    if numeric_columns:

        numeric_summary = pd.DataFrame(
            analytics.get("numeric_summary", {})
        ).T

        numeric_summary.index.name = "Column"

        numeric_summary = numeric_summary.reset_index()

        st.dataframe(
            numeric_summary,
            use_container_width=True
        )

    else:

        st.info(
            "No numeric columns were detected."
        )

    st.markdown("---")

    st.subheader("Automatic Visualizations")

    try:

        charts = create_visualizations(df)

        if charts:

            for index, chart in enumerate(charts):

                st.plotly_chart(
                    chart,
                    use_container_width=True,
                    key=f"analytics_chart_{index}"
                )

        else:

            st.info(
                "Not enough suitable columns to generate charts."
            )

    except Exception as error:

        st.warning(
            f"Chart generation encountered an issue: {error}"
        )

    st.markdown("---")

    st.subheader("Explore Your Own Chart")

    if numeric_columns:

        selected_numeric = st.selectbox(
            "Select numeric column",
            numeric_columns,
            key="custom_numeric"
        )

        if categorical_columns:

            selected_category = st.selectbox(
                "Select category column",
                categorical_columns,
                key="custom_category"
            )

            chart_type = st.selectbox(
                "Chart type",
                ["Bar", "Box", "Violin"],
                key="custom_chart_type"
            )

            try:

                import plotly.express as px

                if chart_type == "Bar":

                    custom_data = (
                        df.groupby(
                            selected_category,
                            dropna=False
                        )[selected_numeric]
                        .sum()
                        .reset_index()
                        .sort_values(
                            selected_numeric,
                            ascending=False
                        )
                        .head(30)
                    )

                    custom_chart = px.bar(
                        custom_data,
                        x=selected_category,
                        y=selected_numeric,
                        title=(
                            f"{selected_numeric} by "
                            f"{selected_category}"
                        )
                    )

                elif chart_type == "Box":

                    custom_chart = px.box(
                        df,
                        x=selected_category,
                        y=selected_numeric,
                        title=(
                            f"{selected_numeric} by "
                            f"{selected_category}"
                        )
                    )

                else:

                    custom_chart = px.violin(
                        df,
                        x=selected_category,
                        y=selected_numeric,
                        title=(
                            f"{selected_numeric} by "
                            f"{selected_category}"
                        ),
                        box=True
                    )

                st.plotly_chart(
                    custom_chart,
                    use_container_width=True,
                    key="custom_explorer_chart"
                )

            except Exception as error:

                st.warning(
                    f"Could not create the selected chart: {error}"
                )

        else:

            st.info(
                "A category column is needed for category comparisons."
            )

            import plotly.express as px

            histogram = px.histogram(
                df,
                x=selected_numeric,
                title=f"Distribution of {selected_numeric}"
            )

            st.plotly_chart(
                histogram,
                use_container_width=True
            )

    else:

        st.info(
            "Custom numeric charts require at least one numeric column."
        )


# =========================================================
# TAB 4 — ANOMALIES
# =========================================================

with tabs[3]:

    st.header("Anomaly Detection")

    st.write(
        "Potential anomalies are identified using the "
        "Interquartile Range (IQR) method."
    )

    st.metric(
        "Potentially Unusual Records",
        f"{anomaly_count:,}"
    )

    if anomaly_count > 0:

        st.warning(
            "These records are statistical outliers, "
            "not necessarily errors or fraud."
        )

        unusual_records = anomaly_df[
            anomaly_df["Anomaly"].fillna(False)
        ]

        st.dataframe(
            unusual_records.head(100),
            use_container_width=True
        )

        st.download_button(
            label="⬇️ Download Anomaly Records",
            data=unusual_records.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="smartbiz_anomalies.csv",
            mime="text/csv"
        )

    else:

        st.success(
            "No potential anomalies were detected by the current method."
        )

    with st.expander("View all records with anomaly flags"):

        st.dataframe(
            anomaly_df.head(500),
            use_container_width=True
        )


# =========================================================
# TAB 5 — INSIGHTS
# =========================================================

with tabs[4]:

    st.header("Business Insights & Recommendations")

    st.subheader("Executive Summary")

    st.write(executive_summary)

    st.markdown("---")

    st.subheader("📌 Automated Insights")

    for index, insight in enumerate(insights, start=1):

        st.markdown(
            f"**{index}.** {insight}"
        )

    st.markdown("---")

    st.subheader("💡 Business Recommendations")

    for index, recommendation in enumerate(
        recommendations,
        start=1
    ):

        st.markdown(
            f"**{index}.** {recommendation}"
        )

    st.markdown("---")

    st.subheader("🔎 Root Cause Exploration")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:

        target_column = st.selectbox(
            "Select a metric to investigate",
            numeric_columns,
            key="root_cause_target"
        )

        if st.button(
            "Analyze Potential Drivers",
            key="root_cause_button"
        ):

            try:

                root_causes = analyze_root_causes(
                    df,
                    target_column
                )

                if (
                    isinstance(root_causes, pd.DataFrame)
                    and not root_causes.empty
                ):

                    st.write(
                        "Groups with the largest differences "
                        "from the overall average:"
                    )

                    st.dataframe(
                        root_causes,
                        use_container_width=True
                    )

                    st.caption(
                        "These are exploratory associations, "
                        "not proof of causation."
                    )

                else:

                    st.info(
                        "No suitable categorical drivers were found "
                        "for this metric."
                    )

            except Exception as error:

                st.warning(
                    f"Root cause analysis failed: {error}"
                )

    else:

        st.info(
            "Root cause exploration requires a numeric target."
        )


# =========================================================
# TAB 6 — PREDICTION
# =========================================================

with tabs[5]:

    st.header("Machine Learning Prediction")

    st.write(
        "Train a Random Forest regression model to estimate "
        "a selected numeric target using other numeric columns."
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(numeric_columns) >= 2 and len(df) >= 20:

        target_column = st.selectbox(
            "Select prediction target",
            numeric_columns,
            key="prediction_target"
        )

        if st.button(
            "🚀 Train Prediction Model",
            key="train_prediction"
        ):

            try:

                result = build_prediction_model(
                    df,
                    target_column
                )

                if result is None:

                    st.warning(
                        "The model could not be trained. "
                        "Choose a suitable numeric target and ensure "
                        "the dataset contains enough valid records."
                    )

                else:

                    st.session_state[
                        "smartbiz_prediction_result"
                    ] = result

                    st.session_state[
                        "smartbiz_prediction_target"
                    ] = target_column

            except Exception as error:

                st.error(
                    f"Prediction model failed: {error}"
                )

        result = st.session_state.get(
            "smartbiz_prediction_result"
        )

        previous_target = st.session_state.get(
            "smartbiz_prediction_target"
        )

        if (
            result is not None
            and previous_target == target_column
        ):

            st.markdown("---")

            st.subheader("Model Performance")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "MAE",
                    format_number(result["mae"])
                )

            with col2:

                st.metric(
                    "RMSE",
                    format_number(result["rmse"])
                )

            with col3:

                st.metric(
                    "R² Score",
                    format_number(result["r2"])
                )

            st.caption(
                "Lower MAE and RMSE generally indicate smaller "
                "prediction errors. R² should be interpreted in "
                "the context of the dataset and test split."
            )

            st.markdown("---")

            st.subheader("Actual vs Predicted")

            actual_predicted = result.get(
                "actual_vs_predicted"
            )

            if isinstance(
                actual_predicted,
                pd.DataFrame
            ):

                st.dataframe(
                    actual_predicted.head(100),
                    use_container_width=True
                )

                import plotly.express as px

                comparison_chart = px.scatter(
                    actual_predicted,
                    x="Actual",
                    y="Predicted",
                    title="Actual vs Predicted Values"
                )

                st.plotly_chart(
                    comparison_chart,
                    use_container_width=True
                )

            st.markdown("---")

            st.subheader("Feature Importance")

            importance = result.get(
                "feature_importance"
            )

            if isinstance(
                importance,
                pd.DataFrame
            ) and not importance.empty:

                st.dataframe(
                    importance,
                    use_container_width=True
                )

                import plotly.express as px

                importance_chart = px.bar(
                    importance,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Feature Importance"
                )

                st.plotly_chart(
                    importance_chart,
                    use_container_width=True
                )

            st.success(
                "Model training and evaluation completed."
            )

    else:

        st.info(
            "Prediction requires at least two numeric columns "
            "and 20 records."
        )


# =========================================================
# TAB 7 — FORECAST
# =========================================================

with tabs[6]:

    st.header("Time-Series Forecasting")

    st.write(
        "Generate a basic ARIMA forecast from historical data."
    )

    date_columns = get_forecast_date_columns(df)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if date_columns and numeric_columns:

        col1, col2 = st.columns(2)

        with col1:

            date_column = st.selectbox(
                "Select date column",
                date_columns,
                key="forecast_date"
            )

        with col2:

            value_column = st.selectbox(
                "Select metric to forecast",
                numeric_columns,
                key="forecast_value"
            )

        periods = st.number_input(
            "Number of future periods",
            min_value=1,
            max_value=90,
            value=7,
            step=1,
            key="forecast_periods"
        )

        if st.button(
            "🔮 Generate Forecast",
            key="generate_forecast"
        ):

            try:

                forecast_result = create_forecast(
                    df,
                    date_column,
                    value_column,
                    periods=int(periods)
                )

                if forecast_result is None:

                    st.warning(
                        "Forecast could not be generated. "
                        "Try a dataset with at least 10 valid "
                        "time points and a suitable numeric metric."
                    )

                else:

                    st.session_state[
                        "smartbiz_forecast_result"
                    ] = forecast_result

                    st.session_state[
                        "smartbiz_forecast_metric"
                    ] = value_column

            except Exception as error:

                st.error(
                    f"Forecast generation failed: {error}"
                )

        forecast_result = st.session_state.get(
            "smartbiz_forecast_result"
        )

        previous_metric = st.session_state.get(
            "smartbiz_forecast_metric"
        )

        if (
            forecast_result is not None
            and previous_metric == value_column
        ):

            historical_data = forecast_result[
                "historical_data"
            ]

            forecast_data = forecast_result[
                "forecast_data"
            ]

            st.subheader("Forecast Results")

            st.dataframe(
                forecast_data,
                use_container_width=True
            )

            import plotly.express as px

            history = historical_data.copy()

            history.columns = [
                "Date",
                "Actual"
            ]

            future = forecast_data.copy()

            future.columns = [
                "Date",
                "Forecast"
            ]

            fig = px.line(
                history,
                x="Date",
                y="Actual",
                title=f"{value_column}: Historical Data"
            )

            fig.add_scatter(
                x=future["Date"],
                y=future["Forecast"],
                mode="lines+markers",
                name="Forecast"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.download_button(
                label="⬇️ Download Forecast CSV",
                data=forecast_data.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="smartbiz_forecast.csv",
                mime="text/csv"
            )

            st.caption(
                "Forecasts are estimates, not guarantees. "
                "The generated dates depend on the detected "
                "time frequency."
            )

    else:

        st.info(
            "Forecasting requires a recognizable date column "
            "and at least one numeric column."
        )


# =========================================================
# TAB 8 — SEGMENTATION
# =========================================================

with tabs[7]:

    st.header("Data Segmentation")

    st.write(
        "Use K-Means clustering to group records by numeric patterns."
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(numeric_columns) >= 2 and len(df) >= 3:

        n_clusters = st.slider(
            "Number of segments",
            min_value=2,
            max_value=min(8, len(df)),
            value=min(3, len(df)),
            key="segment_count"
        )

        if st.button(
            "👥 Generate Segments",
            key="generate_segments"
        ):

            try:

                segmentation_result = create_segments(
                    df,
                    n_clusters=n_clusters
                )

                if segmentation_result is None:

                    st.warning(
                        "Segmentation could not be generated "
                        "from the selected dataset."
                    )

                else:

                    st.session_state[
                        "smartbiz_segmentation_result"
                    ] = segmentation_result

                    st.session_state[
                        "smartbiz_segmentation_clusters"
                    ] = n_clusters

            except Exception as error:

                st.error(
                    f"Segmentation failed: {error}"
                )

        segmentation_result = st.session_state.get(
            "smartbiz_segmentation_result"
        )

        previous_clusters = st.session_state.get(
            "smartbiz_segmentation_clusters"
        )

        if (
            segmentation_result is not None
            and previous_clusters == n_clusters
        ):

            segmented_df = segmentation_result[
                "segmented_data"
            ]

            segment_summary = segmentation_result[
                "segment_summary"
            ]

            st.markdown("---")

            st.subheader("Segment Summary")

            st.dataframe(
                segment_summary,
                use_container_width=True
            )

            st.markdown("---")

            st.subheader("Segment Distribution")

            import plotly.express as px

            segment_counts = (
                segmented_df["Segment"]
                .value_counts()
                .sort_index()
                .rename_axis("Segment")
                .reset_index(name="Record Count")
            )

            chart = px.bar(
                segment_counts,
                x="Segment",
                y="Record Count",
                title="Records per Segment"
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

            st.subheader("Segmented Records")

            st.dataframe(
                segmented_df.head(200),
                use_container_width=True
            )

            st.download_button(
                label="⬇️ Download Segmented Data",
                data=segmented_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="smartbiz_segmented_data.csv",
                mime="text/csv"
            )

            st.caption(
                "Segments are statistical clusters. "
                "Their business meaning requires interpretation."
            )

    else:

        st.info(
            "Segmentation requires at least two numeric columns "
            "and three records."
        )


# =========================================================
# TAB 9 — ASK YOUR DATA
# =========================================================

with tabs[8]:

    st.header("Ask Your Data")

    st.write(
        "Ask questions about rows, columns, missing values, "
        "duplicates, averages, totals, minimums or maximums."
    )

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: What is the average sales?"
        ),
        key="ask_data_question"
    )

    if st.button(
        "💬 Get Answer",
        key="ask_data_button"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            try:

                answer = answer_question(
                    df,
                    question
                )

                st.markdown("### Answer")

                st.info(answer)

            except Exception as error:

                st.error(
                    f"Could not answer the question: {error}"
                )

    st.markdown("---")

    st.subheader("Example Questions")

    example_questions = [
        "How many rows are there?",
        "How many columns are there?",
        "How many missing values?",
        "How many duplicate rows?",
        "What is the average sales?",
        "What is the highest profit?",
        "What is the lowest quantity?",
        "What is the total revenue?"
    ]

    for example in example_questions:

        st.markdown(
            f"- {example}"
        )

    st.caption(
        "This version uses predefined question patterns; "
        "it is not a general-purpose AI chatbot."
    )


# =========================================================
# TAB 10 — EXPORT AND REPORT
# =========================================================

with tabs[9]:

    st.header("Export & Business Report")

    st.write(
        "Download cleaned data, anomaly records, "
        "and a text-based business report."
    )

    st.markdown("---")

    st.subheader("📥 Download Cleaned Data")

    st.download_button(
        label="Download Cleaned CSV",
        data=df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="smartbiz_cleaned_data.csv",
        mime="text/csv",
        key="export_cleaned_data"
    )

    st.markdown("---")

    st.subheader("📄 Generate Business Report")

    try:

        report_text = create_business_report(
            df,
            profile,
            analytics,
            business_mode,
            anomaly_count,
            insights,
            recommendations
        )

        st.text_area(
            "Report Preview",
            report_text,
            height=350,
            key="business_report_preview"
        )

        st.download_button(
            label="⬇️ Download Business Report",
            data=report_text.encode("utf-8"),
            file_name="smartbiz_business_report.txt",
            mime="text/plain",
            key="download_business_report"
        )

    except Exception as error:

        st.error(
            f"Business report generation failed: {error}"
        )

    st.markdown("---")

    st.subheader("📊 Download Anomaly Analysis")

    st.download_button(
        label="Download Anomaly Analysis CSV",
        data=anomaly_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="smartbiz_anomaly_analysis.csv",
        mime="text/csv",
        key="export_anomaly_data"
    )

    st.markdown("---")

    st.subheader("📌 Report Contents")

    st.write(
        "- Dataset overview\n"
        "- Business mode\n"
        "- Data quality and duplicate counts\n"
        "- Detected business KPIs\n"
        "- Automated insights\n"
        "- Business recommendations\n"
        "- Potential anomaly count"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "SmartBiz Analytics | Automated Business Intelligence "
    "& Prediction Platform"
)

st.caption(
    "Predictions, forecasts and anomaly flags are analytical "
    "estimates. Validate results before making business decisions."
)
