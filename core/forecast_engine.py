import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def create_forecast(
    df,
    date_column,
    value_column,
    periods=7
):

    data = df[
        [date_column, value_column]
    ].copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce"
    )

    data = data.dropna()

    if len(data) < 10:
        return None

    data = (
        data.groupby(date_column)[value_column]
        .sum()
        .sort_index()
    )

    if len(data) < 10:
        return None

    try:

        model = ARIMA(
            data,
            order=(1, 1, 1)
        )

        fitted_model = model.fit()

        forecast = fitted_model.forecast(
            steps=periods
        )

        last_date = data.index.max()

        frequency = pd.infer_freq(data.index)

        if frequency is None:
            frequency = "D"

        forecast_dates = pd.date_range(
            start=last_date,
            periods=periods + 1,
            freq=frequency
        )[1:]

        forecast_df = pd.DataFrame({
            "Forecast Date": forecast_dates,
            "Forecast Value": forecast.values
        })

        forecast_df["Forecast Value"] = (
            forecast_df["Forecast Value"]
            .clip(lower=0)
        )

        return {
            "historical_data": data.reset_index(),
            "forecast_data": forecast_df,
            "model": fitted_model,
            "periods": periods
        }

    except Exception:
        return None