import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def build_prediction_model(df, target_column):

    data = df.copy()

    data = data.dropna(subset=[target_column])

    numeric_columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    if target_column not in numeric_columns:
        return None

    feature_columns = [
        column
        for column in numeric_columns
        if column != target_column
    ]

    if not feature_columns:
        return None

    if len(data) < 20:
        return None

    X = data[feature_columns]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    feature_importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    actual_vs_predicted = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    return {
        "model": model,
        "features": feature_columns,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "actual": y_test,
        "predicted": predictions,
        "actual_vs_predicted": actual_vs_predicted,
        "feature_importance": feature_importance
    }