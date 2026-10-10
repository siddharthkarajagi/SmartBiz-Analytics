import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def create_segments(df, n_clusters=3):

    data = df.copy()

    numeric_columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(numeric_columns) < 2:
        return None

    numeric_columns = [
        column
        for column in numeric_columns
        if data[column].notna().any()
    ]

    if len(numeric_columns) < 2:
        return None

    features = data[numeric_columns].copy()

    for column in numeric_columns:
        features[column] = features[column].fillna(
            features[column].median()
        )

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        features
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(
        scaled_features
    )

    result = data.copy()

    result["Segment"] = clusters + 1

    segment_summary = (
        result.groupby("Segment")[numeric_columns]
        .mean()
        .round(2)
        .reset_index()
    )

    segment_size = (
        result["Segment"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    segment_size.columns = [
        "Segment",
        "Customer Count"
    ]

    segment_summary = segment_summary.merge(
        segment_size,
        on="Segment",
        how="left"
    )

    return {
        "segmented_data": result,
        "segment_summary": segment_summary,
        "model": model,
        "features": numeric_columns
    }