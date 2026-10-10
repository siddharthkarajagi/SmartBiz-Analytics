def calculate_data_quality_score(df):

    total_rows = len(df)

    total_cells = df.shape[0] * df.shape[1]

    if total_cells == 0:
        return 0

    missing_cells = int(
        df.isna().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    missing_rate = (
        missing_cells / total_cells
    )

    duplicate_rate = (
        duplicate_rows / total_rows
        if total_rows > 0
        else 0
    )

    quality_score = (
        100
        - (missing_rate * 70)
        - (duplicate_rate * 30)
    )

    quality_score = max(
        0,
        min(100, quality_score)
    )

    return round(
        quality_score,
        2
    )