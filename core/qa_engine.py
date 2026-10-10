import pandas as pd


def answer_question(df, question):

    question = question.lower().strip()

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    # Row count
    if (
        "how many rows" in question
        or "number of rows" in question
        or "total rows" in question
    ):
        return f"The dataset contains {len(df):,} rows."

    # Column count
    if (
        "how many columns" in question
        or "number of columns" in question
        or "total columns" in question
    ):
        return f"The dataset contains {len(df.columns):,} columns."

    # Missing values
    if (
        "missing" in question
        or "null" in question
    ):
        missing = int(
            df.isna().sum().sum()
        )
        return f"The dataset contains {missing:,} missing values."

    # Duplicate rows
    if (
        "duplicate" in question
        or "duplicates" in question
    ):
        duplicates = int(
            df.duplicated().sum()
        )
        return f"The dataset contains {duplicates:,} duplicate rows."

    # Average
    if (
        "average" in question
        or "mean" in question
    ):
        for column in numeric_columns:

            if column.lower() in question:

                value = df[column].mean()

                return (
                    f"The average {column} is "
                    f"{value:,.2f}."
                )

    # Maximum
    if (
        "maximum" in question
        or "highest" in question
        or "max" in question
    ):
        for column in numeric_columns:

            if column.lower() in question:

                value = df[column].max()

                return (
                    f"The highest {column} is "
                    f"{value:,.2f}."
                )

    # Minimum
    if (
        "minimum" in question
        or "lowest" in question
        or "min" in question
    ):
        for column in numeric_columns:

            if column.lower() in question:

                value = df[column].min()

                return (
                    f"The lowest {column} is "
                    f"{value:,.2f}."
                )

    # Sum / Total
    if (
        "total" in question
        or "sum" in question
    ):
        for column in numeric_columns:

            if column.lower() in question:

                value = df[column].sum()

                return (
                    f"The total {column} is "
                    f"{value:,.2f}."
                )

    return (
        "I could not understand this question yet. "
        "Try asking about rows, columns, missing values, "
        "duplicates, average, total, highest, or lowest value."
    )