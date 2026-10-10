import pandas as pd


def detect_business_mode(df):

    columns = [
        str(column).lower().replace(" ", "_")
        for column in df.columns
    ]

    mode_keywords = {

        "Sales": [
            "sales",
            "revenue",
            "order",
            "quantity",
            "unit_price",
            "discount"
        ],

        "Marketing": [
            "marketing",
            "campaign",
            "click",
            "impression",
            "conversion",
            "lead",
            "ad_spend"
        ],

        "Finance": [
            "profit",
            "expense",
            "cost",
            "income",
            "budget",
            "payment",
            "amount"
        ],

        "HR": [
            "employee",
            "salary",
            "department",
            "attendance",
            "experience",
            "performance",
            "attrition"
        ],

        "Inventory": [
            "stock",
            "inventory",
            "warehouse",
            "supplier",
            "reorder",
            "product"
        ],

        "Customer": [
            "customer",
            "client",
            "churn",
            "retention",
            "segment",
            "customer_id"
        ]
    }


    scores = {}

    for mode, keywords in mode_keywords.items():

        score = 0

        for column in columns:

            for keyword in keywords:

                if keyword in column:
                    score += 1

        scores[mode] = score


    sorted_modes = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


    top_mode = sorted_modes[0][0]

    top_score = sorted_modes[0][1]


    if top_score == 0:

        top_mode = "General Business"


    return {
        "mode": top_mode,
        "score": top_score,
        "scores": scores
    }