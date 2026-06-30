import pandas as pd
from prophet import Prophet


def product_forecast(df, product_name):

    # filter product
    product_df = df[
        df['Description'] == product_name
    ]

    # prepare time series
    sales = product_df.groupby(
        'InvoiceDate'
    )['Quantity'].sum().reset_index()

    sales.columns = ['ds', 'y']

    # model
    model = Prophet()

    model.fit(sales)

    # future
    future = model.make_future_dataframe(
        periods=30
    )

    # prediction
    forecast = model.predict(future)

    # next 30 days
    next_30 = forecast.tail(30)

    # total demand
    predicted_demand = round(
        next_30['yhat'].sum()
    )

    # stock
    recommended_stock = round(
        predicted_demand * 1.2
    )

    return (
        predicted_demand,
        recommended_stock,
        forecast,
        model
    )