def churn_pred(snapshot_date,df):
        # last purchase
    last_purchase = df.groupby(
        'Customer ID'
    )['InvoiceDate'].max()

    # churn logic
    churn = (
        snapshot_date - last_purchase
    ).dt.days > 90

    # dataframe
    churn_df = churn.reset_index()

    churn_df.columns = [
        'CustomerID',
        'Churn'
    ]

    # only churn customers
    churn_customers = churn_df[
        churn_df['Churn'] == True
    ]

    # customer ids
    churn_ids = churn_customers[
        'CustomerID'
    ]

    # full churn data
    churn_data = df[df['Customer ID'].isin(churn_ids)].groupby('Customer ID').agg({
    'Country': 'first',
    'Invoice': 'count',
    'TotalPrice': 'sum',
    'InvoiceDate': 'max'
    }).reset_index()

    # total curn customers
    total_churn = churn.sum()


    return(
        total_churn,
        churn_data
    )

