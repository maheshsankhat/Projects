import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prophet import Prophet
from prophet.plot import plot_plotly
from forecast import product_forecast
from churn_data import churn_pred

st.set_page_config(page_title="NeuralRetail - AI Retail Analytics Dashboard", layout="wide")

st.title("📊 NeuralRetail - AI Retail Analytics Dashboard")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Go to",
    [
        "📁 Upload Dataset",
        "📊 Sales Analytics",
        "👥 Customer Segmentation",
        "📈 Demand Forecasting",
        "⚠️ Churn Prediction"
    ]
)

#-----------------------------------
# load data
# ---------------------------------

if page == "📁 Upload Dataset":
    st.header("Upload Retail Dataset")

    uploaded_file = st.file_uploader("Upload your retail dataset (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            upload_df = pd.read_csv(uploaded_file)
        else:
            upload_df = pd.read_excel(uploaded_file)

        st.session_state["data"] = upload_df
        st.success("Dataset uploaded successfully!")

        def load_data(df):
            df = df
            df = df.dropna(subset=['Customer ID'])
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            
            df['TotalPrice'] = df['Quantity'] * df['Price']
            return df

        df=load_data(upload_df)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Information")
        st.write(df.describe())

        st.subheader("📊 KPI Dashboard")

        # sales data
        sales_df = df[
            (df["Quantity"] > 0) &
            (df["Price"] > 0)
        ]

        # return data
        return_df = df[
            df["Quantity"] < 0
        ]

        # revenue
        total_revenue = (
            sales_df['TotalPrice']
            .sum()
        )

        # orders
        total_orders = (
            sales_df['Invoice']
            .nunique()
        )

        # customers
        total_customers = (
            sales_df['Customer ID']
            .nunique()
        )

        # total sales rows
        total_sales = len(
            sales_df
        )

        # total returns
        total_returns = len(
            return_df
        )

        # return rate
        return_rate = (
            total_returns /
            total_sales
        ) * 100

        # -----------------------------------
        # KPI CARDS
        # -----------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "💰 Total Revenue",
            f" {total_revenue:,.0f}"
        )

        col2.metric(
            "🧾 Total Orders",
            total_orders
        )

        col3.metric(
            "👥 Total Customers",
            total_customers
        )

        # second row
        col4, col5, col6 = st.columns(3)

        col4.metric(
            "📦 Total Sales",
            total_sales
        )

        col5.metric(
            "↩ Total Returns",
            total_returns
        )

        col6.metric(
            "⚠ Return Rate",
            f"{return_rate:.2f}%"
        )

# -------------------------------------------------
# LOAD DATA FROM SESSION
# -------------------------------------------------
if "data" in st.session_state:
    df = st.session_state["data"]


    # Basic preprocessing
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month

    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df = df[df["Quantity"] > 0]
        df = df[df["UnitPrice"] > 0]
        df["TotalPrice"] = df["Quantity"] * df["Price"]
                # sales data
       


# ---------------------------------
# "📊 Sales Analytics"
# ---------------------------------

if page =="📊 Sales Analytics" and "data" in st.session_state:

    sales_df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

        # return data
    return_df = df[df["Quantity"] < 0]

        # country select
    selected_country = st.selectbox("Select Country",df['Country'].unique())

    # sales data
    sales_country = sales_df[sales_df['Country'] ==selected_country]

    # returns data
    returns_country = return_df[return_df['Country'] ==selected_country]

    # total sales quantity
    total_sales = (sales_country['Quantity'].sum())

    # total return quantity
    total_returns = abs(returns_country['Quantity'].sum())
    
    # BUSINESS RECOMMENDATIONS
   
    # df["TotalPrice"] = df["Quantity"] * df["Price"]
    st.subheader(
        "💡 Business Recommendations"
    )

    # top product
    top_product = df.groupby('Description')['Quantity'].sum().idxmax()

    # top country
    top_country = df.groupby('Country')['TotalPrice'].sum().idxmax()

    # high return product
    high_return_product = return_df.groupby('Description')['Quantity'].sum().abs().idxmax()

    # churn rate
    churn_rate = (total_returns /total_sales) * 100

    st.info(
        f"""
        🔥 Increase stock for:
        {top_product}

        🌍 Focus marketing in:
        {top_country}

        ↩ High return product:
        {high_return_product}

        ⚠ Current return rate:
        {churn_rate:.2f}%
        """
    )


    # -----------------------------------
    # AUTO INSIGHTS
    # -----------------------------------

    st.subheader(
        "📈 Auto Insights"
    )
   
    # monthly sales
    monthly_sales = df.groupby('Month')['TotalPrice'].sum()

    best_month = monthly_sales.idxmax()

    # top customer
    top_customer = df.groupby('Customer ID')['TotalPrice'].sum().idxmax()

    # highest revenue
    highest_revenue = monthly_sales.max()

    st.success(
        f"""
        📅 Highest sales month:
        {best_month}

        💰 Highest monthly revenue:
        ₹ {highest_revenue:,.0f}

        👤 Top spending customer:
        {top_customer}

        🔥 Best selling product:
        {top_product}

        🌍 Highest revenue country:
        {top_country}
        """
    )
    st.header("Sales Analytics Dashboard")

    col1, col2, col3 = st.columns(3)
    # df["TotalPrice"] = df["Quantity"] * df["Price"]
    total_revenue = df["TotalPrice"].sum()
    total_orders = df["Invoice"].nunique()
    total_customers = df["Customer ID"].nunique()

    col1.metric("Total Revenue", f"₹ {total_revenue:,.0f}")
    col2.metric("Total Orders", total_orders)
    col3.metric("Total Customers", total_customers)

    if st.button("show graph "):

        st.subheader("Daily Sales Trend")

        daily_sales = df.groupby("InvoiceDate")["TotalPrice"].sum()

        fig = plt.figure()
        daily_sales.plot()
        plt.title("Daily Sales Trend")
        plt.xlabel("Date")
        plt.ylabel("Revenue")
        st.pyplot(fig)

        st.subheader("Top 10 Products")

        top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)

        fig2 = plt.figure()
        top_products.plot(kind="bar")
        plt.title("Top 10 Selling Products")
        st.pyplot(fig2)
        
    # country vise sele

    st.subheader("🌍 Country-wise Sales & Returns")

    




    # metrics
    col1, col2 = st.columns(2)

    col1.metric( "📦 Total Sales",round(total_sales))

    col2.metric("↩ Total Returns",round(total_returns))

    #Year-wise Product Sales & Returns


    st.subheader(
        "📦 Year-wise Product Sales & Returns"
    )

    # product select
    selected_product = st.selectbox("Select Product",df['Description'].dropna().unique())

    # sales filter
    sales_product = sales_df[sales_df['Description'] ==selected_product]

    # returns filter
    returns_product = return_df[return_df['Description'] ==selected_product]

    # yearly sales
    sales_year = sales_product.groupby('Year')['Quantity'].sum().reset_index()

    # yearly returns
    returns_year = returns_product.groupby('Year')['Quantity'].sum().abs().reset_index()

    # merge
    year_data = pd.merge(sales_year,returns_year,on='Year',how='outer',suffixes=('_Sales','_Returns'))

    # fill missing
    year_data = year_data.fillna(0)

    # show table
    st.dataframe(year_data)



# -----------------------------------
# 🔮 Product Forecast
# -----------------------------------

if page == "📈 Demand Forecasting" and "data" in st.session_state:


    st.header("Demand Forecasting (30 Days)")
    st.write("Overall demand forcasting , press button ")

    if st.button("Forecast"):
        
        st.write(" it take some time to calculat ")
        daily_sales = df.groupby("InvoiceDate")["TotalPrice"].sum().reset_index()
        daily_sales.columns = ["ds", "y"]

        model = Prophet()
        model.fit(daily_sales)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        st.subheader("Forecast Table")
        st.dataframe(forecast[["ds", "yhat"]].tail(30))

        st.subheader("Forecast Plot")

        fig4 = model.plot(forecast)
        st.pyplot(fig4)

    
    st.subheader("🔮 Product Forecast")

    selected_product = st.selectbox("Select Product",sorted(df['Description'].unique()))

    if st.button("product Forecast"):

        demand, stock, forecast, model = (
            product_forecast(
                df,
                selected_product
            )
        )

        # metrics
        col6, col7 = st.columns(2)

        col6.metric(
            "Predicted Demand",
            demand
        )

        col7.metric(
            "Recommended Stock",
            stock
        )

        # graph
        fig1 = model.plot(forecast)

        st.pyplot(fig1)


# -------------------------------
# CHURN PREDICTION
# -------------------------------

if page =="⚠️ Churn Prediction" and "data" in st.session_state:
    st.subheader("⚠️ Churn Prediction")



    country_list = ["Overall"] + sorted(df['Country'].dropna().unique().tolist())

    selected_country = st.selectbox("Select Country",country_list)

    if selected_country == "Overall":

        filtered_df = df.copy()

    else:

        filtered_df = df[df['Country'] == selected_country]



    snapshot_date = filtered_df['InvoiceDate'].max()


    total_churn, churn_data = churn_pred(snapshot_date,filtered_df)

    st.metric(
        "Total Churn Customers",
        total_churn
    )



    @st.cache_data
    def convert_for_download(df):

        return df.to_csv(index=False).encode("utf-8")

    csv = convert_for_download(churn_data)

    st.write(
        "⬇ Download Churn Customer Information"
    )

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="churn_customers.csv",
        mime="text/csv"
    )

    # SHOW DATA

    st.dataframe(
        churn_data.head(20)
    )


    # -----------------------------------------
    # CHURN CUSTOMER INFORMATION
    # -----------------------------------------

    st.subheader(
        "👤 Churn Customer Information"
    )

    # churn customer ids
    churn_user_id = churn_data["Customer ID"].unique()

    # add overall option
    customer_options = (["Overall"] +list(churn_user_id))

    # select customer
    selected_user = st.selectbox("Select Customer",customer_options)

    # -----------------------------------------
    # FILTER DATA
    # -----------------------------------------

    if selected_user == "Overall":

        filtered_df = df[df['Customer ID'].isin( churn_user_id)]

    else:

        filtered_df = df[df['Customer ID'] ==selected_user]


    # TOTAL PRICE


    filtered_df['TotalPrice'] = (filtered_df['Quantity'] *filtered_df['Price'])


    # CUSTOMER SUMMARY


    user_df = filtered_df.groupby("Customer ID").agg({'Invoice': 'count','Quantity': 'sum','Price': 'mean','TotalPrice': 'sum','Country': 'first'}).reset_index()

    # rename columns
    user_df.columns = ['Customer ID','Total Orders','Total Quantity','Average Price','Total Spending','Country']



    st.dataframe(user_df, use_container_width=True)



# -------------------------------
# CUSTOMER SEGMENTATION
# -------------------------------
if page == "👥 Customer Segmentation" and "data" in st.session_state:
    st.header("Customer Segmentation using RFM + KMeans")
    # df["TotalPrice"] = df["Quantity"] * df["Price"]

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "Invoice": "count",
        "TotalPrice": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    st.subheader("Customer Segmentation Table")
    st.dataframe(rfm.head())

    st.subheader("Cluster Visualization")

    fig3 = plt.figure()
    sns.scatterplot(x="Recency", y="Monetary", hue="Cluster", data=rfm)
    st.pyplot(fig3)
