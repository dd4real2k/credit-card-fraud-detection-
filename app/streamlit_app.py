import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    layout="wide"
)

st.title("Credit Card Fraud Detection Dashboard")
st.write("Fraud risk monitoring dashboard using machine learning predictions.")


@st.cache_data
def load_data():
    return pd.read_csv("../reports/fraud_risk_scored_transactions.csv")


df = load_data()


# Summary metrics
total_transactions = len(df)
actual_fraud = int(df["Actual_Class"].sum())
predicted_fraud = int(df["Predicted_Fraud"].sum())
high_risk = int((df["Risk_Category"] == "High Risk").sum())


col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Actual Fraud Cases", f"{actual_fraud:,}")
col3.metric("Predicted Fraud Alerts", f"{predicted_fraud:,}")
col4.metric("High Risk Transactions", f"{high_risk:,}")


st.divider()


# Risk category distribution
st.subheader("Risk Category Distribution")

risk_counts = df["Risk_Category"].value_counts()

st.bar_chart(risk_counts)


# Fraud probability distribution
st.subheader("Fraud Probability Distribution")

st.line_chart(df["Fraud_Probability"])


# Filter transactions
st.subheader("Transaction Risk Explorer")

risk_filter = st.selectbox(
    "Filter by risk category",
    ["All", "Low Risk", "Medium Risk", "High Risk"]
)

if risk_filter != "All":
    filtered_df = df[df["Risk_Category"] == risk_filter]
else:
    filtered_df = df


st.dataframe(
    filtered_df[
        [
            "Actual_Class",
            "Fraud_Probability",
            "Risk_Score",
            "Risk_Category",
            "Predicted_Fraud"
        ]
    ].sort_values(by="Risk_Score", ascending=False),
    use_container_width=True
)


# High risk transactions
st.subheader("Top 20 Highest Risk Transactions")

top_risk = df.sort_values(by="Risk_Score", ascending=False).head(20)

st.dataframe(
    top_risk[
        [
            "Actual_Class",
            "Fraud_Probability",
            "Risk_Score",
            "Risk_Category",
            "Predicted_Fraud"
        ]
    ],
    use_container_width=True
)
