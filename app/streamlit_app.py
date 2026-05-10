import pandas as pd
import streamlit as st


DATA_PATH = "../reports/fraud_risk_scored_transactions.csv"


st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    layout="wide",
)


@st.cache_data
def load_data():
    """Load fraud risk scored transaction data."""

    return pd.read_csv(DATA_PATH)


def display_summary_metrics(df):
    """Display key dashboard metrics."""

    total_transactions = len(df)
    actual_fraud = int(df["Actual_Class"].sum())
    predicted_fraud = int(df["Predicted_Fraud"].sum())
    high_risk = int((df["Risk_Category"] == "High Risk").sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Actual Fraud Cases", f"{actual_fraud:,}")
    col3.metric("Predicted Fraud Alerts", f"{predicted_fraud:,}")
    col4.metric("High Risk Transactions", f"{high_risk:,}")


def display_risk_distribution(df):
    """Display risk category distribution chart."""

    st.subheader("Risk Category Distribution")
    risk_counts = df["Risk_Category"].value_counts()
    st.bar_chart(risk_counts)


def display_probability_trend(df):
    """Display fraud probability trend chart."""

    st.subheader("Fraud Probability Trend")
    st.line_chart(df["Fraud_Probability"])


def display_transaction_explorer(df):
    """Display filterable transaction risk table."""

    st.subheader("Transaction Risk Explorer")

    risk_filter = st.selectbox(
        "Filter by risk category",
        ["All", "Low Risk", "Medium Risk", "High Risk"],
    )

    if risk_filter != "All":
        filtered_df = df[df["Risk_Category"] == risk_filter]
    else:
        filtered_df = df

    display_columns = [
        "Actual_Class",
        "Fraud_Probability",
        "Risk_Score",
        "Risk_Category",
        "Predicted_Fraud",
    ]

    st.dataframe(
        filtered_df[display_columns].sort_values(
            by="Risk_Score",
            ascending=False,
        ),
        use_container_width=True,
    )


def display_top_risk_transactions(df):
    """Display the highest risk transactions."""

    st.subheader("Top 20 Highest Risk Transactions")

    display_columns = [
        "Actual_Class",
        "Fraud_Probability",
        "Risk_Score",
        "Risk_Category",
        "Predicted_Fraud",
    ]

    top_risk = df.sort_values(by="Risk_Score", ascending=False).head(20)

    st.dataframe(
        top_risk[display_columns],
        use_container_width=True,
    )


def main():
    """Run Streamlit fraud monitoring dashboard."""

    st.title("Credit Card Fraud Detection Dashboard")
    st.write("Fraud risk monitoring dashboard using machine learning predictions.")

    df = load_data()

    display_summary_metrics(df)

    st.divider()

    display_risk_distribution(df)
    display_probability_trend(df)
    display_transaction_explorer(df)
    display_top_risk_transactions(df)


if __name__ == "__main__":
    main()
