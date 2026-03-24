import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# ==========================================================
# LOAD MODEL & DATA
# ==========================================================
model = pickle.load(open("model.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_csv("Telco-Customer-Churn.csv")

df = load_data()

# ==========================================================
# SIDEBAR INPUTS
# ==========================================================
st.sidebar.title("⚙️ Customer Details")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
Dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

tenure = st.sidebar.slider("Tenure", 0, 72, 12)

PhoneService = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes"])

StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 50.0)
TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

predict_clicked = st.sidebar.button("🔍 Predict")

# ==========================================================
# ENCODING FUNCTION
# ==========================================================
def encode(val, mapping):
    return mapping[val]

if predict_clicked:

    input_data = pd.DataFrame([[

        encode(gender, {"Female":0, "Male":1}),
        encode(SeniorCitizen, {"No":0, "Yes":1}),
        encode(Partner, {"No":0, "Yes":1}),
        encode(Dependents, {"No":0, "Yes":1}),
        tenure,
        encode(PhoneService, {"No":0, "Yes":1}),
        encode(MultipleLines, {"No":0, "Yes":1}),
        encode(InternetService, {"DSL":0, "Fiber optic":1, "No":2}),
        encode(OnlineSecurity, {"No":0, "Yes":1}),
        encode(OnlineBackup, {"No":0, "Yes":1}),
        encode(DeviceProtection, {"No":0, "Yes":1}),
        encode(TechSupport, {"No":0, "Yes":1}),
        encode(StreamingTV, {"No":0, "Yes":1}),
        encode(StreamingMovies, {"No":0, "Yes":1}),
        encode(Contract, {"Month-to-month":0, "One year":1, "Two year":2}),
        encode(PaperlessBilling, {"No":0, "Yes":1}),
        encode(PaymentMethod, {
            "Electronic check":0,
            "Mailed check":1,
            "Bank transfer":2,
            "Credit card":3
        }),
        MonthlyCharges,
        TotalCharges

    ]])

    # 👉 PREDICTION
    prediction = model.predict(input_data)[0]

    # 👉 OUTPUT
    if prediction == 1:
        st.error("⚠️ Customer will CHURN")
    else:
        st.success("✅ Customer will STAY")
        
if prediction == 1:
    st.error("⚠️ Customer is likely to CHURN")

    st.markdown("### 💡 Suggestions to Retain Customer:")

    if tenure < 12:
        st.write("👉 Offer loyalty discounts for new customers")

    if Contract == "Month-to-month":
        st.write("👉 Encourage long-term contract plans (1 year / 2 year)")

    if TechSupport == "No":
        st.write("👉 Provide free or discounted tech support")

    if OnlineSecurity == "No":
        st.write("👉 Add security features to increase trust")

    if PaymentMethod == "Electronic check":
        st.write("👉 Suggest automatic payment methods (credit card / bank transfer)")

    if MonthlyCharges > 70:
        st.write("👉 Provide discounts or better pricing plans")

else:
    st.success("✅ Customer is likely to STAY")

    st.markdown("### 🎉 Positive Insights:")
    st.write("👉 Customer is satisfied with services")
    st.write("👉 Low risk of churn")
# ==========================================================
# NAVIGATION
# ==========================================================
if "page" not in st.session_state:
    st.session_state.page = "overview"

col1, col2, col3, col4 = st.columns(4)

if col1.button("🏠 Overview"):
    st.session_state.page = "overview"
if col2.button("📊 Dataset"):
    st.session_state.page = "dataset"
if col3.button("📈 EDA"):
    st.session_state.page = "eda"
if col4.button("💰 Prediction"):
    st.session_state.page = "prediction"

st.markdown("---")

# ==========================================================
# OVERVIEW
# ==========================================================
if st.session_state.page == "overview":
    st.title("📊 Telco Customer Churn Dashboard")
    st.write("This app predicts whether a customer will churn or not using ML.")

# ==========================================================
# DATASET
# ==========================================================
elif st.session_state.page == "dataset":
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

# ==========================================================
# EDA
# ==========================================================
elif st.session_state.page == "eda":

    st.subheader("Churn Distribution")

    fig1 = px.histogram(df, x="tenure", title="Tenure Distribution")
    st.plotly_chart(fig1)

    fig2 = px.bar(df.groupby("Contract")["Churn"].count().reset_index(),
                  x="Contract", y="Churn",
                  title="Churn by Contract")
    st.plotly_chart(fig2)

# ==========================================================
# PREDICTION
# ==========================================================
elif st.session_state.page == "prediction":

    if predict_clicked:
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠️ Customer will churn")
        else:
            st.success("✅ Customer will stay")

        # Gauge chart
        val = 80 if prediction == 1 else 30

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            title={"text": "Churn Risk"},
            gauge={"axis": {"range": [0, 100]}}
        ))

        st.plotly_chart(fig)

    else:
        st.info("Enter details and click Predict")
