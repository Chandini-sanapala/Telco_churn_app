import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Telco Churn App", layout="centered")

st.title("📊 Telco Customer Churn Prediction")
st.markdown("### 🔍 Enter Customer Details")

# -------- INPUTS --------

gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.selectbox("Partner", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])

tenure = st.slider("Tenure (months)", 0, 72, 12)

PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"])
OnlineBackup = st.selectbox("Online Backup", ["No", "Yes"])
DeviceProtection = st.selectbox("Device Protection", ["No", "Yes"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes"])

StreamingTV = st.selectbox("Streaming TV", ["No", "Yes"])
StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# -------- ENCODING --------

def encode(val, mapping):
    return mapping[val]

data = np.array([[
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

# -------- PREDICTION --------

if st.button("🚀 Predict"):

    prediction = model.predict(data)[0]

    # Probability (if model supports it)
    try:
        prob = model.predict_proba(data)[0][1]
    except:
        prob = None

    st.subheader("🔎 Result:")

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    # Show probability
    if prob is not None:
        st.write(f"### 📊 Churn Probability: {prob*100:.2f}%")

        if prob > 0.7:
            st.warning("⚠️ High Risk Customer")
        elif prob > 0.4:
            st.info("⚡ Medium Risk Customer")
        else:
            st.success("✅ Low Risk Customer")

    st.progress(int(prob*100) if prob else 50)

    st.balloons()