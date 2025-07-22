import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("Health Insurance Claim Approval AI")
st.markdown("### Enter Claim Details Below:")

patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
patient_gender = st.selectbox("Gender", ["Male", "Female"])
employee_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed", "Retired"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
diagnosis = st.text_area("Diagnosis Code")
procedure = st.text_area("Procedure Code")
claim_amount = st.number_input("Claim Amount ($)", min_value=0.0, value=1000.0)

if st.button("Submit Claim for Approval"):
    claim_query = f"Patient: {patient_gender.lower()}, {patient_age} years old, {employee_status}, {marital_status}. Diagnosis: {diagnosis}. Procedure: {procedure}."
    response = requests.get(API_URL, params={"query": claim_query})

    if response.status_code == 200:
        decision = response.json()["decision"]
        st.success(f"**Claim** {decision}")
    else:
        st.error("Error: Could not get a response from the API.")
