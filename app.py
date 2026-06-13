import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Healthcare Fraud Detection",
    page_icon="🏥",
    layout="wide"
)

# -------------------------------
# Load Model and Encoders
# -------------------------------
import os

@st.cache_resource
def load_artifacts():
    model_path = os.path.join("models", "xgb_realistic.pkl")
    encoder_path = os.path.join("models", "encoders.pkl")

    model = joblib.load(model_path)
    encoders = joblib.load(encoder_path)

    return model, encoders

model, encoders = load_artifacts()

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("🏥 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Executive Dashboard",
        "Fraud Prediction",
        "Model Performance"
    ]
)

# -------------------------------
# Executive Dashboard
# -------------------------------
if page == "Executive Dashboard":

    st.title("🏥 Healthcare Insurance Fraud Detection")

    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Claims", "10,000")
    col2.metric("Fraud Claims", "829")
    col3.metric("Fraud Rate", "8.29%")
    col4.metric("ROC-AUC", "98.91%")

    st.markdown("---")

    st.write("""
    This dashboard helps investigators identify potentially
    fraudulent healthcare insurance claims using an explainable
    XGBoost model.
    """)

# -------------------------------
# Fraud Prediction Placeholder
# -------------------------------
# Fraud Prediction
# -------------------------------
elif page == "Fraud Prediction":

    st.title("🔍 Claim Investigation")

    st.subheader("Enter Claim Details")

    col1, col2 = st.columns(2)

    with col1:
        patient_age = st.number_input(
            "Patient Age",
            min_value=0,
            max_value=120,
            value=45
        )

        patient_gender = st.selectbox(
            "Patient Gender",
            ["Female", "Male"]
        )

        diagnosis_code = st.selectbox(
            "Diagnosis Code",
            ['E11.9','E78.5','F41.9','I10','I25.10',
             'J06.9','J18.9','K21.9','M54.5','N39.0']
        )

        procedure_code = st.number_input(
            "Procedure Code",
            min_value=0,
            value=100
        )

        claim_amount = st.number_input(
            "Claim Amount",
            min_value=0.0,
            value=500.0
        )

        approved_amount = st.number_input(
            "Approved Amount",
            min_value=0.0,
            value=300.0
        )

        insurance_type = st.selectbox(
            "Insurance Type",
            ['Medicaid', 'Medicare', 'Private', 'Self-Pay']
        )

        provider_claims = st.number_input(
            "Number of Claims per Provider (Monthly)",
            min_value=0,
            value=50
        )

        provider_specialty = st.selectbox(
            "Provider Specialty",
            [
                'Cardiology',
                'General Practice',
                'Internal Medicine',
                'Neurology',
                'Orthopedics',
                'Pulmonology'
            ]
        )

    with col2:

        patient_state = st.selectbox(
            "Patient State",
            ['CA','FL','GA','IL','NY','OH','PA','TX']
        )

        length_of_stay = st.number_input(
            "Length of Stay",
            min_value=0,
            value=2
        )

        visit_type = st.selectbox(
            "Visit Type",
            ['Emergency','Inpatient','Outpatient']
        )

        chronic_flag = st.selectbox(
            "Chronic Condition Flag",
            [0, 1]
        )

        prior_visits = st.number_input(
            "Prior Visits (12 Months)",
            min_value=0,
            value=2
        )

        claim_year = st.number_input(
            "Claim Year",
            min_value=2020,
            max_value=2030,
            value=2024
        )

        claim_month = st.number_input(
            "Claim Month",
            min_value=1,
            max_value=12,
            value=1
        )

        claim_day = st.number_input(
            "Claim Day",
            min_value=1,
            max_value=31,
            value=1
        )

        claim_dayofweek = st.number_input(
            "Claim Day Of Week",
            min_value=0,
            max_value=6,
            value=0
        )

    if st.button("Predict Fraud"):

        # Encode categorical variables
        gender_enc = encoders['Patient_Gender'].transform(
            [patient_gender]
        )[0]

        diagnosis_enc = encoders['Diagnosis_Code'].transform(
            [diagnosis_code]
        )[0]

        insurance_enc = encoders['Insurance_Type'].transform(
            [insurance_type]
        )[0]

        specialty_enc = encoders['Provider_Specialty'].transform(
            [provider_specialty]
        )[0]

        state_enc = encoders['Patient_State'].transform(
            [patient_state]
        )[0]

        visit_enc = encoders['Visit_Type'].transform(
            [visit_type]
        )[0]

        # Create dataframe
        input_df = pd.DataFrame([[
            patient_age,
            gender_enc,
            diagnosis_enc,
            procedure_code,
            claim_amount,
            approved_amount,
            insurance_enc,
            provider_claims,
            specialty_enc,
            state_enc,
            length_of_stay,
            visit_enc,
            chronic_flag,
            prior_visits,
            claim_year,
            claim_month,
            claim_day,
            claim_dayofweek
        ]], columns=[
            'Patient_Age',
            'Patient_Gender',
            'Diagnosis_Code',
            'Procedure_Code',
            'Claim_Amount',
            'Approved_Amount',
            'Insurance_Type',
            'Number_of_Claims_Per_Provider_Monthly',
            'Provider_Specialty',
            'Patient_State',
            'Length_of_Stay',
            'Visit_Type',
            'Chronic_Condition_Flag',
            'Prior_Visits_12m',
            'Claim_Year',
            'Claim_Month',
            'Claim_Day',
            'Claim_DayOfWeek'
        ])

        # Prediction
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        score = prob * 100

        if score < 30:
            risk = "LOW"
        elif score < 70:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        st.markdown("---")

        if pred == 1:
            st.error("🚨 Prediction: FRAUD")
        else:
            st.success("✅ Prediction: GENUINE")

        st.metric(
            "Fraud Probability",
            f"{score:.2f}%"
        )

        st.metric(
            "Risk Level",
            risk
        )

else:

    # Model Performance Code

    st.title("📈 Model Performance")

    metrics = pd.DataFrame({
        "Metric": [
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        "Value": [
            "91%",
            "80%",
            "85%",
            "98.91%"
        ]
    })

    st.table(metrics)