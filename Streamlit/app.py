import streamlit as st
import pandas as pd
import pickle

# Load the AI Brain generated from your custom dataset
with open('hr_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('model_columns.pkl', 'rb') as file:
    model_columns = pickle.load(file)

# UI Setup
st.set_page_config(page_title="Team NEXUS HR Portal", layout="centered")
st.title("📊 HR Attrition Predictor")
st.markdown("Analyze employee flight risk in real-time.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 65, 30)
    income = st.number_input("Monthly Income ($)", 1000, 50000, 5000)
    hours_worked = st.number_input("Average Hours/Week", 20, 80, 40)
    distance = st.number_input("Distance From Home", 1, 50, 5)
    overtime = st.selectbox("Works Overtime?", ["No", "Yes"])

with col2:
    work_life = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
    job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    department = st.selectbox("Department", ['Finance', 'HR', 'Marketing', 'Sales', 'IT'])
    job_role = st.selectbox("Job Role", ['Manager', 'Assistant', 'Analyst', 'Executive'])
    years = st.number_input("Years at Company", 0, 40, 3)

# ==========================================
# 3. PROCESS INPUT & PREDICT
# ==========================================
if st.button("Analyze Risk", type="primary"):
    
    # 1. Establish a "Normal" baseline for all the hidden columns so they don't default to 0
    baseline_employee = {
        'Job_Level': 3,
        'Hourly_Rate': 65,
        'Years_in_Current_Role': 3,
        'Years_Since_Last_Promotion': 1,
        'Performance_Rating': 3,       # Average performance
        'Training_Hours_Last_Year': 3,
        'Project_Count': 4,
        'Absenteeism': 2,
        'Work_Environment_Satisfaction': 3, # Good environment
        'Relationship_with_Manager': 3,     # Good relationship
        'Job_Involvement': 3,               # Highly involved
        'Number_of_Companies_Worked': 2
    }

    # 2. Map the user's UI inputs ON TOP of the baseline
    input_data = {**baseline_employee,
        'Age': age,
        'Monthly_Income': income,
        'Average_Hours_Worked_Per_Week': hours_worked,
        'Distance_From_Home': distance,
        'Work_Life_Balance': work_life,
        'Job_Satisfaction': job_sat,
        'Years_at_Company': years,
        'Overtime': 1 if overtime == "Yes" else 0,
        f'Department_{department}': 1,
        f'Job_Role_{job_role}': 1
    }

    # 3. Now, reindex safely fills the UNUSED Departments and Job Roles with 0
    input_df = pd.DataFrame([input_data]).reindex(columns=model_columns, fill_value=0)

    # Calculate Probability
    risk = model.predict_proba(input_df)[0][1] * 100

    st.divider()
    if risk > 50:
        st.error(f"⚠️ HIGH RISK: {risk:.1f}% chance of turnover.")
    else:
        st.success(f"✅ LOW RISK: {risk:.1f}% chance of turnover.")