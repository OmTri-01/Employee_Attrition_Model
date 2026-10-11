from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [BASE_DIR / "hr_model.pkl", BASE_DIR.parent / "Model" / "hr_model.pkl"]
COLUMN_PATHS = [BASE_DIR / "model_columns.pkl", BASE_DIR.parent / "Model" / "model_columns.pkl"]


def load_model_and_columns():
    model_path = next((path for path in MODEL_PATHS if path.exists()), MODEL_PATHS[0])
    columns_path = next((path for path in COLUMN_PATHS if path.exists()), COLUMN_PATHS[0])

    with model_path.open("rb") as file:
        model = pickle.load(file)

    try:
        model_columns = model.get_booster().feature_names
    except Exception:
        with columns_path.open("rb") as file:
            model_columns = pickle.load(file)

    return model, model_columns


model, model_columns = load_model_and_columns()


def build_input_record(age, income, hours_worked, distance, overtime, work_life, job_sat, department, job_role, years):
    record = {column: 0 for column in model_columns}

    record.update({
        "Age": age,
        "Gender": 0,
        "Job_Level": 2,
        "Monthly_Income": income,
        "Hourly_Rate": 65,
        "Years_at_Company": years,
        "Years_in_Current_Role": max(1, min(years, 15)),
        "Years_Since_Last_Promotion": 1 if years <= 5 else 3,
        "Work_Life_Balance": work_life,
        "Job_Satisfaction": job_sat,
        "Performance_Rating": 3,
        "Training_Hours_Last_Year": 3,
        "Overtime": 1 if overtime == "Yes" else 0,
        "Project_Count": 4,
        "Average_Hours_Worked_Per_Week": hours_worked,
        "Absenteeism": 2,
        "Work_Environment_Satisfaction": 3,
        "Relationship_with_Manager": 3,
        "Job_Involvement": 3,
        "Distance_From_Home": distance,
        "Number_of_Companies_Worked": 2,
        "Marital_Status_Married": 0,
        "Marital_Status_Single": 1,
    })

    dept_map = {
        "HR": "Department_HR",
        "IT": "Department_IT",
        "Marketing": "Department_Marketing",
        "Sales": "Department_Sales",
    }
    role_map = {
        "Assistant": "Job_Role_Assistant",
        "Executive": "Job_Role_Executive",
        "Manager": "Job_Role_Manager",
    }

    if department in dept_map:
        record[dept_map[department]] = 1
    if job_role in role_map:
        record[role_map[job_role]] = 1

    return pd.DataFrame([record]).reindex(columns=model_columns, fill_value=0)


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
    department = st.selectbox("Department", ["HR", "IT", "Marketing", "Sales"])
    job_role = st.selectbox("Job Role", ["Assistant", "Executive", "Manager"])
    years = st.number_input("Years at Company", 0, 40, 3)


if st.button("Analyze Risk", type="primary"):
    input_df = build_input_record(
        age=age,
        income=income,
        hours_worked=hours_worked,
        distance=distance,
        overtime=overtime,
        work_life=work_life,
        job_sat=job_sat,
        department=department,
        job_role=job_role,
        years=years,
    )

    risk = model.predict_proba(input_df)[0][1] * 100

    st.divider()
    if risk > 50:
        st.error(f"⚠️ HIGH RISK: {risk:.1f}% chance of turnover.")
    else:
        st.success(f"✅ LOW RISK: {risk:.1f}% chance of turnover.")
