import streamlit as st
import pandas as pd
import joblib

# ==========================
# IMPORT CUSTOM MODULES
# ==========================

from utils.ui import header, footer
from utils.sidebar import sidebar
from utils.predictor import predict_salary
from utils.explanation import show_prediction_explanation
from utils.recommendations import career_recommendations
from utils.ui import header, hero, footer
from utils.kpi import show_kpis

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(

    page_title="AI Job Salary Predictor",

    page_icon="🤖",

    layout="wide"

)

# ==========================
# LOAD ENCODERS
# ==========================

encoders = joblib.load("models/label_encoders.pkl")

# ==========================
# HEADER
# ==========================

header()
hero()

page = sidebar()
footer()

# ==========================
# HOME PAGE
# ==========================

if page == "🏠 Home":

    st.info(
        "Welcome to AI Job Salary Predictor"
    )

# ==========================
# PREDICTOR PAGE
# ==========================

elif page == "💰 Salary Predictor":

    st.subheader("👤 Candidate Profile")

    col1, col2 = st.columns(2)

    with col1:

        job_title = st.selectbox(

            "Job Title",

            encoders["job_title"].classes_

        )

        experience_level = st.selectbox(

            "Experience",

            encoders["experience_level"].classes_

        )

        years_experience = st.slider(

            "Years",

            0,

            20,

            5

        )

        education = st.selectbox(

            "Education",

            encoders["education_level"].classes_

        )

    with col2:

        company_size = st.selectbox(

            "Company Size",

            encoders["company_size"].classes_

        )

        company_industry = st.selectbox(

            "Industry",

            encoders["company_industry"].classes_

        )

        country = st.selectbox(

            "Country",

            encoders["country"].classes_

        )

        remote_type = st.selectbox(

            "Remote Type",

            encoders["remote_type"].classes_

        )

    st.subheader("💻 Technical Skills")

    c1, c2, c3 = st.columns(3)

    with c1:

        python_skill = st.checkbox("Python")

        sql_skill = st.checkbox("SQL")

    with c2:

        ml_skill = st.checkbox("Machine Learning")

        dl_skill = st.checkbox("Deep Learning")

    with c3:

        cloud_skill = st.checkbox("Cloud")

        urgency = st.selectbox(

            "Hiring Urgency",

            encoders["hiring_urgency"].classes_

        )

    month = st.slider("Posting Month",1,12,6)

    year = st.slider("Posting Year",2020,2026,2026)

    openings = st.slider("Job Openings",1,9,5)

    input_df = pd.DataFrame({

        "job_title":[encoders["job_title"].transform([job_title])[0]],

        "company_size":[encoders["company_size"].transform([company_size])[0]],

        "company_industry":[encoders["company_industry"].transform([company_industry])[0]],

        "country":[encoders["country"].transform([country])[0]],

        "remote_type":[encoders["remote_type"].transform([remote_type])[0]],

        "experience_level":[encoders["experience_level"].transform([experience_level])[0]],

        "years_experience":[years_experience],

        "education_level":[encoders["education_level"].transform([education])[0]],

        "skills_python":[int(python_skill)],

        "skills_sql":[int(sql_skill)],

        "skills_ml":[int(ml_skill)],

        "skills_deep_learning":[int(dl_skill)],

        "skills_cloud":[int(cloud_skill)],

        "job_posting_month":[month],

        "job_posting_year":[year],

        "hiring_urgency":[encoders["hiring_urgency"].transform([urgency])[0]],

        "job_openings":[openings]

    })

    if st.button("🚀 Predict Salary"):

        prediction = predict_salary(input_df)

        st.success(show_kpis(
    prediction,
    experience_level,
    urgency
))

        if prediction < 70000:

            st.error("🔴 Salary Category : Low")

        elif prediction < 110000:

            st.warning("🟡 Salary Category : Average")

        elif prediction < 140000:

            st.info("🟢 Salary Category : High")

        else:

            st.success("🏆 Salary Category : Very High")

        show_prediction_explanation(

            years_experience,

            int(python_skill),

            int(sql_skill),

            int(ml_skill),

            int(cloud_skill),

            int(dl_skill),

            education,

            experience_level

        )

        career_recommendations(

            int(python_skill),

            int(sql_skill),

            int(ml_skill),

            int(cloud_skill),

            int(dl_skill)

        )

        st.balloons()

# ==========================
# ABOUT PAGE
# ==========================

elif page == "ℹ About":

    st.markdown("""

### AI Job Salary Predictor

Developed using

- Python
- Machine Learning
- Streamlit
- Scikit-learn

Developer

K. Durga Nageswararao

M.Sc Statistics

""")

footer()