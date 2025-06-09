import streamlit as st
import joblib
import pandas as pd

model = joblib.load('career_predictor_model.joblib')
le = joblib.load('label_encoder.joblib')

st.title("AI Career Path Predictor for Students")

marks_math = st.slider("Math Marks", 0, 100, 70)
marks_english = st.slider("English Marks", 0, 100, 65)
marks_science = st.slider("Science Marks", 0, 100, 68)

interest_tech = st.checkbox("Interested in Technology?")
interest_art = st.checkbox("Interested in Art?")
interest_sports = st.checkbox("Interested in Sports?")
interest_communication = st.checkbox("Interested in Communication?")

aptitude_score = st.slider("Aptitude Score", 0, 100, 70)

openness = st.slider("Openness (Personality Trait)", 0, 100, 60)
conscientiousness = st.slider("Conscientiousness", 0, 100, 65)
extroversion = st.slider("Extroversion", 0, 100, 55)
agreeableness = st.slider("Agreeableness", 0, 100, 70)
neuroticism = st.slider("Neuroticism", 0, 100, 40)

if st.button("Predict Career"):
    input_df = pd.DataFrame({
        'marks_math': [marks_math],
        'marks_english': [marks_english],
        'marks_science': [marks_science],
        'interest_tech': [int(interest_tech)],
        'interest_art': [int(interest_art)],
        'interest_sports': [int(interest_sports)],
        'interest_communication': [int(interest_communication)],
        'aptitude_score': [aptitude_score],
        'openness': [openness],
        'conscientiousness': [conscientiousness],
        'extroversion': [extroversion],
        'agreeableness': [agreeableness],
        'neuroticism': [neuroticism]
    })
    y_pred_num = model.predict(input_df)
    y_pred_label = le.inverse_transform(y_pred_num)[0]
    st.success(f"Predicted Career Path: **{y_pred_label}**")
