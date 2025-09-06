import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("models/decision_tree_model.pkl")

# Title
st.set_page_config(page_title="Are You Single or Committed?", page_icon="💘")
st.title("💘 Fun Relationship Status Predictor")
st.write("Answer the following lifestyle questions and find out whether you are **Single** or **Committed** (just for fun 😉).")

# Questions (features)
instagram_time = st.slider("📱 Average daily Instagram usage (hours)", 0, 10, 2)
male_friends = st.slider("👨 Number of close male friends", 0, 20, 5)
female_friends = st.slider("👩 Number of close female friends", 0, 20, 5)
party_freq = st.slider("🎉 Parties attended per month", 0, 15, 2)
late_night_talks = st.slider("📞 Average late-night calls/chats in month", 0, 14, 2)
study_hours = st.slider("Study Hours per Day in week(self study)", 0, 25, 4)
Messaging_Apps_Used = st.slider("Messaging app used daily..(count)", 0, 5, 2)
Texts_Per_Day = st.slider(" Number of texts per done day ", 0, 200, 50)
Parents_Strictness_Level = st.slider("Rate your parents on their strictness...", 0, 5, 1)
Coffee_Shop_Visits = st.slider("Coffe shop or cafe visits per month", 0, 10, 1)
gender = st.radio("⚧️ Your gender", ["Female", "Male"])

# Convert gender
gender_value = 0 if gender == "Female" else 1

# Collect input into dataframe
input_data = pd.DataFrame([[
    male_friends, female_friends, instagram_time, party_freq, late_night_talks,
    study_hours, Messaging_Apps_Used, Texts_Per_Day, Parents_Strictness_Level, Texts_Per_Day, gender_value
]], columns=[
    "Num_Male_Friends", "Num_Female_Friends", "Daily_Instagram_Hours", "Party_Frequency",
    "Late_Night_Talks_Per_Week", "Study_Hours", "Messaging_Apps_Used", "Texts_Per_Day",
    "Parents_Strictness_Level", "Coffee_Shop_Visits", "Gender"
])

# Prediction
if st.button("🔮 Predict My Relationship Status"):
    prediction = model.predict(input_data)[0]
    status = "💖 Committed" if prediction == 1 else "😎 Single"

    st.subheader(f"Your predicted status: {status}")

    # Show probability pie chart
    prob = model.predict_proba(input_data)[0]
    fig, ax = plt.subplots()
    ax.pie(prob, labels=["Single", "Committed"], autopct='%1.1f%%', startangle=90, colors=["#66b3ff","#ff99cc"])
    ax.axis('equal')
    st.pyplot(fig)

    st.success("✨ This is just for fun! Don’t take it too seriously 😉")
