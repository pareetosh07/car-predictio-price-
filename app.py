import streamlit as st
import pandas as pd
import pickle as pkl
import math

st.set_page_config(page_title="Car Price Predictor", layout="wide")

# --- CLEAN, SHARP LIGHT LAYOUT WITH LEGIBLE TEXT ---
st.markdown("""
<style>
.stApp {
    background-color: #f1f5f9;
    color: #0f172a;
}
.title {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 10px;
    margin-bottom: 30px;
}
label, p {
    color: #334155 !important;
    font-weight: 600 !important;
}
.box {
    background: #ffffff;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #cbd5e1;
    margin-top: 25px;
    text-align: center;
}
.result {
    font-size: 26px;
    color: #0284c7;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🚗 Car Price Prediction System</div>", unsafe_allow_html=True)

df = pd.read_csv("clean_data.csv")

# Load model cleanly
pipe = pkl.load(open("CPP.pkl", "rb"))

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("🏢 Select Company", sorted(df["company"].unique()))
    name = st.selectbox("🚘 Select Model", sorted(df[df["company"] == company]["name"].unique()))
    fuel_type = st.selectbox("⛽ Fuel Type", sorted(df[df["name"] == name]["fuel_type"].unique()))

with col2:
    year = st.number_input("📅 Year", min_value=2000, max_value=2026, value=2015, step=1)
    km_driven = st.number_input("🛣️ Kilometers Driven", min_value=0, value=10000, step=500)

if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    data = [[name, company, year, km_driven, fuel_type]]
    column = ["name", "company", "year", "kms_driven", "fuel_type"]

    myinput = pd.DataFrame(data=data, columns=column)
    result = pipe.predict(myinput)

    try:
        raw_val = result.flatten()[0] if hasattr(result, "flatten") else result[0]
        value = math.ceil(raw_val)
    except Exception:
        value = math.ceil(float(result))

    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.markdown(f"<div class='result'>💰 Estimated Price: ₹{value:,}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)