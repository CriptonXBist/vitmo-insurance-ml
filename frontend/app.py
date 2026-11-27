import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Предсказание покупки страховки")
st.write("Заполните данные клиента.")

with st.form("prediction"):
    age = st.number_input("Возраст", min_value=18, max_value=100, value=35)
    previously_insured = st.selectbox("Уже есть страховка?", [0, 1])
    annual_premium = st.number_input("Годовой взнос", min_value=0, value=30000)
    vintage = st.number_input("Дней в компании", min_value=0, value=200)
    region_code = st.number_input("Код региона", min_value=0, value=28)
    policy_sales_channel = st.number_input("Код канала продаж", min_value=0, value=26)
    vehicle_age = st.selectbox("Возраст автомобиля", ["< 1 Year", "1-2 Year", "> 2 Years"])
    vehicle_damage = st.selectbox("Были повреждения?", ["Yes", "No"])
    gender = st.selectbox("Пол", ["Male", "Female"])
    driving_license = st.selectbox("Есть водительские права?", [1, 0])

    submitted = st.form_submit_button("Получить прогноз")

if submitted:
    payload = {
        "Age": age,
        "Previously_Insured": previously_insured,
        "Annual_Premium": annual_premium,
        "Vintage": vintage,
        "Region_Code": region_code,
        "Policy_Sales_Channel": policy_sales_channel,
        "Vehicle_Age": vehicle_age,
        "Vehicle_Damage": vehicle_damage,
        "Gender": gender,
        "Driving_License": driving_license,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        st.metric("Вероятность покупки", f"{result['probability']:.1%}")
    except requests.RequestException as exc:
        st.error(f"Не удалось получить прогноз от API: {exc}")