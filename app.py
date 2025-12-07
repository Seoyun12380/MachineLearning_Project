import streamlit as st
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="헬스 케어",
    page_icon="🩺",
    layout="centered",
)

# -----------------------------
# 모델 / 스케일러 / 컬럼 로드
# -----------------------------
@st.cache_resource
def load_model_and_tools():
    model = keras.models.load_model("health_model.h5")
    scaler = joblib.load("scaler.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return model, scaler, feature_cols

model, scaler, feature_cols = load_model_and_tools()

# -----------------------------
# 세션 상태 초기화
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "intro"

if "input_data" not in st.session_state:
    st.session_state.input_data = None

# -----------------------------
# 1페이지: 시작 화면
# -----------------------------
def render_intro_page():
    st.markdown(
        """
        <h1 style='text-align: center; color: #3498db;'>
            헬스 케어 모델
        </h1>
        <h4 style='text-align: center; color: #7f8c8d;'>
            생활 습관 기반 건강 상태 예측 Service
        </h4>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    st.markdown(
        """
        <div style='text-align: center; font-size: 17px; line-height: 1.6; color: #555;'>
            몇 가지 생활 습관을 입력하면<br>
            머신러닝 모델이 <b>현재 건강 상태 (양호 · 주의 · 위험)</b>을 예측해줍니다.<br>
            결과에 따라 <b>생활 습관 개선 팁</b>도 함께 제공됩니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_button = st.button("시작하기", use_container_width=True)

    if start_button:
        st.session_state.page = "input"


# -----------------------------
# 2페이지: 생활 습관 입력
# -----------------------------
def render_input_page():
    st.markdown("##### 2 / 3 단계")
    st.markdown("### 📋 생활 습관 입력")

    st.write("아래 항목들을 입력하면, 다음 페이지에서 건강 상태 예측 결과를 보여줘요.")

    col1, col2 = st.columns(2)

    with col1:
        age_category = st.selectbox(
            "나이대 (AgeCategory)",
            [
                "Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39",
                "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59",
                "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
                "Age 80 or older",
            ],
        )

        height_m = st.number_input(
            "키 (m 단위)", 1.0, 2.2, 1.65, step=0.01
        )
        weight_kg = st.number_input(
            "몸무게 (kg)", 30.0, 200.0, 55.0, step=0.5
        )

        bmi = float(weight_kg / (height_m ** 2))
        st.caption(f"자동 계산된 BMI: **{bmi:.1f}**")

        sleep_hours = st.slider("하루 평균 수면 시간", 3, 12, 7)
        physical_health_days = st.slider("지난 30일 중 몸이 안 좋았던 날", 0, 30, 2)

    with col2:
        physical_activities = st.selectbox("운동 여부", ["Yes", "No"])
        smoker_status = st.selectbox(
            "흡연 상태",
            ["Never smoked", "Former smoker", "Current smoker"],
        )
        alcohol_drinkers = st.selectbox("음주 여부", ["No", "Yes"])

    st.markdown("---")

    input_data = {
        "AgeCategory": age_category,
        "HeightInMeters": float(height_m),
        "WeightInKilograms": float(weight_kg),
        "BMI": bmi,
        "SleepHours": int(sleep_hours),
        "PhysicalHealthDays": int(physical_health_days),
        "PhysicalActivities": physical_activities,
        "SmokerStatus": smoker_status,
        "AlcoholDrinkers": alcohol_drinkers,
    }

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        if st.button("결과 보기 ✅"):
            st.session_state.input_data = input_data
            st.session_state.page = "result"

    with col_btn2:
        if st.button("처음으로 돌아가기 ⬅"):
            st.session_state.page = "intro"


# -----------------------------
# 3페이지: 결과 페이지 (완성본)
# -----------------------------
def render_result_page():
    st.markdown("##### 3 / 3 단계")
    st.markdown("###  건강 상태 예측 결과")

    if st.session_state.input_data is None:
        st.write("먼저 입력 페이지에서 정보를 입력해주세요.")
        return

    data = st.session_state.input_data

    df_input = pd.DataFrame([data])
    df_enc = pd.get_dummies(df_input)

    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[feature_cols]

    X_scaled = scaler.transform(df_enc)
    pred_prob = model.predict(X_scaled)[0]
    pred_class = int(np.argmax(pred_prob))
    labels = {0: "양호", 1: "주의", 2: "위험"}
    risk = labels[pred_class]

    # 결과 표시
    if pred_class == 0:
        st.success(f"현재 건강 상태는 **양호** 입니다. ")
    elif pred_class == 1:
        st.warning(f"현재 건강 상태는 **주의** 입니다. ⚠️")
    else:
        st.error(f"현재 건강 상태는 **위험** 입니다. 🚨")

    st.write(f"예측 확률: {pred_prob}")

    st.markdown("---")
    st.markdown("### 💡 생활 습관 분석 결과")

    improvements = []
    good_habits = []

    # 수면
    if data["SleepHours"] < 7:
        improvements.append("수면 시간이 부족합니다. **7시간 이상** 자도록 노력해보세요.")
    else:
        good_habits.append("수면 시간이 적절한 편이에요.")

    # BMI
    if data["BMI"] < 18.5:
        improvements.append("저체중입니다. 충분한 식사와 영양 공급이 필요해요.")
    elif data["BMI"] > 24.9:
        improvements.append("BMI가 높습니다. 규칙적인 운동과 식단 조절이 도움이 될 수 있어요.")
    else:
        good_habits.append("정상적인 BMI를 유지하고 있어요!")

    # 운동
    if data["PhysicalActivities"] == "No":
        improvements.append("규칙적인 운동을 시작해보세요. **주 2~3회 이상 추천**")
    else:
        good_habits.append("규칙적인 운동을 하고 있어 좋아요.")

    # 흡연
    if data["SmokerStatus"] == "Current smoker":
        improvements.append("흡연 중입니다. 금연을 고려해보세요.")
    else:
        good_habits.append("흡연을 하지 않는 건강한 습관을 갖고 있어요.")

    # 음주
    if data["AlcoholDrinkers"] == "Yes":
        improvements.append("음주 중입니다. 양과 횟수를 줄여보는 것이 좋아요.")
    else:
        good_habits.append("과한 음주를 하지 않아 좋아요.")

    st.markdown("###  개선하면 좋은 습관")
    for item in improvements:
        st.markdown(f"- {item}")

    st.markdown("###  유지하면 좋은 습관")
    for item in good_habits:
        st.markdown(f"- {item}")

    st.markdown("---")
    if st.button("다시 입력하기"):
        st.session_state.page = "input"
    if st.button("처음으로 돌아가기"):
        st.session_state.page = "intro"


# -----------------------------
# 라우팅
# -----------------------------
def main():
    page = st.session_state.page

    if page == "intro":
        render_intro_page()
    elif page == "input":
        render_input_page()
    elif page == "result":
        render_result_page()

if __name__ == "__main__":
    main()
