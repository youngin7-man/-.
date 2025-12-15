# student_study_analysis_app.py
# Streamlit + Plotly 기반 학생 학습 시간과 성적 상관관계 분석 웹앱

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

# -----------------------------
# 1. 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="📘 학생 학습 데이터 분석 웹앱",
    layout="wide"
)

st.title("📘 학생 학습 데이터 분석 대시보드")
st.markdown("""
👋 **환영합니다!**  
이 웹앱은 학생들의 **학습 시간과 성적 데이터**를 분석하여  
시각적으로 쉽게 이해할 수 있도록 제작되었습니다.
""")

# -----------------------------
# 사이드바 메뉴
# -----------------------------
st.sidebar.title("📂 메뉴")
menu = st.sidebar.radio(
    "원하는 분석을 선택하세요 👇",
    [
        "🏠 프로젝트 소개",
        "📊 데이터 확인",
        "🧹 데이터 전처리",
        "📈 시각화 분석",
        "🔍 상관관계 분석",
        "📌 추가 분석",
        "✅ 결론"
    ]
)

# -----------------------------
# 데이터 생성
# -----------------------------
np.random.seed(42)
data_size = 100

study_time = np.random.normal(loc=4, scale=1.5, size=data_size)
study_time = np.clip(study_time, 0.5, 10)

score = study_time * 8 + np.random.normal(0, 10, data_size) + 40
score = np.clip(score, 0, 100)

df = pd.DataFrame({
    "학습 시간(시간)": study_time,
    "성적": score
})

# -----------------------------
# 메뉴별 화면 구성
# -----------------------------

# 🏠 프로젝트 소개
if menu == "🏠 프로젝트 소개":
    st.header("🏠 프로젝트 소개")
    st.write("""
    📌 **분석 주제**: 학생의 학습 시간과 시험 성적의 관계 분석  
    🎯 **분석 목적**: 학습 시간이 성적 향상에 미치는 영향 확인  
    🛠 **사용 기술**: Python, Pandas, Plotly, Streamlit
    """)

# 📊 데이터 확인
elif menu == "📊 데이터 확인":
    st.header("📊 데이터 확인")
    st.dataframe(df.head(10))

# 🧹 데이터 전처리
elif menu == "🧹 데이터 전처리":
    st.header("🧹 데이터 전처리")

    clean_df = df.dropna()

    scaler = MinMaxScaler()
    clean_df["학습 시간(정규화)"] = scaler.fit_transform(
        clean_df[["학습 시간(시간)"]]
    )

    st.success("데이터 전처리가 완료되었습니다 ✅")
    st.dataframe(clean_df.head())

# 📈 시각화 분석
elif menu == "📈 시각화 분석":
    st.header("📈 데이터 시각화")

    # 산점도
    fig1 = px.scatter(
        df,
        x="학습 시간(시간)",
        y="성적",
        title="📈 학습 시간과 성적의 관계",
        trendline="ols"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 히스토그램
    fig2 = px.histogram(
        df,
        x="성적",
        nbins=10,
        title="📊 성적 분포"
    )
    st.plotly_chart(fig2, use_container_width=True)

# 🔍 상관관계 분석
elif menu == "🔍 상관관계 분석":
    st.header("🔍 상관관계 분석")

    corr, p = pearsonr(df["학습 시간(시간)"], df["성적"])

    st.metric("📈 피어슨 상관계수", f"{corr:.2f}")
    st.metric("📉 p-value", f"{p:.4f}")

    if corr > 0.5:
        st.success("강한 양의 상관관계가 있습니다 💡")
    elif corr > 0.3:
        st.info("약한 양의 상관관계가 있습니다 🙂")
    else:
        st.warning("뚜렷한 상관관계가 보이지 않습니다 ⚠️")

# 📌 추가 분석
elif menu == "📌 추가 분석":
    st.header("📌 추가 분석")

    bins = [0, 2, 4, 6, 8, 10, 12]
    labels = ["0~2", "2~4", "4~6", "6~8", "8~10", "10~12"]

    df["학습 시간 구간"] = pd.cut(
        df["학습 시간(시간)"],
        bins=bins,
        labels=labels
    )

    avg_score = df.groupby("학습 시간 구간", observed=False)["성적"].mean().reset_index()

    fig3 = px.bar(
        avg_score,
        x="학습 시간 구간",
        y="성적",
        title="⏱ 학습 시간 구간별 평균 성적"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ✅ 결론
elif menu == "✅ 결론":
    st.header("✅ 분석 결론")
    st.write("""
    ✔ 학습 시간이 증가할수록 성적이 상승하는 경향이 나타났다.  
    ✔ 상관관계 분석 결과, 두 변수 간 양의 상관관계가 확인되었다.  
    ✔ 하지만 성적 향상에는 학습의 질 또한 중요한 요소임을 알 수 있다.
    """)

st.caption("✨ Streamlit + Plotly 기반 데이터 분석 프로젝트 ✨")
