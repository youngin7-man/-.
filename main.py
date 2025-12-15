# student_study_analysis_app.py
# Streamlit 기반 학생 학습 시간과 성적 상관관계 분석 웹앱 (확장 버전)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# -----------------------------
# 한글 폰트 설정 (Streamlit / Matplotlib)
# -----------------------------
try:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)
except:
    rc('font', family='DejaVu Sans')

plt.rcParams['axes.unicode_minus'] = False
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

# -----------------------------
# 1. 페이지 설정
# -----------------------------
st.set_page_config(page_title="📘 학생 학습 데이터 분석 웹앱", layout="wide")

st.title("📘 학생 학습 데이터 분석 대시보드")
st.markdown("""
👋 **환영합니다!**  
이 웹앱은 학생들의 **학습 시간과 성적 데이터**를 다양한 방법으로 분석하고  
그래프와 수치를 통해 쉽게 이해할 수 있도록 제작되었습니다.
""")

# -----------------------------
# 사이드바 메뉴
# -----------------------------
st.sidebar.title("📂 메뉴")
menu = st.sidebar.radio(
    "원하는 분석을 선택하세요 👇",
    ["🏠 프로젝트 소개", "📊 데이터 확인", "🧹 데이터 전처리", "📈 시각화 분석", "🔍 상관관계 분석", "📌 추가 분석", "✅ 결론"]
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
    "study_time": study_time,
    "score": score
})

# -----------------------------
# 메뉴별 화면 구성
# -----------------------------

# 🏠 프로젝트 소개
if menu == "🏠 프로젝트 소개":
    st.header("🏠 프로젝트 소개")
    st.write("""
    📌 **분석 주제**: 학생의 학습 시간과 시험 성적의 관계 분석  
    🎯 **분석 목적**: 공부 시간이 늘어나면 성적이 실제로 향상되는지 데이터로 확인  
    🛠 **사용 기술**: Python, Pandas, Matplotlib, Streamlit
    """)

# 📊 데이터 확인
elif menu == "📊 데이터 확인":
    st.header("📊 데이터 확인")
    st.write("원본 데이터 상위 10개입니다 👇")
    st.dataframe(df.head(10))

# 🧹 데이터 전처리
elif menu == "🧹 데이터 전처리":
    st.header("🧹 데이터 전처리")

    st.subheader("1️⃣ 결측치 처리")
    clean_df = df.dropna()
    st.success("결측치가 제거되었습니다 ✅")

    st.subheader("2️⃣ 이상치 처리")
    clean_df = clean_df[(clean_df['study_time'] >= 0) & (clean_df['study_time'] <= 12)]
    st.success("비정상적인 값이 제거되었습니다 ✅")

    st.subheader("3️⃣ 정규화")
    scaler = MinMaxScaler()
    clean_df[['study_time_norm']] = scaler.fit_transform(clean_df[['study_time']])
    st.write("전처리 후 데이터")
    st.dataframe(clean_df.head())

# 📈 시각화 분석
elif menu == "📈 시각화 분석":
    st.header("📈 데이터 시각화")

    # 1️⃣ 산점도
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(df['study_time'], df['score'])
    ax1.set_xlabel("📘 학습 시간 (시간)")
    ax1.set_ylabel("📝 성적")
    ax1.set_title("📈 학습 시간 vs 성적", fontsize=16, pad=20)

    plt.subplots_adjust(top=0.88)
    st.pyplot(fig1)

    # 2️⃣ 히스토그램
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(df['score'], bins=10)
    ax2.set_title("📊 성적 분포", fontsize=14, pad=20)
    ax2.set_xlabel("성적")
    ax2.set_ylabel("학생 수")

    plt.subplots_adjust(top=0.88)
    st.pyplot(fig2)


# 🔍 상관관계 분석
elif menu == "🔍 상관관계 분석":
    st.header("🔍 상관관계 분석")
    corr, p = pearsonr(df['study_time'], df['score'])
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
    df['time_group'] = pd.cut(df['study_time'], bins=bins, labels=labels)
    avg_score = df.groupby('time_group')['score'].mean()

    fig3, ax3 = plt.subplots()
    avg_score.plot(kind='bar', ax=ax3)
    ax3.set_title("⏱ 학습 시간 구간별 평균 성적", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig3)

# ✅ 결론
elif menu == "✅ 결론":
    st.header("✅ 분석 결론")
    st.write("""
    ✔ 학습 시간이 증가할수록 성적이 전반적으로 상승하는 경향이 나타났다.  
    ✔ 상관계수 분석을 통해 두 변수 간 양의 상관관계를 확인할 수 있었다.  
    ✔ 하지만 공부 시간이 길어도 성적이 낮은 사례가 존재하여 학습의 질 또한 중요함을 알 수 있었다.  

    📌 **결론적으로**, 효율적인 학습 방법과 적절한 학습 시간이 함께 이루어져야 좋은 성과를 낼 수 있다.
    """)

st.caption("✨ Streamlit을 활용한 빅데이터 분석 프로젝트 예시 ✨")
