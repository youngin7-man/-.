# student_study_analysis_app.py
# Streamlit 기반 학생 학습 시간과 성적 상관관계 분석 웹앱 (확장 버전)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings # 경고 메시지 처리를 위해 추가

from matplotlib import font_manager, rc
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

# -----------------------------
# ✅ 한글 폰트 설정 (Streamlit Cloud 환경 대응 최적화)
# -----------------------------

# 💡 수정 전략: 로컬 파일 시스템 경로 대신, 프로젝트 폴더 내 폰트 파일을 찾도록 설정
# 폰트 파일명을 프로젝트 폴더에 저장했다고 가정합니다.
FONT_FILE = "NanumGothic.ttf" 
# (만약 폰트 파일이 없다면, NotoSansKR-Regular.otf 등으로 변경하고 해당 파일을 준비해야 합니다.)

font_path = None

# 1. 프로젝트 폴더 내에서 폰트 파일을 찾습니다.
if os.path.exists(FONT_FILE):
    font_path = FONT_FILE
else:
    # 2. 시스템 경로(Streamlit Cloud 환경에서 작동하지 않을 수 있음)도 시도합니다.
    font_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
    ]
    for path in font_candidates:
        if os.path.exists(path):
            font_path = path
            break

# 3. 폰트 적용
if font_path:
    # 폰트 캐시를 지우고 새로운 폰트를 등록하여 즉시 사용 가능하게 합니다.
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc("font", family=font_name)
    st.sidebar.success(f"✅ 폰트 설정 완료: {font_name}")
else:
    # 모든 시도 실패 시
    rc("font", family="DejaVu Sans")
    warnings.filterwarnings('ignore', category=UserWarning) # 폰트 관련 경고 무시
    st.sidebar.warning("⚠️ 한글 폰트 파일을 찾지 못했습니다. 그래프에서 한글이 깨질 수 있습니다.")

plt.rcParams["axes.unicode_minus"] = False

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
# 사이드바 메뉴 (생략 없이 원본 유지)
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
# 데이터 생성 (원본 유지)
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
# 메뉴별 화면 구성 (Matplotlib 출력 최적화)
# -----------------------------

# 🏠 프로젝트 소개 (원본 유지)
if menu == "🏠 프로젝트 소개":
    st.header("🏠 프로젝트 소개")
    st.write("""
    📌 **분석 주제**: 학생의 학습 시간과 시험 성적의 관계 분석  
    🎯 **분석 목적**: 공부 시간이 늘어나면 성적이 실제로 향상되는지 데이터로 확인  
    🛠 **사용 기술**: Python, Pandas, Matplotlib, Streamlit
    """)

# 📊 데이터 확인 (원본 유지)
elif menu == "📊 데이터 확인":
    st.header("📊 데이터 확인")
    st.write("원본 데이터 상위 10개입니다 👇")
    st.dataframe(df.head(10))

# 🧹 데이터 전처리 (원본 유지)
elif menu == "🧹 데이터 전처리":
    st.header("🧹 데이터 전처리")

    st.subheader("1️⃣ 결측치 처리")
    clean_df = df.dropna()
    st.success("결측치가 제거되었습니다 ✅")

    st.subheader("2️⃣ 이상치 처리")
    clean_df = clean_df[(clean_df["study_time"] >= 0) & (clean_df["study_time"] <= 12)]
    st.success("비정상적인 값이 제거되었습니다 ✅")

    st.subheader("3️⃣ 정규화")
    scaler = MinMaxScaler()
    clean_df["study_time_norm"] = scaler.fit_transform(clean_df[["study_time"]])
    st.write("전처리 후 데이터")
    st.dataframe(clean_df.head())

# 📈 시각화 분석 (Streamlit 출력 방식 최적화)
elif menu == "📈 시각화 분석":
    st.header("📈 데이터 시각화")
    st.markdown("데이터의 경향성을 시각화하여 학습 시간과 성적의 관계를 직관적으로 파악합니다. 

[Image of a scatter plot showing positive correlation]
")

    # 1️⃣ 산점도
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(df["study_time"], df["score"])
    ax1.set_xlabel("📘 학습 시간 (시간)")
    ax1.set_ylabel("📝 성적")
    ax1.set_title("📈 학습 시간 vs 성적", fontsize=16, pad=20)
    plt.subplots_adjust(top=0.88)
    st.pyplot(fig1, use_container_width=True) # Streamlit 권장 방식 적용

    # 2️⃣ 히스토그램
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(df["score"], bins=10)
    ax2.set_title("📊 성적 분포", fontsize=16, pad=20)
    ax2.set_xlabel("성적")
    ax2.set_ylabel("학생 수")
    plt.subplots_adjust(top=0.88)
    st.pyplot(fig2, use_container_width=True) # Streamlit 권장 방식 적용

# 🔍 상관관계 분석 (원본 유지)
elif menu == "🔍 상관관계 분석":
    st.header("🔍 상관관계 분석")
    corr, p = pearsonr(df["study_time"], df["score"])

    st.markdown(f"피어슨 상관계수 ($r$): **{corr:.2f}**") 
    st.markdown(f"p-value: **{p:.4f}**")
    
    st.metric("📈 피어슨 상관계수", f"{corr:.2f}")
    st.metric("📉 p-value", f"{p:.4f}")

    if corr > 0.5:
        st.success("강한 양의 상관관계가 있습니다 💡")
        st.markdown("> **결론:** 학습 시간이 길수록 성적이 높아지는 경향이 매우 뚜렷합니다.")
    elif corr > 0.3:
        st.info("약한 양의 상관관계가 있습니다 🙂")
        st.markdown("> **결론:** 학습 시간과 성적 사이에 어느 정도의 긍정적인 관계가 있습니다.")
    else:
        st.warning("뚜렷한 상관관계가 보이지 않습니다 ⚠️")
        st.markdown("> **결론:** 학습 시간 외 다른 요인(학습의 질, 재능 등)이 성적에 더 큰 영향을 미칠 수 있습니다.")

# 📌 추가 분석 (Streamlit 출력 방식 최적화)
elif menu == "📌 추가 분석":
    st.header("📌 추가 분석")

    bins = [0, 2, 4, 6, 8, 10, 12]
    labels = ["0~2", "2~4", "4~6", "6~8", "8~10", "10~12"]
    # pd.cut에 right=False 옵션 추가 (구간 정의 명확화)
    df["time_group"] = pd.cut(df["study_time"], bins=bins, labels=labels, right=False)

    avg_score = df.groupby("time_group")["score"].mean()

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    avg_score.plot(kind="bar", ax=ax3)
    ax3.set_title("⏱ 학습 시간 구간별 평균 성적", fontsize=16, pad=20)
    ax3.set_xlabel("학습 시간 구간")
    ax3.set_ylabel("평균 성적")
    plt.subplots_adjust(top=0.88)
    st.pyplot(fig3, use_container_width=True) # Streamlit 권장 방식 적용

# ✅ 결론 (원본 유지)
elif menu == "✅ 결론":
    st.header("✅ 분석 결론")
    st.write("""
    ✔ 학습 시간이 증가할수록 성적이 전반적으로 상승하는 경향이 나타났다.  
    ✔ 상관계수 분석을 통해 두 변수 간 양의 상관관계를 확인할 수 있었다.  
    ✔ 하지만 공부 시간이 길어도 성적이 낮은 사례가 존재하여 학습의 질 또한 중요함을 알 수 있었다.  

    📌 **결론적으로**, 효율적인 학습 방법과 적절한 학습 시간이 함께 이루어져야 좋은 성과를 낼 수 있다.
    """)
