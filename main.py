# student_study_analysis_app.py
# Streamlit 기반 학생 학습 시간과 성적 상관관계 분석 웹앱

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

# -----------------------------
# 1. 페이지 설정
# -----------------------------
st.set_page_config(page_title="학생 학습시간-성적 분석", layout="wide")
st.title("📊 학생 학습 시간과 성적의 관계 분석")
st.write("이 웹앱은 학생들의 학습 시간과 시험 성적 간의 관계를 데이터로 분석하고 시각화합니다.")

# -----------------------------
# 2. 데이터 생성 (실제 CSV로 대체 가능)
# -----------------------------
st.subheader("1️⃣ 데이터 불러오기")

np.random.seed(42)
data_size = 100

study_time = np.random.normal(loc=4, scale=1.5, size=data_size)
study_time = np.clip(study_time, 0.5, 10)

score = study_time * 8 + np.random.normal(0, 10, data_size) + 40
score = np.clip(score, 0, 100)

raw_df = pd.DataFrame({
    "study_time": study_time,
    "score": score
})

st.write("원본 데이터 (상위 5개)")
st.dataframe(raw_df.head())

# -----------------------------
# 3. 데이터 전처리
# -----------------------------
st.subheader("2️⃣ 데이터 전처리")

# 결측치 처리
clean_df = raw_df.dropna()

# 이상치 처리 (공부시간 0~12시간)
clean_df = clean_df[(clean_df['study_time'] >= 0) & (clean_df['study_time'] <= 12)]

# 정규화
scaler = MinMaxScaler()
clean_df[['study_time_norm']] = scaler.fit_transform(clean_df[['study_time']])

st.write("전처리 후 데이터 (상위 5개)")
st.dataframe(clean_df.head())

# -----------------------------
# 4. 데이터 시각화
# -----------------------------
st.subheader("3️⃣ 데이터 시각화")

col1, col2 = st.columns(2)

# 산점도
with col1:
    st.write("📌 학습 시간 vs 성적 (산점도)")
    fig1, ax1 = plt.subplots()
    ax1.scatter(clean_df['study_time'], clean_df['score'])
    ax1.set_xlabel("Study Time (hours)")
    ax1.set_ylabel("Score")
    ax1.set_title("Study Time vs Score")
    st.pyplot(fig1)

# 히스토그램
with col2:
    st.write("📌 성적 분포 (히스토그램)")
    fig2, ax2 = plt.subplots()
    ax2.hist(clean_df['score'], bins=10)
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Number of Students")
    ax2.set_title("Score Distribution")
    st.pyplot(fig2)

# -----------------------------
# 5. 상관관계 분석
# -----------------------------
st.subheader("4️⃣ 상관관계 분석")

corr, p_value = pearsonr(clean_df['study_time'], clean_df['score'])

st.write(f"📈 피어슨 상관계수: **{corr:.2f}**")
st.write(f"📉 p-value: **{p_value:.4f}**")

if corr > 0.5:
    st.success("학습 시간과 성적 사이에 강한 양의 상관관계가 있습니다.")
elif corr > 0.3:
    st.info("학습 시간과 성적 사이에 약한 양의 상관관계가 있습니다.")
else:
    st.warning("학습 시간과 성적 사이의 상관관계가 크지 않습니다.")

# -----------------------------
# 6. 추가 분석: 구간별 평균 성적
# -----------------------------
st.subheader("5️⃣ 추가 분석: 학습 시간 구간별 평균 성적")

bins = [0, 2, 4, 6, 8, 10, 12]
labels = ["0~2", "2~4", "4~6", "6~8", "8~10", "10~12"]

clean_df['time_group'] = pd.cut(clean_df['study_time'], bins=bins, labels=labels)

avg_score = clean_df.groupby('time_group')['score'].mean()

fig3, ax3 = plt.subplots()
avg_score.plot(kind='bar', ax=ax3)
ax3.set_xlabel("Study Time Group (hours)")
ax3.set_ylabel("Average Score")
ax3.set_title("Average Score by Study Time Group")
st.pyplot(fig3)

# -----------------------------
# 7. 결론
# -----------------------------
st.subheader("6️⃣ 분석 결론")
st.write("""
- 학습 시간이 증가할수록 성적이 전반적으로 상승하는 경향이 나타났다.
- 산점도와 상관계수를 통해 두 변수 사이에 양의 상관관계가 있음을 확인했다.
- 하지만 공부 시간이 길어도 성적이 낮은 경우가 존재하여 학습의 질 또한 중요함을 알 수 있다.
- 따라서 효율적인 학습 전략과 적절한 공부 시간이 함께 필요하다는 결론을 도출하였다.
""")

st.caption("📌 본 웹앱은 Streamlit과 Python을 활용한 데이터 분석 프로젝트 예시입니다.")
