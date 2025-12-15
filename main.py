import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="학습 시간과 성적 상관관계 분석", layout="centered")

st.title("📘 학생의 학습 시간과 성적 간 상관관계 분석")
st.write("학생의 학습 시간(시간)과 시험 성적 간의 관계를 시각화하고 상관계수를 확인합니다.")

# Sample data (can be replaced with CSV upload later)
data = {
    "Study_Time": [1, 2, 3, 4, 5, 6, 7, 8],  # hours
    "Score": [50, 55, 60, 68, 72, 78, 85, 90]
}

df = pd.DataFrame(data)

st.subheader("📊 데이터")
st.dataframe(df)

# Correlation
correlation = df["Study_Time"].corr(df["Score"])
st.subheader("📈 상관계수")
st.write(f"학습 시간과 성적의 피어슨 상관계수: **{correlation:.2f}**")

# Scatter plot
st.subheader("📉 산점도")
fig, ax = plt.subplots()
ax.scatter(df["Study_Time"], df["Score"])
ax.set_xlabel("학습 시간 (시간)")
ax.set_ylabel("성적")
ax.set_title("학습 시간 vs 성적")

st.pyplot(fig)

st.caption("실행 방법: streamlit run app.py")
