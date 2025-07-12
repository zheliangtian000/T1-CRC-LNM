import streamlit as st
import joblib
import numpy as np

# 加载训练好的模型
model = joblib.load('lgb_model.joblib')

st.set_page_config(page_title="Predicting lymph node metastasis in T1 colorectal cancer")

st.title("Predicting lymph node metastasis in T1 colorectal cancer")

st.markdown("""
Please fill in the following pathological information. The model will predict the risk of lymph node metastasis in patients with T1 colorectal cancer.
""")

# 用户输入区域
morphology = st.selectbox(
    "Morphology",
    options=[0, 1],
    format_func=lambda x: "Pedunculated (1)" if x == 1 else "Non-pedunculated (0)"
)

histological_grade = st.selectbox(
    "Histological grade",
    options=[1, 2],
    format_func=lambda x: "Well-differentiated (1)" if x == 1 else "Moderate/Poorly-differentiated (2)"
)

depth_of_invasion = st.selectbox(
    "Depth of invasion",
    options=[0, 1],
    format_func=lambda x: "Superficial (0)" if x == 0 else "Deep (1)"
)

lvi = st.selectbox(
    "Lymphovascular Invasion (LVI)",
    options=[0, 1],
    format_func=lambda x: "Absent (0)" if x == 0 else "Present (1)"
)

tumor_budding = st.selectbox(
    "Tumor budding",
    options=[1, 2],
    format_func=lambda x: "BD1 (1)" if x == 1 else "BD2/3 (2)"
)

ki67 = st.number_input(
    "Ki67 (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

if st.button("Predict"):
    # 按照模型要求的顺序组织输入数据
    input_data = np.array([[morphology, histological_grade, depth_of_invasion, lvi, tumor_budding, ki67]])
    # 预测概率（假设是二分类，返回为1的概率）
    prob = model.predict_proba(input_data)[0][1]
    st.success(f"Predicted probability of lymph node metastasis: {prob:.2%}")
