import streamlit as st 
import joblib
import numpy as np
import pandas as pd

# --- 1. 增强模型加载的健壮性 ---
@st.cache_resource
def load_model():
    """
    加载预训练模型，并缓存，避免重复加载。
    """
    try:
        model = joblib.load('lgb_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Error: Model file 'lgb_model.joblib' not found. Please check if it exists in the app root directory.")
        return None

# --- 2. 模型特征顺序声明 ---
# 注意：顺序必须和训练模型时完全一致！
EXPECTED_FEATURE_ORDER = [
    'Morphology',
    'Histological grade',
    'Depth of invasion',
    'LVI',
    'Tumor budding',
    'Ki67'
]

# --- 页面配置 ---
st.set_page_config(page_title="Predicting lymph node metastasis in T1 colorectal cancer")

# 加载模型
model = load_model()

# --- 页面内容 ---
st.title("Predicting lymph node metastasis in T1 colorectal cancer")

st.markdown("""
Please fill in the following pathological information. The model will predict the risk of lymph node metastasis in patients with T1 colorectal cancer.
""")

# --- 免责声明 ---
st.info("⚠️ Disclaimer: This tool is for academic research and auxiliary reference only. The prediction results should NOT be regarded as professional medical diagnosis. All clinical decisions must be made by licensed physicians.")

# 只有模型加载成功才显示输入界面
if model:
    # --- 用户输入区域 ---
    morphology = st.selectbox(
        "Morphology",
        options=[0, 1],
        format_func=lambda x: "Non-pedunculated (0)" if x == 0 else "Pedunculated (1)"
    )

    histological_grade = st.selectbox(
        "Histological grade",
        options=[1, 2],
        format_func=lambda x: "G1 (1)" if x == 1 else "G2/G3 (2)"
    )

    depth_of_invasion = st.selectbox(
        "Depth of invasion",
        options=[0, 1],
        format_func=lambda x: "Superficial (0)" if x == 0 else "Deep (1)"
    )

    lvi = st.selectbox(
        "Lymphovascular Invasion (LVI)",
        options=[0, 1],
        format_func=lambda x: "Negative (0)" if x == 0 else "Positive (1)"
    )

    tumor_budding = st.selectbox(
        "Tumor budding",
        options=[1, 2],
        format_func=lambda x: "BD1 (1)" if x == 1 else "BD2/3 (2)"
    )

    # Ki67输入，保证为整数型
    ki67 = st.number_input(
        "Ki67 (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        format="%d",  # 仅整数显示，需 Streamlit 1.24+
        help="Enter an integer between 0 and 100"
    )

    # --- 预测按钮与逻辑 ---
    if st.button("Predict"):
        # --- 用DataFrame确保特征顺序一致 ---
        input_data = pd.DataFrame({
            'Morphology': [morphology],
            'Histological grade': [histological_grade],
            'Depth of invasion': [depth_of_invasion],
            'LVI': [lvi],
            'Tumor budding': [tumor_budding],
            'Ki67': [ki67]
        })
        # 保证列顺序
        input_data_ordered = input_data[EXPECTED_FEATURE_ORDER]

        # 预测概率并输出
        try:
            prob = model.predict_proba(input_data_ordered)[0][1]
            st.markdown(
                f'<span style="font-size:1.6rem;">Predicted probability of lymph node metastasis: <span style="color:tomato; font-weight:bold;">{prob:.2%}</span></span>',
                unsafe_allow_html=True
            )
            st.warning("Reminder: This result is for reference only. Please consult a medical professional for clinical decisions.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
