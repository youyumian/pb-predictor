import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# 页面配置
st.set_page_config(page_title="PB Predictor", layout="centered")
st.title("儿童MP肺炎-塑型性支气管炎预测模型")
st.markdown("输入7个临床变量，模型将预测是否存在PB风险，并解释其决策依据。")


# 加载模型与背景数据
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

@st.cache_resource
def load_background():
    return joblib.load("shap_background_data.pkl")


model = load_model()
background_data = load_background()
explainer = shap.Explainer(model, background_data)

# 用户输入
age = st.number_input("Age", value=5.0, step=0.1)
wbc_nlr = st.selectbox("WBC > 6.2 且 NLR > 1.95", [0, 1])
il6 = st.number_input("IL-6", value=15.0, step=0.1)
ddi = st.number_input("D-二聚体（DDI）", value=0.5, step=0.01)
crp_ldh = st.selectbox("CRP > 42.5 且 LDH > 325", [0, 1])
stenosis = st.selectbox("是否气道狭窄", [0, 1])
cavity = st.selectbox("是否肺部空洞", [0, 1])

input_df = pd.DataFrame([{
    "Age": age,
    "WBC>6.2 & NLR>1.95": wbc_nlr,
    "IL-6": il6,
    "DDI": ddi,
    "CRP>42.5 & LDH>325": crp_ldh,
    "Tracheal stenosis": stenosis,
    "Cavity": cavity
}])

# 预测与解释
if st.button("预测风险"):
    prob = model.predict_proba(input_df)[0][1]
    st.success(f"预测PB风险为：**{prob * 100:.2f}%**")

    # st.subheader("SHAP 力图解释")
    shap_values = explainer(input_df)

    # 修正后的SHAP力图生成
    fig= shap.plots.force(
        explainer.expected_value[1],  # 使用模型的基础期望值（正类）
        shap_values.values[0, :, 1],  # 获取正类的SHAP值
        input_df.iloc[0],  # 特征值
        feature_names=input_df.columns.tolist(),
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)