import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model_path = 'Streamlit应用程序/XGBoostCI.pkl'
model = joblib.load(model_path)

# 定义特征名称
feature_names = [
    'CI_age', 'CI_height', 'CI_weight', 'CI_BMI', 'CI_triglycerides',
    'CI_total cholesterol', 'CI_HDL', 'CI_LDL', 'CI_free fatty acids',
    'CI_CA125', 'CI_HE4', 'CI_endometrial thickness', 'CI_menopause', 
    'CI_HRT', 'CI_diabetes', 'CI_hypertension', 'CI_endometrial heterogeneity',
    'CI_uterine cavity occupation', 'CI_uterine cavity occupying lesion with rich blood flow',
    'CI_uterine cavity fluid'
]

# Streamlit用户界面
st.title("子宫内膜癌风险预测")

# 创建用户输入的字段
input_data = []
for feature in feature_names:
    if feature in ['CI_menopause', 'CI_HRT', 'CI_diabetes', 'CI_hypertension', 
                   'CI_endometrial heterogeneity', 'CI_uterine cavity occupation',
                   'CI_uterine cavity occupying lesion with rich blood flow', 
                   'CI_uterine cavity fluid']:
        value = st.selectbox(f"{feature} (0=否, 1=是):", options=[0, 1])
    else:
        value = st.number_input(f"{feature}:", min_value=0.0, step=0.1)
    input_data.append(value)

# 将输入数据转换为numpy数组
features = np.array([input_data])

if st.button("预测"):
    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**预测类别:** {predicted_class}")
    st.write(f"**预测概率:** {predicted_proba}")

    # 生成基于预测结果的建议
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"根据我们的模型，您可能存在子宫内膜癌的高风险。"
            f"模型预测您患有子宫内膜癌的概率为 {probability:.1f}%。"
            "虽然这只是一个估计，但建议您尽快咨询妇科医生，"
            "进行进一步的检查和评估，以确保及时获得准确的诊断和必要的治疗。"
        )
    else:
        advice = (
            f"根据我们的模型，您患子宫内膜癌的风险较低。"
            f"模型预测您没有子宫内膜癌的概率为 {probability:.1f}%。"
            "尽管如此，保持定期的健康检查仍然非常重要，"
            "并在出现任何症状时及时就医。"
        )

    st.write(advice)

    # 计算SHAP值并显示force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([input_data], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([input_data], columns=feature_names),
                    matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")
