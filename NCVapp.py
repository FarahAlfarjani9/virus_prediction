import streamlit as st
import numpy as np
import joblib
from PIL import Image
pca = joblib.load("pca_model.joblib")
model = joblib.load("svm_model.joblib")

page_style = """
<style>
/* خلفية عامة للصفحة */
[data-testid="stAppViewContainer"] {
    background-color: #E8F5E9;   /* أخضر فاتح */
}

/* تخصيص منطقة المحتوى (المنتصف) */
[data-testid="stVerticalBlock"] {
    background-color: #FFFFFF;   /* أبيض */
    border-radius: 15px;
    padding: 20px;
    box-shadow: 2px 2px 10px rgba(0, 168, 107, 0.6);
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# {0: 'covid', 1: 'normal', 2: 'virus'}
label_map = {0: 'مصابة بفايروس كورونا تحديدا', 1: 'ليست مصابة بأي فايروس', 2: 'مصابة بفايروس'}

st.markdown("""
    <style>
    .bg-title {
        background-color: #00A86B;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        margin-bottom: 20px;
    }
     .bg-card {
        background-color: #3498db;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        margin-bottom: 20px;
    }
     .info {
        color: black;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# العنوان
st.markdown('<div class="bg-title">تحليل صور الأشعة باستخدام الذكاء الاصطناعي</div>', unsafe_allow_html=True)

st.markdown('<div  class="info" > Browse File رجاء ارفق صورة الأشعة بالضغط على </div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

# المعالجة والتشخيص
if uploaded_file:
    img = Image.open(uploaded_file).convert("L").resize((512, 512))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_flatten = img_array.flatten().reshape(1, -1)
    img_pca = pca.transform(img_flatten)
    prediction = model.predict(img_pca)[0]

    st.image(img, caption=" الصورة التي تم إرفاقها", width=660)

# تحديد اللون حسب الحالة
    if prediction == 0:  # COVID
        bg_color = "#ffcccc"  # أحمر فاتح
    elif prediction == 1:  # Normal
        bg_color = "#ccffcc"  # أخضر فاتح
    elif prediction == 2:  # Virus
        bg_color = "#fff3cd"  # أصفر فاتح
    st.markdown(f"""
    <div style="background-color:{bg_color};padding:10px;border-left:6px solid #00A86B;border-right:6px solid #00A86B;
    border-radius:10px;box-shadow:2px 2px 10px rgba(0,0,0,0.1);text-align:center;margin-top:10px;">
        <h4 style="color:#2c3e50;">التشخيص الطبي للحالة هو</h4>
        <p style="font-size:24px;font-weight:bold;">{label_map[prediction]}</p>
    </div>
""", unsafe_allow_html=True)

    st.write("القيمة المتوقعة من النموذج:", prediction)