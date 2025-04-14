import time
from pickle import GLOBAL

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import streamlit as st
from PIL import Image


# Load Data
heart_data = pd.read_csv('dataset.csv')
heart_data.head()
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, stratify=Y, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy On Train Set
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# Accuracy On Test Set
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# Web App
st.title('💕 Heart Disease Prediction Model')
st.header('By Uncle\'s Snack')
img = Image.open('heart.png')
st.image(img,width=300,)

if "default_values" not in st.session_state:
    st.session_state.default_values = [None] * 13

default_values = st.session_state.default_values

st.header("📝 Please enter the following information:")

# 2 Columns
col1, col2 = st.columns(2)
input_list = []

# Left Column
with col1:
    age = st.number_input("Age", 1, 120, int(default_values[0]) if default_values[0] else 30)

    sex_raw = int(default_values[1]) if default_values[1] in [0, 1] else 0
    sex_label = "Male" if sex_raw == 1 else "Female"
    sex = st.selectbox("Sex", ["Male", "Female"], index=["Male", "Female"].index(sex_label))
    sex = 1 if sex == "Male" else 0

    cp_index = int(default_values[2]) if default_values[2] in [0, 1, 2, 3] else 0
    cp_options = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]
    cp = st.selectbox("Chest Pain Type", cp_options, index=cp_index)
    cp = cp_options.index(cp)

    trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, int(default_values[3]) if default_values[3] else 120)

    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, int(default_values[4]) if default_values[4] else 200)

    fbs_raw = int(default_values[5]) if default_values[5] in [0, 1] else 0
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"], index=fbs_raw)
    fbs = 1 if fbs == "Yes" else 0

    restecg_idx = int(default_values[6]) if default_values[6] in [0, 1, 2] else 0
    restecg_options = ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"]
    restecg = st.selectbox("Resting ECG", restecg_options, index=restecg_idx)
    restecg = restecg_options.index(restecg)

# Right Column
with col2:
    thalach = st.number_input("Max Heart Rate", 60, 220, int(default_values[7]) if default_values[7] else 150)

    exang_raw = int(default_values[8]) if default_values[8] in [0, 1] else 0
    exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"], index=exang_raw)
    exang = 1 if exang == "Yes" else 0

    oldpeak = st.number_input("ST Depression", 0.0, 10.0, float(default_values[9]) if default_values[9] else 1.0,
                              step=0.1)

    slope_idx = int(default_values[10]) if default_values[10] in [0, 1, 2] else 0
    slope_options = ["Upsloping", "Flat", "Downsloping"]
    slope = st.selectbox("Slope of ST Segment", slope_options, index=slope_idx)
    slope = slope_options.index(slope)

    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3],
                      index=int(default_values[11]) if default_values[11] in [0, 1, 2, 3] else 0)

    thal_raw = int(default_values[12]) if default_values[12] in [1, 2, 3] else 1
    thal_options = ["Normal", "Fixed defect", "Reversible defect"]
    thal = st.selectbox("Thalassemia", thal_options, index=thal_raw - 1)
    thal = thal_options.index(thal) + 1

# Paste Data Directly
st.markdown("---")
st.subheader("📋 Paste Data (Optional)")

if "paste_error" not in st.session_state:
    st.session_state.paste_error = ""
if "show_error" not in st.session_state:
    st.session_state.show_error = False

raw_input = st.text_input(
    "Paste 13 comma-separated values to auto-fill the form",
    placeholder="e.g. 63,1,3,145,233,1,0,150,0,2.3,0,0,1"
)

error_placeholder = st.empty()

if raw_input:
    try:
        parsed = list(map(float, raw_input.strip().split(",")))
        if len(parsed) == 13:
            st.session_state.default_values = parsed
            st.session_state.paste_error = ""
            st.session_state.show_error = False
        else:
            st.session_state.paste_error = "❗ Please enter exactly 13 values."
            st.session_state.show_error = True
    except ValueError:
        st.session_state.paste_error = "❗ Invalid input format. Use only numbers separated by commas."
        st.session_state.show_error = True
    except AttributeError:
        pass

# Show error and auto-hide after 2 seconds
if st.session_state.show_error:
    with error_placeholder:
        st.error(st.session_state.paste_error)
        time.sleep(3)
        error_placeholder.empty()
        st.session_state.show_error = False

st.markdown("---")
# Submit button

st.markdown("""
    <style>
        .stButton>button {
            background-color: #63A6FF; 
            color: white; 
            border: none; 
            padding: 5px 20px; 
            font-size: 12px; 
        }
        .stButton>button:hover {
            background-color: #5590E1; 
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.subheader("🔍 See Result")
if st.button("✔ Submit"):
    input_list = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,thalach, exang, oldpeak, slope, ca, thal]])

try :
    np_df = np.asarray(input_list, dtype=float)
    reshaped_df = np_df.reshape(1, -1)
    prediction = model.predict(reshaped_df)
    st.subheader(f"🎯 Prediction ( Accuracy: {test_data_accuracy:.2%} )")
    if prediction[0] == 0:
        st.markdown(f"""
        <div style='
            background-color: #fDfDfD;
            color: #28A745;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 20px;
            text-align: left;
            box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.25);
        '>
            ✅ &nbsp This person does not have a heart disease
        </div>
        <br>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='
            background-color: #fDfDfD;
            color: #FFC107;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 20px;
            text-align: left;
            box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.25);
        '>
            ⚠️ &nbsp This person has a heart disease
        </div>
        <br>
        """, unsafe_allow_html=True)

    check_row = np.append(np_df,prediction)
    exists = any((heart_data.iloc[::-1] == check_row).all(axis=1))
    matched_rows = heart_data[heart_data.eq(check_row).all(axis=1)]

    if exists:
        st.subheader(f"🧾 Outcome")

        actual = matched_rows.iloc[0, -1]
        predicted = prediction[0]

        if actual == predicted:
            st.markdown(f"""
            <div style='
                background-color: #e6f4ea;
                color: #28A745;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 20px;
                text-align: left;
                box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.25);
            '>
                ✅ &nbsp Correct Prediction
            </div>
            <br>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='
                background-color: #fdecea;
                color: #e53935;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 20px;
                text-align: left;
                box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.25);
            '>
                ❌ &nbsp Incorrect Prediction
            </div>
            <br>
            """, unsafe_allow_html=True)
    else:
        st.subheader(f"🧾 Outcome")
        st.markdown(f"""
                    <div style='
                        background-color: #fDfDfD;
                        color: #999;
                        padding: 10px 20px;
                        border-radius: 5px;
                        font-size: 20px;
                        text-align: left;
                        box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.25);
                    '>
                        📭 &nbsp Data Not Found
                    </div>
                    <br>
                    """, unsafe_allow_html=True)

except ValueError:
    pass

st.markdown("---")

st.subheader("🗂️ About Data")
st.write(heart_data)
st.subheader("🧠 Model Performance on Train Data")
st.markdown(f"""
<div style='
    background-color: #fDfDfD;
    color: #4CAF50;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 20px;
    text-align: left;
    box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.25);
'>
    {training_data_accuracy}
</div>
""", unsafe_allow_html=True)

st.subheader("📈 Model Performance on Test Data")
st.markdown(f"""
<div style='
    background-color: #fDfDfD;
    color: #4CAF50;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 20px;
    text-align: left;
    box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.25);
'>
    {test_data_accuracy}
</div>
""", unsafe_allow_html=True)