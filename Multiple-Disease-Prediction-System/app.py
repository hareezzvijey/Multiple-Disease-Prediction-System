import streamlit as st
from streamlit_option_menu import option_menu
import joblib as jb
import numpy as np
from sklearn.preprocessing import StandardScaler

# loading the saved models
diabetes_model = jb.load('Multiple-Disease-Prediction-System/ensemble_model_diabetes.pkl')
heart_disease_model = jb.load('Multiple-Disease-Prediction-System/Heart_disease_Model.pkl')
parkinsons_model = jb.load('Multiple-Disease-Prediction-System/Parkinson_Disease_Model.pkl')

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Disease Prediction",
        options=["Diabetes", "Heart Disease", "Parkinson's"],
        icons=["activity", "heart", "person"],
        menu_icon="cast",
        default_index=0,
    )
# prediction functions
def predict(data, model):
    sc = StandardScaler()
    data = np.array(data).reshape(1, -1) 
    data = sc.fit_transform(data)      
    prediction = model.predict(data)
    return prediction

def diabetes_page():
    st.title("Diabetes Prediction")
    st.write("Enter the following details to predict diabetes:")
    # Add your input fields and prediction logic here

    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0

    age = st.number_input("Age", min_value=0, max_value=100, step=1)

    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    hypertension = 1 if hypertension == "Yes" else 0

    heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
    heart_disease = 1 if heart_disease == "Yes" else 0

    smoking_history = st.selectbox("Smoking History", ["Never","No_info","Current", "Former", "Ever", "Not Current"])

    smoking_onehot = [0,0,0,0,0]

    if smoking_history != "Never":
        index_map = {"No_info":0,"Current": 1, "Former": 2, "Ever": 3, "Not Current": 4}
        smoking_onehot[index_map[smoking_history]] = 1

    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.5)

    HbA1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=10.0, step=0.5)
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0, max_value=500, step=1)
    if st.button("Predict"):
        data = smoking_onehot + [gender, age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level]
        data = np.array(data).reshape(-1,1)
        prediction = predict(data, diabetes_model)
        if prediction[0] == 1:
            st.error("The person is diabetic")
            st.write("Please consult a doctor immediately")
        else:
            st.success("The person is not diabetic")
            st.write("You are safe, but please maintain a healthy lifestyle")

def heart_page():
    st.title("Heart Disease Prediction")
    # Add your input fields and prediction logic here
    # age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  ca  thal  
    age = st.number_input("Age",min_value=0,
                          max_value=100,step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0
    cp =  st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    if cp == "Typical Angina":
        cp = 0
    elif cp == "Atypical Angina":
        cp = 1
    elif cp == "Non-Anginal Pain":
        cp = 2
    elif cp == "Asymptomatic":
        cp = 3

    trestbps = st.number_input("Resting Blood Pressure",min_value=0,max_value=300,step=1)
    chol = st.number_input("Serum Cholestoral in mg/dl",min_value=0,max_value=600,step=1)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    fbs = 1 if fbs == "Yes" else 0
    restecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    if restecg == "Normal":
        restecg = 0
    elif restecg == "ST-T Wave Abnormality":
        restecg = 1
    elif restecg == "Left Ventricular Hypertrophy":
        restecg = 2
    thalach = st.number_input("Maximum Heart Rate Achieved",min_value=0,max_value=300,step=1)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    exang = 1 if exang == "Yes" else 0
    oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest",min_value=0.0,max_value=10.0,step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    if slope == "Upsloping":
        slope = 0
    elif slope == "Flat":
        slope = 1
    elif slope == "Downsloping":
        slope = 2
    ca = st.number_input("Number of Major Vessels Colored by Flourosopy",min_value=0,max_value=4,step=1)
    thal = st.selectbox("Thal", ["Normal", "Fixed Defect", "Reversable Defect"])
    if thal == "Normal":
        thal = 0
    elif thal == "Fixed Defect":
        thal = 1
    elif thal == "Reversable Defect":
        thal = 2

    if st.button("Predict"):
        data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        prediction = predict(data, heart_disease_model)

        if prediction[0] == 1:
            st.error("The person is likely to have heart disease")
            st.write("Please consult a doctor immediately")
        else:
            st.success("The person is likely to be healthy")
            st.write("You are safe, but please maintain a healthy lifestyle")

def parkinson_page():
    st.title("Parkinson's Disease Prediction")
    # Add your input fields and prediction logic here

     # ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
     #       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
     #       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
     #       'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'status', 'RPDE', 'DFA',
     #       'spread1', 'spread2', 'D2', 'PPE'] 
    # Add your input fields and prediction logic here
    fo = st.number_input("MDVP:Fo(Hz)",min_value=0.0000,max_value=300.0,step=0.1)
    fhi = st.number_input("MDVP:Fhi(Hz)",min_value=0.0000,max_value=300.0,step=0.1)
    flo = st.number_input("MDVP:Flo(Hz)",min_value=0.0000,max_value=300.0,step=0.1)
    jitter = st.number_input("MDVP:Jitter(%)",min_value=0.0000,max_value=100.0,step=0.1)
    jitter_abs = st.number_input("MDVP:Jitter(Abs)",min_value=0.0000,max_value=100.0,step=0.1)
    rap = st.number_input("MDVP:RAP",min_value=0.0000,max_value=100.0,step=0.1)
    ppq = st.number_input("MDVP:PPQ",min_value=0.0000,max_value=100.0,step=0.1)
    ddp = st.number_input("Jitter:DDP",min_value=0.0000,max_value=100.0,step=0.1)
    shimmer = st.number_input("MDVP:Shimmer",min_value=0.0000,max_value=100.0,step=0.1)
    shimmer_db = st.number_input("MDVP:Shimmer(dB)",min_value=0.0000,max_value=100.0,step=0.1)
    apq3 = st.number_input("Shimmer:APQ3",min_value=0.0000,max_value=100.0,step=0.1)
    apq5 = st.number_input("Shimmer:APQ5",min_value=0.0000,max_value=100.0,step=0.1)
    apq = st.number_input("MDVP:APQ",min_value=0.0000,max_value=100.0,step=0.1)
    dda = st.number_input("Shimmer:DDA",min_value=0.0000,max_value=100.0,step=0.1)
    nhr = st.number_input("NHR",min_value=0.0000,max_value=100.0,step=0.1)
    hnr = st.number_input("HNR",min_value=0.0000,max_value=100.0,step=0.1)
    rpde = st.number_input("RPDE",min_value=0.0000,max_value=100.0,step=0.1)
    dfa  = st.number_input("DFA",min_value=0.0000,max_value=100.0,step=0.1)
    spread1 = st.number_input("spread1",min_value=-100.0000,max_value=100.0,step=0.1)
    spread2 = st.number_input("spread2",min_value=-100.0000,max_value=100.0000,step=0.1)
    d2 = st.number_input("D2",min_value=0.0000,max_value=100.0,step=0.1)
    ppe = st.number_input("PPE",min_value=0.0000,max_value=100.0,step=0.1)

    if st.button("Predict"):
        data = [fo, fhi, flo, jitter, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
        prediction = predict(data, parkinsons_model)
        if prediction[0] == 1:
            st.error("The person is likely to have Parkinson's disease")
            st.write("Please consult a doctor immediately")
        else:
            st.success("The person is likely to be healthy")
            st.write("You are safe, but please maintain a healthy lifestyle")

# Show page based on menu choice
if selected == "Diabetes":
    diabetes_page()
elif selected == "Heart Disease":
    heart_page()
elif selected == "Parkinson's":
    parkinson_page()