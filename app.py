import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"./diabetes.csv")

# Updated CSS for the header with smoother gradients and refined styles
st.markdown("""
    <style>
        .header {
            font-size: 42px;
            font-family: 'Arial Black', sans-serif;
            text-align: left;
            background: linear-gradient(to right, #FF6F61, #FFB74D);
            -webkit-background-clip: text;
            color: transparent;
            margin-bottom: 0.5em;
        }
        .sub-header {
            font-size: 20px;
            font-family: 'Arial', sans-serif;
            color: #606060;
            margin-bottom: 1.5em;
        }
        .highlight {
            font-size: 18px;
            color: #D32F2F;
            font-weight: bold;
            margin-top: 1em;
        }
        .divider {
            border-top: 2px solid #FF7043;
            margin: 1.5em 0;
        }
    </style>
    <h1 class="header">Diabetes Risk Checker</h1>
    <p class="sub-header">Find out if you're at risk based on your health information.</p>
    <div class="divider"></div>
""", unsafe_allow_html=True)

# Improved sidebar design with cleaner and more modern styles
st.sidebar.title("🩺 Patient Health Information")
st.sidebar.write("Please enter your health data below:")

def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', min_value=0, max_value=17, value=3, format="%d")
    bp = st.sidebar.slider('Blood Pressure (mm Hg)', min_value=0, max_value=122, value=70, format="%d")
    bmi = st.sidebar.slider('BMI (Body Mass Index)', min_value=0.0, max_value=67.0, value=20.0, format="%.1f")
    glucose = st.sidebar.slider('Glucose Level (mg/dL)', min_value=0, max_value=200, value=120, format="%d")
    skinthickness = st.sidebar.slider('Skin Thickness (mm)', min_value=0, max_value=100, value=20, format="%d")
    dpf = st.sidebar.slider('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47, format="%.2f")
    insulin = st.sidebar.slider('Insulin Level (IU/mL)', min_value=0, max_value=846, value=79, format="%d")
    age = st.sidebar.slider('Age (years)', min_value=21, max_value=88, value=33, format="%d")

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

user_data = get_user_input()

# Data Summary with improved color and border styles
st.markdown("<h2 style='color: #FF7043;'>Health Data Overview</h2>", unsafe_allow_html=True)
st.dataframe(user_data.style.set_properties({
    'background-color': '#FAFAFA',
    'color': '#333333',
    'border-color': '#FF7043'
}))

# Split the data
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Model training
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Enhanced button styles for a modern look
st.markdown("""
    <style>
        .btn-primary {
            background-color: #FF6F61;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s ease-in-out;
        }
        .btn-primary:hover {
            background-color: #E64A19;
            color: white;
        }
        .btn-secondary {
            background-color: #757575;
            color: white;
            font-size: 16px;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0px 3px 5px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Button for prediction
if st.button('🔮 Check My Risk', key="primary", help="Click to predict diabetes risk"):
    st.markdown("<h3 style='text-align: center;'>🔄 Analyzing your data...</h3>", unsafe_allow_html=True)
    
    progress = st.progress(0)
    for percent in range(100):
        progress.progress(percent + 1)
    
    prediction = rf.predict(user_data)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #FF7043;'>Prediction Result</h2>", unsafe_allow_html=True)
    result = 'You are not diabetic.' if prediction[0] == 0 else 'You are **at risk of diabetes**.'
    st.markdown(f"<p class='highlight'>{result}</p>", unsafe_allow_html=True)
    
    # Display model accuracy
    accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
    st.markdown(f"<p style='color: #606060; font-size: 18px;'>Model Accuracy: {accuracy:.2f}%</p>", unsafe_allow_html=True)

else:
    st.markdown("<h3 style='text-align: center;'>👈 Enter your data and click 'Check My Risk'</h3>", unsafe_allow_html=True)
