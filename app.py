import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"./diabetes.csv")

# CSS Styling
st.markdown("""
    <style>
        /* Main header style with a vibrant gradient */
        .main-header {
            text-align: center;
            font-size: 50px;
            color: #FF6F61;
            font-family: 'Arial Black', sans-serif;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
        }

        /* Subheader style for the introduction text */
        .intro-text {
            text-align: center;
            font-size: 20px;
            color: #606060;
            font-family: 'Arial', sans-serif;
            margin-bottom: 20px;
        }

        /* Sidebar header styling */
        .sidebar-header {
            font-size: 18px;
            font-weight: bold;
            color: #FF9F40;
        }

        /* Style the number input boxes */
        .stNumberInput > div {
            font-size: 16px;
            color: #404040;
            background-color: #f9f9f9;
            border-radius: 5px;
        }

        /* Subheader for data summary */
        .subheader {
            font-size: 25px;
            color: #333;
            font-family: 'Arial', sans-serif;
        }

        /* Prediction result styling */
        .prediction-result {
            text-align: center;
            font-size: 28px;
            font-family: 'Arial Black', sans-serif;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }

        /* Model accuracy styling */
        .accuracy {
            font-size: 22px;
            color: #2D87BB;
            font-family: 'Arial', sans-serif;
            text-align: center;
            margin-top: 10px;
        }

        /* Progress bar styling */
        .stProgress > div > div {
            background-color: #FF9F40 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown("<h1 class='main-header'>Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='intro-text'>This app predicts whether a patient is diabetic based on their health data.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar header
st.sidebar.markdown("<h2 class='sidebar-header'>Enter Patient Data</h2>", unsafe_allow_html=True)
st.sidebar.write("Please provide the following details for a diabetes checkup:")

# Function to get user input
def calc():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3)
    bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
    bmi = st.sidebar.number_input('BMI', min_value=0, max_value=67, value=20)
    glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
    skinthickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47)
    insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=79)
    age = st.sidebar.number_input('Age', min_value=21, max_value=88, value=33)

    output = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bp': bp,
        'skinthickness': skinthickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age
    }
    report_data = pd.DataFrame(output, index=[0])
    return report_data

# Collect user data
user_data = calc()

# Display user data summary
st.markdown("<h2 class='subheader'>Patient Data Summary</h2>", unsafe_allow_html=True)
st.write(user_data)

# Split the dataset
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train the model with a progress bar
progress = st.progress(0)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
progress.progress(100)

# Make the prediction
result = rf.predict(user_data)

# Display prediction result with styling
prediction_text = 'You are not Diabetic' if result[0] == 0 else 'You are Diabetic'
prediction_color = '#4CAF50' if result[0] == 0 else '#FF4136'
st.markdown(f"<div class='prediction-result' style='background-color: {prediction_color}; color: white;'>{prediction_text}</div>", unsafe_allow_html=True)

# Calculate and display model accuracy
accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
st.markdown(f"<div class='accuracy'>Model Accuracy: {accuracy:.2f}%</div>", unsafe_allow_html=True)
