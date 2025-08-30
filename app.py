import joblib
import streamlit as st
import pandas as pd

# load the model (joblib for robustness)
def load_model():
    model = joblib.load("KMeans_model.pkl")
    return model

# prediction display function (no decorator)
def display(gender, age, income, score, pred):
    st.subheader("Prediction Result")
    st.markdown(f"""
        **Gender** : {gender}  
        **Age**: {age}  
        **Annual Income**: $ {income*1000}  
        **Spending Score**: {score}  
    """)
    if pred == 0:
        st.success("Low income and Average Spending Score")
    elif pred == 1:
        st.success("High income and High Spending Score")
    elif pred == 2:
        st.success("Low income and High Spending Score")
    elif pred == 3:
        st.success("High income and Low Spending Score")

# load model
model = load_model()

st.set_page_config(page_title="customer segmentation", page_icon="🧑‍🤝‍🧑", layout="wide")

# sidebar
with st.sidebar:
    st.subheader("Customer Segmentation")
    st.write('---')
    st.image("segment.png")

# columns
col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender: ", options=['Male', 'Female'], horizontal=True)
    age = st.number_input("Age: ", min_value=1, max_value=100, step=1)
    income = st.number_input("Annual Income (in $): ", min_value=15000, step=1000) / 1000
    score = st.number_input("Spending Score: ", min_value=1, max_value=100, step=1)
    data = [[1 if gender == "Male" else 0, age, income, score]]
    c1, c2, c3 = st.columns(3)
    if c2.button("Predict"):
        pred = model.predict(data)
        display(gender, age, income, score, int(pred[0]))

with col2:
    try:
        df = pd.read_csv("cleaned.csv")
        st.write("Customer clusters scatter (Annual Income vs Spending Score)")
        # safer scatter_chart usage
        st.scatter_chart(df, x='Annual Income (k$)', y='Spending Score (1-100)')
    except Exception as e:
        st.error(f"Could not load cleaned.csv: {e}")
