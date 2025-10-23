import streamlit as st
from src.pipeline.predict_pipeline import PredictPipeline,CustomData

# page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
        .result {
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            font-size: 1.1rem;
        }
        .success {
            background-color: #d4edda;
            color: #155724;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
        }
        .prob {
            font-size: 1.4rem;
            font-weight: bold;
            color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Header
st.title("💼 Customer Churn Prediction Dashboard")
st.markdown("Predict whether a customer is likely to leave the bank based on their profile and activity.")

st.markdown("---")

# Sidebar Inputs
st.sidebar.header("Customer Information")

credit_score = st.sidebar.number_input("Credit Score",value=650)
geography = st.sidebar.selectbox("Geography",["France","Germany","Spain"])
gender = st.sidebar.selectbox("Gender",["Male","Female"])
age = st.sidebar.slider("Age",18,100,35)
tenure = st.sidebar.slider("Tenure(Years with Bank)",0,10,5)
balance = st.sidebar.number_input("Account Balance",min_value=0.0,value=10000.0)
num_of_products = st.sidebar.slider("Number of Products",1,4,1)
has_cr_card = st.sidebar.selectbox("Has Credit Card",[1, 0],format_func=lambda x: "Yes" if x==1 else "No")
is_active_member = st.sidebar.selectbox("Is Active Member",[1, 0],format_func=lambda x: "Yes" if x==1 else "No")
estimated_salary = st.sidebar.number_input("Estimated Salary",min_value=0.0,value=50000.0)

st.markdown("### Model Prediction")

# Prediction Logic
if st.button("Predict Churn",use_container_width=True):
    try:
        # Prepare data
        data = CustomData(
            credit_score=credit_score,
            geography=geography,
            gender=gender,
            age=age,
            tenure=tenure,
            balance=balance,
            num_of_products=num_of_products,
            has_cr_card=has_cr_card,
            is_active_member=is_active_member,
            estimated_salary=estimated_salary
        )

        input_df = data.to_dataframe()

        model = PredictPipeline()
        prediction = model.predict(input_df)

        # Get churn probability
        prob = prediction[0][0]

        st.markdown(f"<p class='prob'>Predicted Churn Probability: <b>{prob:.2f}</b></p>", unsafe_allow_html=True)

        if prob > 0.5:
            st.markdown("<div class='result error'>The customer is <b>likely to churn</b>.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result success'>The customer is <b>not likely to churn</b>.</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

# Show user input summary
with st.expander("View Input Data"):
    input_summary = {
        "Credit Score": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "Num of Products": num_of_products,
        "Has Credit Card": "Yes" if has_cr_card == 1 else "No",
        "Is Active Member": "Yes" if is_active_member == 1 else "No",
        "Estimated Salary": estimated_salary
    }
    st.table(input_summary)

