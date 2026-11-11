
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diwali Sales AI Predictor",
    page_icon="🛍️",
    layout="centered"
)

# Load saved AI model and feature columns
@st.cache_resource
def load_model():
    model = joblib.load("diwali_sales_ai_model.pkl")
    columns = joblib.load("diwali_feature_columns.pkl")
    return model, columns

model, feature_columns = load_model()

st.title("🛍️ Diwali Sales AI Predictor")
st.write("Enter customer details to predict the purchase amount using AI.")

# Customer input form
with st.form("customer_form"):

    st.subheader("Customer Information")

    gender = st.selectbox("Gender", ["F", "M"])
    age = st.number_input("Customer Age", min_value=1, max_value=100, value=30)
    marital = st.selectbox("Marital Status", [0, 1], format_func=lambda x: "Single" if x == 0 else "Married")
    orders = st.number_input("Number of Orders", min_value=1, value=3)

    state = st.selectbox("State", [
        "Maharashtra", "Andhra Pradesh", "Uttar Pradesh",
        "Karnataka", "Gujarat", "Himachal Pradesh", "Delhi",
        "Jharkhand", "Kerala", "Haryana", "Madhya Pradesh",
        "Bihar", "Rajasthan", "Uttarakhand", "Telangana", "Punjab"
    ])

    zone = st.selectbox("Zone", [
        "Western", "Southern", "Central", "Northern", "Eastern"
    ])

    occupation = st.selectbox("Occupation", [
        "Healthcare", "Govt", "Automobile", "Construction",
        "Food Processing", "Lawyer", "Media", "Banking",
        "Retail", "IT Sector", "Aviation", "Hospitality",
        "Agriculture", "Textile", "Chemical"
    ])

    product = st.selectbox("Product Category", [
        "Auto", "Hand & Power Tools", "Stationery", "Tupperware",
        "Footwear & Shoes", "Furniture", "Food", "Games & Toys",
        "Sports Products", "Books", "Electronics & Gadgets",
        "Decor", "Clothing & Apparel", "Beauty",
        "Household items", "Pet Care", "Veterinary", "Office"
    ])

    submitted = st.form_submit_button("Predict Purchase Amount")

if submitted:

    if age <= 17:
        age_group = "0-17"
    elif age <= 25:
        age_group = "18-25"
    elif age <= 35:
        age_group = "26-35"
    elif age <= 45:
        age_group = "36-45"
    elif age <= 50:
        age_group = "46-50"
    elif age <= 55:
        age_group = "51-55"
    else:
        age_group = "55+"

    # Prepare input in the exact training column order
    customer = pd.DataFrame(0, columns=feature_columns, index=[0])

    customer["Age"] = age
    customer["Marital_Status"] = marital
    customer["Orders"] = orders

    def set_category(prefix, value):
        target = value.strip().replace("\xa0", " ").lower()

        for col in customer.columns:
            if col.startswith(prefix + "_"):
                category = col[len(prefix) + 1:]
                if category.strip().replace("\xa0", " ").lower() == target:
                    customer.loc[0, col] = 1
                    return

    set_category("Gender", gender)
    set_category("Age Group", age_group)
    set_category("State", state)
    set_category("Zone", zone)
    set_category("Occupation", occupation)
    set_category("Product_Category", product)

    prediction = model.predict(customer)[0]

    st.success("Prediction completed successfully!")

    st.subheader("Predicted Purchase Amount")
    st.metric("AI Prediction", f"₹ {prediction:,.2f}")

    st.caption("This is an estimated amount, not a guaranteed actual purchase.")
