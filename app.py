
import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Diwali Sales AI",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f4f8ff;
}
.main-title {
    color: #174ea6;
    font-size: 38px;
    font-weight: 800;
    text-align: center;
}
.subtitle {
    color: #52647d;
    text-align: center;
    font-size: 17px;
}
div.stButton > button {
    background-color: #1769e0;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #0d47a1;
    color: white;
}
.result-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #d9e7ff;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# Load AI model
@st.cache_resource
def load_model():
    model = joblib.load("diwali_sales_ai_model.pkl")
    columns = joblib.load("diwali_feature_columns.pkl")
    return model, columns


model, feature_columns = load_model()


# Website heading
st.markdown(
    '<div class="main-title">🛍️ Diwali Sales AI Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict customer purchase amount using Machine Learning</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

# Information section
col1, col2, col3 = st.columns(3)

with col1:
    st.info("🤖 AI Powered Prediction")

with col2:
    st.info("📊 Customer Analysis")

with col3:
    st.info("⚡ Instant Results")

st.write("")

# Customer form
st.subheader("👤 Enter Customer Details")
st.write("Fill in the information below to predict the purchase amount.")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["F", "M"])

        age = st.number_input(
            "Customer Age",
            min_value=1,
            max_value=100,
            value=30
        )

        marital = st.selectbox(
            "Marital Status",
            [0, 1],
            format_func=lambda x: "Single" if x == 0 else "Married"
        )

        orders = st.number_input(
            "Number of Orders",
            min_value=1,
            value=3
        )

        state = st.selectbox("State", [
            "Maharashtra", "Andhra Pradesh", "Uttar Pradesh",
            "Karnataka", "Gujarat", "Himachal Pradesh", "Delhi",
            "Jharkhand", "Kerala", "Haryana", "Madhya Pradesh",
            "Bihar", "Rajasthan", "Uttarakhand", "Telangana", "Punjab"
        ])

    with col2:
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

    submitted = st.form_submit_button(
        "🔮 Predict Purchase Amount",
        use_container_width=True
    )


# Prediction
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

    customer = pd.DataFrame(
        0,
        columns=feature_columns,
        index=[0]
    )

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

    st.write("")
    st.divider()

    st.markdown(
        '<div class="result-card">'
        '<h2 style="color:#174ea6;">🎉 Prediction Result</h2>'
        '</div>',
        unsafe_allow_html=True
    )

    st.metric(
        "Estimated Customer Purchase Amount",
        f"₹ {prediction:,.2f}"
    )

    st.success("Your AI prediction is ready!")

    st.caption(
        "Note: This is an estimated purchase amount, "
        "not a guaranteed actual purchase."
    )

# Footer
st.divider()

st.markdown(
    "<p style='text-align:center;color:#718096;'>"
    "Developed with Python, Streamlit & Machine Learning"
    "</p>",
    unsafe_allow_html=True
)
