
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diwali Sales AI",
    page_icon="🛍️",
    layout="wide"
)

# Design
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
.hero {
    background: linear-gradient(120deg, #174ea6, #4285f4);
    padding: 35px;
    border-radius: 18px;
    color: white;
    text-align: center;
}
.hero h1 {
    color: white;
}
.feature-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #d9e7ff;
    min-height: 140px;
}
</style>
""", unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_model():
    model = joblib.load("diwali_sales_ai_model.pkl")
    columns = joblib.load("diwali_feature_columns.pkl")
    return model, columns

model, feature_columns = load_model()


# Header
st.markdown(
    '<div class="main-title">🛍️ Diwali Sales AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Smart Customer Purchase Prediction</div>',
    unsafe_allow_html=True
)

st.write("")

# Navigation
page = st.radio(
    "Navigation",
    ["Home", "Predict", "About"],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()


# HOME PAGE
if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to Diwali Sales AI</h1>
        <h3>Predict. Analyze. Discover.</h3>
        <p>
        Use Artificial Intelligence to estimate
        customer shopping amounts.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.subheader("✨ Our Features")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
        <h3>🤖 AI Prediction</h3>
        <p>Estimate customer purchase amounts.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
        <h3>📊 Smart Analysis</h3>
        <p>Analyze customer shopping information.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
        <h3>⚡ Instant Results</h3>
        <p>Get estimated results in seconds.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.info("Go to the Predict page to try our AI model.")


# PREDICTION PAGE
elif page == "Predict":

    st.header("👤 Customer Purchase Predictor")
    st.write("Enter customer details below.")

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
                format_func=lambda x:
                    "Single" if x == 0 else "Married"
            )

            orders = st.number_input(
                "Number of Orders",
                min_value=1,
                value=3
            )

            state = st.selectbox("State", [
                "Maharashtra", "Andhra Pradesh",
                "Uttar Pradesh", "Karnataka",
                "Gujarat", "Himachal Pradesh", "Delhi",
                "Jharkhand", "Kerala", "Haryana",
                "Madhya Pradesh", "Bihar", "Rajasthan",
                "Uttarakhand", "Telangana", "Punjab"
            ])

        with col2:
            zone = st.selectbox("Zone", [
                "Western", "Southern", "Central",
                "Northern", "Eastern"
            ])

            occupation = st.selectbox("Occupation", [
                "Healthcare", "Govt", "Automobile",
                "Construction", "Food Processing",
                "Lawyer", "Media", "Banking",
                "Retail", "IT Sector", "Aviation",
                "Hospitality", "Agriculture",
                "Textile", "Chemical"
            ])

            product = st.selectbox("Product Category", [
                "Auto", "Hand & Power Tools",
                "Stationery", "Tupperware",
                "Footwear & Shoes", "Furniture",
                "Food", "Games & Toys",
                "Sports Products", "Books",
                "Electronics & Gadgets", "Decor",
                "Clothing & Apparel", "Beauty",
                "Household items", "Pet Care",
                "Veterinary", "Office"
            ])

        submitted = st.form_submit_button(
            "🔮 Predict Purchase Amount",
            use_container_width=True
        )

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
            target = value.strip().replace(
                "\xa0", " "
            ).lower()

            for col in customer.columns:
                if col.startswith(prefix + "_"):
                    category = col[len(prefix) + 1:]

                    if category.strip().replace(
                        "\xa0", " "
                    ).lower() == target:
                        customer.loc[0, col] = 1
                        return

        set_category("Gender", gender)
        set_category("Age Group", age_group)
        set_category("State", state)
        set_category("Zone", zone)
        set_category("Occupation", occupation)
        set_category("Product_Category", product)

        prediction = model.predict(customer)[0]

        st.divider()
        st.subheader("🎉 Prediction Result")

        st.metric(
            "Estimated Purchase Amount",
            f"₹ {prediction:,.2f}"
        )

        st.success("AI prediction completed successfully!")

        st.caption(
            "This is an estimated amount, not a guaranteed purchase."
        )


# ABOUT PAGE
elif page == "About":

    st.header("📖 About This Project")

    st.write("""
    Diwali Sales AI is a Machine Learning project
    that predicts customer purchase amounts using
    customer information and shopping patterns.

    The model was trained using the Diwali Sales dataset.
    """)

    st.subheader("🤖 Machine Learning Model")

    st.write("Model: HistGradientBoostingRegressor")

    c1, c2, c3 = st.columns(3)

    c1.metric("R² Score", "63.60%")
    c2.metric("MAE", "₹2,373.81")
    c3.metric("RMSE", "₹3,157.65")

    st.info(
        "R² Score is a regression performance metric, "
        "not classification accuracy."
    )

    st.subheader("🛠️ Technologies Used")

    st.write("""
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    - Machine Learning
    """)


# Footer
st.divider()

st.markdown(
    "<p style='text-align:center;color:#718096;'>"
    "Diwali Sales AI | Built with Python and Streamlit"
    "</p>",
    unsafe_allow_html=True
)
