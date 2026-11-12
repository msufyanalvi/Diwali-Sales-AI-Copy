
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Diwali Sales AI",
    page_icon="🛍️",
    layout="wide"
)

# ---------------- BLACK + GOLD THEME ----------------
st.markdown("""
<style>

.stApp {
    background-color: #101010;
    color: #F5F5F5;
}

[data-testid="stHeader"] {
    background-color: #101010;
}

[data-testid="stSidebar"] {
    background-color: #181818;
}

.main-title {
    color: #D4AF37;
    font-size: 42px;
    font-weight: 900;
    text-align: center;
}

.subtitle {
    color: #C8B77B;
    text-align: center;
    font-size: 18px;
}

.hero {
    background: linear-gradient(120deg, #181818, #39301C);
    padding: 40px;
    border-radius: 18px;
    border: 1px solid #D4AF37;
    text-align: center;
}

.hero h1 {
    color: #E8C65A;
    font-size: 38px;
}

.hero h3 {
    color: #FFFFFF;
}

.hero p {
    color: #E5D9B0;
    font-size: 17px;
}

.feature-card {
    background-color: #1C1C1C;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #A88932;
    min-height: 150px;
}

.feature-card h3 {
    color: #E8C65A;
}

.feature-card p {
    color: #DDDDDD;
}

div.stButton > button {
    background: linear-gradient(120deg, #B38B22, #E8C65A);
    color: #111111;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    padding: 12px 25px;
}

div.stButton > button:hover {
    background: #FFFFFF;
    color: #111111;
    border: 1px solid #D4AF37;
}

div[data-testid="stMetric"] {
    background-color: #1C1C1C;
    border: 1px solid #D4AF37;
    padding: 20px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    color: #E8C65A;
}

.stSelectbox label,
.stNumberInput label {
    color: #E8C65A !important;
}

hr {
    border-color: #8C7027;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD AI MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("diwali_sales_ai_model.pkl")
    columns = joblib.load("diwali_feature_columns.pkl")
    return model, columns

model, feature_columns = load_model()


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">🛍️ DIWALI SALES AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Premium AI Customer Purchase Predictor</div>',
    unsafe_allow_html=True
)

st.write("")


# ---------------- NAVIGATION ----------------
page = st.radio(
    "Navigation",
    ["Home", "Predict", "About"],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()


# ==================================================
# HOME PAGE
# ==================================================
if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to Diwali Sales AI</h1>
        <h3>Predict. Analyze. Discover.</h3>
        <p>
        Discover customer shopping patterns with
        Artificial Intelligence and Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.subheader("✨ Explore Our Features")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI Prediction</h3>
            <p>Predict estimated customer purchase amounts.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Smart Analysis</h3>
            <p>Explore customer information and shopping patterns.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Instant Results</h3>
            <p>Get AI-based estimates in just a few seconds.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.info("Select the Predict page above to try the AI model.")


# ==================================================
# PREDICTION PAGE
# ==================================================
elif page == "Predict":

    st.header("👤 Customer Purchase Predictor")

    st.write(
        "Enter customer information to estimate their purchase amount."
    )

    with st.form("customer_form"):

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["F", "M"]
            )

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
            "✨ Predict Purchase Amount",
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

        st.markdown("""
        <div class="hero">
            <h1>🎉 Prediction Completed!</h1>
            <p>Your estimated customer purchase amount</p>
        </div>
        """, unsafe_allow_html=True)

        st.metric(
            "Estimated Purchase Amount",
            f"₹ {prediction:,.2f}"
        )

        st.success("Your AI prediction is ready!")

        st.caption(
            "This is an estimated amount, not a guaranteed actual purchase."
        )


# ==================================================
# ABOUT PAGE
# ==================================================
elif page == "About":

    st.header("📖 About Diwali Sales AI")

    st.write("""
    Diwali Sales AI is a Machine Learning project
    designed to estimate customer purchase amounts
    using customer information and shopping patterns.

    This project uses the Diwali Sales dataset and
    a trained regression model.
    """)

    st.divider()

    st.subheader("🤖 AI Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("R² Score", "63.60%")
    c2.metric("MAE", "₹2,373.81")
    c3.metric("RMSE", "₹3,157.65")

    st.info(
        "R² is a regression performance metric, "
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


# ---------------- FOOTER ----------------
st.divider()

st.markdown("""
<p style="text-align:center;color:#C8B77B;">
    🛍️ DIWALI SALES AI
    <br>
    Developed with Python, Streamlit & Machine Learning
</p>
""", unsafe_allow_html=True)
