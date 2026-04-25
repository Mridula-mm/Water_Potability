# ==========================================
# WATER POTABILITY PROJECT (DEPLOY READY)
# ==========================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

# Streamlit
import streamlit as st


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Water AI", layout="wide")

# ==========================================
# FRONT PAGE (INTRO)
# ==========================================

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.title("💧 Smart Water Quality Monitoring System")
    st.image("Water.jpg", width=500)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("""
    ### 🌍 About This Project
    This AI-powered system predicts whether water is safe for drinking based on chemical properties.

    ### 🚀 Features
    - Machine Learning model comparison
    - Automatic best model selection
    - Interactive dashboard
    - Feature importance insights

    ### 🧠 Real-World Use
    - Smart cities water monitoring
    - IoT-based water testing
    - Public health safety systems
    """)

st.markdown("---")


# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    df.fillna(df.mean(), inplace=True)
    return df


# ==========================================
# PREPROCESS
# ==========================================
def preprocess(df):
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    smote = SMOTE()
    X, y = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler


# ==========================================
# TRAIN MODELS
# ==========================================
def train_models(X_train, y_train):
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier()
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


# ==========================================
# EVALUATE
# ==========================================
def evaluate(models, X_test, y_test):

    results = []
    roc_data = {}

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results.append({"Model": name, "Accuracy": acc})

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr)

    return pd.DataFrame(results), roc_data, best_model, best_name, best_score


# ==========================================
# MAIN LOGIC
# ==========================================
df = load_data()
X, y, scaler = preprocess(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = train_models(X_train, y_train)
acc_df, roc_data, best_model, best_name, best_score = evaluate(
    models, X_test, y_test
)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.success(f"🏆 Best Model: {best_name} | Accuracy: {best_score:.2f}")

st.markdown("---")


# ==========================================
# IMAGE + INPUT
# ==========================================
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("water2.jpg", width=500)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("## 🔍 Enter Water Parameters")

    ph = st.slider("pH", 0.0, 14.0, 7.0)
    hardness = st.slider("Hardness", 0.0, 500.0, 200.0)
    solids = st.slider("Solids", 0.0, 50000.0, 10000.0)
    chloramines = st.slider("Chloramines", 0.0, 20.0, 5.0)
    sulfate = st.slider("Sulfate", 0.0, 500.0, 300.0)
    conductivity = st.slider("Conductivity", 0.0, 1000.0, 400.0)
    organic_carbon = st.slider("Organic Carbon", 0.0, 50.0, 10.0)
    trihalomethanes = st.slider("Trihalomethanes", 0.0, 200.0, 80.0)
    turbidity = st.slider("Turbidity", 0.0, 10.0, 3.0)

    if st.button("🚀 Predict Water Quality"):
        input_data = np.array([[ph, hardness, solids, chloramines,
                                sulfate, conductivity, organic_carbon,
                                trihalomethanes, turbidity]])

        input_data = scaler.transform(input_data)
        result = best_model.predict(input_data)

        if result[0] == 1:
            st.success("✅ Safe to Drink")
        else:
            st.error("❌ Not Safe")

st.markdown("---")


# ==========================================
# DASHBOARD
# ==========================================

fig = px.bar(acc_df, x="Model", y="Accuracy",
             color="Accuracy", text="Accuracy")
fig.update_layout(height=350, width=600)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.subheader("📊 Model Accuracy Comparison")
    st.plotly_chart(fig)

st.markdown("<br>", unsafe_allow_html=True)

fig2 = go.Figure()
for name, (fpr, tpr) in roc_data.items():
    fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name))

fig2.add_trace(go.Scatter(x=[0,1], y=[0,1],
                         mode='lines', name='Random',
                         line=dict(dash='dash')))
fig2.update_layout(height=350, width=600)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.subheader("📈 ROC Curve Comparison")
    st.plotly_chart(fig2)

st.markdown("---")


# ==========================================
# FEATURE IMPORTANCE
# ==========================================
if hasattr(best_model, "feature_importances_"):
    features = df.drop("Potability", axis=1).columns
    importance = best_model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig3 = px.bar(imp_df, x="Feature", y="Importance",
                  color="Importance")
    fig3.update_layout(height=350, width=600)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.subheader("🔥 Feature Importance")
        st.plotly_chart(fig3)

st.markdown("---")



# ==========================================
# FOOTER IMAGE CARD (CLICK TO SWAP)
# ==========================================

st.markdown(
    "<h2 style='text-align: center;'>📈 ROC Curve Comparison</h2>",
    unsafe_allow_html=True
)

# ---------- CSS ----------
st.markdown("""
<style>
.footer-card {
    background-color: #0e1117;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    text-align: center;
}
.footer-title {
    font-size: 20px;
    font-weight: bold;
    margin-top: 10px;
    color: white;
}
.footer-desc {
    font-size: 14px;
    margin-top: 8px;
    color: #cccccc;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ---------- IMAGE DATA ----------
images = [
    "water4.png",
    "water5.png",
    "water6.png",
    "water7.png",
    "water8.png",
    "water9.png"
]

titles = [
    "Drink Healthy",
    "Optimal Physical Energy and Vitality",
    "Enhanced Digestion and Nutrient Absorption",
    "Clear Skin and Complexion",
    "Efficient Kidney Function and Detoxification",
    "Robust Immune System Support"
]

descriptions = [
    "clean water is the cornerstone of health and wellness",
    "The clear water is light and pure. Choosing the clean water is the first step toward sustained physical vitality.",
    "Water is crucial for the breakdown of food and the absorption of nutrients in the intestinal tract. It helps dissolve vitamins, minerals, and other nutrients, making them accessible to the body.",
    "Proper hydration with hygienic water helps flush toxins from the body and maintains skin elasticity.",
    "The kidneys are the body's main filtration system, processing approximately 200 quarts of fluid daily to remove waste products and toxins. Adequate intake of hygienic water is essential for this process.",
    "Water is the primary component of lymph, the fluid that circulates immune cells throughout the body. Hygienic water maintains the production of lymph and supports the overall function of the immune system."
]


# ---------- SESSION STATE ----------
if "img_index" not in st.session_state:
    st.session_state.img_index = 0


# ---------- BUTTON ----------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    if st.button("➡️ Next Image"):
        st.session_state.img_index = (st.session_state.img_index + 1) % len(images)


# ---------- DISPLAY CARD ----------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown('<div class="footer-card">', unsafe_allow_html=True)

    st.image(images[st.session_state.img_index], width=450)

    # TITLE
    st.markdown(
        f'<div class="footer-title">{titles[st.session_state.img_index]}</div>',
        unsafe_allow_html=True
    )

    # ✅ DESCRIPTION (THIS IS WHAT YOU NEEDED)
    st.markdown(
        f'<div class="footer-desc">{descriptions[st.session_state.img_index]}</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


st.markdown(
    "<h2 style='text-align: center;'>🚀 Built with Machine Learning & Streamlit</h2>",
    unsafe_allow_html=True
)
