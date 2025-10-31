import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------- TITLE ----------------------
st.set_page_config(page_title="💧 Water Quality Prediction", layout="wide")
st.title("💧 Water Quality Prediction App")
st.markdown("This app predicts whether water is **Safe (Potable)** or **Unsafe (Non-potable)** based on various chemical properties.")

# ---------------------- LOAD DATA ----------------------
df = pd.read_csv("water_quality_dataset.csv")

st.success("✅ Dataset loaded successfully!")
st.write(df.head())

# Dataset Info
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()

st.subheader("📊 Dataset Info")
st.text(info_str)

st.subheader("🔍 Missing Values")
st.write(df.isnull().sum())

st.subheader("📈 Data Summary")
st.write(df.describe())

# ---------------------- VISUALIZATIONS ----------------------
st.subheader("📊 Data Visualizations")

# Potability Count Plot
fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.countplot(x='Potability', hue='Potability', data=df, palette='coolwarm', legend=False, ax=ax1)
ax1.set_title('Water Potability Distribution')
ax1.set_xlabel('Potability (0 = Unsafe, 1 = Safe)')
ax1.set_ylabel('Count')
st.pyplot(fig1)

# Correlation Heatmap
fig2, ax2 = plt.subplots(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt=".2f", ax=ax2)
ax2.set_title("Feature Correlation Heatmap")
st.pyplot(fig2)

# ---------------------- MODEL TRAINING ----------------------
X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("🎯 Model Performance")
accuracy = accuracy_score(y_test, y_pred) * 100
st.write(f"**Accuracy:** {accuracy:.2f}%")

st.write("**Confusion Matrix:**")
st.write(confusion_matrix(y_test, y_pred))

st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred))

# ---------------------- FEATURE IMPORTANCE ----------------------
st.subheader("🔥 Feature Importance")
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
fig3, ax3 = plt.subplots(figsize=(8, 6))
feature_importances.sort_values().plot(kind='barh', color='teal', ax=ax3)
ax3.set_title("Feature Importance (Random Forest)")
st.pyplot(fig3)

# ---------------------- USER INPUT ----------------------
st.subheader("💧 Predict Your Own Sample")

col1, col2, col3 = st.columns(3)
with col1:
    pH = st.number_input("pH (3–9)", min_value=3.0, max_value=9.0, value=6.5)
    Hardness = st.number_input("Hardness (0–200)", min_value=0.0, max_value=200.0, value=100.0)
    Solids = st.number_input("Solids (100–600)", min_value=100.0, max_value=600.0, value=300.0)
with col2:
    Chloramines = st.number_input("Chloramines (1–8)", min_value=1.0, max_value=8.0, value=5.0)
    Sulfate = st.number_input("Sulfate (0–400)", min_value=0.0, max_value=400.0, value=200.0)
    Conductivity = st.number_input("Conductivity (300–1200)", min_value=300.0, max_value=1200.0, value=750.0)
with col3:
    Organic_carbon = st.number_input("Organic Carbon (2–30)", min_value=2.0, max_value=30.0, value=10.0)
    Trihalomethanes = st.number_input("Trihalomethanes (0–150)", min_value=0.0, max_value=150.0, value=70.0)
    Turbidity = st.number_input("Turbidity (1–8)", min_value=1.0, max_value=8.0, value=4.0)

if st.button("🔮 Predict Water Quality"):
    user_df = pd.DataFrame([{
        'pH': pH,
        'Hardness': Hardness,
        'Solids': Solids,
        'Chloramines': Chloramines,
        'Sulfate': Sulfate,
        'Conductivity': Conductivity,
        'Organic_carbon': Organic_carbon,
        'Trihalomethanes': Trihalomethanes,
        'Turbidity': Turbidity
    }])

    prediction = model.predict(user_df)[0]

    if prediction == 1:
        st.success("✅ The water is SAFE for drinking (Potable).")
    else:
        st.error("⚠️ The water is UNSAFE for drinking (Non-potable).")

