# WATER_QUALITY_FINAL_CLEAN.py
# -------------------------------------------------
# Water Quality Prediction (Streamlit + Data + Visualizations)
# Dataset: water_quality_dataset.csv
# -------------------------------------------------

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="💧 Water Quality Prediction", layout="centered")

st.title("💧 Water Quality Prediction System")
st.markdown("""
This web app predicts whether water is **safe (potable)** or **unsafe (non-potable)** 
based on chemical characteristics.
""")

# Step 1: Load dataset
df = pd.read_csv("water_quality_dataset.csv")
st.success("✅ Dataset loaded successfully!")

# Step 2: Dataset overview
with st.expander("📊 Dataset Overview"):
    st.write(df.head())
    st.write("### Dataset Info")
    buffer = []
    df.info(buf=buffer)
    info_str = "\n".join(map(str, buffer))
    st.text(info_str)
    st.write("### Missing Values", df.isnull().sum())
    st.write("### Data Summary", df.describe())

# Step 3: Visualizations
with st.expander("📈 Data Visualizations"):
    st.subheader("Potability Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Potability', hue='Potability', data=df, palette='coolwarm', legend=False, ax=ax1)
    ax1.set_title('Water Potability Distribution')
    ax1.set_xlabel('Potability (0 = Unsafe, 1 = Safe)')
    ax1.set_ylabel('Count')
    st.pyplot(fig1)

    st.subheader("Feature Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt=".2f", ax=ax2)
    st.pyplot(fig2)

# Step 4: Data preparation
X = df.drop('Potability', axis=1)
y = df['Potability']

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate model
y_pred = model.predict(X_test)

st.subheader("🎯 Model Performance")
st.write(f"**Accuracy:** {round(accuracy_score(y_test, y_pred) * 100, 2)} %")
st.write("**Confusion Matrix:**")
st.write(confusion_matrix(y_test, y_pred))
st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred))

# Step 8: Feature importance
st.subheader("🔥 Feature Importance")
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
fig3, ax3 = plt.subplots(figsize=(8, 6))
feature_importances.sort_values().plot(kind='barh', color='teal', ax=ax3)
st.pyplot(fig3)

# Step 9: User input for prediction
st.subheader("💧 Test with Your Own Water Sample")

pH = st.number_input("pH (3–9)", 3.0, 9.0, 7.0)
hardness = st.number_input("Hardness (0–200)", 0.0, 200.0, 100.0)
solids = st.number_input("Solids (100–600)", 100.0, 600.0, 300.0)
chloramines = st.number_input("Chloramines (1–8)", 1.0, 8.0, 4.0)
sulfate = st.number_input("Sulfate (0–400)", 0.0, 400.0, 200.0)
conductivity = st.number_input("Conductivity (300–1200)", 300.0, 1200.0, 600.0)
organic_carbon = st.number_input("Organic Carbon (2–30)", 2.0, 30.0, 10.0)
trihalomethanes = st.number_input("Trihalomethanes (0–150)", 0.0, 150.0, 75.0)
turbidity = st.number_input("Turbidity (1–8)", 1.0, 8.0, 3.0)

if st.button("🔍 Predict Potability"):
    user_data = pd.DataFrame([{
        'pH': pH,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }])

    prediction = model.predict(user_data)[0]

    if prediction == 1:
        st.success("✅ The water is **SAFE** for drinking (Potable).")
    else:
        st.error("⚠️ The water is **UNSAFE** for drinking (Non-potable).")

st.info("✅ Process completed successfully.")
