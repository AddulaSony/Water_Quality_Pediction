# 💧 Water Quality Prediction App

A **Streamlit-based web application** that predicts whether water is **safe (potable)** or **unsafe (non-potable)** for drinking based on various water quality parameters.

This project uses **machine learning (Random Forest Classifier)** to analyze water quality data and provide predictions along with data visualizations and model performance insights.

---

## 🚀 Features

* 🧠 Machine learning–based **water potability prediction**
* 📊 Interactive **data exploration and visualizations**
* 🔍 **Feature importance** insights using Random Forest
* 💬 User-friendly **Streamlit interface**
* ⚡ Works both **locally** and on **Streamlit Cloud**

---

## 🧩 Tech Stack

* **Python 3.8+**
* **Streamlit** – Web App Framework
* **Pandas** – Data Manipulation
* **Seaborn & Matplotlib** – Data Visualization
* **Scikit-learn** – Machine Learning

---

## 📂 Project Structure

```
water_quality_prediction/
│
├── ip.py                      # Main Streamlit app
├── water_quality_dataset.csv  # Dataset used for training/testing
├── requirements.txt           # Required Python packages
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/water_quality_prediction.git
cd water_quality_prediction
```

### 2️⃣ Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On macOS/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App Locally

```bash
streamlit run app.py
```

App will open in the browser at:
👉 https://waterqualitypediction-ggz8jt4mehserfkvl8se22.streamlit.app/

---

## 🌐 Deploying to Streamlit Cloud

1. Push all files (`app.py`, `requirements.txt`, `water_quality_dataset.csv`, `README.md`) to your GitHub repository.
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Click **“New app”**, connect your GitHub repo, and select:

   * **Main file path:** `app.py`
   * **Branch:** main (or whichever branch you use)
4. Click **Deploy** 🎉

---

## 📊 Dataset Information

* The dataset contains various features like:

  * pH
  * Hardness
  * Solids
  * Chloramines
  * Sulfate
  * Conductivity
  * Organic Carbon
  * Trihalomethanes
  * Turbidity
* Target column: **Potability** (0 = Unsafe, 1 = Safe)

---

## 🧠 Model Details

* **Algorithm Used:** Random Forest Classifier
* **Accuracy Achieved:** ~70–80% (depending on dataset)
* **Evaluation Metrics:** Accuracy, Confusion Matrix, Classification Report

---

## 📸 App Preview

Example components of the app:

* Water potability distribution chart
* Correlation heatmap
* Model accuracy metrics
* Custom input section for water testing

---

## 🤝 Contributing

Feel free to fork this repository, raise issues, and submit pull requests to improve the project!

---

## 🧑‍💻 Author

**Developed by:** Sonyreddy Addula
🌐 GitHub: https://github.com/AddulaSony

---

## 🪪 License

This project is open-source under the **MIT License**.
