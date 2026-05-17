# 📞 Telecom Customer Churn Prediction using ANN & Machine Learning

## 🚀 Project Overview

This project predicts whether a telecom customer is likely to churn using Artificial Neural Networks (ANN) and Machine Learning models.

The project compares ANN performance with traditional ML algorithms such as:

* Random Forest
* XGBoost

An interactive Streamlit web application was also developed for real-time churn prediction.

---

# 🎯 Problem Statement

Customer churn is a major challenge in the telecom industry. Predicting potential churners helps companies improve customer retention strategies and reduce revenue loss.

This project aims to:

* Predict customer churn
* Compare ANN with ML models
* Analyze model performance using advanced evaluation metrics
* Deploy the model using Streamlit

---

# 🧠 Models Used

## Machine Learning Models

* Random Forest
* XGBoost

## Deep Learning Model

* Artificial Neural Network (ANN)

---

# ⚙️ Features Implemented

✅ Data preprocessing & feature engineering
✅ Handling imbalanced data using SMOTE
✅ ANN with Dense & Dropout layers
✅ EarlyStopping for overfitting control
✅ Model evaluation using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

✅ ANN Learning Curves
✅ Confusion Matrix
✅ ROC Curve & Precision-Recall Curve
✅ Streamlit deployment

---

# 🛠 Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* XGBoost
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

# 📊 Results

The project compares ANN with ML models using multiple evaluation metrics and visualization techniques.

Key findings:

* Ensemble models performed strongly on structured telecom data
* ANN demonstrated competitive nonlinear learning capability

---

# 📂 Project Structure

```bash
Telecom-Customer-Churn-Prediction/
│
├── data/
│   └── telecom_churn.csv
│
├── models/
│   ├── ann_churn_model.h5
│   └── scaler.pkl
│
├── notebooks/
│   └── telecom_churn_ann.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ▶️ Run the Streamlit App

Install dependencies:

```bash
pip install -r requirements.txt
```

Run app:

```bash
streamlit run app.py
```

---

# 📌 Future Improvements

* SHAP Explainability
* Hyperparameter tuning
* Real-time API integration
* Advanced ANN architectures

---

