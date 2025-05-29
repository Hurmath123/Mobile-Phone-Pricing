# Mobile Price Range Predictor

A Streamlit web app that predicts the price range of a mobile phone based on its specifications using a machine learning model.

---

## Features

- Accepts user input for top 5 different phone important features
- Uses a stacked ensemble model (Logistic Regression, XGBoost, SVM)
- Predicts price range: **Low**, **Medium**, **High**, or **Very High**
- Clean, interactive UI built with **Streamlit**

---

## Model Details

- Final Model: **StackingClassifier**
- Base Models: Tuned Logistic Regression, XGBoost, and SVM
- Accuracy: **98%**
- F1 Score: **0.98**
- Scaled input using `StandardScaler`

---

## How to Use

### 1. Clone this repository
```bash
git clone https://github.com/Hurmath123/Mobile-Phone-Pricing.git
cd Mobile-Phone-Pricing
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app locally
```bash
Python -m streamlit run app.py
```

---

## Deploy on Streamlit Cloud

1. Push this repo to your GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New App" and select your repo
4. Set `app.py` as the main file
5. Deploy!

---


