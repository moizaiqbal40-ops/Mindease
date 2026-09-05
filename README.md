<div align="center">

# 🧠 MindEase

### A transparent ML prototype for stress-aware language analysis.

<p>
  <a href="https://mindease-asvwrt2y6admlbex4gy43m.streamlit.app/">🌐 Live Demo</a>
</p>

</div>

## 💡 What It Is

MindEase is a Streamlit app that analyzes written language and estimates stress using three ML approaches built with NumPy. I built it to explore feature engineering, model implementation, and how ML can become a usable product instead of just a notebook.

> Educational prototype only — the result is not a medical or psychological diagnosis.

## 🛠️ Tech Stack

- **Python** · NumPy
- **Streamlit** · Matplotlib · Pillow
- Hand-built text features with Python/Regex
- Linear Regression · Linear SVM · Decision Tree

## ⚙️ How It Works

```text
User Text
   ↓
12 Text Features
   ↓
Regression + SVM + Decision Tree
   ↓
Ensemble Estimate
   ↓
Visual Feedback
```

The models are implemented from scratch rather than using ready-made ML estimators, making the learning and prediction pipeline easy to inspect.

## 🚀 Run Locally

```bash
git clone https://github.com/moizaiqbal40-ops/Mindease.git
cd Mindease
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## ✨ What I Learned / Challenges

The biggest challenge was implementing the ML fundamentals myself and then connecting the models, feature engineering, evaluation, and UI into one working application.

## 📸 Screenshots

<p align="center">
<img src="https://github.com/user-attachments/assets/c7410d2a-2ca0-40fb-b622-b4e74537da57" width="31%" alt="MindEase application screenshot 1">
<img src="https://github.com/user-attachments/assets/7f2e7140-2aca-4e7b-ac29-106bf047628d" width="31%" alt="MindEase application screenshot 2">
<img src="https://github.com/user-attachments/assets/72c8224a-d225-4ffa-a337-b1da4fe9b816" width="31%" alt="MindEase application screenshot 3">
</p>
