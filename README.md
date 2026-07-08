# 🧠 MindEase

**An AI-assisted mental health companion that detects stress from natural language, using three classic ML algorithms implemented from scratch with NumPy.**

MindEase lets a user describe how they're feeling in plain text. Three independently-trained models — a linear regressor, a support vector machine, and a decision tree — each score the input, and an ensemble combines them into a single stress read. A small CBT/mindfulness engine then responds with a validating message and a matching coping technique.

---

## 📸 Screenshots

<img width="1350" height="645" alt="MindEase home screen" src="https://github.com/user-attachments/assets/7b603ab5-61d7-483a-9e8c-cf85f8ba32e4" />
<img width="1352" height="640" alt="MindEase chat mode" src="https://github.com/user-attachments/assets/d08a05e4-581a-421b-b89a-59975150688b" />
<img width="1366" height="591" alt="MindEase wellness check" src="https://github.com/user-attachments/assets/ae55cfd7-8b08-4731-b867-6680c8a5c070" />

---

## 🔍 How It Works

```
User text
   │
   ▼
1. Feature Extraction (12 signals)
   negative/positive word counts, negation-aware flips,
   exclamation & caps ratio, intensifiers, absolutist
   ("always"/"never") language, first-person ratio, etc.
   │
   ▼
2. Three models run in parallel, each trained from scratch
   ┌───────────────────────┬──────────────────────┬───────────────────────┐
   │  Linear Regression    │  Support Vector       │  Decision Tree        │
   │  (gradient descent)   │  Machine (hinge loss) │  (greedy Gini splits) │
   │  → continuous 0–10    │  → STRESSED / NOT     │  → LOW…HIGH category  │
   └───────────────────────┴──────────────────────┴───────────────────────┘
   │
   ▼
3. Ensemble score (weighted vote, weights set from validation performance)
   │
   ▼
4. CBT/mindfulness response engine
   picks a severity-matched message + technique, and gently names any
   detected cognitive distortion (all-or-nothing thinking, catastrophizing)
   │
   ▼
5. Visual feedback — gauge, mood breakdown, category radar/bar charts
```

Every model is trained on a hand-labeled 100-example dataset (`dataset.py`) with an 80/20 train/test split — see `train.py` to reproduce the numbers below from scratch.

---

## 📊 Model Performance

Measured on a held-out 20-example test set (never seen during training), by re-running `train.py`:

| Model | Task | Metric | Score |
|---|---|---|---|
| Linear Regression | Stress score (0–10, regression) | MAE | **1.21 points** |
| | | R² | **0.71** |
| SVM | Stressed / Not Stressed (binary) | Accuracy | **85%** |
| | | F1 | **0.82** |
| Decision Tree | LOW / LOW-MED / MEDIUM / HIGH (4-class) | Exact accuracy | **55%** |
| | | Adjacent (±1 level) accuracy | **90%** |
| Ensemble | 4-class (thresholded from ensemble score) | Accuracy | **60%** |

**Honest context, because a portfolio should hold up under questions:**
- The dataset is small (100 examples) and self-authored, not a public benchmark — treat these as directionally meaningful, not clinically validated.
- The decision tree's 4-class exact accuracy (55%) is the weakest link: with only 12 features and ordinal, overlapping categories, a shallow greedy tree confuses *adjacent* stress levels (e.g. MEDIUM vs. HIGH) far more than distant ones — hence the much higher 90% ±1-level accuracy. That's a real, documented limitation, not a rounding trick.
- The SVM's binary task (genuinely stressed vs. not) is easier and the model performs solidly on it.
- Retrain anytime with `python train.py` — it prints all of these numbers fresh.

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **NumPy** — all three ML models implemented from scratch (batch gradient descent, hinge-loss subgradient descent, greedy Gini-impurity tree splitting) — no scikit-learn
- **Streamlit** — UI
- **Matplotlib + Pillow** — gauge, radar, bar, and timeline charts

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain the models on the included dataset and regenerate the metrics table above:
```bash
python train.py
```

## 📁 Project Structure

```
app.py               # Streamlit app + trained model parameters
dataset.py           # Hand-labeled training examples (100 samples)
features.py          # From-scratch feature extraction
train.py             # Trains LR / SVM / Decision Tree, evaluates, saves params
trained_params.json  # Output of the last training run
requirements.txt
```

## 🧘 Psychology Frameworks

- **CBT (Cognitive Behavioural Therapy)** — validates before advising, and flags simple cognitive distortions (all-or-nothing thinking, catastrophizing, self-blame) detected in the user's phrasing.
- **Mindfulness-Based Stress Reduction** — grounding techniques (4-7-8 breathing, 5-4-3-2-1) matched to detected severity.
- **Positive Psychology** — reinforces existing strengths and coping capacity rather than only flagging deficits.

## ⚠️ Disclaimer

MindEase is a student/portfolio project and is **not** a substitute for professional mental health care. If you are experiencing severe distress, please reach out to a professional or a crisis line in your area.

## 📜 License

Apache License 2.0 — see `LICENSE` for details.

---

*Built by a BS Computer Science student at Superior University, Lahore.*
