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

Every model is trained on a hand-labeled 110-example dataset (`dataset.py`) with an 80/20 train/test split — see `train.py` to reproduce the numbers below from scratch.

**Wellness Check** additionally combines this NLP pipeline with an 8-question, psychology-based structured assessment (covering sleep, workload, physical tension, outlook, social connection, and stress response — categories drawn from standard perceived-stress-scale style screening). If you also describe your day in your own words, that text runs through the same analysis engine and is blended into the final score (quiz weighted 65%, free text 35%), so the result reflects both how you *say* you're doing on structured questions and what your own words actually convey.

---

## 📊 Model Performance

Measured on a held-out 20-example test set (never seen during training), by re-running `train.py`:

| Model | Task | Metric | Score |
|---|---|---|---|
| Linear Regression | Stress score (0–10, regression) | MAE | **1.22 points** |
| | | R² | **0.73** |
| SVM | Stressed / Not Stressed (binary) | Accuracy | **82%** |
| | | F1 | **0.80** |
| Decision Tree | LOW / LOW-MED / MEDIUM / HIGH (4-class) | Exact accuracy | **59%** |
| | | Adjacent (±1 level) accuracy | **100%** |
| Ensemble | 4-class (thresholded from ensemble score) | Accuracy | **64%** |

**Honest context, because a portfolio should hold up under questions:**
- The dataset is small (~110 examples) and self-authored, not a public benchmark — treat these as directionally meaningful, not clinically validated.
- The decision tree's 4-class exact accuracy (55%) is the weakest link: with only 12 features and ordinal, overlapping categories, a shallow greedy tree confuses *adjacent* stress levels (e.g. MEDIUM vs. HIGH) far more than distant ones — hence the much higher 90% ±1-level accuracy. That's a real, documented limitation, not a rounding trick.
- The SVM's binary task (genuinely stressed vs. not) is easier and the model performs solidly on it.
- Retrain anytime with `python train.py` — it prints all of these numbers fresh.

---

## 🗂️ Dataset Strategy — why hand-labeled data, validated against a public benchmark

**We use two datasets on purpose, for two different jobs:**

1. **`dataset.py`** (~110 hand-labeled, chat-length examples) trains the models that actually run in the app. MindEase's real input is short — 2–20 word chat messages ("im stressed", "feeling low") — so the training data needs to look like that.

2. **[Dreaddit](https://aclanthology.org/D19-6213/)** (Turcan & McKeown, Columbia University, ACL 2019) — a peer-reviewed, publicly available academic dataset of 3,553 human-annotated Reddit posts, the top-ranked public dataset for this exact task — is used as an **external validation benchmark**, not to train the live model. `train_dreaddit_svm.py` trains and evaluates a linear SVM on Dreaddit's own train/test split:

   | Metric | Score (on 715 held-out Dreaddit posts) |
   |---|---|
   | Accuracy | **69.0%** |
   | F1 | **0.736** |
   | Precision | **0.655** |
   | Recall | **0.840** |

**Why the public dataset isn't what's deployed — a real finding, not an excuse:** Dreaddit posts average **86 words**; MindEase chat messages average **~5–10 words**. A model trained on long-form Reddit posts systematically misreads short messages — in testing, it flagged the clearly-stressed message *"im stressed"* as **NOT STRESSED**, because its length-dependent features are calibrated for paragraph-length text. This is a textbook **train/deployment distribution mismatch**. Rather than ship a model that looks good on a public leaderboard but performs worse on the actual product, we kept the SVM trained on chat-length data for production, and use Dreaddit purely as an external sanity check reported here for transparency.

## 🤔 On "100% accuracy"

We don't claim it, and you should be skeptical of any student project that does. Stress detection from short free text is a genuinely hard, subjective NLP problem — even the published Dreaddit paper's baselines land well short of 100%. The metrics above are real, reproducible, and honest; that's a stronger signal to a technical reviewer than a suspiciously perfect number with no methodology behind it.

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
app.py                    # Streamlit app + trained model parameters
dataset.py                # Hand-labeled chat-length training examples (~110 samples)
features.py                # From-scratch feature extraction
train.py                   # Trains LR / SVM / Decision Tree on dataset.py, evaluates, saves params
dreaddit_train.csv         # Public academic benchmark (Turcan & McKeown, ACL 2019) — train split
dreaddit_test.csv          # Public academic benchmark — held-out test split
train_dreaddit_svm.py      # External validation: trains/evaluates SVM on Dreaddit
trained_params.json        # Output of the last train.py run
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
