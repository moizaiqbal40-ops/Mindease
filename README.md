<div align="center">

# 🧠 MindEase

### AI-assisted stress estimation from natural language — with transparent ML and safety-aware design.

MindEase is a Streamlit mental-wellness prototype that analyzes written language and estimates stress using three machine-learning approaches implemented from scratch with NumPy. The system combines interpretable text features, multiple prediction models, visual feedback, and psychology-informed guidance while clearly positioning itself as a supportive software prototype rather than a diagnostic tool.

<p>

<a href="https://mindease-asvwrt2y6admlbex4gy43m.streamlit.app/">🌐 Live Demo</a> •
<a href="https://github.com/moizaiqbal40-ops/Mindease">💻 Source Code</a>

</p>

</div>

---

## 🎯 Problem

Stress can be difficult to communicate through fixed questionnaires alone. Many lightweight wellness applications provide generic responses, while formal assessment requires professional context.

MindEase explores a middle ground: let a user describe their experience in natural language, extract transparent language signals, estimate a stress level, and return supportive guidance without presenting the result as a clinical diagnosis.

## 💡 Solution

The application follows a multi-model pipeline:

```text
User Text
   ↓
Text Feature Extraction
   ↓
Feature Normalization
   ↓
┌──────────────────┬──────────────────┬──────────────────┐
│ Linear Regression│    Linear SVM    │  Decision Tree   │
│   stress score   │  stressed/not    │ stress level 0–3 │
└──────────────────┴──────────────────┴──────────────────┘
                    ↓
              Ensemble Estimate
                    ↓
        Visual Feedback + Guidance
```

The training pipeline uses a deterministic 80/20 train-test split after shuffling the dataset with a fixed random seed. The trained parameters and evaluation metrics are exported to `trained_params.json` for the application to use.

---

## 🧠 Machine Learning Pipeline

### 1. Feature Engineering

MindEase does not depend on a large NLP framework for its core feature extraction. `features.py` derives 12 numerical signals using Python regex/string processing and NumPy, including:

- negative and positive lexical signals
- text length
- exclamation/question counts
- capitalization ratio
- lexical diversity
- intensifiers
- negation-aware flips
- first-person ratio
- absolutist language
- elongated/repeated characters

Features are z-score normalized using training-set statistics so raw scales such as word count do not dominate the models.

### 2. Linear Regression — From Scratch

A linear regression model predicts a continuous stress score from 0–10 using batch gradient descent with L2 regularization.

### 3. Linear SVM — From Scratch

A binary linear SVM uses hinge loss and subgradient descent to estimate whether the text belongs to the stressed/non-stressed class.

### 4. Decision Tree — From Scratch

A decision tree is constructed using greedy Gini-impurity splits, with configurable depth and minimum-sample constraints.

### 5. Ensemble

The application combines the model outputs into a final bounded score and maps that score into four stress levels. This demonstrates how different model outputs can be combined rather than relying on a single algorithm.

---

## 📊 Evaluation

The training script calculates model-specific metrics on the held-out test set, including:

| Model | Evaluation |
|---|---|
| Linear Regression | MAE, RMSE, R² |
| Linear SVM | Accuracy, F1, Precision, Recall |
| Decision Tree | Accuracy, Macro-F1, ±1 adjacent accuracy |
| Ensemble | Accuracy, Macro-F1 |

Run the training script to generate the current values rather than hard-coding potentially stale results into the README:

```bash
python train.py
```

This keeps the repository honest and makes the evaluation reproducible from the checked-in training code.

---

## ✨ Key Features

| Feature | What it demonstrates |
|---|---|
| 🧠 Natural-language analysis | Text-based feature engineering without a black-box NLP pipeline |
| ⚙️ Three ML algorithms | Regression, SVM classification, and decision-tree classification implemented with NumPy |
| 🔀 Ensemble scoring | Combines independent model outputs into one estimate |
| 📈 Visual feedback | Stress gauges, charts, and wellness summaries in the Streamlit UI |
| 🤝 Supportive guidance | Psychology-informed coping suggestions alongside the estimate |
| 🛡️ Safety-aware routing | Dedicated safety logic can prioritize supportive resources instead of normal prediction |
| 🔍 Transparent engineering | Feature definitions, training logic, metrics, and limitations are visible in the repository |

---

## 🖥️ Application

MindEase is built as a Streamlit application with a custom light, calming interface. The live prototype provides an interactive text-analysis experience rather than a static model demo.

**Live:**

https://mindease-asvwrt2y6admlbex4gy43m.streamlit.app/

> **Important:** MindEase is an educational/software engineering prototype. Its stress estimate should not be treated as a medical or psychological diagnosis, and the application is not a replacement for qualified professional support.

---

## 🗂️ Project Structure

```text
Mindease/
├── app.py                       # Streamlit application and UI
├── train.py                     # Training, evaluation, and parameter export
├── dataset.py                   # Dataset loading/preparation
├── features.py                  # Hand-built text feature extraction
├── Train dreaddit svm.py        # Additional SVM training experiment
├── requirements.txt             # Python dependencies
├── trained_params.json          # Generated model parameters/metrics
├── MindEase_Documentation.pdf   # Project documentation
├── MindEase_Presentation.pptx   # Project presentation
├── assets/                      # Application assets
└── LICENSE                      # MIT License
```

---

## 🛠️ Tech Stack

- **Python**
- **NumPy** — numerical computation and ML implementation
- **Streamlit** — interactive web application
- **Matplotlib** — visualizations
- **Pillow** — image handling
- **Regex / Python standard library** — text feature extraction

No scikit-learn model implementation is required for the core three algorithms; the training logic is written directly in the repository.

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/moizaiqbal40-ops/Mindease.git
cd Mindease
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the models

```bash
python train.py
```

This creates/updates `trained_params.json` with the learned parameters and evaluation metrics.

### 5. Launch MindEase

```bash
streamlit run app.py
```

---

## 🔬 Engineering Decisions

### Why build the models from scratch?

The project is intentionally designed to demonstrate understanding of the mechanics behind common ML algorithms: gradient descent, hinge loss, Gini impurity, feature normalization, prediction, and evaluation.

### Why hand-built features?

The feature extractor makes the model easier to inspect and debug. Each signal has a clear relationship to the text rather than being hidden inside a large pretrained language model.

### Why an ensemble?

The three models approach the problem differently. Combining their outputs provides an engineering exercise in coordinating regression and classification signals into a single application-level estimate.

### Why keep the limitations visible?

Mental-wellness software requires extra care around uncertainty, dataset quality, and interpretation. Documenting these constraints is part of the engineering work, not an afterthought.

---

## ⚠️ Current Limitations

- The feature space is relatively small compared with modern NLP systems.
- Lexical signals can miss context, sarcasm, cultural nuance, and complex language.
- The dataset and labels limit how generalizable the model can be.
- The train/test strategy is a simple shuffled holdout rather than cross-validation.
- The ensemble weights are heuristic rather than learned through a separate optimization procedure.
- The system should not be interpreted as a clinical assessment tool.

---

## 🔮 Future Improvements

- Add stronger linguistic representations such as word/character n-grams or embeddings.
- Compare the hand-built models against established ML baselines.
- Introduce cross-validation and stronger experiment tracking.
- Add calibration and uncertainty reporting.
- Expand and audit the dataset for broader language variation.
- Separate application, model, and safety logic into dedicated modules.
- Add automated unit and integration tests.
- Improve accessibility and responsive UI behavior.

---

## 📸 Screenshots

<p align="center">

<img src="https://github.com/user-attachments/assets/c7410d2a-2ca0-40fb-b622-b4e74537da57" width="31%" alt="MindEase application screenshot 1">

<img src="https://github.com/user-attachments/assets/7f2e7140-2aca-4e7b-ac29-106bf047628d" width="31%" alt="MindEase application screenshot 2">

<img src="https://github.com/user-attachments/assets/72c8224a-d225-4ffa-a337-b1da4fe9b816" width="31%" alt="MindEase application screenshot 3">

</p>

---

## 🎓 What This Project Demonstrates

- Machine-learning fundamentals implemented without relying on high-level model libraries
- Feature engineering and numerical preprocessing
- Gradient-based optimization
- Binary and multiclass-style prediction workflows
- Ensemble reasoning
- Model evaluation and reproducibility
- Streamlit application development
- Responsible communication of model limitations
- Translating an ML concept into a usable product experience

---

## 📄 Documentation

- [`MindEase_Documentation.pdf`](MindEase_Documentation.pdf)
- [`MindEase_Presentation.pptx`](MindEase_Presentation.pptx)

---

## 📜 License

MIT License. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

**MindEase — a transparent ML engineering project exploring stress-aware language analysis.**

</div>
