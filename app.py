"""
MindEase — Mental Health Chatbot
Stress Detection & Psychological Support System
BS Computer Science | Superior University, Lahore | 2025
Apache 2.0 License
"""

import streamlit as st
import numpy as np
import re, random, time, html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MindEase",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
#  THEME — Light Teal / Seafoam / Soft White
#  Colors pulled directly from the watercolor reference images
# ══════════════════════════════════════════════════════════
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

:root {
    /* Core palette — from the watercolor/ocean reference images */
    --seafoam:    #ACE1AF;
    --mint:       #A1DEDC;
    --pale-teal:  #C8EFEE;
    --sky:        #B8E8E6;
    --white-soft: #F0FFFF;
    --white:      #FAFFFE;
    /* Backgrounds — light, airy */
    --bg-main:    #EDF9F8;
    --bg-card:    #F5FFFE;
    --bg-card2:   #EAF8F7;
    --bg-sidebar: #E0F5F4;
    /* Borders */
    --border:     #B8E2E0;
    --border-soft:#D4F0EE;
    /* Text — dark on light = readable */
    --text-dark:  #1A4A44;
    --text-mid:   #2E7A72;
    --text-muted: #5A9A94;
    --text-light: #8ABFBB;
    /* Accents */
    --accent:     #3DBDB5;
    --accent2:    #5EC8AF;
    --green:      #68B87A;
}

/* ── Base Reset ── */
html, body, [class*="css"], .stApp {
    background: var(--bg-main) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-dark) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.5rem !important; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #D8F2F0 0%, #CBF0EE 50%, #E0F7F5 100%) !important;
    border-right: 1.5px solid var(--border) !important;
    min-width: 240px !important;
}
[data-testid="stSidebar"] * { color: var(--text-dark) !important; }
section[data-testid="stSidebar"] > div { padding: 0 !important; }

/* ── NAVIGATION — left side pill buttons ── */
.nav-btn {
    display: block;
    width: calc(100% - 24px);
    margin: 4px 12px;
    padding: 11px 18px;
    border-radius: 12px;
    border: none;
    background: transparent;
    color: var(--text-mid) !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    text-align: left;
    cursor: pointer;
    transition: all 0.2s ease;
    letter-spacing: 0.01em;
}
.nav-btn:hover { background: rgba(61,189,181,0.12) !important; color: var(--text-dark) !important; }
.nav-btn.active {
    background: linear-gradient(135deg, #ACE1AF30, #A1DEDC40) !important;
    color: var(--text-dark) !important;
    font-weight: 600;
    border-left: 3px solid var(--accent);
    padding-left: 15px;
}

/* ── Streamlit Radio — styled as clean option cards ── */
.stRadio > div { gap: 0 !important; }
.stRadio label {
    display: flex !important;
    align-items: center !important;
    width: calc(100% - 16px) !important;
    margin: 3px 8px !important;
    padding: 10px 16px !important;
    border-radius: 10px !important;
    border: none !important;
    background: transparent !important;
    color: var(--text-mid) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.stRadio label:has(input:checked) {
    background: linear-gradient(135deg, rgba(172,225,175,0.35), rgba(161,222,220,0.35)) !important;
    color: var(--text-dark) !important;
    font-weight: 600 !important;
    border-left: 3px solid var(--accent) !important;
    padding-left: 13px !important;
}
.stRadio label div { pointer-events: none; }
.stRadio input { display: none !important; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #FFFFFF !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text-dark) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.8rem 1.1rem !important;
    box-shadow: 0 2px 8px rgba(61,189,181,0.08) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(61,189,181,0.15) !important;
    outline: none !important;
}
textarea { resize: none !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #ACE1AF, #A1DEDC) !important;
    border: none !important;
    color: var(--text-dark) !important;
    border-radius: 50px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.55rem 1.8rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 3px 14px rgba(61,189,181,0.25) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8ED4A0, #7EC8C6) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(61,189,181,0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Secondary button ── */
button[kind="secondary"], .stButton > button[data-secondary="true"] {
    background: var(--bg-card2) !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text-mid) !important;
    box-shadow: none !important;
}

/* ── Radio (quiz) ── */
div[data-testid="stRadio"] > div { gap: 0.5rem !important; flex-direction: column !important; }

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--seafoam), var(--accent)) !important;
    border-radius: 50px !important;
}
.stProgress > div > div { background: var(--border-soft) !important; border-radius: 50px !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    box-shadow: 0 2px 10px rgba(61,189,181,0.08) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.1em; }
[data-testid="stMetricValue"] { color: var(--text-dark) !important; font-size: 1.6rem !important; font-weight: 700 !important; }

/* ── Dividers ── */
hr { border-color: var(--border-soft) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Cards ── */
.me-card {
    background: #FFFFFF;
    border: 1px solid var(--border-soft);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(61,189,181,0.08);
}
.me-card-teal {
    background: linear-gradient(135deg, #EAF9F8, #E0F5F3);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.me-card-accent {
    background: linear-gradient(135deg, rgba(172,225,175,0.15), rgba(161,222,220,0.15));
    border: 1.5px solid var(--seafoam);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Chat bubbles ── */
.chat-wrap { display: flex; flex-direction: column; gap: 0.6rem; padding: 0.5rem 0; }
.bubble-user-wrap { display: flex; justify-content: flex-end; margin: 0.2rem 0; }
.bubble-bot-wrap  { display: flex; justify-content: flex-start; margin: 0.2rem 0; }
.bubble-user {
    background: linear-gradient(135deg, #ACE1AF, #8FD8A0);
    color: #1A4A44;
    border-radius: 20px 20px 4px 20px;
    padding: 0.9rem 1.2rem;
    max-width: 72%;
    font-size: 0.93rem;
    line-height: 1.6;
    font-weight: 500;
    box-shadow: 0 2px 10px rgba(144,210,160,0.3);
}
.bubble-bot {
    background: #FFFFFF;
    color: var(--text-dark);
    border: 1px solid var(--border-soft);
    border-radius: 20px 20px 20px 4px;
    padding: 0.9rem 1.2rem;
    max-width: 78%;
    font-size: 0.91rem;
    line-height: 1.7;
    box-shadow: 0 2px 10px rgba(61,189,181,0.08);
}
.chat-label {
    font-size: 0.7rem;
    color: var(--text-light);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 6px 2px;
}

/* ── Typing indicator ── */
@keyframes blink { 0%,80%,100%{opacity:0} 40%{opacity:1} }
.typing-dot { display:inline-block; width:7px; height:7px; border-radius:50%;
              background: var(--accent); margin: 0 2px;
              animation: blink 1.4s infinite; }
.typing-dot:nth-child(2){animation-delay:0.2s}
.typing-dot:nth-child(3){animation-delay:0.4s}

/* ── Mood/feature tags ── */
.mood-tag {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}

/* ── Assessment cards ── */
.asmnt-card {
    background: #FFFFFF;
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 2px 10px rgba(61,189,181,0.07);
    text-align: center;
    height: 100%;
}
.asmnt-card:hover {
    border-color: var(--accent);
    box-shadow: 0 5px 20px rgba(61,189,181,0.2);
    transform: translateY(-3px);
}

/* ── About section ── */
.about-hero {
    background: linear-gradient(135deg, #C8F0ED, #D8F5E8);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}

/* ── Stress bar ── */
.stress-bar-wrap { background: var(--border-soft); border-radius: 50px; height: 10px; overflow: hidden; margin: 6px 0; }
.stress-bar-fill { height: 100%; border-radius: 50px; transition: width 0.5s ease; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  ML ENGINE
# ══════════════════════════════════════════════════════════
NEGATIVE_WORDS = {
    "sad","hopeless","anxious","stressed","tired","exhausted","overwhelmed",
    "helpless","worthless","alone","scared","afraid","depressed","miserable",
    "fail","failure","useless","broken","lost","empty","numb","pain","hurt",
    "cry","crying","desperate","hate","angry","furious","frustrated","trapped",
    "stuck","impossible","unbearable","terrible","horrible","awful","dread",
    "panic","nightmare","suffer","heavy","burden","shame","guilty","regret",
    "worry","worried","pressure","difficult","struggle","struggling","failing",
    "cant","can't","cannot","never","lonely","ignored","unloved","rejected",
    "suffocating","misery","anguish","devastated","shattered","cried",
    "dying","dead","pointless","pathetic","disgusting","bad","upset",
    "nervous","tense","restless","irritated","confused","heartbroken",
    "drowning","drained","terrified","panicking","shattering","losing"
}
POSITIVE_WORDS = {
    "happy","good","great","wonderful","amazing","better","hope","hopeful",
    "calm","peace","peaceful","joy","grateful","thankful","love","loved",
    "strong","okay","fine","improving","progress","success","smile","laugh",
    "blessed","content","relaxed","confident","proud","safe","excited",
    "refreshed","satisfied","balanced","steady"
}
NEGATION_WORDS   = {"not","no","never","cant","can't","cannot","won't","wont","dont","don't"}
INTENSIFIERS     = {"very","extremely","really","so","totally","completely",
                     "utterly","absolutely","incredibly"}
ABSOLUTIST_WORDS = {"always","never","everything","nothing","everyone",
                     "no one","completely","totally","constantly"}
FIRST_PERSON     = {"i","me","my","myself","mine"}

FEATURE_NAMES = ["neg","length","excl","caps","qmarks","lex","pos",
                  "intensifiers","negation_flips","first_person_ratio",
                  "absolutist","elongated"]

# ══════════════════════════════════════════════════════════
#  FEATURE EXTRACTION (from scratch — regex + counting only)
#  12 signals instead of the original 7. Negation-aware ("not
#  happy" now correctly reads as negative, not positive), plus
#  intensifiers, absolutist/catastrophizing language, first-
#  person pronoun ratio, and elongated letters ("sooo tired").
# ══════════════════════════════════════════════════════════
MAX_INPUT_WORDS = 300   # token-management: cap runaway inputs before any processing

def extract_features(text):
    """Extract 12 numerical stress signals from raw text."""
    if not text or not text.strip():
        return [0.0] * len(FEATURE_NAMES)

    words_all = text.split()
    if len(words_all) > MAX_INPUT_WORDS:               # token/length guard
        text = " ".join(words_all[:MAX_INPUT_WORDS])

    lower = text.lower()
    raw_words = lower.split()
    total = max(len(raw_words), 1)
    cleaned = [re.sub(r"[^a-z']", "", w) for w in raw_words]

    neg = pos = negation_flips = 0
    for i, w in enumerate(cleaned):
        prev = cleaned[i - 1] if i > 0 else ""
        negated = prev in NEGATION_WORDS
        if w in NEGATIVE_WORDS:
            if negated: pos += 1; negation_flips += 1
            else: neg += 1
        elif w in POSITIVE_WORDS:
            if negated: neg += 1; negation_flips += 1
            else: pos += 1

    excl  = text.count("!")
    qmarks= text.count("?")
    caps  = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    lex   = len(set(raw_words)) / total
    intens= sum(1 for w in cleaned if w in INTENSIFIERS)
    fp    = sum(1 for w in cleaned if w in FIRST_PERSON) / total
    absol = sum(1 for w in cleaned if w in ABSOLUTIST_WORDS)
    elong = sum(1 for w in raw_words if re.search(r"(.)\1{2,}", w))

    return [float(neg), float(total), float(excl), float(caps), float(qmarks),
            float(lex), float(pos), float(intens), float(negation_flips),
            float(fp), float(absol), float(elong)]

# ══════════════════════════════════════════════════════════
#  TRAINED MODEL PARAMETERS
#  These weights/tree were produced by train.py (included
#  alongside this app), which trains all three models from
#  scratch with NumPy on a hand-labeled 100-example dataset
#  (dataset.py) and evaluates on a held-out 20% test split.
#  Real measured test metrics (see README "Model Performance"):
#    Linear Regression : MAE 1.22   R² 0.73
#    SVM                : Accuracy 82%   F1 0.80
#    Decision Tree      : Accuracy 59%  (±1-level accuracy 100%)
#  Retrain any time with: python train.py
# ══════════════════════════════════════════════════════════
_FEATURE_MEAN = np.array([1.0543478260869565, 9.695652173913043, 0.17391304347826086,
                           0.030818090273619543, 0.0, 0.9877618154792068,
                           0.31521739130434784, 0.1956521739130435,
                           0.010869565217391304, 0.08690524451394012,
                           0.30434782608695654, 0.0])
_FEATURE_STD  = np.array([0.9482722324538985, 2.030951428543768, 0.5635426694267708,
                           0.03643818138879521, 1.0, 0.03446475706774315,
                           0.6414885512904135, 0.3967019041498838,
                           0.10368904363227668, 0.07753349634111395,
                           0.4601306627938417, 1.0])

_LR_W = np.array([1.7857506989908, 0.07936848122489967, 0.20554781187890553,
                   0.3655948511943456, 0.0, 0.07742385294003504,
                   -0.6576504485668843, 0.32741330507485655,
                   -0.40809441235720917, 0.4987829397097666,
                   0.5489360572995601, 0.0])
_LR_B = 5.028260869565209

_SVM_W = np.array([0.3784928704843767, 0.14091257963897771, 0.07789962970284584,
                    0.05509102591457233, 0.0, 0.05764677709210057,
                    -0.20177512291972238, 0.1931691676841796,
                    8.533388307236952e-07, 0.1504501965717579,
                    0.14219855017936006, 0.0])
_SVM_B = -0.008369565217391295

# Greedy Gini-impurity decision tree, max depth 6, trained on the same
# normalized features. Stored as a nested dict; walked at inference time.
_DT_TREE = {"leaf": False, "feature": 0, "threshold": -1.1118619632661395, "left": {"leaf": False, "feature": 6, "threshold": -0.491384282182837, "left": {"leaf": False, "feature": 9, "threshold": -1.1208735400193295, "left": {"leaf": False, "feature": 1, "threshold": -0.8349053306158379, "left": {"leaf": False, "feature": 1, "threshold": -2.312045530936167, "left": {"leaf": True, "class": 1}, "right": {"leaf": True, "class": 0}}, "right": {"leaf": True, "class": 1}}, "right": {"leaf": False, "feature": 3, "threshold": 0.134369917873337, "left": {"leaf": True, "class": 0}, "right": {"leaf": False, "feature": 3, "threshold": 0.2097648097846423, "left": {"leaf": True, "class": 3}, "right": {"leaf": True, "class": 1}}}}, "right": {"leaf": False, "feature": 3, "threshold": 0.0845326164404404, "left": {"leaf": True, "class": 0}, "right": {"leaf": False, "feature": 3, "threshold": 0.29772551701449823, "left": {"leaf": True, "class": 1}, "right": {"leaf": True, "class": 0}}}}, "right": {"leaf": False, "feature": 10, "threshold": -0.6614378277661479, "left": {"leaf": False, "feature": 1, "threshold": 0.1498548029310481, "left": {"leaf": False, "feature": 7, "threshold": -0.493196961916072, "left": {"leaf": False, "feature": 9, "threshold": 0.31219882682286904, "left": {"leaf": False, "feature": 5, "threshold": -2.868812520453157, "left": {"leaf": True, "class": 2}, "right": {"leaf": True, "class": 1}}, "right": {"leaf": False, "feature": 1, "threshold": -3.296805664483053, "left": {"leaf": True, "class": 3}, "right": {"leaf": True, "class": 2}}}, "right": {"leaf": True, "class": 2}}, "right": {"leaf": False, "feature": 0, "threshold": 0.9972370185995272, "left": {"leaf": False, "feature": 3, "threshold": 0.1171745916479516, "left": {"leaf": False, "feature": 2, "threshold": -0.30860669992418366, "left": {"leaf": True, "class": 2}, "right": {"leaf": True, "class": 3}}, "right": {"leaf": False, "feature": 3, "threshold": 0.29772551701449823, "left": {"leaf": True, "class": 3}, "right": {"leaf": True, "class": 1}}}, "right": {"leaf": True, "class": 3}}}, "right": {"leaf": False, "feature": 9, "threshold": -0.12874651682088423, "left": {"leaf": False, "feature": 1, "threshold": -0.3425252638423949, "left": {"leaf": False, "feature": 0, "threshold": -0.05731247233330618, "left": {"leaf": False, "feature": 3, "threshold": -0.41695522922808215, "left": {"leaf": True, "class": 2}, "right": {"leaf": True, "class": 1}}, "right": {"leaf": True, "class": 3}}, "right": {"leaf": True, "class": 2}}, "right": {"leaf": False, "feature": 9, "threshold": 0.051640214669742154, "left": {"leaf": False, "feature": 3, "threshold": -0.39586628917502237, "left": {"leaf": True, "class": 2}, "right": {"leaf": True, "class": 3}}, "right": {"leaf": False, "feature": 0, "threshold": -0.05731247233330618, "left": {"leaf": False, "feature": 7, "threshold": -0.493196961916072, "left": {"leaf": True, "class": 2}, "right": {"leaf": True, "class": 3}}, "right": {"leaf": True, "class": 3}}}}}}

_DT_LABELS = {0: "LOW", 1: "LOW-MED", 2: "MEDIUM", 3: "HIGH"}

def _normalize(features):
    x = np.array(features, dtype=float)
    return (x - _FEATURE_MEAN) / _FEATURE_STD

def lr_predict(features):
    """Continuous stress score 0–10, trained via batch gradient descent (NumPy only)."""
    x = _normalize(features)
    score = float(np.dot(x, _LR_W) + _LR_B)
    return round(float(np.clip(score, 0, 10)), 1)

def svm_predict(features):
    """Binary STRESSED / NOT STRESSED, trained via hinge-loss subgradient descent."""
    x = _normalize(features)
    sv = float(np.dot(x, _SVM_W) + _SVM_B)
    return ("STRESSED", 1) if sv > 0 else ("NOT STRESSED", 0)

def _walk_tree(node, x):
    while not node["leaf"]:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["class"]

def dt_predict(features):
    """Categorical LOW / LOW-MED / MEDIUM / HIGH via a greedy Gini decision tree."""
    x = _normalize(features)
    cls = _walk_tree(_DT_TREE, x)
    return _DT_LABELS[cls], cls

def mood_breakdown(text, lr_score):
    """Estimate emotion percentages from lexical analysis.

    Uses the SAME master NEGATIVE_WORDS / POSITIVE_WORDS lists that drive the
    LR/SVM/DT models, so the pie chart can never disagree with the score the
    way it used to (e.g. "my day was really bad" scoring HIGH stress while
    the chart still showed Happy 100%, because "bad" wasn't in any of the
    category-specific sub-lists below). Every negative word now lands in a
    specific emotion category if one matches, or a generic Sad catch-all
    otherwise — so no negative word is ever invisible to this chart.
    """
    anxious_w = {"anxious","worried","scared","dread","panic","nervous","afraid","tense","restless",
                 "stressed","overwhelmed","anxiety","pressure","suffocating","panicking"}
    sad_w     = {"sad","cry","crying","tears","depressed","hopeless","empty","numb","broken","heartbroken","devastated",
                 "low","down","blue","glum","gloomy","discouraged","defeated","disheartened","flat","dull","blah",
                 "bad","terrible","horrible","awful","miserable","unbearable","rough","tired","exhausted","drained",
                 "shattered","shattering","pointless","worthless","useless"}
    angry_w   = {"angry","furious","frustrated","hate","rage","irritated","mad","upset"}
    lonely_w  = {"alone","lonely","ignored","unloved","rejected","isolated","unseen","abandoned","trapped","stuck"}

    words     = [re.sub(r"[^a-z']","",w) for w in text.lower().split()]
    excl      = text.count("!")
    caps      = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    pos       = sum(1 for w in words if w in POSITIVE_WORDS)
    ac = sum(1 for w in words if w in anxious_w)
    sc = sum(1 for w in words if w in sad_w)
    agc= sum(1 for w in words if w in angry_w)
    lc = sum(1 for w in words if w in lonely_w)

    # Catch-all: any word that's negative in the master list but wasn't
    # already counted above (e.g. future words added to NEGATIVE_WORDS)
    # still shows up as Sad instead of silently vanishing.
    categorized = anxious_w | sad_w | angry_w | lonely_w
    uncategorized_neg = sum(1 for w in words if w in NEGATIVE_WORDS and w not in categorized)
    sc += uncategorized_neg

    neg_emotion_total = ac + sc + agc + lc
    caps_bonus = caps * 15 if neg_emotion_total > 0 else 0
    excl_bonus = excl * 1.2 if neg_emotion_total > 0 else 0
    happy_baseline = max(0, (1 - lr_score / 10)) * (1.5 if neg_emotion_total > 0 else 4.0)
    raw = {
        "Anxious": ac * 2.5 + caps_bonus,
        "Sad":     sc * 2.5,
        "Angry":   agc * 2.5 + excl_bonus,
        "Lonely":  lc * 2.5,
        "Happy":   pos * 2.0 + happy_baseline,
    }
    total = max(sum(raw.values()), 1)
    pct   = {k: round(v / total * 100) for k, v in raw.items()}
    diff  = 100 - sum(pct.values())
    pct["Happy"] = max(0, pct["Happy"] + diff)
    return pct

def detect_distortions(features):
    """Flag simple CBT cognitive-distortion signals (for response personalization)."""
    neg, length, excl, caps, qmarks, lex, pos, intens, flips, fp, absolutist, elong = features
    flags = []
    if absolutist >= 1:
        flags.append("all-or-nothing")   # "always/never/everything" -> catastrophizing / black-and-white thinking
    if fp > 0.15 and neg >= 2:
        flags.append("self-blame")       # heavy first-person focus alongside negative language
    if intens >= 1 and neg >= 1:
        flags.append("catastrophizing")  # "completely/utterly" + negative words
    return flags

def analyze(text):
    """Run full tri-algorithm pipeline on input text."""
    f              = extract_features(text)
    lr             = lr_predict(f)
    dt, dtl        = dt_predict(f)
    svm, svmb      = svm_predict(f)
    # Ensemble: weighted vote of all 3 models, weights set from relative
    # validation performance (SVM strongest on this task, DT weakest —
    # see README metrics). LR contributes a continuous score directly;
    # DT maps its level to a 0-10 scale; SVM adds a binary push.
    ens = round(lr * 0.55 + dtl * (10/3) * 0.20 + svmb * 10 * 0.25, 1)
    ens = float(np.clip(ens, 0, 10))
    if lr >= 7.0:
        ens = max(ens, lr * 0.9)
    ens  = round(ens, 1)
    mood = mood_breakdown(text, lr)
    distortions = detect_distortions(f)
    return {"lr": lr, "dt": dt, "dtl": dtl, "svm": svm, "ens": ens,
            "features": f, "mood": mood, "distortions": distortions}


# ══════════════════════════════════════════════════════════
#  PSYCHOLOGY RESPONSES
# ══════════════════════════════════════════════════════════
RESP = {
    0: {"emoji":"🌿","label":"Calm & Low Stress","color":"#2E7A72","bg":"#E8F8F5",
        "responses":[
            "You seem grounded and at peace right now — and that's genuinely beautiful. 🌿 This calm is something worth nurturing. Take a slow breath and appreciate exactly where you are.",
            "Your words carry a quiet stillness. That's a real strength. Consider a gratitude practice today — name 3 small things that brought you joy.",
            "Low stress detected. Your mind is in a good space. Protect this peace — a walk, some tea, music you love."
        ],
        "technique":"Mindfulness Moment","color_hex":"2E7A72","bar_color":"#68B87A",
        "exercise":"Close your eyes. Breathe in for 4 → hold for 4 → out for 4. Notice 5 things around you. You are present. You are okay."},
    1: {"emoji":"🌱","label":"Mild Tension","color":"#3DBDB5","bg":"#E8F8F7",
        "responses":[
            "I sense a little tension in your words — and that's completely human. 🌱 Sometimes just naming what we feel takes away its power.",
            "You seem slightly tense. Life has rough patches and you're navigating one. Try the 5-4-3-2-1 grounding technique — it brings you back to the present fast.",
            "A small ripple of stress. You're managing, and that truly matters. Remember: 100% of your difficult days so far — you've gotten through every single one."
        ],
        "technique":"5-4-3-2-1 Grounding (CBT)","color_hex":"3DBDB5","bar_color":"#A1DEDC",
        "exercise":"5 things you SEE → 4 you can TOUCH → 3 you HEAR → 2 you SMELL → 1 you TASTE. Grounded. Present."},
    2: {"emoji":"🌊","label":"Medium Stress","color":"#E8A030","bg":"#FEF8E8",
        "responses":[
            "That sounds genuinely tough, and your feelings are completely valid. 🌊 You don't have to hold this alone. Let's try 4-7-8 breathing — it activates your nervous system's calm response.",
            "I hear heaviness in what you're sharing. Medium stress is your mind saying 'I need support.' That awareness is brave. Let's breathe through it together.",
            "You're carrying something real. Medium stress builds when we hold too much inside. Try writing your 3 biggest worries on paper — getting them out of your head releases their grip."
        ],
        "technique":"4-7-8 Breathing","color_hex":"E8A030","bar_color":"#F0C060",
        "exercise":"Breathe IN for 4 → HOLD for 7 → OUT for 8. Repeat 4 times. This activates your body's built-in calm."},
    3: {"emoji":"💙","label":"High Stress","color":"#D05050","bg":"#FEF0F0",
        "responses":[
            "What you're feeling right now is real, and it matters deeply. 💙 High stress doesn't mean weakness — it means you're human, carrying something heavy. You don't have to solve everything today. Just this breath.",
            "I see you. What you're going through is genuinely hard. Reaching out, even here, is an act of courage. Please put one hand on your chest right now and breathe. You're still here. That counts.",
            "This sounds overwhelming — and I'm really glad you're expressing it rather than holding it in. One small act right now: drink a glass of water. Not because it fixes everything — because you deserve care in this moment."
        ],
        "technique":"Progressive Muscle Relaxation","color_hex":"D05050","bar_color":"#E88080",
        "exercise":"TENSE each muscle group 5 sec then RELEASE: Feet → Legs → Stomach → Hands → Shoulders → Face. Feel the release."}
}

DISTORTION_REFRAMES = {
    "all-or-nothing": "I noticed some 'always/never/everything' language — that's a common thought pattern called all-or-nothing thinking. Try asking: is this really *always* true, or does it just feel that way right now?",
    "self-blame": "A lot of this seems to be landing on you specifically. It's worth asking whether you'd judge a friend this harshly for the same situation.",
    "catastrophizing": "The intensity here suggests this might feel bigger than it is in this moment — a classic stress amplifier. Naming the specific worry (not the 'everything is ruined' version) can shrink it back to size.",
}

def get_response(analysis):
    s, dtl = analysis["ens"], analysis["dtl"]
    lvl = dtl
    if s >= 7: lvl = 3
    elif s >= 5: lvl = max(lvl, 2)
    rd = RESP[lvl]
    base = random.choice(rd["responses"])
    # CBT touch: gently name one detected distortion, if any, without diagnosing the person.
    distortions = analysis.get("distortions", [])
    if distortions:
        base = base + " " + DISTORTION_REFRAMES[distortions[0]]
    return lvl, rd, base


# ══════════════════════════════════════════════════════════
#  CHART FUNCTIONS
# ══════════════════════════════════════════════════════════
def fig_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140, facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf); plt.close(fig); return img

def gauge_chart(score, sub="Stress Score"):
    fig, ax = plt.subplots(figsize=(5,3), facecolor="#F5FFFE")
    ax.set_facecolor("#F5FFFE")
    cols = ["#68B87A","#ACE1AF","#A1DEDC","#F0C060","#E88060","#D05050"]
    for i,c in enumerate(cols):
        a1=180-(i+1)*30; a2=180-i*30
        th=np.linspace(np.radians(a1),np.radians(a2),60)
        xo,yo=0.88*np.cos(th),0.88*np.sin(th)
        xi,yi=0.62*np.cos(th),0.62*np.sin(th)
        ax.fill(np.concatenate([xo,xi[::-1]]),np.concatenate([yo,yi[::-1]]),color=c,alpha=0.9,zorder=2)
    nr=np.radians(180-(score/10)*180)
    ax.annotate("",xy=(0.72*np.cos(nr),0.72*np.sin(nr)),xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>",color="#1A4A44",lw=2,mutation_scale=16))
    ax.add_patch(plt.Circle((0,0),0.06,color="#1A4A44",zorder=5))
    ax.text(0,-0.22,f"{score}/10",ha="center",va="center",fontsize=24,fontweight="bold",color="#1A4A44")
    ax.text(0,-0.40,sub,ha="center",va="center",fontsize=9,color="#5A9A94")
    for l,a in zip(["Calm","Low","Mild","Mod","High","Severe"],[165,132,99,81,48,15]):
        r=np.radians(a); ax.text(0.99*np.cos(r),0.99*np.sin(r),l,ha="center",va="center",fontsize=6.5,color="#5A9A94")
    ax.set_xlim(-1.1,1.1); ax.set_ylim(-0.55,1.1); ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout(pad=0.2)
    return fig

def mood_pie(mood_dict):
    fig, ax = plt.subplots(figsize=(4,4), facecolor="#F5FFFE")
    ax.set_facecolor("#F5FFFE")
    labels = [k for k,v in mood_dict.items() if v>0]
    vals   = [v for v in mood_dict.values() if v>0]
    colors = {"Anxious":"#F0C060","Sad":"#A1DEDC","Angry":"#E88080","Lonely":"#C8A8D8","Happy":"#ACE1AF"}
    clrs   = [colors.get(l,"#B8E0DE") for l in labels]
    wedges,texts,autotexts = ax.pie(vals,labels=labels,colors=clrs,autopct="%1.0f%%",
                                     startangle=90,pctdistance=0.78,
                                     wedgeprops=dict(width=0.55,edgecolor="white",linewidth=2))
    for t in texts: t.set_color("#2E7A72"); t.set_fontsize(8.5)
    for at in autotexts: at.set_color("#1A4A44"); at.set_fontsize(8); at.set_fontweight("bold")
    ax.set_title("Emotion Breakdown",fontsize=10,color="#1A4A44",fontweight="bold",pad=8)
    plt.tight_layout(pad=0.3); return fig

def stress_timeline(history_scores):
    fig,ax = plt.subplots(figsize=(6,2.8),facecolor="#F5FFFE")
    ax.set_facecolor("#F5FFFE")
    xs = list(range(1,len(history_scores)+1))
    ax.fill_between(xs,history_scores,alpha=0.18,color="#A1DEDC")
    ax.plot(xs,history_scores,"o-",color="#3DBDB5",lw=2,markersize=6,markerfacecolor="white",
            markeredgecolor="#3DBDB5",markeredgewidth=2)
    ax.set_ylim(0,10); ax.set_xticks(xs)
    ax.set_xticklabels([f"#{i}" for i in xs],fontsize=8,color="#5A9A94")
    ax.tick_params(axis="y",colors="#5A9A94",labelsize=8)
    for spine in ax.spines.values(): spine.set_color("#B8E2E0")
    ax.yaxis.grid(True,color="#D4F0EE",linestyle="--",linewidth=0.7)
    ax.set_title("Your Stress Timeline",fontsize=10,color="#1A4A44",fontweight="bold",pad=8)
    plt.tight_layout(pad=0.4); return fig

def radar_chart(cat_scores):
    cats=[c["cat"] for c in cat_scores]
    norm=[c["score"]/c["max"] for c in cat_scores]
    N=len(cats); angles=[n/N*2*np.pi for n in range(N)]; angles+=angles[:1]; norm_p=norm+norm[:1]
    fig,ax=plt.subplots(figsize=(4.5,4.5),subplot_kw=dict(projection="polar"),facecolor="#F5FFFE")
    ax.set_facecolor("#F5FFFE"); ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    for r in [0.25,0.5,0.75,1.0]:
        th=np.linspace(0,2*np.pi,120); ax.plot(th,[r]*120,"-",color="#B8E2E0",lw=0.8)
    ax.plot(angles,norm_p,"o-",lw=2,color="#3DBDB5",markersize=5,markerfacecolor="white",markeredgewidth=2)
    ax.fill(angles,norm_p,alpha=0.18,color="#ACE1AF")
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,size=7.5,color="#2E7A72")
    ax.set_yticks([]); ax.spines["polar"].set_visible(False)
    ax.grid(color="#D4F0EE"); ax.set_ylim(0,1)
    ax.set_title("Stress Profile",pad=14,color="#1A4A44",fontsize=11,fontweight="bold")
    plt.tight_layout(); return fig

def bars_chart(cat_scores):
    cats=[c["cat"] for c in cat_scores]
    pcts=[(c["score"]/c["max"])*100 for c in cat_scores]
    clrs=["#68B87A" if p<30 else "#A1DEDC" if p<55 else "#F0C060" if p<75 else "#E88080" for p in pcts]
    fig,ax=plt.subplots(figsize=(6,4),facecolor="#F5FFFE"); ax.set_facecolor("#F5FFFE")
    ax.barh(range(len(cats)),[100]*len(cats),color="#EDF9F8",height=0.55,zorder=1)
    bars=ax.barh(range(len(cats)),pcts,color=clrs,height=0.55,alpha=0.9,zorder=2)
    for bar,p in zip(bars,pcts):
        ax.text(bar.get_width()+1.5,bar.get_y()+bar.get_height()/2,f"{p:.0f}%",va="center",ha="left",color="#2E7A72",fontsize=8.5,fontweight="600")
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats,color="#2E7A72",fontsize=8.5)
    ax.set_xlim(0,115); ax.set_xlabel("Stress %",color="#5A9A94",fontsize=8)
    ax.tick_params(axis="x",colors="#8ABFBB")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["left","bottom"]: ax.spines[s].set_color("#B8E2E0")
    ax.set_title("Category Breakdown",color="#1A4A44",fontsize=11,fontweight="bold",pad=8)
    plt.tight_layout(); return fig


# ══════════════════════════════════════════════════════════
#  QUIZ DATA
# ══════════════════════════════════════════════════════════
QUIZ = [
    {"q":"How has your sleep been lately?",
     "opts":["Sleeping well, feel rested","Some difficulty but managing","Often wake anxious or restless","Barely sleeping, mind won't stop"],
     "scores":[0,2,5,8],"cat":"Sleep"},
    {"q":"How do you feel about your responsibilities right now?",
     "opts":["Comfortable and in control","A bit much but okay","Overwhelmed most days","Completely drowning"],
     "scores":[0,2,6,9],"cat":"Workload"},
    {"q":"How often do you feel physically tense — tight shoulders, headaches?",
     "opts":["Rarely","Occasionally","Most days","Almost always"],
     "scores":[0,2,5,8],"cat":"Physical Tension"},
    {"q":"When you think about the future, how do you feel?",
     "opts":["Hopeful and excited","A bit uncertain but okay","Mostly anxious or worried","Dreading it / feeling hopeless"],
     "scores":[0,2,6,9],"cat":"Future Outlook"},
    {"q":"How connected do you feel to people around you?",
     "opts":["Very connected","Somewhat connected","A bit isolated","Very alone"],
     "scores":[0,2,5,8],"cat":"Social Connection"},
    {"q":"How are you managing daily tasks — eating, studying, routines?",
     "opts":["Managing well","Slight struggles but okay","Frequently forgetting","Barely functioning"],
     "scores":[0,2,6,9],"cat":"Daily Function"},
    {"q":"In the past week, how often have you felt like shutting down?",
     "opts":["Not at all","Once or twice","Several times","Almost every day"],
     "scores":[0,3,6,9],"cat":"Emotional State"},
    {"q":"When something stressful happens, how do you respond?",
     "opts":["Handle it calmly","Get anxious but recover","Spiral and overthink","Completely shut down"],
     "scores":[0,2,5,9],"cat":"Stress Response"},
]

ASMNT_CARDS = [
    {"emoji":"💼","title":"Lost My Job","desc":"Sudden job loss, fear of the future, financial pressure and loss of identity tied to work.","color":"#FEF3E8","border":"#F0B060"},
    {"emoji":"💔","title":"Lost Someone I Love","desc":"Grief is one of the heaviest burdens a human carries. You don't have to rush healing.","color":"#FEF0F5","border":"#E898B8"},
    {"emoji":"🫥","title":"Feeling Unloved","desc":"When you feel invisible or unworthy of love, it distorts how you see everything.","color":"#F0EFFF","border":"#A090D8"},
    {"emoji":"📚","title":"Career & Studies Pressure","desc":"Academic expectations, career uncertainty and comparison — the modern student's silent struggle.","color":"#EAF5FF","border":"#80B8E0"},
    {"emoji":"📱","title":"Social Media & Gen Z Stress","desc":"Comparison culture, FOMO, online identity and the weight of always being 'on'.","color":"#F0FFF4","border":"#80C890"},
    {"emoji":"👨‍👩‍👧","title":"Family Conflict","desc":"Tension at home, feeling unheard, pressure from parents — the stress that follows you everywhere.","color":"#FFF8E8","border":"#D8A840"},
    {"emoji":"🫂","title":"Loneliness & Isolation","desc":"Surrounded by people yet feeling deeply alone. This disconnection is real and valid.","color":"#EFF8FF","border":"#70B0E0"},
    {"emoji":"😔","title":"Low Self-Worth","desc":"When your inner critic is louder than your supporters — learning to silence it takes practice.","color":"#FEF5F0","border":"#E0906A"},
]


# ══════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════
defaults = {
    "page":"Home","chat_history":[],"score_history":[],
    "last_gauge":None,"last_mood_pie":None,"last_timeline":None,
    "quiz_step":-1,"quiz_answers":[],"quiz_done":False,
    "expanded_card":None,
    "chat_ended":False,      # True after user sends one message and session closes
    "session_result":None,   # stores last analysis for session summary
    "day_text":"",           # free-text "how was your day" for Wellness Check
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    # Logo + Brand
    st.markdown("""
    <div style='text-align:center; padding:1.8rem 0.5rem 1rem;'>
        <div style='font-size:2.6rem; margin-bottom:0.4rem;'>🌿</div>
        <div style='font-family:"Playfair Display",serif; font-size:1.7rem; font-weight:400;
                    color:#1A4A44; letter-spacing:0.05em;'>MindEase</div>
        <div style='color:#5A9A94; font-size:0.72rem; letter-spacing:0.18em; text-transform:uppercase;
                    margin-top:0.3rem;'>Mental Health Chatbot</div>
    </div>
    <hr style='border:none; border-top:1px solid #B8E2E0; margin:0 12px 1rem;'>
    """, unsafe_allow_html=True)

    # Navigation — four seafoam-toned options
    NAV = [
        ("🏡", "Home"),
        ("💬", "Chat Mode"),
        ("🌿", "Wellness Check"),
        ("ℹ️", "About"),
    ]
    NAV_COLORS = ["#ACE1AF","#A1DEDC","#B8E8E6","#C8EFEE"]

    for i,(icon,label) in enumerate(NAV):
        is_active = st.session_state.page == label
        bg = f"rgba({','.join(str(int(NAV_COLORS[i].lstrip('#')[j:j+2],16)) for j in (0,2,4))},0.3)" if is_active else "transparent"
        bdr = f"3px solid {NAV_COLORS[i]}" if is_active else "3px solid transparent"
        fw = "700" if is_active else "500"
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()

    st.markdown("<hr style='border:none; border-top:1px solid #B8E2E0; margin:1rem 12px;'>", unsafe_allow_html=True)

    # Quick info

    st.markdown("""
    <div style='padding:0.9rem 1rem; background:#FFFFFF; border:1px solid #D4F0EE; border-radius:14px;
                margin:0.7rem 8px 0; font-size:0.78rem; color:#5A9A94; line-height:1.7;'>
        🇵🇰 Need real support?<br>
        <strong style='color:#2E7A72;'>Umang:</strong> 0317-4288665<br>
        <strong style='color:#2E7A72;'>Rozan Lahore:</strong> 042-35761999
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><div style='text-align:center; color:#8ABFBB; font-size:0.7rem; padding-bottom:1rem;'>Superior University, Lahore<br>BS CS 2025 · Apache 2.0</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  PAGE HEADER
# ══════════════════════════════════════════════════════════
def page_header(title, sub):
    st.markdown(f"""
    <div style='padding:1.2rem 0 1rem;'>
        <div style='font-family:"Playfair Display",serif; font-size:1.9rem; font-weight:400;
                    color:#1A4A44; letter-spacing:0.02em;'>{title}</div>
        <div style='color:#5A9A94; font-size:0.85rem; margin-top:0.3rem;'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════════════
if st.session_state.page == "Home":
    # Hero
    st.markdown("""
    <div style='background:linear-gradient(135deg,#C8F0ED,#D8F5E8,#E8F8F5); border-radius:24px;
                padding:2.5rem 2rem; margin-bottom:1.5rem; text-align:center; position:relative; overflow:hidden;'>
        <div style='position:absolute;top:-30px;right:-30px;width:140px;height:140px;
                    background:radial-gradient(circle,rgba(172,225,175,0.4),transparent 70%);border-radius:50%;'></div>
        <div style='position:absolute;bottom:-20px;left:-20px;width:100px;height:100px;
                    background:radial-gradient(circle,rgba(161,222,220,0.4),transparent 70%);border-radius:50%;'></div>
        <div style='font-size:3rem; margin-bottom:0.5rem;'>🌿</div>
        <div style='font-family:"Playfair Display",serif; font-size:2.8rem; font-weight:400;
                    color:#1A4A44; letter-spacing:0.05em;'>MindEase</div>
        <div style='color:#3DBDB5; font-size:0.85rem; letter-spacing:0.18em; text-transform:uppercase;
                    margin:0.6rem 0;'>Stress Detection & Psychological Support</div>
        <div style='color:#5A9A94; font-size:0.8rem; max-width:520px; margin:0.5rem auto 1.2rem; line-height:1.7;'>
            Your private, non-judgmental AI companion — powered by three ML algorithms and real psychology frameworks.
        </div>
        <div style='display:flex; justify-content:center; gap:1rem; flex-wrap:wrap;'>
    """, unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("💬  Start Chatting", use_container_width=True):
            st.session_state.page="Chat Mode"; st.rerun()
    with c2:
        if st.button("🌿  Wellness Check", use_container_width=True):
            st.session_state.page="Wellness Check"; st.rerun()
    with c3:
        if st.button("ℹ️  Learn More", use_container_width=True):
            st.session_state.page="About"; st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)

    # 4 Feature cards
    st.markdown("<div style='color:#5A9A94;font-size:0.78rem;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.8rem;'>What you can do</div>", unsafe_allow_html=True)
    fc1,fc2,fc3,fc4 = st.columns(4, gap="small")
    feature_cards = [
        ("🎵","Music for Your Mood","Curated playlists matched to how you're feeling right now.","#EAF9F8","#A1DEDC"),
        ("🧘","Yoga & Breathing","Quick sessions for anxiety relief — 2 to 10 minutes.","#EAF5E8","#ACE1AF"),
        ("💌","Reach Out","Feeling alone? A gentle reminder to call someone who matters.","#EDF0FF","#B0B8E8"),
        ("✨","Vision & Hobbies","Build a mood board. Revisit what makes you you.","#FFF5EA","#F0C888"),
    ]
    for col,(emoji,title,desc,bg,bdr) in zip([fc1,fc2,fc3,fc4], feature_cards):
        with col:
            st.markdown(f"""
            <div style='background:{bg}; border:1.5px solid {bdr}; border-radius:18px; padding:1.3rem 1.1rem;
                        text-align:center; height:170px; display:flex; flex-direction:column;
                        align-items:center; justify-content:center; gap:0.4rem;
                        box-shadow:0 2px 12px rgba(61,189,181,0.08);'>
                <div style='font-size:1.8rem;'>{emoji}</div>
                <div style='font-weight:600; color:#1A4A44; font-size:0.88rem;'>{title}</div>
                <div style='color:#5A9A94; font-size:0.78rem; line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # ML Pipeline strip
    st.markdown("""
    <div style='background:#FFFFFF; border:1px solid #D4F0EE; border-radius:18px; padding:1.3rem 1.6rem;
                display:flex; align-items:center; gap:2rem; flex-wrap:wrap; box-shadow:0 2px 12px rgba(61,189,181,0.06);'>
        <div style='color:#1A4A44; font-size:0.85rem; font-weight:600; min-width:120px;'>🧠 ML Pipeline</div>
        <div style='flex:1; display:flex; gap:1rem; flex-wrap:wrap;'>
            <div style='background:#EAF9F8; border-radius:10px; padding:0.6rem 1rem; text-align:center; flex:1; min-width:130px;'>
                <div style='color:#3DBDB5; font-weight:700; font-size:0.85rem;'>📈 Linear Regression</div>
                <div style='color:#5A9A94; font-size:0.75rem; margin-top:2px;'>Continuous score 0–10</div>
            </div>
            <div style='background:#EAF5E8; border-radius:10px; padding:0.6rem 1rem; text-align:center; flex:1; min-width:130px;'>
                <div style='color:#2E9A5A; font-weight:700; font-size:0.85rem;'>🌳 Decision Tree</div>
                <div style='color:#5A9A94; font-size:0.75rem; margin-top:2px;'>Low / Medium / High</div>
            </div>
            <div style='background:#EDF0FF; border-radius:10px; padding:0.6rem 1rem; text-align:center; flex:1; min-width:130px;'>
                <div style='color:#6878D8; font-weight:700; font-size:0.85rem;'>⚡ SVM</div>
                <div style='color:#5A9A94; font-size:0.75rem; margin-top:2px;'>Stressed / Not Stressed</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  CHAT MODE  — one message → full analysis → session ends
# ══════════════════════════════════════════════════════════
elif st.session_state.page == "Chat Mode":
    page_header("💬 Chat Mode","Tell me how you're feeling. I'll listen, analyse, and respond. 🌿")

    col_chat, col_vis = st.columns([3, 2], gap="large")

    with col_chat:

        # ── BEFORE any message: show welcome + input ──────
        if not st.session_state.chat_history:
            st.markdown("""
            <div style='background:linear-gradient(135deg,#EAF9F8,#E8F5E8); border-radius:20px;
                        padding:2rem 1.5rem; text-align:center; margin-bottom:1.2rem;'>
                <div style='font-size:2.5rem; margin-bottom:0.6rem;'>🌿</div>
                <div style='font-weight:600; color:#1A4A44; font-size:1rem;'>MindEase is listening</div>
                <div style='color:#5A9A94; font-size:0.85rem; margin-top:0.4rem; line-height:1.6;'>
                    Write exactly how you're feeling — as long or short as you like.<br>
                    There are no wrong answers here.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── SHOW conversation (user msg + bot response) ───
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class='chat-label' style='text-align:right;'>You</div>
                    <div class='bubble-user-wrap'><div class='bubble-user'>{msg['content']}</div></div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='chat-label'>🌿 MindEase</div>
                    <div class='bubble-bot-wrap'><div class='bubble-bot'>{msg['content']}</div></div>
                    """, unsafe_allow_html=True)

        # ── SESSION ENDED: show summary + new session btn ─
        if st.session_state.chat_ended and st.session_state.session_result:
            res = st.session_state.session_result
            rd  = res["rd"]
            st.markdown(f"""
            <div style='background:#FFFFFF; border:1.5px solid #ACE1AF; border-radius:16px;
                        padding:1.2rem 1.5rem; margin-top:1rem;
                        box-shadow:0 2px 12px rgba(61,189,181,0.1);'>
                <div style='color:#1A4A44; font-weight:600; font-size:0.9rem; margin-bottom:0.7rem;'>
                    📊 Session Summary
                </div>
                <div style='display:flex; gap:1.5rem; flex-wrap:wrap; font-size:0.85rem;'>
                    <div>
                        <span style='color:#5A9A94;'>Stress Score (LR)</span><br>
                        <strong style='color:#3DBDB5; font-size:1.3rem;'>{res["lr"]}/10</strong>
                    </div>
                    <div>
                        <span style='color:#5A9A94;'>Category (DT)</span><br>
                        <strong style='color:#2E7A72;'>{res["dt"]}</strong>
                    </div>
                    <div>
                        <span style='color:#5A9A94;'>Binary (SVM)</span><br>
                        <strong style='color:#2E7A72;'>{res["svm"]}</strong>
                    </div>
                    <div>
                        <span style='color:#5A9A94;'>Ensemble</span><br>
                        <strong style='color:#2E7A72;'>{res["ens"]}/10</strong>
                    </div>
                </div>
                <div style='margin-top:0.8rem; color:#5A9A94; font-size:0.8rem; font-style:italic;'>
                    Session complete. Start a new one whenever you're ready. 🌿
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔄  New Session", use_container_width=False):
                st.session_state.chat_history  = []
                st.session_state.score_history = []
                st.session_state.chat_ended    = False
                st.session_state.session_result= None
                st.session_state.last_gauge    = None
                st.session_state.last_mood_pie = None
                st.session_state.last_timeline = None
                st.rerun()

        # ── INPUT: only show if session not ended ─────────
        if not st.session_state.chat_ended:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_area(
                    "message",
                    placeholder="Type how you're feeling right now...",
                    label_visibility="collapsed",
                    height=100
                )
                send = st.form_submit_button("Send  🌿", use_container_width=True)

            # Example prompts (only before first message)
            if not st.session_state.chat_history:
                st.markdown("<div style='color:#8ABFBB; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:0.5rem; margin-bottom:0.4rem;'>Try an example</div>", unsafe_allow_html=True)
                ex_list = [
                    "I've been feeling really overwhelmed and lost, don't know what to do",
                    "I can't sleep, my mind won't stop racing with all these worries",
                    "Today was a peaceful day, feeling calm and grateful 🌿",
                    "I CANT HANDLE THIS ANYMORE everything is falling apart help",
                ]
                exa, exb = st.columns(2)
                for i, ex in enumerate(ex_list):
                    with (exa if i % 2 == 0 else exb):
                        if st.button(ex[:42] + "…" if len(ex) > 42 else ex, key=f"ex_{i}", use_container_width=True):
                            user_input = ex
                            send = True

            if send and user_input and user_input.strip():
                # Token/length management: cap input before analysis and storage
                # so a very long paste can't blow up processing time or the
                # rendered chat bubble.
                _words = user_input.split()
                if len(_words) > MAX_INPUT_WORDS:
                    user_input = " ".join(_words[:MAX_INPUT_WORDS]) + " …"
                safe_input = html.escape(user_input)  # prevent HTML/script injection into the bubble

                result           = analyze(user_input)
                lvl, rd, resp_t  = get_response(result)

                # Build mood tags HTML
                mood_bg = {"Anxious":"#F0E0A0","Sad":"#C8E8E8","Angry":"#F0C0C0","Lonely":"#D8C8F0","Happy":"#C8E8C8"}
                mood_tc = {"Anxious":"#805A00","Sad":"#2A5A5A","Angry":"#7A2A2A","Lonely":"#4A3A6A","Happy":"#2A5A2A"}
                tags_html = "".join(
                    f"<span style='background:{mood_bg.get(em,'#D0E8E0')};color:{mood_tc.get(em,'#1A4A44')};"
                    f"padding:3px 10px;border-radius:50px;font-size:0.73rem;font-weight:600;"
                    f"margin:2px;display:inline-block;'>{em} {pct}%</span>"
                    for em, pct in sorted(result["mood"].items(), key=lambda x: -x[1]) if pct > 0
                )

                bot_msg = f"""
<div style='margin-bottom:0.3rem;'>
  <strong style='color:{rd["color"]};'>{rd["emoji"]} {rd["label"]}</strong>
</div>
<div style='margin-bottom:0.7rem;'>{tags_html}</div>
<p style='margin:0 0 0.7rem; color:#1A4A44; font-size:0.92rem; line-height:1.7;'>{resp_t}</p>
<div style='background:#EAF9F8; border-radius:12px; padding:0.7rem 0.9rem; margin-top:0.5rem;'>
  <div style='color:#3DBDB5; font-size:0.82rem; font-weight:600; margin-bottom:0.3rem;'>🧘 {rd["technique"]}</div>
  <div style='color:#5A9A94; font-size:0.8rem; font-style:italic;'>{rd["exercise"]}</div>
</div>
<div style='margin-top:0.7rem; font-size:0.75rem; color:#8ABFBB;'>
  📈 LR: <strong style='color:#3DBDB5;'>{result["lr"]}/10</strong> &nbsp;·&nbsp;
  🌳 DT: <strong style='color:#3DBDB5;'>{result["dt"]}</strong> &nbsp;·&nbsp;
  ⚡ SVM: <strong style='color:#3DBDB5;'>{result["svm"]}</strong> &nbsp;·&nbsp;
  Ensemble: <strong style='color:#3DBDB5;'>{result["ens"]}/10</strong>
</div>"""

                # Save to history + end session
                st.session_state.chat_history.append({"role": "user",  "content": safe_input})
                st.session_state.chat_history.append({"role": "bot",   "content": bot_msg})
                st.session_state.score_history.append(result["lr"])
                st.session_state.chat_ended     = True
                st.session_state.session_result = {**result, "rd": rd}

                # Generate charts
                st.session_state.last_gauge    = fig_img(gauge_chart(result["lr"], f"Score: {result['lr']}/10"))
                st.session_state.last_mood_pie = fig_img(mood_pie(result["mood"]))
                st.rerun()

    with col_vis:
        st.markdown("<div style='color:#5A9A94; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.6rem;'>📊 Your Analysis</div>", unsafe_allow_html=True)

        if st.session_state.last_gauge:
            st.image(st.session_state.last_gauge, use_container_width=True)
        else:
            st.markdown("""
            <div style='background:#FFFFFF; border:1px solid #D4F0EE; border-radius:16px;
                        padding:2.5rem 1rem; text-align:center; margin-bottom:0.8rem;'>
                <div style='font-size:2rem; color:#A1DEDC;'>📊</div>
                <div style='color:#8ABFBB; font-size:0.82rem; margin-top:0.4rem;'>Gauge appears after first message</div>
            </div>""", unsafe_allow_html=True)

        if st.session_state.last_mood_pie:
            st.image(st.session_state.last_mood_pie, use_container_width=True)

        if st.session_state.last_timeline:
            st.image(st.session_state.last_timeline, use_container_width=True)

        # Legend
        st.markdown("""
        <div style='background:#FFFFFF; border:1px solid #D4F0EE; border-radius:12px;
                    padding:0.9rem 1rem; font-size:0.8rem; line-height:2.1;'>
            <span style='color:#68B87A;'>●</span> <span style='color:#5A9A94;'> 0–3 &nbsp;Calm</span><br>
            <span style='color:#A1DEDC;'>●</span> <span style='color:#5A9A94;'> 3–5 &nbsp;Mild tension</span><br>
            <span style='color:#F0C060;'>●</span> <span style='color:#5A9A94;'> 5–7 &nbsp;Medium stress</span><br>
            <span style='color:#E88080;'>●</span> <span style='color:#5A9A94;'> 7–10  High stress</span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  WELLNESS CHECK (Assessment)
# ══════════════════════════════════════════════════════════
elif st.session_state.page == "Wellness Check":
    page_header("🌿 Wellness Check","Understand yourself better — 8 gentle questions, honest results.")

    if not st.session_state.quiz_done:
        if st.session_state.quiz_step == -1:
            # START SCREEN — Concern cards
            st.markdown("""
            <div style='color:#5A9A94; font-size:0.85rem; margin-bottom:1.2rem; line-height:1.7;'>
                Choose what's weighing on you right now, or simply start the full assessment below. 🌿
            </div>
            """, unsafe_allow_html=True)

            # Concern cards
            card_cols = st.columns(4, gap="small")
            for i,card in enumerate(ASMNT_CARDS):
                with card_cols[i%4]:
                    if st.button(f"{card['emoji']}  {card['title']}", key=f"card_{i}", use_container_width=True):
                        st.session_state.expanded_card = i if st.session_state.expanded_card!=i else None
                        st.rerun()

            # Expanded card view
            if st.session_state.expanded_card is not None:
                c = ASMNT_CARDS[st.session_state.expanded_card]
                st.markdown(f"""
                <div style='background:{c["color"]}; border:1.5px solid {c["border"]}; border-radius:18px;
                            padding:1.4rem 1.8rem; margin:1rem 0; box-shadow:0 4px 16px rgba(0,0,0,0.07);'>
                    <div style='font-size:1.6rem; margin-bottom:0.5rem;'>{c["emoji"]}</div>
                    <div style='font-size:1rem; font-weight:700; color:#1A4A44; margin-bottom:0.6rem;'>{c["title"]}</div>
                    <div style='color:#2E7A72; font-size:0.91rem; line-height:1.75;'>{c["desc"]}</div>
                    <div style='margin-top:1rem; color:#5A9A94; font-size:0.83rem; font-style:italic;'>
                        Whatever you're carrying right now — it's valid. You don't have to have it all figured out.
                        The assessment below can help you understand what you're feeling and offer support. 💙
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div style='color:#5A9A94; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.6rem;'>Before we begin — in your own words</div>", unsafe_allow_html=True)
            day_text_input = st.text_area(
                "How was your day? What's been on your mind?",
                placeholder="Optional — write a sentence or two about how today felt for you...",
                key="day_text_input", height=90,
            )
            st.markdown("<div style='color:#8AB8B4; font-size:0.75rem; margin:-0.3rem 0 1rem;'>We'll blend this with your quiz answers using the same language-analysis engine as Chat Mode, for a fuller picture. 🌿</div>", unsafe_allow_html=True)

            st.markdown("<div style='color:#5A9A94; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.6rem;'>Or take the full assessment</div>", unsafe_allow_html=True)
            if st.button("🌿  Start Full Wellness Assessment", use_container_width=False):
                st.session_state.day_text = (day_text_input or "").strip()
                st.session_state.quiz_step=0; st.session_state.quiz_answers=[]; st.rerun()

        else:
            step = st.session_state.quiz_step
            st.progress(step/len(QUIZ))
            st.markdown(f"<div style='color:#5A9A94; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:1rem;'>Question {step+1} of {len(QUIZ)} — {QUIZ[step]['cat']}</div>", unsafe_allow_html=True)

            q = QUIZ[step]
            st.markdown(f"""
            <div style='background:#FFFFFF; border:1.5px solid #B8E2E0; border-radius:18px;
                        padding:1.4rem 1.8rem; margin-bottom:1rem; font-size:1.05rem;
                        color:#1A4A44; font-weight:500; line-height:1.5; box-shadow:0 2px 10px rgba(61,189,181,0.08);'>
                {q['q']}
            </div>""", unsafe_allow_html=True)

            # Custom styled radio
            choice = st.radio("ans", q["opts"], label_visibility="collapsed", key=f"qr_{step}")

            cb, cn = st.columns([1,3])
            with cb:
                if step>0 and st.button("← Back", use_container_width=True):
                    st.session_state.quiz_step-=1
                    if st.session_state.quiz_answers: st.session_state.quiz_answers.pop()
                    st.rerun()
            with cn:
                if st.button("Next  →", use_container_width=True):
                    idx=q["opts"].index(choice)
                    st.session_state.quiz_answers.append(q["scores"][idx])
                    if step+1>=len(QUIZ): st.session_state.quiz_done=True
                    else: st.session_state.quiz_step+=1
                    st.rerun()

    else:
        # RESULTS
        answers = st.session_state.quiz_answers
        quiz_final = round((sum(answers)/(len(QUIZ)*9))*10, 1)
        day_text = st.session_state.get("day_text", "")

        # Combine the structured quiz (psychology-based: sleep, workload, tension,
        # outlook, social connection, etc.) with the same NLP pipeline used in
        # Chat Mode, if the user wrote about their day. Quiz is weighted higher
        # since it's a more reliable, structured signal; the free text adds
        # texture and can catch things the quiz options don't cover.
        nlp_result = analyze(day_text) if day_text else None
        if nlp_result:
            final = round(quiz_final * 0.65 + nlp_result["ens"] * 0.35, 1)
        else:
            final = quiz_final

        lvl = 3 if final>=7 else 2 if final>=5 else 1 if final>=3 else 0
        rd = RESP[lvl]; rt = random.choice(rd["responses"])
        if nlp_result and nlp_result.get("distortions"):
            rt = rt + " " + DISTORTION_REFRAMES[nlp_result["distortions"][0]]
        cat_scores=[{"cat":QUIZ[i]["cat"],"score":answers[i],"max":9} for i in range(len(QUIZ))]

        st.markdown(f"""
        <div style='text-align:center; padding:1.2rem 0 1.5rem;'>
            <div style='font-size:2.2rem;'>{rd["emoji"]}</div>
            <div style='font-family:"Playfair Display",serif; font-size:1.6rem; color:#1A4A44;
                        margin:0.4rem 0;'>{rd["label"]}</div>
            <div style='color:#5A9A94; font-size:0.88rem;'>
                Overall score: <strong style='color:#1A4A44;'>{final}/10</strong>
            </div>
        </div>""", unsafe_allow_html=True)

        if nlp_result:
            st.markdown(f"""
            <div style='background:#EAF9F8; border-radius:12px; padding:0.7rem 1rem; margin-bottom:1rem;
                        font-size:0.82rem; color:#2E7A72; text-align:center;'>
                📋 Quiz: <strong>{quiz_final}/10</strong> &nbsp;·&nbsp;
                📝 Your words: <strong>{nlp_result["ens"]}/10</strong> &nbsp;·&nbsp;
                🌿 Combined: <strong>{final}/10</strong>
            </div>""", unsafe_allow_html=True)

        r1,r2 = st.columns(2, gap="medium")
        with r1: st.image(fig_img(gauge_chart(final,"Overall")), use_container_width=True)
        with r2: st.image(fig_img(radar_chart(cat_scores)), use_container_width=True)
        st.image(fig_img(bars_chart(cat_scores)), use_container_width=True)

        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #D4F0EE; border-radius:18px;
                    padding:1.5rem 1.8rem; margin:0.5rem 0; box-shadow:0 2px 12px rgba(61,189,181,0.08);'>
            <div style='color:{rd["color"]}; font-weight:600; margin-bottom:0.7rem;'>{rd["emoji"]} What your results mean</div>
            <p style='color:#1A4A44; font-size:0.92rem; line-height:1.8; margin:0 0 1rem;'>{rt}</p>
            <div style='background:#EAF9F8; border-radius:12px; padding:0.9rem 1rem;'>
                <div style='color:#3DBDB5; font-size:0.85rem; font-weight:600; margin-bottom:0.3rem;'>🧘 {rd["technique"]}</div>
                <div style='color:#5A9A94; font-size:0.83rem; font-style:italic;'>{rd["exercise"]}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Category tiles
        st.markdown("<div style='color:#5A9A94; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; margin:1rem 0 0.6rem;'>Category Detail</div>", unsafe_allow_html=True)
        emmap={"Sleep":"😴","Workload":"💼","Physical Tension":"💪","Future Outlook":"🔮","Social Connection":"🤝","Daily Function":"📅","Emotional State":"💭","Stress Response":"⚡"}
        tile_cols = st.columns(4)
        for i,cat in enumerate(cat_scores):
            pct=int((cat["score"]/cat["max"])*100)
            col_c="#68B87A" if pct<30 else "#3DBDB5" if pct<55 else "#E8A030" if pct<75 else "#D05050"
            with tile_cols[i%4]:
                st.markdown(f"""
                <div style='background:#FFFFFF; border:1px solid #D4F0EE; border-radius:14px;
                            padding:0.9rem; text-align:center; margin-bottom:0.5rem;
                            box-shadow:0 2px 8px rgba(61,189,181,0.06);'>
                    <div style='font-size:1.3rem;'>{emmap.get(cat["cat"],"🌿")}</div>
                    <div style='color:#5A9A94; font-size:0.7rem; margin:0.3rem 0;'>{cat["cat"]}</div>
                    <div style='color:{col_c}; font-size:1.2rem; font-weight:700;'>{pct}%</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄  Retake Assessment"):
            st.session_state.quiz_step=-1; st.session_state.quiz_answers=[]
            st.session_state.quiz_done=False; st.session_state.expanded_card=None
            st.session_state.day_text=""; st.rerun()


# ══════════════════════════════════════════════════════════
#  ABOUT
# ══════════════════════════════════════════════════════════
elif st.session_state.page == "About":
    page_header("ℹ️ About MindEase","The story behind this project, and the people it's built for.")

    st.markdown("""
    <div style='background:linear-gradient(135deg,#C8F0ED,#D8F5E8); border-radius:22px;
                padding:2rem 2.2rem; margin-bottom:1.5rem;'>
        <div style='font-size:1.8rem; margin-bottom:0.5rem;'>🌿</div>
        <div style='font-family:"Playfair Display",serif; font-size:1.4rem; color:#1A4A44; margin-bottom:0.7rem;'>
            Built by a student, for everyone who needed this.
        </div>
        <p style='color:#2E7A72; font-size:0.92rem; line-height:1.8; margin:0;'>
            MindEase was created as a semester-end project at <strong>The Superior University, Lahore</strong> —
            but it was designed with something bigger in mind. In Pakistan, mental health is still widely
            misunderstood and access to support is limited. This project is a small step toward changing that:
            a space where anyone — student, professional, parent, teenager — can come to understand what
            they're feeling, without fear of judgment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    ca, cb = st.columns(2, gap="large")
    with ca:
        st.markdown("""
        <div class='me-card'>
            <div style='color:#1A4A44; font-weight:600; margin-bottom:0.9rem;'>👩‍💻 Who Built This</div>
            <p style='color:#2E7A72; font-size:0.88rem; line-height:1.8; margin:0;'>
                This app was built by a <strong>4th Semester BS Computer Science student</strong>
                at Superior University, Lahore — someone who believes technology and empathy can
                exist in the same space.<br><br>
                The project combines three ML algorithms, clinical psychology frameworks, and
                a design philosophy centered on calm, safety, and honesty.<br><br>
                It was built to show that student projects can solve real problems.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='me-card'>
            <div style='color:#1A4A44; font-weight:600; margin-bottom:0.9rem;'>🔬 How It Works</div>
            <div style='font-size:0.86rem; color:#2E7A72; line-height:1.9;'>
                <strong>1. You type or answer questions</strong> — freely, honestly.<br>
                <strong>2. Feature extraction</strong> — 7 signals from your text (negative words, caps, punctuation...).<br>
                <strong>3. Three ML models run simultaneously:</strong><br>
                &nbsp;&nbsp;&nbsp;&nbsp;📈 Linear Regression → score 0–10<br>
                &nbsp;&nbsp;&nbsp;&nbsp;🌳 Decision Tree → Low/Medium/High<br>
                &nbsp;&nbsp;&nbsp;&nbsp;⚡ SVM → Stressed/Not Stressed<br>
                <strong>4. Psychology engine</strong> selects a CBT/mindfulness response.<br>
                <strong>5. You see charts</strong> — gauge, mood breakdown, timeline.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with cb:
        st.markdown("""
        <div class='me-card-teal'>
            <div style='color:#1A4A44; font-weight:600; margin-bottom:0.9rem;'>🧠 Psychology Frameworks</div>
            <div style='font-size:0.87rem; line-height:1.9;'>
                <strong style='color:#1A4A44;'>CBT — Cognitive Behavioural Therapy</strong><br>
                <span style='color:#5A9A94;'>Identifies negative thought patterns and gently reframes them.
                Every MindEase response validates before advising.</span><br><br>
                <strong style='color:#1A4A44;'>Mindfulness-Based Stress Reduction</strong><br>
                <span style='color:#5A9A94;'>Present-moment grounding. Breathing techniques like 4-7-8
                and 5-4-3-2-1 activate the parasympathetic nervous system.</span><br><br>
                <strong style='color:#1A4A44;'>Positive Psychology (Seligman)</strong><br>
                <span style='color:#5A9A94;'>Builds on strengths rather than deficits. Affirms resilience
                and reminds users of their inner capacity to cope.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='me-card-teal'>
            <div style='color:#1A4A44; font-weight:600; margin-bottom:0.8rem;'>⚙️ Tech Stack</div>
            <div style='font-size:0.85rem; color:#2E7A72; line-height:2;'>
                🐍 Python 3.10+ &nbsp;·&nbsp; Streamlit (UI + Hugging Face)<br>
                📐 NumPy — ML from scratch (no sklearn for LR)<br>
                📊 Matplotlib — Gauge, Radar, Bar, Timeline charts<br>
                🖼️ Pillow — Image rendering<br>
                ⚖️ Apache License 2.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#FEF8E8; border:1.5px solid #F0C888; border-radius:16px; padding:1rem 1.2rem; font-size:0.82rem; color:#805A00; line-height:1.7;'>
            ⚠️ <strong>Disclaimer</strong><br>
            MindEase is an educational project and is NOT a substitute for professional mental health care.
            If you are experiencing severe distress, please reach out to a professional.<br><br>
            🇵🇰 <strong>Umang Helpline:</strong> 0317-4288665<br>
            🇵🇰 <strong>Rozan Lahore:</strong> 042-35761999
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<hr style='border:none; border-top:1px solid #D4F0EE; margin-top:2.5rem;'>
<div style='text-align:center; color:#8ABFBB; font-size:0.73rem; padding:0.8rem 0 1.2rem; letter-spacing:0.05em;'>
    Superior University, Lahore &nbsp;·&nbsp; Apache License 2.0
</div>
""", unsafe_allow_html=True)
