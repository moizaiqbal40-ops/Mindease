import re
import numpy as np

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
    "drowning","drained","terrified","panicking","shattering","losing",
    "low","down","blue","meh","unwell","glum","gloomy","discouraged",
    "defeated","disheartened","flat","dull","blah","rough",
    "fight","fighting","divorce","divorced","divorcing","argue","arguing",
    "argument","conflict","breakup","cheat","cheated","cheating","betrayed",
    "quit","quitting","fired","unemployed","toxic","abusive","abuse",
    "neglected","insecure","unstable","chaos","chaotic","dysfunctional",
    "resent","resentful","bully","bullied","humiliated","embarrassed","abandon"
}
POSITIVE_WORDS = {
    "happy","good","great","wonderful","amazing","better","hope","hopeful",
    "calm","peace","peaceful","joy","grateful","thankful","love","loved",
    "strong","okay","fine","improving","progress","success","smile","laugh",
    "blessed","content","relaxed","confident","proud","safe","excited",
    "refreshed","satisfied","balanced","steady"
}
NEGATION_WORDS = {"not", "no", "never", "cant", "can't", "cannot", "won't", "wont", "dont", "don't"}
INTENSIFIERS = {"very", "extremely", "really", "so", "totally", "completely",
                 "utterly", "absolutely", "incredibly"}
ABSOLUTIST_WORDS = {"always", "never", "everything", "nothing", "everyone",
                     "no one", "completely", "totally", "constantly"}
FIRST_PERSON = {"i", "me", "my", "myself", "mine"}

FEATURE_NAMES = [
    "neg", "length", "excl", "caps", "qmarks", "lex", "pos",
    "intensifiers", "negation_flips", "first_person_ratio",
    "absolutist", "elongated",
]
NUM_FEATURES = len(FEATURE_NAMES)


def _stem(w):
    """Lightweight suffix-stripping fallback (see app.py for rationale)."""
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 5 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 5 and w.endswith("es"):
        return w[:-2]
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w

def _matches(word, word_set):
    return word in word_set or _stem(word) in word_set

def extract_features(text):
    """Extract stress-relevant numerical signals from raw text, from scratch (NumPy/regex only)."""
    if not text or not text.strip():
        return [0.0] * NUM_FEATURES

    lower = text.lower()
    raw_words = lower.split()
    total = max(len(raw_words), 1)
    cleaned = [re.sub(r"[^a-z']", "", w) for w in raw_words]

    neg = 0
    pos = 0
    negation_flips = 0
    # Negation-aware scan: "not happy" should count as negative, not positive.
    for i, w in enumerate(cleaned):
        prev = cleaned[i - 1] if i > 0 else ""
        negated = prev in NEGATION_WORDS
        if _matches(w, NEGATIVE_WORDS):
            if negated:
                pos += 1          # "not sad" -> mild positive signal
                negation_flips += 1
            else:
                neg += 1
        elif _matches(w, POSITIVE_WORDS):
            if negated:
                neg += 1           # "not happy" -> negative signal
                negation_flips += 1
            else:
                pos += 1

    excl = text.count("!")
    qmarks = text.count("?")
    caps = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    lex = len(set(raw_words)) / total
    intensifiers = sum(1 for w in cleaned if w in INTENSIFIERS)
    fp = sum(1 for w in cleaned if w in FIRST_PERSON) / total
    absolutist = sum(1 for w in cleaned if w in ABSOLUTIST_WORDS)
    elongated = sum(1 for w in raw_words if re.search(r"(.)\1{2,}", w))

    return [
        float(neg), float(total), float(excl), float(caps), float(qmarks),
        float(lex), float(pos), float(intensifiers), float(negation_flips),
        float(fp), float(absolutist), float(elongated),
    ]


def normalize_matrix(X, mean=None, std=None):
    """Z-score normalize features so no single raw scale (e.g. word count) dominates training."""
    X = np.array(X, dtype=float)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std
