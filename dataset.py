# dataset.py
# Hand-authored labeled examples for MindEase stress detection.
# Each example: text, level (0=Calm,1=Mild,2=Medium,3=High), score (0-10 ground truth),
# stressed (bool, for the SVM binary task).
#
# Levels are assigned by the author based on realistic clinical framing (CBT screening
# language), not derived from the feature extractor itself, so training against them
# is a genuine (if small-scale, hand-built) supervised learning problem.

DATA = [
    # ---------- LEVEL 0: Calm / Low stress ----------
    ("Today was a peaceful day, feeling calm and grateful.", 0, 1.0),
    ("I'm doing okay, things feel manageable right now.", 0, 1.5),
    ("Had a great walk this morning, feeling refreshed and happy.", 0, 0.8),
    ("Life feels balanced lately, I'm content with where I am.", 0, 1.2),
    ("Slept well and woke up feeling good today.", 0, 1.0),
    ("I feel relaxed after spending time with my family.", 0, 0.9),
    ("Everything is going smoothly at work this week.", 0, 1.4),
    ("I'm proud of how I handled things today, feeling confident.", 0, 1.1),
    ("Just had a calm evening reading a book, very peaceful.", 0, 0.7),
    ("Feeling thankful and safe, nothing is bothering me right now.", 0, 0.6),
    ("I finished my tasks early and feel quite satisfied.", 0, 1.3),
    ("A quiet weekend helped me recharge, feeling great.", 0, 1.0),
    ("My mood has been steady and positive this week.", 0, 1.5),
    ("I feel strong and hopeful about the future.", 0, 1.8),
    ("Things are fine, I'm in a good headspace today.", 0, 1.6),

    # ---------- LEVEL 1: Mild tension ----------
    ("I've been a little tired lately but nothing serious.", 1, 3.0),
    ("Work has been a bit busy, I'm slightly stressed about deadlines.", 1, 3.8),
    ("I feel a small knot of worry about tomorrow's meeting.", 1, 3.5),
    ("Some days are harder than others, today is one of them.", 1, 3.2),
    ("I'm a bit anxious about my exam next week.", 1, 4.0),
    ("Feeling a little overwhelmed with my to-do list today.", 1, 3.7),
    ("I've had trouble focusing, feeling mildly stressed.", 1, 3.4),
    ("There's a slight tension at home, nothing major though.", 1, 3.1),
    ("I feel a bit nervous about the presentation tomorrow.", 1, 3.6),
    ("Things are okay but I'm a little on edge lately.", 1, 3.3),
    ("I'm somewhat worried about money this month.", 1, 3.9),
    ("Feeling tired and a touch stressed after a long week.", 1, 3.5),
    ("I keep thinking about a small mistake I made today.", 1, 3.2),
    ("A little restless tonight, can't quite relax.", 1, 3.6),
    ("I'm mildly frustrated with how slow things are moving.", 1, 3.4),

    # ---------- LEVEL 2: Medium stress ----------
    ("I've been feeling really overwhelmed and lost, don't know what to do.", 2, 6.0),
    ("I can't sleep, my mind won't stop racing with all these worries.", 2, 6.3),
    ("I'm so stressed about work, I feel exhausted all the time.", 2, 6.5),
    ("Everything feels heavy lately and I'm struggling to keep up.", 2, 6.2),
    ("I feel anxious almost every day now, it's hard to focus.", 2, 6.4),
    ("I'm carrying a lot of pressure and I don't know who to talk to.", 2, 6.6),
    ("I've been crying more than usual and feel really drained.", 2, 6.8),
    ("My chest feels tight with worry and I can't relax at all.", 2, 6.1),
    ("I feel stuck and frustrated, nothing seems to be working out.", 2, 6.3),
    ("I'm struggling to get through each day, it feels unbearable some nights.", 2, 6.7),
    ("I keep worrying about failing and it's exhausting me.", 2, 6.0),
    ("I feel really tense and on edge, my body feels heavy with stress.", 2, 6.5),
    ("I'm overwhelmed with responsibilities and feel like I'm drowning.", 2, 6.9),
    ("I've been so worried I can barely eat properly.", 2, 6.2),
    ("Everything feels difficult right now and I feel very tired of it all.", 2, 6.4),

    # ---------- LEVEL 3: High stress ----------
    ("I CANT HANDLE THIS ANYMORE everything is falling apart help!!", 3, 9.2),
    ("I feel completely hopeless and I don't know how to keep going.", 3, 8.8),
    ("I am so exhausted and broken, nothing ever gets better!!", 3, 9.0),
    ("I feel trapped and terrified, like there's no way out at all.", 3, 8.9),
    ("Everything is unbearable right now, I can't stop crying!!", 3, 9.1),
    ("I feel worthless and numb, like nothing matters anymore.", 3, 8.7),
    ("I'm panicking, my heart is racing and I can't calm down!!", 3, 9.3),
    ("I feel utterly overwhelmed, like I'm suffocating under everything.", 3, 8.8),
    ("I can't take this pressure anymore, I feel like I'm shattering.", 3, 9.0),
    ("Nothing feels okay anymore, I feel devastated and alone.", 3, 8.9),
    ("I am terrified and desperate, everything feels like too much!!", 3, 9.2),
    ("I feel like I'm losing control completely, this is unbearable.", 3, 9.1),
    ("I'm so angry and furious and exhausted, I just want it to stop!!", 3, 8.9),
    ("I feel empty and hopeless, like I've given up on everything.", 3, 8.6),
    ("I can't breathe through this panic, everything feels impossible!!", 3, 9.3),
]

MORE_DATA = [
    # ---------- LEVEL 0 additions ----------
    ("I had a lovely quiet morning and feel very much at ease.", 0, 0.9),
    ("Things have been going well, I feel grounded and clear-headed.", 0, 1.2),
    ("I'm feeling optimistic and light today, nothing weighing on me.", 0, 1.0),
    ("Spent time in the garden, feeling calm and centered.", 0, 0.8),
    ("I feel well-rested and my mood is genuinely good today.", 0, 1.1),
    ("Work is manageable and I feel in control of my schedule.", 0, 1.3),
    ("I feel safe and supported by the people around me.", 0, 0.7),
    ("Today went smoothly, I feel pretty happy with how things are.", 0, 1.4),
    ("I'm feeling steady, nothing is really bothering me right now.", 0, 1.0),
    ("I took a long walk and feel refreshed and at peace.", 0, 0.9),

    # ---------- LEVEL 1 additions ----------
    ("I'm a little worried about an upcoming appointment.", 1, 3.4),
    ("Feeling somewhat tired, work has kept me busy this week.", 1, 3.6),
    ("A bit of tension with a friend today, nothing too big.", 1, 3.3),
    ("I'm mildly anxious about starting a new project.", 1, 3.9),
    ("Things are alright but I feel a little unsettled tonight.", 1, 3.5),
    ("I've been a bit distracted and slightly stressed about bills.", 1, 3.8),
    ("Feeling a small amount of pressure with school deadlines.", 1, 3.7),
    ("I'm somewhat nervous about a conversation I need to have.", 1, 3.6),
    ("A little restless today, hard to settle my thoughts.", 1, 3.4),
    ("I feel mildly overwhelmed but I think I can manage it.", 1, 3.8),

    # ---------- LEVEL 2 additions ----------
    ("I feel like I'm barely keeping up and it's wearing me down.", 2, 6.3),
    ("My anxiety has been high this week and sleep is difficult.", 2, 6.5),
    ("I'm juggling too much and I feel constantly on edge.", 2, 6.2),
    ("I feel emotionally drained and it's hard to find motivation.", 2, 6.4),
    ("I've been snapping at people because I'm so stressed out.", 2, 6.6),
    ("I feel like I'm failing at everything and it's exhausting.", 2, 6.1),
    ("My mind keeps racing and I can't seem to switch off.", 2, 6.3),
    ("I feel weighed down by pressure from work and family.", 2, 6.5),
    ("I'm struggling to stay positive, everything feels harder lately.", 2, 6.2),
    ("I feel tense all the time and my sleep has really suffered.", 2, 6.4),

    # ---------- LEVEL 3 additions ----------
    ("I feel like I'm completely falling apart and can't cope!!", 3, 9.0),
    ("Everything is too much right now, I feel like I'm breaking down.", 3, 8.8),
    ("I can't stop shaking, I feel utterly terrified and alone.", 3, 9.1),
    ("I feel like giving up, nothing I do matters anymore.", 3, 8.7),
    ("I'm overwhelmed to the point I can't think straight at all!!", 3, 9.2),
    ("I feel like I'm drowning and no one can help me.", 3, 8.9),
    ("My whole body feels tense with panic, I can't calm down.", 3, 9.0),
    ("I feel completely hopeless, like there's no way through this.", 3, 8.8),
    ("I'm exhausted beyond belief and everything feels impossible now.", 3, 8.9),
    ("I feel shattered and desperate, I don't know what to do anymore.", 3, 9.1),
    ("I've been feeling really low lately, everything feels dull.", 2, 6.0),
    ("I'm feeling a bit down today, not sure why.", 1, 3.6),
    ("Feeling kind of blue this week, just a bit gloomy.", 1, 3.8),
    ("I feel so low and defeated, nothing feels worth it.", 3, 8.5),
    ("I'm just feeling flat and off today, hard to explain.", 1, 3.5),
    ("I feel really down and discouraged about everything right now.", 2, 6.3),

    # ---------- Short, colloquial phrasing (brevity coverage) ----------
    ("Feeling great today.", 0, 1.0),
    ("I'm doing fine, thanks.", 0, 1.2),
    ("Feeling a bit off today.", 1, 3.6),
    ("I'm kind of stressed.", 1, 3.8),
    ("I am feeling low.", 2, 5.8),
    ("Feeling really down.", 2, 6.0),
    ("I feel hopeless.", 3, 8.3),
    ("I can't cope anymore.", 3, 8.7),
]

def get_dataset():
    """Returns list of (text, level, score, stressed_bool)."""
    out = []
    for text, level, score in DATA + MORE_DATA:
        stressed = score >= 5.0
        out.append((text, level, score, stressed))
    return out
