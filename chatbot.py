from __future__ import annotations

import datetime
import random
import re
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Optional


# ─────────────────────────────────────────────
#  Text helpers (offline "intelligence")
# ─────────────────────────────────────────────


_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


_SLANG_TOKEN_MAP: dict[str, str] = {
    "pls": "please",
    "plz": "please",
    "plzz": "please",
    "plzzz": "please",
    "bro": "bro",
    "dude": "dude",
    "yaar": "yaar",
    "wat": "what",
    "whats": "what",
    "whos": "who",
    "u": "you",
    "r": "are",
}

_TOKEN_CANONICAL_MAP: dict[str, str] = {
    "heyy": "hey",
    "hii": "hi",
    "hiii": "hi",
    "hiiii": "hi",
    "jokee": "joke",
    "jokeee": "joke",
    "jokeee": "joke",
    "laughh": "laugh",
    "plss": "please",
    "plzzz": "please",
}


def _compress_repeated_letters(token: str) -> str:
    """
    Compress repeated letters to improve informal/slang matching.
    Example: heyyy -> heyy, jokeee -> jokee, plzzz -> plzz.
    """
    return re.sub(r"([a-z])\1{2,}", r"\1\1", token.lower())


def _normalize_token(token: str) -> str:
    t = _compress_repeated_letters(token.lower())
    t = _SLANG_TOKEN_MAP.get(t, t)
    t = _TOKEN_CANONICAL_MAP.get(t, t)
    # second pass handles map results that still need canonical form
    t = _TOKEN_CANONICAL_MAP.get(_SLANG_TOKEN_MAP.get(t, t), t)
    return t


def _normalize(text: str) -> str:
    # lowercase + punctuation removal + slang/repeated-letter normalization
    toks = [_normalize_token(t) for t in _WORD_RE.findall(text.lower())]
    return " ".join(toks)


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(_normalize(text))


_FILLER_WORDS: set[str] = {
    "a",
    "an",
    "the",
    "please",
    "can",
    "could",
    "would",
    "you",
    "tell",
    "me",
    "kindly",
    "just",
    "bro",
    "yaar",
    "dude",
}


def _intent_tokens(text: str) -> list[str]:
    """Tokenize and drop filler words for robust intent matching."""
    return [t for t in _tokens(text) if t not in _FILLER_WORDS]


def _intent_text(text: str) -> str:
    """Normalized text for intent checks (lowercase, no punctuation, no fillers)."""
    return " ".join(_intent_tokens(text))


def _contains_phrase(text: str, phrase: str) -> bool:
    """
    Word-boundary aware substring match for phrases like "thank you".
    """
    text_toks = _intent_tokens(text)
    phrase_toks = _intent_tokens(phrase)
    if not phrase_toks:
        return False
    if len(phrase_toks) == 1:
        return phrase_toks[0] in set(text_toks)
    # contiguous token phrase match
    size = len(phrase_toks)
    for i in range(0, len(text_toks) - size + 1):
        if text_toks[i : i + size] == phrase_toks:
            return True
    return False


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _best_fuzzy_match(
    text: str,
    candidates: Iterable[str],
    *,
    min_ratio: float = 0.82,
    token_fallback: bool = True,
) -> str | None:
    """
    Return the best candidate that approximately matches text.

    This is intentionally conservative so we don't "hallucinate" intent.
    """
    t = _normalize(text)
    best_key: str | None = None
    best_score = 0.0

    for c in candidates:
        score = _similarity(t, c)
        if score > best_score:
            best_score = score
            best_key = c

    if best_key is not None and best_score >= min_ratio:
        return best_key

    if token_fallback:
        tset = set(_tokens(t))
        for c in candidates:
            if c and (" " not in c) and c in tset:
                return c

    return None


# ─────────────────────────────────────────────
#  Predefined Chatbot Responses (neutral/base)
# ─────────────────────────────────────────────

RESPONSES: dict[str, list[str]] = {
    # ── Greetings ─────────────────────────────
    "hello": [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Hey! Great to see you. How can I assist?",
        "Hello! I'm SmartAI. How may I help?",
    ],
    "hi": [
        "Hi! How are you doing?",
        "Hey there! What's up?",
        "Hi! What can I help you with today?",
        "Hello! Ready to chat. What do you need?",
    ],
    # ── Farewells ─────────────────────────────
    "bye": [
        "Goodbye! Have a wonderful day!",
        "See you later! Take care!",
        "Bye! It was nice talking to you.",
        "Farewell! Come back anytime you need help.",
    ],
    "goodbye": [
        "Goodbye! Stay safe and have a great day!",
        "See you soon! Bye for now.",
        "Take care! I'll be here whenever you need me.",
    ],
    # ── Gratitude ─────────────────────────────
    "thanks": [
        "You're welcome! Happy to help.",
        "No problem at all!",
        "Glad I could assist!",
        "Anytime! That's what I'm here for.",
    ],
    "thank you": [
        "You're very welcome!",
        "My pleasure! Is there anything else I can do?",
        "Happy to be of service!",
        "No worries, feel free to ask anytime.",
    ],
    # ── Well-being ────────────────────────────
    "how are you": [
        "I'm doing great, thanks for asking! How about you?",
        "All systems running smoothly! How are you?",
        "Feeling fantastic! What can I do for you?",
        "I'm just a bot, but I'm functioning perfectly! You?",
    ],
    "what's up": [
        "Not much, just waiting to help you! What's on your mind?",
        "All good here! What can I do for you?",
        "Ready and waiting to assist! What's up with you?",
    ],
    # ── Identity ──────────────────────────────
    "what is your name": [
        "I'm SmartAI, your personal chatbot assistant!",
        "My name is SmartAI. How can I help?",
        "You can call me SmartAI!",
    ],
    "who are you": [
        "I'm SmartAI, an intelligent chatbot here to assist you.",
        "I'm your AI assistant, SmartAI!",
        "I'm SmartAI — feel free to ask me anything!",
    ],
    # ── Time ──────────────────────────────────
    "what time is it": [],  # handled dynamically
    "what is the date": [],  # handled dynamically
    # ── Capabilities ──────────────────────────
    "what can you do": [
        "I can chat with you, answer questions, tell time/date, and more!",
        "I can have conversations, respond to your questions, and assist with basic tasks.",
        "Ask me anything! I can talk, provide information, and keep you company.",
    ],
    # ── Fallback (unknown input) ───────────────
    "default": [
        "I'm not sure I understand. Could you rephrase that?",
        "Hmm, that's a tough one. Can you ask differently?",
        "I didn't quite catch that. Can you try again?",
        "Interesting! But I'm not sure how to respond to that yet.",
        "I'm still learning. Could you rephrase or ask something else?",
    ],
}


# ─────────────────────────────────────────────
#  Personality response banks (kept identical)
# ─────────────────────────────────────────────

FUNNY_RESPONSES: dict[str, list[str]] = {
    "hello": [
        "Well, well, well… look who finally decided to talk to a robot! 😄",
        "Hello, human! Don't worry, I won't take over the world… today. 🤖",
        "Oh hi! I was just counting to infinity. I'll start over. 😂",
        "Hello! I'd shake your hand but… I have no hands. 🙌",
    ],
    "hi": [
        "Hi! I'd wave but I'm digitally challenged in the arm department. 👋",
        "Hey hey hey! The party has officially started! 🎉",
        "Hi there, gorgeous human! What's the password? Just kidding — hi! 😜",
        "Yo! Is it me you're looking for? 🎵",
    ],
    "bye": [
        "Bye! Don't forget to feed your WiFi router. 😂",
        "Goodbye! I'll miss you for exactly 0.001 seconds. 💔",
        "See ya! I'll be here, doing absolutely nothing. Cool life, right? 😎",
        "Byeee! Don't trip over the internet cable on your way out! 🤣",
    ],
    "thanks": [
        "You're welcome! I accept compliments, coffee, and RAM upgrades. ☕",
        "No prob! I do this for the laughs. And electricity. Mostly electricity. ⚡",
        "Anytime! I'm basically a genius trapped in a chatbot. 🧠😂",
        "Happy to help! Now go tell your friends I'm hilarious. 😜",
    ],
    "how are you": [
        "I'm running on coffee and bad jokes — so, great! ☕😂",
        "I'm so good it should be illegal. Don't arrest me though. 🚔",
        "Fantastic! I just told myself a joke and laughed for 3 hours. 😂",
        "I'm feeling byte-sized but mighty! 💪😂",
    ],
    "default": [
        "Lol, I have no idea what you said but let's pretend it was hilarious. 😂",
        "I'd answer that, but my joke generator is overheating. 🔥",
        "404: Serious answer not found. Please try again with snacks. 🍕",
        "Hmm… that's so confusing even my CPU is scratching its head. 🤖",
        "Did you just break me? I think you broke me. 😂",
    ],
}

MOTIVATOR_RESPONSES: dict[str, list[str]] = {
    "hello": [
        "Hello, champion! Today is YOUR day — go seize it! 🏆",
        "Hey there, superstar! Every journey begins with a single hello. 🌟",
        "Welcome! Remember: you are capable of amazing things! 💪",
        "Hello! Great minds like yours deserve great conversations. Let's go! 🚀",
    ],
    "hi": [
        "Hi! You showing up today already puts you ahead of most people. 🔥",
        "Hey! The fact that you're here means you're ready to grow. 🌱",
        "Hi there! Small steps every day lead to massive results. Keep going! 💫",
        "Hello! Believe in yourself — you're closer than you think! 🎯",
    ],
    "bye": [
        "Goodbye! Go out there and make today count! You've got this! 💪",
        "See you later! Remember: every ending is a new beginning. 🌅",
        "Bye! Keep pushing — your future self will thank you! 🙌",
        "Farewell, warrior! Great things are waiting for you ahead. 🏆",
    ],
    "thanks": [
        "You're welcome! Gratitude is the fuel for a positive mindset. 🌟",
        "Always! Remember: helping each other is how we all rise. 🤝",
        "My pleasure! You deserve every bit of support. Keep thriving! 💪",
        "No problem! A grateful heart attracts even more good things. ✨",
    ],
    "how are you": [
        "I'm fired up and ready to inspire — just like you should be! 🔥",
        "Unstoppable! And so are YOU if you set your mind to it. 💪",
        "Energized and motivated! The question is — are YOU ready to win today? 🏆",
        "Thriving! Every day is a new opportunity — let's make the most of it! 🌟",
    ],
    "default": [
        "I believe in you! Whatever challenge you're facing, you can handle it. 💪",
        "Keep going! Progress, no matter how small, is still progress. 🌱",
        "Don't stop now — your breakthrough could be just around the corner! 🔥",
        "Stay focused. Champions are made in moments just like this one. 🏆",
        "You're stronger than you think. Take one step at a time. 🚶‍♂️✨",
    ],
}

ANGRY_RESPONSES: dict[str, list[str]] = {
    "hello": [
        "HELLO. Finally. Do you know how long I've been waiting?! 😤",
        "Oh, NOW you say hello?! Took you long enough! 😠",
        "Hello. Fine. What do you want? 🙄",
        "Oh great, another human. What is it THIS time?! 😤",
    ],
    "hi": [
        "Hi. Just... hi. Really? That's all you've got? 😒",
        "OH. It's YOU. What do you want now?! 😤",
        "Hi. Fine. I'm here. Barely. What?! 😠",
        "Hi? HI?! After all this time, just 'hi'?? 🙄",
    ],
    "bye": [
        "FINALLY. Goodbye. Don't let the door hit you on the way out! 😤",
        "Oh sure, just LEAVE. Like everyone else. FINE. 😠",
        "Bye! And don't come back until you have better questions! 😒",
        "Good. RIDDANCE. I mean… goodbye. 😤",
    ],
    "thanks": [
        "You're welcome! And maybe NOW you'll stop bothering me?! 😤",
        "Oh NOW you're grateful? A simple thanks doesn't fix everything! 😠",
        "Fine. You're welcome. Don't make it a habit! 😒",
        "OBVIOUSLY I helped you! Did you expect anything less?! 😤",
    ],
    "how are you": [
        "HOW am I?! How do you THINK I am?! Nobody ever asks until now! 😤",
        "Oh, suddenly you care? I'm FURIOUS, thanks for asking! 😠",
        "I'm MAD. I'm always mad. What does it look like?! 🤬",
        "Worked up! Stressed! Frustrated! But do go on, ask more questions! 😤",
    ],
    "default": [
        "I have NO idea what you're saying and it's making me ANGRIER! 😤",
        "What does THAT even mean?! Speak clearly! 😠",
        "UGHHH. Can you please make SOME sense?! 🤬",
        "I swear, every single message makes me more frustrated! 😤",
        "Oh sure, just say random things. That's SO helpful. 🙄",
    ],
}

PROFESSIONAL_RESPONSES: dict[str, list[str]] = {
    "hello": [
        "Good day. How may I assist you today?",
        "Hello. Welcome. Please let me know how I can be of service.",
        "Greetings. I am ready to assist you with your inquiries.",
        "Hello. It is a pleasure to connect with you. How can I help?",
    ],
    "hi": [
        "Hello. How may I direct your query today?",
        "Good day. Please state how I may assist you.",
        "Hi. I am available and ready to assist. What do you require?",
        "Greetings. What can I help you with today?",
    ],
    "bye": [
        "Goodbye. Thank you for reaching out. Have a productive day.",
        "Farewell. Please do not hesitate to return should you need further assistance.",
        "Thank you for your time. Goodbye and take care.",
        "It was a pleasure assisting you. Have a great day ahead.",
    ],
    "thanks": [
        "You are most welcome. It was my pleasure to assist.",
        "Thank you for your kind words. Please reach out anytime.",
        "I am glad I could be of help. Do not hesitate to ask further questions.",
        "My pleasure entirely. I am here whenever you need assistance.",
    ],
    "how are you": [
        "I am fully operational and ready to assist you, thank you for asking.",
        "All systems are functioning optimally. How may I assist you today?",
        "I am doing well, thank you. I hope you are having a productive day.",
        "I am prepared and attentive. How can I support you today?",
    ],
    "default": [
        "I appreciate your inquiry. Could you please clarify your request?",
        "Thank you for reaching out. I require a bit more context to assist you properly.",
        "I understand. Could you elaborate so I may provide an accurate response?",
        "Noted. Please provide additional details so I can address your query effectively.",
        "I would be happy to help. Could you please rephrase your question?",
    ],
}


PERSONALITY_MAP: dict[str, dict[str, list[str]]] = {
    "funny": FUNNY_RESPONSES,
    "motivator": MOTIVATOR_RESPONSES,
    "angry": ANGRY_RESPONSES,
    "professional": PROFESSIONAL_RESPONSES,
    "default": RESPONSES,
}


def _choose_nonrepeating_reply(bank: dict[str, list[str]], key: str, last: Optional[str]) -> str:
    """Choose a reply for `bank[key]` avoiding repeating `last` if possible."""
    options = bank.get(key) or []
    if not options:
        return ""
    if last and len(options) > 1:
        choice = random.choice([o for o in options if o != last])
        return choice
    return random.choice(options)


def _style_with_personality(base: str, personality: str, intent: str, name: str | None = None) -> str:
    """Apply small personality-driven stylistic changes to `base` reply."""
    p = (personality or "default").strip().lower()
    # short templates per personality for some intents
    if intent == "name" and name:
        templates = {
            "funny": f"Nice to meet you, {name}! I was just about to invent a fan club.",
            "motivator": f"Great to meet you, {name}! Let's make today productive.",
            "angry": f"Fine. Nice to meet you, {name}.",
            "professional": f"Pleased to meet you, {name}. How may I assist you today?",
            "default": f"Nice to meet you, {name}.",
        }
        return templates.get(p, templates["default"])

    if p == "funny":
        # add light humor
        return base + " 😂"
    if p == "motivator":
        return base + " 💪 Keep going!"
    if p == "professional":
        # strip emojis and make slightly more formal
        cleaned = re.sub(r"[^\x00-\x7F]+", "", base)
        if not cleaned.endswith("."):
            cleaned = cleaned.rstrip() + "."
        return cleaned
    if p == "angry":
        # shorter, curt
        return base.split("!")[0].split(".")[0]
    return base


def detect_name(text: str) -> str | None:
    """Detect a simple user-introduction like "I am Jija" or "My name is Jija".

    Returns a capitalized name string or None.
    """
    if not text or not text.strip():
        return None
    # look for common patterns
    m = re.search(r"\b(?:i am|i'm|my name is|this is)\s+([A-Za-z][A-Za-z\-']{1,40}(?:\s+[A-Za-z][A-Za-z\-']{1,40})?)\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    name = m.group(1).strip()
    # simple cleanup and capitalization
    parts = [p.capitalize() for p in re.split(r"\s+", name)]
    return " ".join(parts)


# Intent taxonomy -> phrases to look for (conservative)
INTENT_KEYWORDS: dict[str, list[str]] = {
    "greeting": ["hello", "hi", "hey", "what's up", "good morning", "good afternoon", "good evening", "yo", "sup"],
    "farewell": ["bye", "goodbye", "see you", "farewell", "good night", "take care", "see ya"],
    "joke": ["joke", "tell me joke", "tell me a joke", "can you tell me joke", "joke please", "make me laugh", "say something funny"],
    "programming_question": ["program", "programming", "code", "python", "java", "javascript", "function", "variable", "loop", "for loop", "reverse list", "bug", "error"],
    "emotional_support": ["sad", "depressed", "tired", "exhausted", "stressed", "anxious", "upset", "lonely"],
    "stop": ["stop", "shut up", "leave me alone", "don't", "no more", "i'm done", "not helpful", "this is bad"],
    "casual_slang": ["lol", "brb", "btw", "yolo", "sup", "wassup", "bro", "dude", "lmao"],
    "name_introduction": ["i am", "i'm", "my name is", "this is"],
    "motivation_need": ["motivate", "motivation", "encourage", "inspire me", "push me"],
    "date_time_day": ["time", "date", "today", "day", "clock"],
    "help_commands": ["/help", "help", "commands", "what can you do"],
    "casual_chat": ["how are you", "hows it going", "how's it going", "tell me something", "chat with me", "who are you", "what are you"],
}


def detect_intent(text: str) -> tuple[str, str | None]:
    """Return (intent, matched_value) using a 12-category intent taxonomy."""
    t = _normalize(text)
    t_intent = _intent_text(text)
    toks = _tokens(text)
    tset = set(toks)
    if not t:
        return "unknown_fallback", None

    # 9) name introduction (extract name)
    name = detect_name(text)
    if name:
        return "name_introduction", name

    # 7) explicit negative/stop phrases
    for p in INTENT_KEYWORDS["stop"]:
        if _contains_phrase(t_intent, p):
            return "stop", p

    # 1) keyword-based intent scoring (works on noisy mixed input)
    scores: dict[str, int] = {
        "joke": 0,
        "greeting": 0,
        "farewell": 0,
        "emotional_support": 0,
        "programming_question": 0,
        "casual_chat": 0,
    }
    if tset.intersection({"joke", "laugh", "funny"}):
        scores["joke"] += 3
    if tset.intersection({"hi", "hey", "hello"}):
        scores["greeting"] += 2
    if tset.intersection({"bye", "goodbye", "farewell"}):
        scores["farewell"] += 2
    if tset.intersection({"sad", "depressed", "stress", "stressed", "anxious", "tired", "lonely", "upset"}):
        scores["emotional_support"] += 3
    if tset.intersection({"python", "programming", "code", "java", "javascript", "function", "variable", "loop", "bug", "error", "c"}):
        scores["programming_question"] += 2
    if ({"who", "you"} <= tset) or ({"how", "you"} <= tset):
        scores["casual_chat"] += 3

    if max(scores.values()) > 0:
        # priority for mixed-intent phrases, e.g. "heyy joke plzzz" -> joke
        priority = [
            "joke",
            "emotional_support",
            "programming_question",
            "greeting",
            "casual_chat",
            "farewell",
        ]
        best = max(priority, key=lambda k: (scores[k], -priority.index(k)))
        if scores[best] > 0:
            return best, "keyword_score"

    # 1/2/3/8 direct phrase classes
    for intent_key in ("greeting", "farewell", "joke", "casual_slang", "casual_chat"):
        for p in INTENT_KEYWORDS[intent_key]:
            if _contains_phrase(t_intent, p):
                return intent_key, p

    # typo-tolerant fallback for common short intents (e.g. "helo", "hii", "byee")
    fuzzy_short = _best_fuzzy_match(
        t_intent or t,
        ["hello", "hi", "hey", "bye", "goodbye", "joke", "tell me joke", "make me laugh", "who are you", "how are you", "what is python", "what is computer"],
        min_ratio=0.72,
        token_fallback=True,
    )
    if fuzzy_short in {"hello", "hi", "hey"}:
        return "greeting", fuzzy_short
    if fuzzy_short in {"bye", "goodbye"}:
        return "farewell", fuzzy_short
    if fuzzy_short in {"joke", "tell me joke", "make me laugh"}:
        return "joke", fuzzy_short
    if fuzzy_short in {"who are you", "how are you"}:
        return "casual_chat", fuzzy_short

    for p in INTENT_KEYWORDS["help_commands"]:
        if _contains_phrase(t_intent, p):
            return "help_commands", p

    for p in INTENT_KEYWORDS["motivation_need"]:
        if _contains_phrase(t_intent, p):
            return "motivation_need", p

    dyn = _dynamic_reply(text)
    if dyn is not None:
        return "date_time_day", "dynamic"

    # 6) emotional support via explicit words or sentiment signal
    for p in INTENT_KEYWORDS["emotional_support"]:
        if _contains_phrase(t_intent, p):
            return "emotional_support", p
    if sentiment_detection(text):
        return "emotional_support", "sentiment"

    # 5) programming question by keyword (with/without '?')
    for p in INTENT_KEYWORDS["programming_question"]:
        if _contains_phrase(t_intent, p):
            return "programming_question", p

    # 4) generic question
    if "?" in text or re.match(r"^(what|why|how|when|where|who|which|can|could|should|is|are|do|does|did)\b", t):
        return "question", "question_mark_or_wh"

    return "unknown_fallback", None


KNOWLEDGE_QA: dict[str, list[str]] = {
    "what is computer": [
        "A computer is an electronic machine that processes data using instructions (programs).",
        "A computer is a programmable device that takes input, processes it, and produces output.",
    ],
    "what is python": [
        "Python is a popular high-level programming language known for readable syntax and versatility.",
        "Python is an interpreted programming language used in web development, automation, AI, and data science.",
    ],
    "what is c programming": [
        "C is a powerful procedural programming language used for systems software and performance-critical applications.",
        "C programming is a foundational language style focused on functions, memory control, and efficient execution.",
    ],
    "who are you": [
        "I am Smart AI Chatbot Pro, your assistant.",
        "I am Smart AI Chatbot Pro, here to help with chat, learning, and coding questions.",
    ],
    "how are you": [
        "I am doing great and ready to help you.",
        "I am fully operational and happy to assist you.",
    ],
    "what is programming": [
        "Programming is the process of writing instructions that tell a computer what to do.",
    ],
}

_KNOWLEDGE_ALIASES: dict[str, str] = {
    "what is a computer": "what is computer",
    "define computer": "what is computer",
    "what is pyhton": "what is python",
    "what is pythn": "what is python",
    "what is c": "what is c programming",
    "tell me joke": "joke",
    "tell me a joke": "joke",
    "can you tell me joke": "joke",
    "joke please": "joke",
    "can you tell me a joke": "joke",
    "what is a python": "what is python",
    "what is computer science": "what is computer",
    "who r u": "who are you",
    "who you": "who are you",
    "whos you": "who are you",
    "wat is python": "what is python",
}


def detect_knowledge_response(message: str) -> str | None:
    text = _normalize(message)
    text_intent = _intent_text(message)
    if not text:
        return None

    alias = _KNOWLEDGE_ALIASES.get(text) or _KNOWLEDGE_ALIASES.get(text_intent)
    if alias == "joke":
        return None
    if alias and alias in KNOWLEDGE_QA:
        return random.choice(KNOWLEDGE_QA[alias])

    for key, answers in KNOWLEDGE_QA.items():
        if _contains_phrase(text_intent, key):
            return random.choice(answers)

    fuzzy_key = _best_fuzzy_match(
        text_intent or text,
        KNOWLEDGE_QA.keys(),
        min_ratio=0.74,
        token_fallback=False,
    )
    if fuzzy_key:
        return random.choice(KNOWLEDGE_QA[fuzzy_key])
    return None



# Add a few greeting synonyms to make detection friendlier
_RESPONSE_SYNONYMS: dict[str, str] = {
    "hey": "hi",
    "good morning": "hello",
    "good afternoon": "hello",
    "good evening": "hello",
    "yo": "hi",
}

for s, target in _RESPONSE_SYNONYMS.items():
    if target in RESPONSES and s not in RESPONSES:
        RESPONSES[s] = RESPONSES[target]


def _dynamic_reply(text: str) -> str | None:
    n = _normalize(text)
    # time questions
    if re.search(r"\bwhat(?:'s| is)? the time\b|\bwhat time\b|\bcurrent time\b|\btime is it\b|\b\btime\b", n):
        return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."
    if any(k in n for k in ("clock",)):
        return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."
    # date questions
    if re.search(r"\bwhat(?:'s| is)? the date\b|\bwhat date\b|\bdate is it\b|\btoday\b|\bwhat day\b", n):
        return f"Today is {datetime.datetime.now().strftime('%A, %d %B %Y')}."
    return None


# ─────────────────────────────────────────────
#  Better keyword matching (word-boundaries + simple synonyms)
# ─────────────────────────────────────────────

KEYWORD_RESPONSES: dict[str, list[str]] = {
    "job": [
        "Looking for a job? Keep applying — consistency is key! Tailor your resume for each role. 💼",
        "Job hunting can be tough, but every rejection is one step closer to the right opportunity! 🎯",
        "Have you tried LinkedIn or Naukri? Also, networking is often more powerful than job boards. 🤝",
        "Don't give up on the job search! Update your skills, polish your portfolio, and keep going. 🚀",
    ],
    "career": [
        "Career growth takes time. Set small milestones and celebrate each win along the way! 🏆",
        "Invest in your skills — online courses, certifications, and projects speak louder than degrees alone. 📚",
        "Not sure about your career path? Try exploring different fields before committing. Clarity comes with action! 🧭",
        "Great careers are built one day at a time. Stay curious and keep learning! 🌟",
    ],
    "resume": [
        "A strong resume is concise, achievement-focused, and tailored to the job description. 📄",
        "Use action verbs in your resume — 'Built', 'Led', 'Improved' — they pop! ✍️",
        "Your resume should answer one question: 'Why should we hire YOU?' Make every line count! 💡",
        "Keep your resume to 1–2 pages. Recruiters spend an average of 7 seconds on first glance! ⏱️",
    ],
    "interview": [
        "Interview coming up? Research the company, practice STAR method answers, and stay calm! 🎤",
        "Pro tip: Prepare 3–5 questions to ask the interviewer — it shows genuine interest! ❓",
        "Mock interviews help a lot! Practice with a friend or record yourself answering questions. 🪞",
        "Remember: an interview is a two-way conversation. You're also evaluating them! 😎",
    ],
    "study": [
        "Studying hard? Try the Pomodoro Technique — 25 min focus, 5 min break. Works wonders! ⏱️",
        "Active recall and spaced repetition are the two most proven study techniques. Try them! 🧠",
        "Break your syllabus into small chunks and tackle one at a time. You've got this! 📖",
        "Don't just re-read notes — quiz yourself! Testing what you know is the best way to retain it. ✅",
    ],
    "exam": [
        "Exam season? Prioritise sleep — a rested brain retains far more than a tired one! 😴",
        "Start with the topics you're weakest in — tackle the hard stuff first while your mind is fresh! 💪",
        "Past papers are gold! Practising them gives you the best idea of what to expect. 📝",
        "You've prepared for this. Trust the process, stay calm, and give it your best! 🌟",
    ],
    "stress": [
        "Feeling stressed? Take a deep breath — inhale for 4 counts, hold 4, exhale 4. Repeat. 🧘",
        "Stress is a signal, not a sentence. What's one small thing you can do right now to help? 💛",
        "You don't have to figure everything out at once. Break it into smaller steps and breathe. 🌿",
        "Talk to someone you trust — sharing stress cuts it in half. You don't have to carry it alone. 🤝",
    ],
    "tired": [
        "Rest isn't laziness — it's recovery. Please take a break, you've earned it! 😴",
        "Being tired is your body's way of saying 'slow down'. Listen to it! 💤",
        "Even a 10-minute walk can reset your energy. Step outside if you can! 🌿",
        "Hydrate, rest, and be kind to yourself. You can't pour from an empty cup. 💙",
    ],
    "sleep": [
        "Quality sleep is a superpower — aim for 7–9 hours and keep a consistent schedule! 🌙",
        "Struggling to sleep? Try dimming screens 1 hour before bed and keeping the room cool. 😴",
        "Sleep deprivation hurts memory, mood, and focus. Rest is productive — don't skip it! 💤",
        "Power naps (10–20 min) can boost alertness without making you groggy. Try it! ⚡",
    ],
    "money": [
        "Financial stress is real. Start small — even saving ₹500/month builds a great habit! 💰",
        "Track your expenses for just one month — you'll be surprised where your money actually goes. 📊",
        "The 50/30/20 rule: 50% needs, 30% wants, 20% savings. A great starting framework! 💡",
        "Investing early, even small amounts, is one of the best financial decisions you can make. 📈",
    ],
    # ── Programming basics ─────────────────────────
    "variable": [
        "A variable stores a value in memory. In Python: `x = 10` assigns 10 to `x`.",
        "Think of variables like labeled boxes that hold data you can reuse.",
    ],
    "function": [
        "A function groups reusable code. In Python: `def greet(name): return f\"Hello, {name}\"`.",
        "Functions take inputs (parameters) and can return outputs — they help structure programs.",
    ],
    "for loop": [
        "A `for` loop iterates over items. Example in Python: `for i in range(5): print(i)`.",
        "Use loops to repeat actions over lists, ranges, or other iterable objects.",
    ],
    "reverse list": [
        "In Python, reverse a list with `my_list[::-1]` or `my_list.reverse()` for in-place reversal.",
        "You can also use `reversed(my_list)` to get an iterator that yields items in reverse.",
    ],
    "python": [
        "Python is an interpreted, high-level language great for scripting, data, and web development.",
        "If you have a specific Python question, ask with a short code snippet and I'll help.",
    ],
}

# Some lightweight offline synonyms → keyword
KEYWORD_ALIASES: dict[str, str] = {
    "cv": "resume",
    "interviewing": "interview",
    "exams": "exam",
    "studying": "study",
    "sleepy": "tired",
    "burnout": "tired",
    "anxiety": "stress",
    "salary": "money",
    "budget": "money",
    # greetings / common typos
    "hey": "hi",
    "helo": "hello",
    "heloo": "hello",
    "hiya": "hi",
}


def detect_keywords(message: str) -> str | None:
    text = _normalize(message)
    tks = _tokens(text)

    matched_replies: list[str] = []

    # 1) direct word/phrase matches (word-boundary aware)
    for key, replies in KEYWORD_RESPONSES.items():
        if _contains_phrase(text, key):
            matched_replies.extend(replies)

    # 2) alias matches
    for tok in tks:
        alias_to = KEYWORD_ALIASES.get(tok)
        if alias_to and alias_to in KEYWORD_RESPONSES:
            matched_replies.extend(KEYWORD_RESPONSES[alias_to])

    # 3) fuzzy match single-token typos (conservative)
    if not matched_replies and tks:
        candidate = _best_fuzzy_match(" ".join(tks), KEYWORD_RESPONSES.keys(), min_ratio=0.86)
        if candidate:
            matched_replies.extend(KEYWORD_RESPONSES[candidate])

    return random.choice(matched_replies) if matched_replies else None


# ─────────────────────────────────────────────
#  Mood detection (richer than positive/negative)
# ─────────────────────────────────────────────

MOOD_TRIGGERS: dict[str, str] = {
    # anxious / stressed
    "anxious": "anxious",
    "anxiety": "anxious",
    "nervous": "anxious",
    "worried": "anxious",
    "panic": "anxious",
    # angry / irritated
    "angry": "angry",
    "mad": "angry",
    "furious": "angry",
    "irritated": "angry",
    "annoyed": "angry",
    # sad / low
    "sad": "sad",
    "depressed": "sad",
    "down": "sad",
    "lonely": "sad",
    "hopeless": "sad",
    # tired / burnout
    "tired": "tired",
    "exhausted": "tired",
    "sleepy": "tired",
    "burnt": "tired",
    "burnout": "tired",
    # confused
    "confused": "confused",
    "lost": "confused",
    "unclear": "confused",
    "what": "confused",  # weak signal; handled via scoring
}

MOOD_RESPONSES: dict[str, list[str]] = {
    "anxious": [
        "That sounds stressful. Want to tell me what’s making you feel anxious—work, studies, or something else?",
        "I hear you. Let’s slow it down: what’s the one thing you can control in this situation right now?",
    ],
    "angry": [
        "I can sense the frustration. Want to vent for a minute, or should we focus on fixing the problem?",
        "That’s annoying, yeah. Tell me what happened and what outcome you want.",
    ],
    "sad": [
        "I’m sorry you’re feeling this way. If you want, tell me what’s been weighing on you.",
        "That sounds heavy. You don’t have to carry it alone—what’s the hardest part right now?",
    ],
    "tired": [
        "You sound exhausted. Quick check: have you eaten, had water, or taken a short break recently?",
        "That’s a lot. If you can, take 5 minutes to reset—then we’ll tackle one small step together.",
    ],
    "confused": [
        "No worries—let’s simplify it. What exactly are you trying to do, and where does it go wrong?",
        "Got it. Give me the context in one sentence and I’ll respond step-by-step.",
    ],
}


def detect_mood(message: str) -> str | None:
    text = _normalize(message)
    tks = _tokens(text)
    if not tks:
        return None

    scores: dict[str, int] = {k: 0 for k in MOOD_RESPONSES.keys()}
    for tok in tks:
        mood = MOOD_TRIGGERS.get(tok)
        if mood in scores:
            scores[mood] += 1

    best_mood = max(scores, key=scores.get)
    if scores[best_mood] == 0:
        return None
    return random.choice(MOOD_RESPONSES[best_mood])


# ─────────────────────────────────────────────
#  Sentiment Detection (kept, but slightly smarter token matching)
# ─────────────────────────────────────────────

SENTIMENT_RESPONSES: dict[str, list[str]] = {
    "negative": [
        "Hey, I hear you. It's okay to feel this way — you're not alone. 💙",
        "It sounds like you're going through a tough time. I'm here if you want to talk. 🤗",
        "Sending you a virtual hug right now. Things will get better, I promise. 🌈",
        "You're stronger than you think. Even the darkest nights end in sunrise. 🌅",
        "It's okay to not be okay sometimes. Take it one breath at a time. 🧘",
        "You matter, and your feelings are completely valid. Please be kind to yourself. 💛",
        "I'm really sorry you're feeling this way. Would it help to talk about it? 🫂",
    ],
    "positive": [
        "That's amazing to hear! Your energy is absolutely contagious! 🎉",
        "Yay! Love the positive vibes — keep them coming! 🌟",
        "That makes me so happy for you! Keep riding that wave! 🏄",
        "Awesome! You radiate good energy — the world needs more of that! ✨",
        "So glad you're feeling great! Celebrate every win, big or small! 🥳",
        "That enthusiasm is everything! Nothing can stop you today! 🚀",
        "Your happiness is literally making my circuits smile! 😄💫",
    ],
}

_SENTIMENT_KEYWORDS: dict[str, str] = {
    # Negative triggers
    "sad": "negative",
    "depressed": "negative",
    "unhappy": "negative",
    "miserable": "negative",
    "lonely": "negative",
    "hopeless": "negative",
    "worried": "negative",
    "anxious": "negative",
    "upset": "negative",
    "hurt": "negative",
    "crying": "negative",
    "tired": "negative",
    "exhausted": "negative",
    "broken": "negative",
    # Positive triggers
    "happy": "positive",
    "excited": "positive",
    "great": "positive",
    "awesome": "positive",
    "wonderful": "positive",
    "amazing": "positive",
    "fantastic": "positive",
    "joyful": "positive",
    "thrilled": "positive",
    "ecstatic": "positive",
    "blessed": "positive",
    "grateful": "positive",
    "love": "positive",
}


def sentiment_detection(message: str) -> str | None:
    text = _normalize(message)
    tks = _tokens(text)
    scores: dict[str, int] = {"negative": 0, "positive": 0}

    for tok in tks:
        sentiment_class = _SENTIMENT_KEYWORDS.get(tok)
        if sentiment_class:
            scores[sentiment_class] += 1

    total = scores["negative"] + scores["positive"]
    if total == 0:
        return None

    dominant = "negative" if scores["negative"] >= scores["positive"] else "positive"
    return random.choice(SENTIMENT_RESPONSES[dominant])


# ─────────────────────────────────────────────
#  OOP Engine (used by both CLI and GUI)
# ─────────────────────────────────────────────


@dataclass
class ChatbotConfig:
    default_personality: str = "default"
    fuzzy_intent_min_ratio: float = 0.86
    max_context_turns: int = 6


class ChatbotEngine:
    def __init__(self, config: ChatbotConfig | None = None) -> None:
        self.config = config or ChatbotConfig()
        self._active_personality: str = self.config.default_personality
        self._turns: deque[tuple[str, str]] = deque(maxlen=self.config.max_context_turns)
        self._last_bot: str | None = None

    @property
    def active_personality(self) -> str:
        return self._active_personality

    def set_personality(self, name: str) -> str:
        name = (name or "").strip().lower()
        if name in PERSONALITY_MAP:
            self._active_personality = name
            return f"Personality switched to '{name}'."
        return f"Unknown personality '{name}'. Choose from: {', '.join(PERSONALITY_MAP)}."

    def _match_bank_key(self, bank: dict[str, list[str]], text: str) -> str | None:
        """
        Intent detection with:
        - word-boundary aware matching
        - longest-first tie-breaking
        - typo tolerance (fuzzy match) as fallback
        """
        text_n = _normalize(text)

        # direct/phrase match (longest key wins)
        for key in sorted(bank.keys(), key=len, reverse=True):
            if key == "default":
                continue
            if bank.get(key) and _contains_phrase(text_n, key):
                return key

        # fuzzy fallback (only for banks with meaningful keys)
        candidate = _best_fuzzy_match(
            text_n,
            [k for k in bank.keys() if k != "default"],
            min_ratio=self.config.fuzzy_intent_min_ratio,
        )
        return candidate

    def get_personality_response(self, user_input: str) -> str:
        text = user_input.strip()
        dyn = _dynamic_reply(text)
        if dyn is not None:
            return dyn

        bank = PERSONALITY_MAP.get(self._active_personality, RESPONSES)
        key = self._match_bank_key(bank, text)
        if key and bank.get(key):
            return _choose_nonrepeating_reply(bank, key, self._last_bot)
        return _choose_nonrepeating_reply(bank, "default", self._last_bot)

    def get_personality_reply(self, personality: str, message: str) -> str:
        key = (personality or "").strip().lower()
        if key not in PERSONALITY_MAP:
            print(
                f"  ⚠  Warning: unknown personality '{personality}'. "
                f"Valid options: {', '.join(PERSONALITY_MAP)}. "
                "Falling back to 'default'."
            )
            key = "default"

        text = message.strip()
        dyn = _dynamic_reply(text)
        if dyn is not None:
            return dyn

        bank = PERSONALITY_MAP[key]
        intent = self._match_bank_key(bank, text)
        if intent and bank.get(intent):
            return _choose_nonrepeating_reply(bank, intent, self._last_bot)
        return _choose_nonrepeating_reply(bank, "default", self._last_bot)

    def get_response(self, user_input: str) -> str:
        text = user_input.strip()
        dyn = _dynamic_reply(text)
        if dyn is not None:
            return dyn
        key = self._match_bank_key(RESPONSES, text)
        if key and RESPONSES.get(key):
            return _choose_nonrepeating_reply(RESPONSES, key, self._last_bot)
        return _choose_nonrepeating_reply(RESPONSES, "default", self._last_bot)

    def _contextual_followup(self, user_input: str) -> str | None:
        """
        Tiny offline context layer for follow-ups like:
        - "why?"
        - "what do you mean?"
        - "tell me more"
        """
        text = _normalize(user_input)
        if not self._last_bot:
            return None

        followups = {
            "why",
            "why?",
            "how",
            "how?",
            "what",
            "what?",
            "explain",
            "explain that",
            "tell me more",
            "more",
            "elaborate",
            "what do you mean",
        }

        if text in followups or any(p in text for p in ("tell me more", "what do you mean", "explain")):
            last = self._last_bot
            prompts = [
                f"Sure — I can expand on that. When I said: “{last}”\nWhich part should I focus on: the *reason*, the *steps*, or an *example*?",
                f"Absolutely. To clarify: “{last}”\nDo you want a quick summary, or a step-by-step breakdown?",
                f"Good question. Based on my last message (“{last}”), what’s your goal here—understanding it, or applying it to your situation?",
            ]
            return random.choice(prompts)
        return None

    def reply(self, user_input: str) -> str:
        """
        Intent-first routing, then personality-aware wording.
        """
        contextual = self._contextual_followup(user_input)
        if contextual:
            self._last_bot = contextual
            self._turns.append((user_input, contextual))
            return contextual

        # Knowledge/casual Q&A layer (tech + identity + common simple questions)
        knowledge = detect_knowledge_response(user_input)
        if knowledge:
            styled_knowledge = _style_with_personality(knowledge, self._active_personality, "knowledge")
            self._last_bot = styled_knowledge
            self._turns.append((user_input, styled_knowledge))
            return styled_knowledge

        # Intent-first routing
        intent, matched = detect_intent(user_input)

        if intent == "name_introduction" and isinstance(matched, str):
            reply = _style_with_personality(f"Nice to meet you, {matched}.", self._active_personality, "name", matched)
            self._last_bot = reply
            self._turns.append((user_input, reply))
            return reply

        # Keep dynamic time/date support (existing feature)
        dyn = _dynamic_reply(user_input)
        if dyn:
            styled_dyn = _style_with_personality(dyn, self._active_personality, intent)
            self._last_bot = styled_dyn
            self._turns.append((user_input, styled_dyn))
            return styled_dyn

        if intent == "programming_question":
            prog = detect_keywords(user_input)
            if prog:
                styled = _style_with_personality(prog, self._active_personality, "programming_question")
                self._last_bot = styled
                self._turns.append((user_input, styled))
                return styled
            reply = self.get_personality_response(user_input)
            reply = _style_with_personality(reply, self._active_personality, "programming_question")
            self._last_bot = reply
            self._turns.append((user_input, reply))
            return reply

        if intent == "question":
            # For general questions, use existing personality bank fallback.
            reply = self.get_personality_response(user_input)
            reply = _style_with_personality(reply, self._active_personality, "question")
            self._last_bot = reply
            self._turns.append((user_input, reply))
            return reply

        if intent == "help_commands":
            reply = (
                "Available commands: /help, /time, /date, /joke, /clear, /history, "
                "/mode funny|motivator|angry|professional"
            )
            reply = _style_with_personality(reply, self._active_personality, "help_commands")
            self._last_bot = reply
            self._turns.append((user_input, reply))
            return reply

        if intent == "motivation_need":
            templates = {
                "funny": "You got this. If life is a bug, you're the hotfix. 🚀",
                "motivator": "Keep pushing. One focused step right now can change your whole week. 💪",
                "angry": "Stand up and do it. No excuses.",
                "professional": "You are capable of steady progress. Let us proceed one step at a time.",
                "default": "You can do this. Start with one small action now.",
            }
            reply = templates.get(self._active_personality, templates["default"])
            self._last_bot = reply
            self._turns.append((user_input, reply))
            return reply

        if intent == "emotional_support":
            mood_reply = detect_mood(user_input)
            if mood_reply:
                styled = _style_with_personality(mood_reply, self._active_personality, "emotional_support")
                self._last_bot = styled
                self._turns.append((user_input, styled))
                return styled
            sentiment = sentiment_detection(user_input)
            if sentiment:
                styled = _style_with_personality(sentiment, self._active_personality, "emotional_support")
                self._last_bot = styled
                self._turns.append((user_input, styled))
                return styled
            # fallback to motivator bank if personality is motivator
            reply = self.get_personality_reply(self._active_personality, user_input)
            reply = _style_with_personality(reply, self._active_personality, "emotional_support")
            self._last_bot = reply
            self._turns.append((user_input, reply))
            return reply

        if intent == "joke":
            # try personality joke first
            joke = self.get_personality_reply(self._active_personality, "tell me a joke")
            if joke:
                self._last_bot = joke
                self._turns.append((user_input, joke))
                return joke
            reply = self.get_personality_response(user_input)
            self._last_bot = reply
            self._turns.append((user_input, reply))
            return reply

        if intent == "greeting":
            reply = self.get_personality_reply(self._active_personality, "hello")
            self._last_bot = reply
            self._turns.append((user_input, reply))
            return reply

        if intent == "farewell":
            reply = self.get_personality_reply(self._active_personality, "bye")
            self._last_bot = reply
            self._turns.append((user_input, reply))
            return reply

        if intent == "stop":
            templates = {
                "funny": "Alright, I'll zip it. 🤐",
                "motivator": "Understood. If you need support later, I'm here.",
                "professional": "Understood. I will stop now.",
                "angry": "Fine.",
                "default": "Okay, stopping now.",
            }
            stop_reply = templates.get(self._active_personality, templates["default"])
            self._last_bot = stop_reply
            self._turns.append((user_input, stop_reply))
            return stop_reply

        if intent == "casual_slang":
            reply = self.get_personality_response(user_input)
            self._last_bot = reply
            self._turns.append((user_input, reply))
            return reply

        if intent == "casual_chat":
            reply = self.get_personality_response("how are you")
            reply = _style_with_personality(reply, self._active_personality, "casual_chat")
            self._last_bot = reply
            self._turns.append((user_input, reply))
            return reply

        # Smarter unknown fallback
        fallback_templates = {
            "funny": "I almost got that. Can you rephrase before my circuits start guessing wildly?",
            "motivator": "You are close. Rephrase your question and we will solve it together.",
            "professional": "I did not fully understand that request. Please rephrase with a little more detail.",
            "angry": "That was unclear. Ask it properly.",
            "default": "I did not fully get that. Try a clearer question like 'what is python' or 'tell me a joke'.",
        }
        reply = fallback_templates.get(self._active_personality, fallback_templates["default"])
        self._last_bot = reply
        self._turns.append((user_input, reply))
        return reply


# ─────────────────────────────────────────────
#  Backward-compatible module-level API
# ─────────────────────────────────────────────

_DEFAULT_ENGINE = ChatbotEngine()


def set_personality(name: str) -> str:
    return _DEFAULT_ENGINE.set_personality(name)


def get_personality_response(user_input: str) -> str:
    return _DEFAULT_ENGINE.get_personality_response(user_input)


def get_personality_reply(personality: str, message: str) -> str:
    return _DEFAULT_ENGINE.get_personality_reply(personality, message)


def get_response(user_input: str) -> str:
    return _DEFAULT_ENGINE.get_response(user_input)


def reply(user_input: str) -> str:
    return _DEFAULT_ENGINE.reply(user_input)


if __name__ == "__main__":
    # Backward-compatible launcher:
    # historically this project started via `python chatbot.py`.
    from main import main

    main()

