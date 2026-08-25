# ==============================================================================
# 0. IMPORTS & DEPENDENCIES
# ==============================================================================
import os
import io
import re
import csv
import json
import time
import math
import uuid
import base64
import random
import pickle
import shutil
import sqlite3
import colorsys
import hashlib
import datetime as dt
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    silhouette_score,
    davies_bouldin_score,
)
from scipy.spatial.distance import pdist, squareform

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    import streamlit as st
except ImportError:
    st = None


# ==============================================================================
# 1. CONSTANTS, VAD ANCHORS & SEED DATASETS
# ==============================================================================
CATEGORY_ANCHORS_VAD = {
    "stress":      (-0.55,  0.75, -0.30),
    "anger":       (-0.75,  0.90,  0.50),
    "calm":        ( 0.55, -0.50,  0.40),
    "clarity":     ( 0.65,  0.15,  0.60),
    "focus":       ( 0.35,  0.05,  0.55),
    "sad":         (-0.70, -0.25, -0.60),
    "fog":         (-0.35, -0.20, -0.40),
    "heavy":       (-0.50, -0.60, -0.55),
    "hope":        ( 0.50,  0.30,  0.10),
    "joy":         ( 0.85,  0.70,  0.55),
    "light":       ( 0.65, -0.10,  0.20),
    "fear":        (-0.70,  0.75, -0.60),
    "surprise":    ( 0.20,  0.85,  0.05),
    "disgust":     (-0.65,  0.40,  0.20),
    "guilt":       (-0.50,  0.10, -0.45),
    "pride":       ( 0.75,  0.55,  0.75),
    "boredom":     (-0.30, -0.75, -0.10),
    "love":        ( 0.85,  0.35,  0.40),
    "gratitude":   ( 0.70,  0.05,  0.25),
    "loneliness":  (-0.65, -0.40, -0.50),
    "excitement":  ( 0.75,  0.90,  0.45),
    "frustration": (-0.60,  0.60,  0.25),
    "nostalgia":   ( 0.15, -0.15, -0.05),
}

CATEGORIES = list(CATEGORY_ANCHORS_VAD.keys())

_READING_MAP = {
    "stress": "A charged, restless current",
    "anger": "Heat rising to the surface",
    "calm": "Settling into quiet ground",
    "clarity": "A clearing — light finding form",
    "focus": "A sharp, concentrated beam",
    "sad": "A muted, inward turn",
    "fog": "Lost in the shadows of the head",
    "heavy": "Weight, slowly loosening",
    "hope": "Something fragile lifting",
    "joy": "A bright, effortless opening",
    "light": "Touched by gentle gold",
    "fear": "A sudden tremor underfoot",
    "surprise": "An unexpected shift in the air",
    "disgust": "A sharp, recoiling edge",
    "guilt": "Looking backward through shadow",
    "pride": "Standing taller, quietly certain",
    "boredom": "A flat, unmoving stillness",
    "love": "Warmth filling all the corners",
    "gratitude": "Noticing the ground that holds you",
    "loneliness": "A quiet, hollow expanse",
    "excitement": "A spark catching into flame",
    "frustration": "Pressing against closed doors",
    "nostalgia": "A faded photograph, half-smiling",
}
READING_MAP = _READING_MAP

STYLE_NAMES = ["cloud", "silk", "prism", "aurora", "ink", "nebula"]

SEED_EXAMPLES = {
    "stress": [
        "Back-to-back deadlines and intense work pressure give me heavy stress.",
        "Overwhelmed, anxious, panicking, under extreme workload stress.",
        "Constant stress and tension, racing pulse and urgent pressure.",
        "Too much stress on my plate, strained and feeling stressed.",
    ],
    "anger": [
        "Furious and full of burning rage, extremely angry right now.",
        "Boiling with anger and furious hostility, snapping with rage.",
        "Infuriated and angry, bitter resentment and fierce fury.",
        "Pure anger and hot wrath, hostile and severely angered.",
    ],
    "calm": [
        "A serene and calm morning, breathing quietly in peaceful stillness.",
        "Grounded, tranquil, and calm, resting in quiet meditation.",
        "Deep calm settled over me, gently resting peacefully at ease.",
        "Relaxed, calm, unbothered, enjoying quiet tranquility.",
    ],
    "clarity": [
        "Sharp mental clarity, bright precision and no confusion.",
        "Lucid thoughts with total clarity, clear-headed and certain.",
        "Clean transparent clarity, sharp vision of the path forward.",
        "Total cognitive clarity, lucid and decisive with clear reasoning.",
    ],
    "focus": [
        "Laser focus and intense concentration on finishing tasks.",
        "Deeply focused, productive workflow, disciplined and locked in.",
        "Sustained task focus, executing plans with methodical concentration.",
        "Driven focus and disciplined attention on the work at hand.",
    ],
    "sad": [
        "Deep sad ache in my heart, weeping and sorrowful.",
        "Tearful, depressed, mournful, and deeply sad.",
        "Grief and sorrow, feeling downcast, unhappy, and sad.",
        "Crying softly in sorrow, downhearted and deeply sad.",
    ],
    "fog": [
        "Brain fog makes my mind hazy, cloudy, and disoriented.",
        "Murky thoughts and thick mental fog, discombobulated.",
        "Confused, hazy mind, completely lost in a mental fog.",
        "Unclear and spaced out, heavy brain fog clouding my mind.",
    ],
    "heavy": [
        "Exhausted, lethargic, carrying a heavy burden of fatigue.",
        "Heavy sluggish limbs, drained and completely devoid of energy.",
        "Weighed down by exhaustion, physically heavy and depleted.",
        "Severe burnout, weary, dragging my heavy exhausted body.",
    ],
    "hope": [
        "A fragile hope is rising, feeling hopeful about the future.",
        "Optimistic and full of hope, believing things will improve.",
        "A hopeful perspective, looking forward with bright optimism.",
        "Renewed with genuine hope, positive anticipation for tomorrow.",
    ],
    "joy": [
        "Bursting with pure joy and cheerful delight, so happy.",
        "Radiant joy and boundless happiness, laughing out loud.",
        "Ecstatic joy, celebratory, cheerful, and full of smiles.",
        "Blissful joy and vibrant delight, joyful exuberance.",
    ],
    "light": [
        "Gentle golden light shining softly and illuminating everything.",
        "Luminous brightness, airy glow, radiant warmth of sunlight.",
        "A bright golden light casting a soft, gentle shine.",
        "Sunlit and luminous, basking in glowing golden light.",
    ],
    "fear": [
        "Terrified and trembling with fear, cold dread in my spine.",
        "Paralyzed by sheer horror and fear, panicking and terrified.",
        "Gripped by sudden fright, fearing impending danger.",
        "Scared to death, trembling fear, feeling deeply endangered.",
    ],
    "surprise": [
        "Stunned in surprise, jaw dropped at this unexpected event.",
        "Completely caught off guard by the surprise announcement.",
        "Startled and shocked, an astonishing and sudden surprise.",
        "Astounded by the unexpected surprise, totally blindsided.",
    ],
    "disgust": [
        "Revolting, sickening, and nauseating, filled with utter disgust.",
        "Repulsed by the foul smell, grossed out in disgust.",
        "Pure revulsion and disgust, making my stomach churn.",
        "Appalled and disgusted by that repulsive sight.",
    ],
    "guilt": [
        "Consumed by remorse and guilt, blaming myself entirely.",
        "Ashamed and guilty, regretful over letting everyone down.",
        "Weighed down by conscience and deep personal guilt.",
        "Regretful, apologetic, and feeling terribly guilty for my error.",
    ],
    "pride": [
        "Proud of my hard-earned victory and triumphant success.",
        "Holding my head high with immense pride in this achievement.",
        "Proud and accomplished, celebrating this earned milestone.",
        "A deep sense of pride, self-respect, and triumphant glory.",
    ],
    "boredom": [
        "Tedious boredom, watching the seconds tick on the clock.",
        "Unstimulated, monotonous boredom, nothing interesting to do.",
        "Stuck in dull, uneventful boredom, restless and disengaged.",
        "Flat, dreary boredom, completely uninspired and bored.",
    ],
    "love": [
        "Deep affection, tenderness, and heartfelt love for my partner.",
        "Warmly cherishing my beloved family with devoted love.",
        "Heart full of romantic love, passion, and deep fondness.",
        "Affectionate, devoted, and overflowing with warm love.",
    ],
    "gratitude": [
        "Deeply thankful, expressing sincere gratitude for the kindness.",
        "Counting my blessings with heartfelt gratitude and thanks.",
        "Full of appreciation and gratitude for this generous support.",
        "Gratefully acknowledging the help, deeply appreciative.",
    ],
    "loneliness": [
        "Isolated in an empty room, aching with cold loneliness.",
        "Nobody to speak with, abandoned in solitary loneliness.",
        "Feeling deserted, disconnected, and painfully lonely.",
        "Solitude and painful loneliness, wishing for companionship.",
    ],
    "excitement": [
        "Hyped up, enthusiastic, and buzzing with thrilling excitement.",
        "Pumping with adrenaline and eager excitement for tomorrow!",
        "Electric thrills, cheering, and ecstatic excitement.",
        "Can barely sit still from bubbly, energized excitement.",
    ],
    "frustration": [
        "Annoyed by repeated roadblocks, hitting constant frustration.",
        "Exasperated and irritated, sheer frustration with these delays.",
        "Aggravated and frustrated that nothing is executing properly.",
        "Fed up with obstacles, venting my deep frustration.",
    ],
    "nostalgia": [
        "Bittersweet nostalgia reminiscing about cherished childhood years.",
        "Fond nostalgic memories of the past, wistful and sentimental.",
        "Looking back at vintage keepsakes with warm nostalgia.",
        "Longing for bygone days with tender, nostalgic sentimentality.",
    ],
}


# ==============================================================================
# 2. EMOTIONAL MAPPER & CURATED PALETTE SYSTEM
# ==============================================================================
class EmotionalMapper:
    def __init__(self, category_anchors=None):
        self.anchors = category_anchors or CATEGORY_ANCHORS_VAD

    def map(self, posterior):
        valence = arousal = dominance = 0.0
        for category, p in posterior.items():
            v, a, d = self.anchors.get(category, (0.0, 0.0, 0.0))
            valence += p * v
            arousal += p * a
            dominance += p * d

        clarity = float(np.clip(
            0.5 + posterior.get("clarity", 0) * 0.5 + posterior.get("light", 0) * 0.3
            - posterior.get("fog", 0) * 0.6 - posterior.get("stress", 0) * 0.2,
            0, 1,
        ))
        turbulence = float(np.clip(
            posterior.get("stress", 0) * 0.6 + posterior.get("anger", 0) * 0.6
            - posterior.get("calm", 0) * 0.4 + (arousal + 1) / 2 * 0.2,
            0, 1,
        ))

        themes = sorted(posterior.keys(), key=lambda c: -posterior[c])
        top_category = themes[0]

        calm_w = posterior.get("calm", 0) + 0.4 * posterior.get("sad", 0)
        prism_w = posterior.get("clarity", 0) + posterior.get("focus", 0) + 0.6 * posterior.get("light", 0)
        silk_w = posterior.get("heavy", 0) + posterior.get("fog", 0) + posterior.get("stress", 0) + 0.1
        
        if calm_w >= prism_w and calm_w >= silk_w and calm_w > 0.25:
            style = "cloud"
        elif prism_w >= silk_w and prism_w > 0.3:
            style = "prism"
        else:
            style = "silk"

        if arousal < -0.25 and valence >= -0.1:
            style = "cloud"

        reading = _READING_MAP.get(top_category) or (
            "A gentle brightening" if valence > 0.25 else
            "A muted, inward turn" if valence < -0.25 else
            "A quiet, even keel"
        )

        return {
            "valence": float(valence),
            "arousal": float(arousal),
            "dominance": float(dominance),
            "clarity": clarity,
            "turbulence": turbulence,
            "top_category": top_category,
            "style": style,
            "reading": reading,
            "themes": themes,
        }


def hsl_to_rgb01(h, s, l):
    h = (h % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return np.array([r, g, b])


def rgb01_to_hex(rgb01):
    return "#{:02x}{:02x}{:02x}".format(*(np.clip(rgb01, 0, 1) * 255).astype(int))


def deterministic_palette(valence, energy01, clarity, turbulence, theme_scores):
    T = theme_scores or {}
    if T.get("anger", 0) > 0.5 or T.get("stress", 0) > 1.0:
        hue, accent_hue, sat = 8, 350, 0.85
    elif T.get("calm", 0) > 0.4:
        hue, accent_hue, sat = 255, 285, 0.45
    elif T.get("clarity", 0) > 0.3 or T.get("light", 0) > 0.3:
        hue, accent_hue, sat = 45, 320, 0.6
    elif T.get("focus", 0) > 0.4:
        hue, accent_hue, sat = 42, 270, 0.55
    elif T.get("sad", 0) > 0.4:
        hue, accent_hue, sat = 225, 260, 0.4
    elif T.get("fog", 0) > 0.3 or T.get("heavy", 0) > 0.3:
        hue, accent_hue, sat = 265, 300, 0.4
    elif T.get("hope", 0) > 0.3 or T.get("joy", 0) > 0.3:
        hue, accent_hue, sat = 330, 45, 0.6
    elif T.get("fear", 0) > 0.4:
        hue, accent_hue, sat = 300, 260, 0.55
    elif T.get("surprise", 0) > 0.4:
        hue, accent_hue, sat = 190, 45, 0.6
    elif T.get("disgust", 0) > 0.4:
        hue, accent_hue, sat = 95, 40, 0.5
    elif T.get("guilt", 0) > 0.4:
        hue, accent_hue, sat = 15, 350, 0.35
    elif T.get("pride", 0) > 0.4:
        hue, accent_hue, sat = 20, 45, 0.6
    elif T.get("boredom", 0) > 0.4:
        hue, accent_hue, sat = 200, 220, 0.18
    elif T.get("love", 0) > 0.4:
        hue, accent_hue, sat = 345, 40, 0.65
    elif T.get("gratitude", 0) > 0.4:
        hue, accent_hue, sat = 55, 330, 0.55
    elif T.get("loneliness", 0) > 0.4:
        hue, accent_hue, sat = 210, 240, 0.35
    elif T.get("excitement", 0) > 0.4:
        hue, accent_hue, sat = 315, 45, 0.75
    elif T.get("frustration", 0) > 0.4:
        hue, accent_hue, sat = 0, 350, 0.7
    elif T.get("nostalgia", 0) > 0.4:
        hue, accent_hue, sat = 280, 200, 0.3
    else:
        hue, accent_hue, sat = 270, 320, 0.45

    hue += valence * 12
    sat = np.clip(sat + turbulence * 0.2 - (0.05 if valence < 0 else 0), 0.25, 0.9)

    base_l = 0.05 + (1 - energy01) * 0.02
    c0 = hsl_to_rgb01(hue + 200, sat * 0.4, base_l)
    c1 = hsl_to_rgb01(accent_hue, sat * 0.7, 0.18 + turbulence * 0.06)
    c2 = hsl_to_rgb01(hue, sat, 0.5 + clarity * 0.08)
    hi_l = 0.72 + clarity * 0.14
    c3 = hsl_to_rgb01(hue + 18, max(0.18, sat * 0.45), hi_l)
    return np.concatenate([c0, c1, c2, c3])


def deterministic_palette_hex(valence, energy01, clarity, turbulence, theme_scores):
    raw = deterministic_palette(valence, energy01, clarity, turbulence, theme_scores)
    return [rgb01_to_hex(s) for s in raw.reshape(4, 3)]


PRESET_DEFS = {
    "calm":    dict(label="Calm",    theme_scores={"calm": 1.2},                valence=0.6,  energy01=0.25, clarity=0.5,  turbulence=0.05),
    "clarity": dict(label="Clarity", theme_scores={"clarity": 1, "light": 0.6}, valence=0.6,  energy01=0.5,  clarity=0.85, turbulence=0.1),
    "warmth":  dict(label="Warmth",  theme_scores={"hope": 0.8, "joy": 0.6},    valence=0.75, energy01=0.55, clarity=0.6,  turbulence=0.15),
    "fog":     dict(label="Fog",     theme_scores={"fog": 0.8, "heavy": 0.5},   valence=-0.3, energy01=0.3,  clarity=0.25, turbulence=0.3),
    "stress":  dict(label="Stress",  theme_scores={"stress": 1.1, "anger": 0.3}, valence=-0.5, energy01=0.85, clarity=0.3,  turbulence=0.75),
    "sorrow":  dict(label="Sorrow",  theme_scores={"sad": 0.9},                 valence=-0.6, energy01=0.2,  clarity=0.35, turbulence=0.2),
}

CATEGORY_CORRECTION_DEFS = {
    "stress":      dict(label="Stress",      theme_scores={"stress": 1.1, "anger": 0.3},   valence=-0.5,  energy01=0.85,  clarity=0.3,  turbulence=0.75),
    "anger":       dict(label="Anger",       theme_scores={"anger": 0.9},                  valence=-0.7,  energy01=0.9,   clarity=0.3,  turbulence=0.85),
    "calm":        dict(label="Calm",        theme_scores={"calm": 1.2},                   valence=0.6,   energy01=0.25,  clarity=0.5,  turbulence=0.05),
    "clarity":     dict(label="Clarity",     theme_scores={"clarity": 1, "light": 0.6},    valence=0.6,   energy01=0.5,   clarity=0.85, turbulence=0.1),
    "focus":       dict(label="Focus",       theme_scores={"focus": 0.7},                  valence=0.55,  energy01=0.7,   clarity=0.7,  turbulence=0.15),
    "sad":         dict(label="Sorrow",      theme_scores={"sad": 0.9},                    valence=-0.6,  energy01=0.2,   clarity=0.35, turbulence=0.2),
    "fog":         dict(label="Fog",         theme_scores={"fog": 0.8, "heavy": 0.3},      valence=-0.3,  energy01=0.3,   clarity=0.25, turbulence=0.3),
    "heavy":       dict(label="Heavy",       theme_scores={"heavy": 0.7, "fog": 0.2},      valence=-0.45, energy01=0.35,  clarity=0.3,  turbulence=0.25),
    "hope":        dict(label="Warmth",      theme_scores={"hope": 0.8, "joy": 0.4},       valence=0.7,   energy01=0.5,   clarity=0.6,  turbulence=0.15),
    "joy":         dict(label="Joy",         theme_scores={"joy": 0.9, "hope": 0.2},       valence=0.78,  energy01=0.8,   clarity=0.6,  turbulence=0.1),
    "light":       dict(label="Light",       theme_scores={"light": 0.7, "clarity": 0.2},  valence=0.63,  energy01=0.675, clarity=0.75, turbulence=0.05),
    "fear":        dict(label="Fear",        theme_scores={"fear": 0.9},                   valence=-0.64, energy01=0.81,  clarity=0.25, turbulence=0.75),
    "surprise":    dict(label="Surprise",    theme_scores={"surprise": 0.8},               valence=0.20,  energy01=0.90,  clarity=0.4,  turbulence=0.55),
    "disgust":     dict(label="Disgust",     theme_scores={"disgust": 0.8},                valence=-0.60, energy01=0.675, clarity=0.35, turbulence=0.4),
    "guilt":       dict(label="Guilt",       theme_scores={"guilt": 0.7},                  valence=-0.55, energy01=0.55,  clarity=0.3,  turbulence=0.45),
    "pride":       dict(label="Pride",       theme_scores={"pride": 0.8},                  valence=0.65,  energy01=0.725, clarity=0.7,  turbulence=0.15),
    "boredom":     dict(label="Boredom",     theme_scores={"boredom": 0.7},                valence=-0.35, energy01=0.175, clarity=0.3,  turbulence=0.15),
    "love":        dict(label="Love",        theme_scores={"love": 0.8},                   valence=0.85,  energy01=0.675, clarity=0.6,  turbulence=0.15),
    "gratitude":   dict(label="Gratitude",   theme_scores={"gratitude": 0.8},              valence=0.75,  energy01=0.625, clarity=0.65, turbulence=0.1),
    "loneliness":  dict(label="Loneliness",  theme_scores={"loneliness": 0.8},             valence=-0.60, energy01=0.4,   clarity=0.3,  turbulence=0.25),
    "excitement":  dict(label="Excitement",  theme_scores={"excitement": 0.8},             valence=0.70,  energy01=0.925, clarity=0.5,  turbulence=0.4),
    "frustration": dict(label="Frustration", theme_scores={"frustration": 0.8},            valence=-0.55, energy01=0.825, clarity=0.3,  turbulence=0.7),
    "nostalgia":   dict(label="Nostalgia",   theme_scores={"nostalgia": 0.8},              valence=0.10,  energy01=0.45,  clarity=0.45, turbulence=0.2),
}


def category_correction_palette(category):
    d = CATEGORY_CORRECTION_DEFS[category]
    return deterministic_palette_hex(d["valence"], d["energy01"], d["clarity"], d["turbulence"], d["theme_scores"])


def preset_palette(preset_key):
    d = PRESET_DEFS[preset_key]
    return deterministic_palette_hex(d["valence"], d["energy01"], d["clarity"], d["turbulence"], d["theme_scores"])


# ==============================================================================
# 3. DIARY MOOD CLASSIFIER & REGRESSION MODELS
# ==============================================================================
class DiaryMoodClassifier:
    def __init__(self, seed_examples=None):
        seed_examples = seed_examples or SEED_EXAMPLES
        self.texts = []
        self.labels = []
        for category, examples in seed_examples.items():
            for text in examples:
                self.texts.append(text)
                self.labels.append(category)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.model = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
        self.mapper = EmotionalMapper(CATEGORY_ANCHORS_VAD)
        self._baseline_valence = 0.0
        self._fit()

    def _fit(self):
        X = self.vectorizer.fit_transform(self.texts)
        self.model.fit(X, self.labels)
        zero_vec = np.zeros((1, len(self.vectorizer.vocabulary_)))
        proba = self.model.predict_proba(zero_vec)[0]
        self._baseline_valence = sum(
            p * CATEGORY_ANCHORS_VAD.get(str(c).strip().lower(), (0.0, 0.0, 0.0))[0] 
            for c, p in zip(self.model.classes_, proba)
        )

    def get_state(self):
        return {
            "texts": self.texts, "labels": self.labels,
            "vectorizer": self.vectorizer, "model": self.model,
            "baseline_valence": self._baseline_valence,
        }

    def load_state(self, state):
        self.texts = state.get("texts", [])
        self.labels = state.get("labels", [])
        self.vectorizer = state.get("vectorizer")
        self.model = state.get("model")
        self._baseline_valence = state.get("baseline_valence", 0.0)
        self.mapper = EmotionalMapper(CATEGORY_ANCHORS_VAD)

    def learn(self, text, category):
        cat_key = str(category).strip().lower()
        if cat_key not in CATEGORY_ANCHORS_VAD:
            raise ValueError(f"Unknown category: {category}")
        self.texts.append(text)
        self.labels.append(cat_key)
        self._fit()

    def analyze(self, text):
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        classes = [str(c).strip().lower() for c in self.model.classes_]
        posterior = dict(zip(classes, proba))
        mapped = self.mapper.map(posterior)
        mapped["energy"] = mapped["arousal"]
        mapped["posterior"] = posterior
        return mapped

    def word_score(self, word):
        X = self.vectorizer.transform([word])
        if X.nnz == 0:
            return 0.0
        proba = self.model.predict_proba(X)[0]
        classes = [str(c).strip().lower() for c in self.model.classes_]
        raw = sum(
            p * CATEGORY_ANCHORS_VAD.get(cat, (0.0, 0.0, 0.0))[0]
            for cat, p in zip(classes, proba)
        )
        base = getattr(self, "_baseline_valence", 0.0)
        return float(raw - base)


def _random_theme_scores(rng):
    scores = {c: 0.0 for c in CATEGORIES}
    dominant = rng.choice(CATEGORIES)
    scores[dominant] = rng.uniform(0.3, 1.4)
    for _ in range(rng.integers(0, 3)):
        scores[rng.choice(CATEGORIES)] += rng.uniform(0.05, 0.4)
    return scores


def _theme_scores_from_posterior(posterior, rng, noise=0.15):
    if not posterior:
        return {}
    top_category = max(posterior, key=posterior.get)
    scores = {c: 0.0 for c in CATEGORIES}
    scores[top_category] = float(np.clip(posterior[top_category] * 3.0 + rng.uniform(-noise, noise), 0.3, 1.4))
    for cat, p in posterior.items():
        if cat != top_category and p > 0.15:
            scores[cat] = float(np.clip(p * 2.0 + rng.uniform(0, noise), 0.05, 0.6))
    return scores


class ArtColorModel:
    def __init__(self, n_synthetic_samples=800, random_state=42):
        rng = np.random.default_rng(random_state)
        self._X = []
        self._y = []
        self.n_synthetic_samples = n_synthetic_samples
        for _ in range(n_synthetic_samples):
            valence = rng.uniform(-1, 1)
            energy01 = rng.uniform(0, 1)
            clarity = rng.uniform(0, 1)
            turbulence = rng.uniform(0, 1)
            theme_scores = _random_theme_scores(rng)
            features = self._features(valence, energy01, clarity, turbulence, theme_scores)
            target = deterministic_palette(valence, energy01, clarity, turbulence, theme_scores)
            self._X.append(features)
            self._y.append(target.tolist())
        self.model = LinearRegression()
        self.n_real_corrections = 0
        self._fit()

    @staticmethod
    def _features(valence, energy01, clarity, turbulence, theme_scores):
        theme_scores = theme_scores or {}
        return [valence, energy01, clarity, turbulence] + [theme_scores.get(c, 0.0) for c in CATEGORIES]

    @staticmethod
    def _hex_to_rgb01(hexcolor):
        h = hexcolor.lstrip("#")
        return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]

    def _fit(self):
        self.model.fit(np.array(self._X), np.array(self._y))

    def get_state(self):
        return {
            "X": self._X, "y": self._y, "model": self.model,
            "n_synthetic_samples": self.n_synthetic_samples,
            "n_real_corrections": self.n_real_corrections,
        }

    def load_state(self, state):
        self._X = state["X"]
        self._y = state["y"]
        self.model = state["model"]
        self.n_synthetic_samples = state["n_synthetic_samples"]
        self.n_real_corrections = state["n_real_corrections"]

    def predict_palette(self, valence, energy01, clarity, turbulence, theme_scores):
        features = [self._features(valence, energy01, clarity, turbulence, theme_scores)]
        raw = np.clip(self.model.predict(np.array(features))[0], 0, 1)
        stops = raw.reshape(4, 3)
        return ["#{:02x}{:02x}{:02x}".format(*(stop * 255).astype(int)) for stop in stops]

    def learn(self, valence, energy01, clarity, turbulence, theme_scores, chosen_hex_palette):
        if len(chosen_hex_palette) != 4:
            raise ValueError("Expected exactly 4 hex colours (base, mid, accent, highlight).")
        features = self._features(valence, energy01, clarity, turbulence, theme_scores)
        target = [c for hexcolor in chosen_hex_palette for c in self._hex_to_rgb01(hexcolor)]
        self._X.append(features)
        self._y.append(target)
        self.n_real_corrections += 1
        self._fit()

    def training_set_size(self):
        return len(self._X)

    def data_summary(self):
        return {
            "synthetic_examples": self.n_synthetic_samples,
            "real_corrections": self.n_real_corrections,
            "total_training_examples": len(self._X),
            "fraction_real": self.n_real_corrections / max(1, len(self._X)),
        }


# ==============================================================================
# 4. PALETTE CVAE ARCHITECTURE & LATENT DIFFUSION
# ==============================================================================
COND_DIM = 5
X_DIM = 12
LATENT_DIM = 4


class _CVAEEmoMapper:
    def map(self, posterior):
        v = sum(p * ((i % 5) - 2) / 2 for i, p in enumerate(posterior.values()))
        a = sum(p * ((i % 3) - 1) for i, p in enumerate(posterior.values()))
        d = sum(p * ((i % 4) - 1.5) / 1.5 for i, p in enumerate(posterior.values()))
        return {
            "valence": float(np.clip(v, -1.0, 1.0)),
            "arousal": float(np.clip(a, -1.0, 1.0)),
            "dominance": float(np.clip(d, -1.0, 1.0)),
            "clarity": float(np.clip(max(posterior.values()), 0.0, 1.0)),
            "turbulence": float(np.clip(1.0 - min(posterior.values()), 0.0, 1.0))
        }


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(X_DIM + COND_DIM, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2)
        )
        self.mu = nn.Linear(32, LATENT_DIM)
        self.logvar = nn.Linear(32, LATENT_DIM)

    def forward(self, x, c):
        h = self.net(torch.cat([x, c], dim=-1))
        return self.mu(h), self.logvar(h)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LATENT_DIM + COND_DIM, 64)
        self.fc2 = nn.Linear(64, 32)
        self.film_gamma = nn.Linear(COND_DIM, 32)
        self.film_beta = nn.Linear(COND_DIM, 32)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)
        self.out_fc = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, X_DIM),
            nn.Sigmoid()
        )

    def forward(self, z, c):
        h = F.leaky_relu(self.fc1(torch.cat([z, c], dim=-1)), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        gamma = self.film_gamma(c)
        beta = self.film_beta(c)
        h = gamma * h + beta
        return self.out_fc(h)


class CVAEArtModel:
    def __init__(self, n_synthetic_samples=2500, epochs=5, lr=1e-3, batch_size=64, random_state=42):
        torch.manual_seed(random_state)
        self.encoder = _Encoder()
        self.decoder = _Decoder()
        self.mapper = _CVAEEmoMapper()
        self.batch_size = batch_size
        self.n_synthetic_samples = n_synthetic_samples
        self.n_real_corrections = 0
        self._X, self._C = self._build_synthetic_dataset(n_synthetic_samples, random_state)
        if epochs > 0:
            self._train(epochs, lr, verbose=False)

    @staticmethod
    def _cvae_deterministic_palette(valence, energy01, clarity, turbulence, theme_scores, rng=None):
        T = theme_scores or {}
        if T.get("anger", 0) > 0.5:
            base_hue = 8.0
        elif T.get("calm", 0) > 0.4:
            base_hue = 255.0
        elif T.get("clarity", 0) > 0.3:
            base_hue = 45.0
        else:
            base_hue = 270.0

        base_hue += valence * 15.0
        if rng is not None:
            hue = base_hue + rng.normal(0, 35.0)
            hue_step = rng.uniform(40.0, 140.0)
            sat_jitter = rng.uniform(-0.25, 0.25)
            lum_shift = rng.uniform(-0.15, 0.15)
        else:
            hue = base_hue
            hue_step = 90.0
            sat_jitter = 0.0
            lum_shift = 0.0

        base_sat = np.clip(0.5 + turbulence * 0.25 + sat_jitter, 0.15, 0.95)
        colors = []
        for i in range(4):
            h_offset = (hue + i * hue_step) % 360
            s = np.clip(base_sat * (0.35 + i * 0.2), 0.05, 0.95)
            l = np.clip(0.25 + (i * 0.18) + clarity * 0.12 + lum_shift, 0.05, 0.95)
            c = (1.0 - abs(2.0 * l - 1.0)) * s
            x = c * (1.0 - abs(((h_offset / 60.0) % 2.0) - 1.0))
            m = l - c / 2.0

            if h_offset < 60:
                r, g, b = c, x, 0.0
            elif h_offset < 120:
                r, g, b = x, c, 0.0
            elif h_offset < 180:
                r, g, b = 0.0, c, x
            elif h_offset < 240:
                r, g, b = 0.0, x, c
            elif h_offset < 300:
                r, g, b = x, 0.0, c
            else:
                r, g, b = c, 0.0, x

            colors.extend([
                np.clip(r + m, 0.0, 1.0),
                np.clip(g + m, 0.0, 1.0),
                np.clip(b + m, 0.0, 1.0)
            ])
        return np.array(colors, dtype=np.float32)

    def _build_synthetic_dataset(self, n, random_state):
        rng = np.random.default_rng(random_state)
        X, C = [], []
        for _ in range(n):
            raw = rng.dirichlet(np.ones(len(CATEGORIES)) * rng.uniform(0.3, 3.0))
            posterior = dict(zip(CATEGORIES, raw))
            mapped = self.mapper.map(posterior)
            cond = [mapped["valence"], mapped["arousal"], mapped["dominance"],
                    mapped["clarity"], mapped["turbulence"]]
            theme_scores = _theme_scores_from_posterior(posterior, rng)
            energy01 = (mapped["arousal"] + 1) / 2
            target = CVAEArtModel._cvae_deterministic_palette(
                mapped["valence"], energy01, mapped["clarity"],
                mapped["turbulence"], theme_scores, rng=rng
            )
            X.append(target.tolist())
            C.append(cond)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(C, dtype=torch.float32)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _train(self, epochs, lr, verbose=True):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        loader = DataLoader(TensorDataset(self._X, self._C), batch_size=self.batch_size, shuffle=True)
        log_path = Path("cvae_training_fixed.csv")
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["epoch", "reconstruction_loss", "kl_loss", "total_loss", "color_std", "beta"])
        free_bits = 0.15

        loss_before = 0.0
        for epoch in range(epochs):
            self.encoder.train()
            self.decoder.train()
            warmup = int(0.3 * epochs)
            beta = min(0.04, (epoch / max(1, warmup)) * 0.04)
            epoch_recon, epoch_kl, epoch_total = 0.0, 0.0, 0.0

            for bx, bc in loader:
                opt.zero_grad()
                mu, logvar = self.encoder(bx, bc)
                z = self.reparameterize(mu, logvar)
                recon = self.decoder(z, bc)
                recon_loss = F.mse_loss(recon, bx, reduction="none").sum(dim=-1).mean()
                kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = torch.clamp(kl_per_dim, min=free_bits).sum(dim=-1).mean()
                loss = recon_loss + beta * kl_loss
                loss.backward()
                opt.step()

                epoch_recon += recon_loss.item() * bx.size(0)
                epoch_kl += kl_loss.item() * bx.size(0)
                epoch_total += loss.item() * bx.size(0)

            n = len(self._X)
            epoch_recon /= n
            epoch_kl /= n
            epoch_total /= n
            if epoch == 0:
                loss_before = epoch_total

            self.decoder.eval()
            with torch.no_grad():
                z_test = torch.randn(len(self._C), LATENT_DIM)
                recon_test = self.decoder(z_test, self._C)
                std_dev = recon_test.std(dim=0).mean().item()

            log_writer.writerow([epoch, f"{epoch_recon:.6f}", f"{epoch_kl:.6f}", f"{epoch_total:.6f}", f"{std_dev:.6f}", f"{beta:.4f}"])
            log_file.flush()

        log_file.close()
        return loss_before, epoch_total

    def sample(self, mood_vad, n_samples=5):
        cond = torch.tensor([[mood_vad["valence"], mood_vad["arousal"], mood_vad["dominance"],
                              mood_vad["clarity"], mood_vad["turbulence"]]] * n_samples, dtype=torch.float32)
        self.decoder.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, LATENT_DIM)
            out = self.decoder(z, cond).cpu().numpy()

        hex_palettes = []
        for row in out:
            stops = np.clip(row, 0, 1).reshape(4, 3)
            hex_palettes.append([f"#{int(s[0]*255):02x}{int(s[1]*255):02x}{int(s[2]*255):02x}" for s in stops])
        return hex_palettes

    def data_summary(self):
        return {
            "synthetic_examples": self.n_synthetic_samples,
            "total_training_examples": len(self._X),
        }

    def fine_tune(self, mood_vad, target_hex_palette, steps=15, lr=1e-3):
        target = torch.tensor([c for hexcolor in target_hex_palette for c in ArtColorModel._hex_to_rgb01(hexcolor)], dtype=torch.float32).unsqueeze(0)
        cond = torch.tensor([[mood_vad["valence"], mood_vad["arousal"], mood_vad["dominance"],
                              mood_vad["clarity"], mood_vad["turbulence"]]], dtype=torch.float32)
        params = list(self.decoder.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        self.decoder.train()
        for _ in range(steps):
            opt.zero_grad()
            z = torch.zeros(1, LATENT_DIM)
            recon = self.decoder(z, cond)
            loss = F.mse_loss(recon, target)
            loss.backward()
            opt.step()
        self._X = torch.cat([self._X, target], dim=0)
        self._C = torch.cat([self._C, cond], dim=0)
        self.n_real_corrections += 1

    def maybe_periodic_retrain(self):
        if self.n_real_corrections > 0 and self.n_real_corrections % 10 == 0:
            loss_before, loss_after = self._train(epochs=40, lr=1e-3, verbose=False)
            return True, loss_before, loss_after
        return False, 0.0, 0.0

    def get_state(self):
        return {
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "X": self._X, "C": self._C,
            "n_synthetic_samples": self.n_synthetic_samples,
            "n_real_corrections": self.n_real_corrections
        }

    def load_state(self, state):
        self.encoder = _Encoder()
        self.encoder.load_state_dict(state["encoder_state_dict"])
        self.decoder = _Decoder()
        self.decoder.load_state_dict(state["decoder_state_dict"])
        self.mapper = _CVAEEmoMapper()
        self._X = state["X"]
        self._C = state["C"]
        self.n_synthetic_samples = state["n_synthetic_samples"]
        self.n_real_corrections = state.get("n_real_corrections", 0)


# ==============================================================================
# 5. K-MEANS ARCHIVE & TEMPORAL LSTM FORECASTER
# ==============================================================================
class ArchiveClustering:
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.cluster_labels = {}

    def fit(self, entries):
        if len(entries) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} entries to form {self.n_clusters} clusters.")
        X = np.array([[e["valence"], e["arousal"], e["clarity"], e["turbulence"]] for e in entries])
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        assignments = self.model.fit_predict(X)
        self._name_clusters()
        return [int(a) for a in assignments]

    def _name_clusters(self):
        for cluster_id, centroid in enumerate(self.model.cluster_centers_):
            c_valence, c_energy = centroid[0], centroid[1]
            nearest = min(
                CATEGORY_ANCHORS_VAD.items(),
                key=lambda kv: (kv[1][0] - c_valence) ** 2 + (kv[1][1] - c_energy) ** 2,
            )
            self.cluster_labels[cluster_id] = nearest[0]


class MoodLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=24, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = x[:, -1, :]
        delta = self.fc(out[:, -1, :])
        return torch.clamp(last_step + delta, -1.0, 1.0)


def _generate_synthetic_vad_trajectories(n_sequences=1200, seq_len=6, random_state=42):
    rng = np.random.default_rng(random_state)
    sequences = []
    for _ in range(n_sequences):
        vad = rng.uniform(-0.5, 0.5, size=3)
        momentum = rng.normal(0, 0.05, size=3)
        seq = [vad.copy()]
        for _ in range(seq_len - 1):
            momentum = 0.75 * momentum + rng.normal(0, 0.05, size=3)
            vad = np.clip(vad + momentum, -1.0, 1.0)
            seq.append(vad.copy())
        sequences.append(seq)
    return np.array(sequences, dtype=np.float32)


class MoodTemporalForecaster:
    def __init__(self, hidden_dim=24, epochs=5, lr=2e-3, batch_size=32, val_fraction=0.15, random_state=42):
        torch.manual_seed(random_state)
        self.hidden_dim = hidden_dim
        self.net = MoodLSTM(hidden_dim=hidden_dim)
        self.train_losses = []
        self.val_losses = []
        if epochs > 0:
            self._train(epochs, lr, batch_size, val_fraction, random_state)

    def _train(self, epochs, lr, batch_size, val_fraction, random_state):
        data = _generate_synthetic_vad_trajectories(n_sequences=1500, seq_len=6, random_state=random_state)
        n_val = int(len(data) * val_fraction)
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(data))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_train = torch.tensor(data[train_idx][:, :-1, :], dtype=torch.float32)
        y_train = torch.tensor(data[train_idx][:, -1, :], dtype=torch.float32)
        X_val = torch.tensor(data[val_idx][:, :-1, :], dtype=torch.float32)
        y_val = torch.tensor(data[val_idx][:, -1, :], dtype=torch.float32)

        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()

        for epoch in range(epochs):
            self.net.train()
            epoch_loss = 0.0
            for bx, by in loader:
                opt.zero_grad()
                pred = self.net(bx)
                loss = loss_fn(pred, by)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * bx.size(0)

            epoch_loss /= len(X_train)
            self.train_losses.append(epoch_loss)

            self.net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net(X_val), y_val).item()
            self.val_losses.append(val_loss)

    def predict(self, history_vad_sequence):
        self.net.eval()
        arr = np.array(history_vad_sequence, dtype=np.float32)
        x = torch.from_numpy(arr).unsqueeze(0)
        with torch.no_grad():
            return self.net(x).squeeze(0).cpu().numpy()

    def get_state(self):
        return {"net_state_dict": self.net.state_dict(), "hidden_dim": self.hidden_dim}

    def load_state(self, state):
        self.hidden_dim = state["hidden_dim"]
        self.net = MoodLSTM(hidden_dim=self.hidden_dim)
        self.net.load_state_dict(state["net_state_dict"])


def evaluate_forecaster_vs_baseline(forecaster, history_vad_sequence, verbose=True):
    if len(history_vad_sequence) < 4:
        return np.array(history_vad_sequence[-1])
    data = np.array(history_vad_sequence, dtype=np.float32)
    inputs, target = data[:-1], data[-1]
    pred = forecaster.predict(inputs)
    sma_pred = data[-4:-1].mean(axis=0)
    if verbose:
        print(f"Forecaster MSE: {np.mean((pred - target) ** 2):.4f} | SMA MSE: {np.mean((sma_pred - target) ** 2):.4f}")
    return pred


# ==============================================================================
# 6. PROCEDURAL RENDERING & IMAGE CVAE ENGINE
# ==============================================================================
def _hex_to_rgb(hexcolor):
    h = hexcolor.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _layer_cloud(size, pal, rng):
    w, h = size
    img = Image.new("RGB", size, pal[0])
    draw = ImageDraw.Draw(img)
    colors = [pal[1], pal[1], pal[2], pal[2], pal[3]]
    for i in range(12):
        color = rng.choice(colors)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(min(w, h) * 0.15, min(w, h) * 0.45) * (1 - i / 20)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return img


def _layer_silk(size, pal, rng):
    w, h = size
    img = Image.new("RGB", size, pal[0])
    draw = ImageDraw.Draw(img)
    colors = [pal[1], pal[2], pal[2], pal[3]]
    for i in range(6):
        color = colors[i % len(colors)]
        y0 = rng.uniform(-h * 0.2, h * 1.2)
        amp = rng.uniform(h * 0.08, h * 0.22)
        freq = rng.uniform(1.2, 2.4)
        phase = rng.uniform(0, 2 * math.pi)
        width = rng.uniform(w * 0.06, w * 0.16)
        points = []
        for s in range(41):
            x = w * s / 40
            y = y0 + amp * math.sin(freq * (x / w) * 2 * math.pi + phase)
            points.append((x, y))
        draw.line(points, fill=color, width=int(width), joint="curve")
    return img


def _layer_prism(size, pal, rng):
    w, h = size
    img = Image.new("RGB", size, pal[0])
    draw = ImageDraw.Draw(img)
    colors = [pal[0], pal[1], pal[2], pal[3]]
    pts = [(rng.uniform(-w * 0.1, w * 1.1), rng.uniform(-h * 0.1, h * 1.1)) for _ in range(22)]
    for _ in range(34):
        p1, p2, p3 = rng.sample(pts, 3)
        color = rng.choice(colors)
        jitter = rng.uniform(0.85, 1.15)
        color = tuple(min(255, int(c * jitter)) for c in color)
        draw.polygon([p1, p2, p3], fill=color)
    return img


def _layer_aurora(size, pal, rng):
    w, h = size
    base = Image.new("RGB", size, pal[0])
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [pal[1], pal[2], pal[3]]
    for i in range(5):
        color = colors[i % len(colors)] + (rng.randint(70, 140),)
        x0 = rng.uniform(0, w)
        sway = rng.uniform(w * 0.08, w * 0.2)
        freq = rng.uniform(0.8, 1.6)
        phase = rng.uniform(0, 2 * math.pi)
        width = rng.uniform(w * 0.05, w * 0.12)
        points_left, points_right = [], []
        for s in range(31):
            y = h * s / 30
            x = x0 + sway * math.sin(freq * (y / h) * 2 * math.pi + phase)
            points_left.append((x - width / 2, y))
            points_right.append((x + width / 2, y))
        draw.polygon(points_left + points_right[::-1], fill=color)
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def _layer_ink(size, pal, rng):
    w, h = size
    img = Image.new("RGB", size, pal[0])
    draw = ImageDraw.Draw(img)
    colors = [pal[1], pal[2], pal[3]]
    for i in range(4):
        color = colors[i % len(colors)]
        cx, cy = rng.uniform(w * 0.15, w * 0.85), rng.uniform(h * 0.15, h * 0.85)
        base_r = rng.uniform(min(w, h) * 0.18, min(w, h) * 0.32)
        points = []
        for k in range(14):
            angle = 2 * math.pi * k / 14
            wobble = base_r * rng.uniform(0.55, 1.35)
            points.append((cx + wobble * math.cos(angle), cy + wobble * math.sin(angle)))
        draw.polygon(points, fill=color)
    return img


def _layer_nebula(size, pal, rng):
    w, h = size
    img = Image.new("RGB", size, tuple(int(c * 0.7) for c in pal[0]))
    draw = ImageDraw.Draw(img)
    colors = [pal[1], pal[2], pal[3]]
    for i in range(7):
        color = rng.choice(colors)
        cx, cy = rng.uniform(w * 0.25, w * 0.75), rng.uniform(h * 0.25, h * 0.75)
        r = rng.uniform(min(w, h) * 0.2, min(w, h) * 0.5)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    n_stars = int(0.0015 * w * h)
    for _ in range(n_stars):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        bright = rng.random()
        r = 0.6 + bright * 2.8
        color = tuple(min(255, int(c + (255 - c) * bright)) for c in pal[3])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return img


_STYLE_LAYERS = {
    "cloud": _layer_cloud, "silk": _layer_silk, "prism": _layer_prism,
    "aurora": _layer_aurora, "ink": _layer_ink, "nebula": _layer_nebula,
}

_STYLE_BLUR_FRACTION = {
    "cloud": 0.018, "silk": 0.007, "prism": 0.004,
    "aurora": 0.009, "ink": 0.005, "nebula": 0.004,
}


def render_abstract_art(palette_hex, seed=None, size=(640, 400), style="cloud", blur_radius=None):
    if style not in _STYLE_LAYERS:
        style = "cloud"
    rgb_palette = [_hex_to_rgb(c) for c in palette_hex]
    w, h = size

    if seed is not None:
        seed_int = int(hashlib.sha256(str(seed).encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed_int)
    else:
        rng = random.Random()

    base = _STYLE_LAYERS[style](size, rgb_palette, rng)
    if blur_radius is None:
        blur_radius = max(1, int(min(w, h) * _STYLE_BLUR_FRACTION[style]))
    blurred = base.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    sharpen_percent = 60 if style in ("prism", "ink") else 140
    blurred = blurred.filter(ImageFilter.UnsharpMask(radius=2, percent=sharpen_percent, threshold=2))

    if style not in ("prism", "ink", "nebula"):
        accent_layer = Image.new("RGBA", size, (0, 0, 0, 0))
        adraw = ImageDraw.Draw(accent_layer)
        for _ in range(5):
            color = rgb_palette[3] + (rng.randint(40, 100),)
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            r = rng.uniform(8, 28) * (min(w, h) / 400)
            adraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        accent_layer = accent_layer.filter(ImageFilter.GaussianBlur(radius=max(1, int(min(w, h) * 0.015))))
        blurred = Image.alpha_composite(blurred.convert("RGBA"), accent_layer).convert("RGB")

    return blurred


def render_thumbnail(palette_hex, seed=None, size=(120, 120), style="cloud"):
    return render_abstract_art(palette_hex, seed=seed, size=size, style=style)


def render_meditation_gif(palette_hex, seed=None, size=(640, 400), style="cloud", n_frames=24, duration_ms=90):
    if style not in _STYLE_LAYERS:
        style = "cloud"
    rgb_palette = [_hex_to_rgb(c) for c in palette_hex]
    w, h = size
    seed_int = int(hashlib.sha256(str(seed if seed is not None else random.random()).encode()).hexdigest(), 16) % (2**32)
    blur_radius = max(1, int(min(w, h) * _STYLE_BLUR_FRACTION[style]))
    base_seed = seed_int
    sharpen_percent = 60 if style in ("prism", "ink") else 140
    frames = []
    for f in range(n_frames):
        t = 2 * math.pi * f / n_frames
        frame_rng = random.Random(base_seed)
        drift_draws = int(4 + 3 * math.sin(t))
        for _ in range(drift_draws):
            frame_rng.random()
        layer = _STYLE_LAYERS[style](size, rgb_palette, frame_rng)
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        layer = layer.filter(ImageFilter.UnsharpMask(radius=2, percent=sharpen_percent, threshold=2))
        frames.append(layer)
    return frames


def frames_to_gif_bytes(frames, duration_ms=90):
    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=False,
    )
    buf.seek(0)
    return buf.getvalue()


IMG_SIZE = 48
IMG_VAD_DIM = 5
IMG_STYLE_DIM = len(STYLE_NAMES)
IMG_COND_DIM = IMG_VAD_DIM + IMG_STYLE_DIM
IMG_LATENT_DIM = 16


def _style_onehot(style):
    vec = [0.0] * IMG_STYLE_DIM
    if style in STYLE_NAMES:
        vec[STYLE_NAMES.index(style)] = 1.0
    else:
        vec[0] = 1.0
    return vec


def _mood_vad_to_list(mood_vad):
    return [mood_vad["valence"], mood_vad["arousal"], mood_vad["dominance"],
            mood_vad["clarity"], mood_vad["turbulence"]]


class _ImgEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.flat_dim = 64 * 6 * 6
        self.fc = nn.Linear(self.flat_dim + IMG_COND_DIM, 128)
        self.mu = nn.Linear(128, IMG_LATENT_DIM)
        self.logvar = nn.Linear(128, IMG_LATENT_DIM)

    def forward(self, x, c):
        h = self.conv(x).flatten(1)
        h = F.relu(self.fc(torch.cat([h, c], dim=-1)))
        return self.mu(h), self.logvar(h)


class _ImgDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(IMG_LATENT_DIM + IMG_COND_DIM, 64 * 6 * 6)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z, c):
        h = F.relu(self.fc(torch.cat([z, c], dim=-1)))
        h = h.view(-1, 64, 6, 6)
        return self.deconv(h)


class _ImageCVAENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _ImgEncoder()
        self.decoder = _ImgDecoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, c), mu, logvar


def _image_cvae_loss(recon, x, mu, logvar, beta=0.0005):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.item(), kl.item()


class GenerativeArtImageModel:
    def __init__(self, n_synthetic_samples=480, epochs=2, lr=1e-3, random_state=42):
        torch.manual_seed(random_state)
        self.net = _ImageCVAENet()
        self.mapper = EmotionalMapper()
        self.n_synthetic_samples = n_synthetic_samples
        self.n_real_examples = 0
        self._X, self._C = self._build_synthetic_dataset(n_synthetic_samples, random_state)
        if epochs > 0:
            self._train(epochs, lr, verbose=False)

    def _build_synthetic_dataset(self, n, random_state):
        rng = np.random.default_rng(random_state)
        images, conds = [], []
        for i in range(n):
            style = STYLE_NAMES[i % len(STYLE_NAMES)]
            concentration = rng.uniform(0.3, 3.0)
            raw = rng.dirichlet(np.ones(11) * concentration)
            posterior = dict(zip(CATEGORIES, raw))
            mapped = self.mapper.map(posterior)
            cond = [mapped["valence"], mapped["arousal"], mapped["dominance"], mapped["clarity"], mapped["turbulence"]] + _style_onehot(style)
            theme_scores = _theme_scores_from_posterior(posterior, rng)
            energy01 = (mapped["arousal"] + 1) / 2
            palette_raw = deterministic_palette(mapped["valence"], energy01, mapped["clarity"], mapped["turbulence"], theme_scores)
            palette_hex = ["#{:02x}{:02x}{:02x}".format(*(np.clip(s, 0, 1) * 255).astype(int)) for s in palette_raw.reshape(4, 3)]

            img = render_abstract_art(palette_hex, seed=int(rng.integers(0, 10**9)), size=(IMG_SIZE, IMG_SIZE), style=style)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)

            images.append(arr)
            conds.append(cond)
        return torch.tensor(np.array(images), dtype=torch.float32), torch.tensor(conds, dtype=torch.float32)

    def _train(self, epochs, lr, verbose=True):
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        batch_size = 32
        n = self._X.shape[0]
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                recon, mu, logvar = self.net(self._X[idx], self._C[idx])
                loss, recon_l, kl_l = _image_cvae_loss(recon, self._X[idx], mu, logvar)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(idx)

    def get_state(self):
        return {
            "net_state_dict": self.net.state_dict(),
            "X": self._X, "C": self._C,
            "n_synthetic_samples": self.n_synthetic_samples,
            "n_real_examples": getattr(self, "n_real_examples", 0),
        }

    def data_summary(self):
        total = self._X.shape[0]
        return {
            "synthetic_examples": self.n_synthetic_samples,
            "real_examples": self.n_real_examples,
            "total_training_examples": total,
            "fraction_real": self.n_real_examples / max(1, total),
        }

    def load_state(self, state):
        self.net = _ImageCVAENet()
        self.net.load_state_dict(state["net_state_dict"])
        self.mapper = EmotionalMapper()
        self._X = state["X"]
        self._C = state["C"]
        self.n_synthetic_samples = state["n_synthetic_samples"]
        self.n_real_examples = state["n_real_examples"]

    @staticmethod
    def _tensor_to_image(t, upscale_to=240):
        arr = (t.clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")
        return img.resize((upscale_to, upscale_to), Image.BICUBIC)

    def sample(self, mood_vad, style="cloud", n_samples=1, upscale_to=240):
        cond_row = _mood_vad_to_list(mood_vad) + _style_onehot(style)
        cond = torch.tensor([cond_row] * n_samples, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, IMG_LATENT_DIM)
            out = self.net.decoder(z, cond)
        return [self._tensor_to_image(out[i], upscale_to) for i in range(n_samples)]

    def sample_animation(self, mood_vad, style="cloud", n_frames=24, upscale_to=240, radius=1.4, random_state=None):
        rng = np.random.default_rng(random_state)
        e1 = rng.normal(size=IMG_LATENT_DIM)
        e2 = rng.normal(size=IMG_LATENT_DIM)
        e2 = e2 - e1 * (e1 @ e2) / (e1 @ e1)
        e1 = e1 / np.linalg.norm(e1)
        e2 = e2 / np.linalg.norm(e2)

        cond_row = _mood_vad_to_list(mood_vad) + _style_onehot(style)
        cond = torch.tensor([cond_row] * n_frames, dtype=torch.float32)

        angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
        z_path = np.array([radius * (np.cos(a) * e1 + np.sin(a) * e2) for a in angles])
        z = torch.tensor(z_path, dtype=torch.float32)

        self.net.eval()
        with torch.no_grad():
            out = self.net.decoder(z, cond)
        return [self._tensor_to_image(out[i], upscale_to) for i in range(n_frames)]

    def fine_tune(self, mood_vad, target_image, style="cloud", steps=25, lr=3e-4):
        img_resized = target_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        arr = np.asarray(img_resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        target = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        cond_row = _mood_vad_to_list(mood_vad) + _style_onehot(style)
        cond = torch.tensor([cond_row], dtype=torch.float32)

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        for _ in range(steps):
            opt.zero_grad()
            recon, mu, logvar = self.net(target, cond)
            loss, _, _ = _image_cvae_loss(recon, target, mu, logvar)
            loss.backward()
            opt.step()

        self._X = torch.cat([self._X, target], dim=0)
        self._C = torch.cat([self._C, cond], dim=0)
        self.n_real_examples = getattr(self, "n_real_examples", 0) + 1


# ==============================================================================
# 7. MULTI-USER STORAGE & CACHED LOADERS
# ==============================================================================
USER_DIR = os.environ.get(
    "MOOD_APP_DATA_DIR",
    os.path.join(os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(), "user_data")
)


def user_dir(user_id):
    d = os.path.join(USER_DIR, user_id)
    os.makedirs(d, exist_ok=True)
    return d


def _user_path(user_id, filename):
    return os.path.join(user_dir(user_id), filename)


def _model_state_to_b64(obj):
    return base64.b64encode(pickle.dumps(obj.get_state())).decode("ascii")


def _model_state_from_b64(b64_str):
    return pickle.loads(base64.b64decode(b64_str))


def _load_pickle(path, cls, *init_args, user_id=None, model_name=None, **init_kwargs):
    supabase = _get_supabase_client()
    if supabase and user_id and model_name:
        try:
            res = (supabase.table("model_state").select("state_b64")
                   .eq("user_id", user_id).eq("model_name", model_name).execute())
            if res.data:
                obj = cls.__new__(cls)
                obj.load_state(_model_state_from_b64(res.data[0]["state_b64"]))
                return obj
        except Exception as e:
            print(f"WARNING: Supabase model-state read failed ({model_name}): {e}")

    if os.path.exists(path):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj.load_state(state)
        return obj

    obj = cls(*init_args, **init_kwargs)
    _save_pickle(path, obj, user_id=user_id, model_name=model_name)
    return obj


def _save_pickle(path, obj, user_id=None, model_name=None):
    with open(path, "wb") as f:
        pickle.dump(obj.get_state(), f)

    supabase = _get_supabase_client()
    if supabase and user_id and model_name:
        try:
            supabase.table("model_state").upsert({
                "user_id": user_id,
                "model_name": model_name,
                "state_b64": _model_state_to_b64(obj),
            }, on_conflict="user_id,model_name").execute()
        except Exception as e:
            print(f"WARNING: Supabase model-state write failed ({model_name}): {e}")


def load_diary_classifier(user_id):
    return _load_pickle(_user_path(user_id, "diary_classifier.pkl"), DiaryMoodClassifier,
                         user_id=user_id, model_name="diary_classifier")


def save_diary_classifier(user_id, clf):
    _save_pickle(_user_path(user_id, "diary_classifier.pkl"), clf, user_id=user_id, model_name="diary_classifier")


def load_art_model(user_id):
    return _load_pickle(_user_path(user_id, "art_model.pkl"), ArtColorModel,
                         user_id=user_id, model_name="art_model")


def save_art_model(user_id, model):
    _save_pickle(_user_path(user_id, "art_model.pkl"), model, user_id=user_id, model_name="art_model")


GLOBAL_MODEL_USER = "__global__"


def _model_state_exists(path, user_id, model_name):
    if os.path.exists(path):
        return True
    supabase = _get_supabase_client()
    if supabase:
        try:
            res = (supabase.table("model_state").select("model_name")
                   .eq("user_id", user_id).eq("model_name", model_name).limit(1).execute())
            return bool(res.data)
        except Exception:
            return False
    return False


def load_cvae_model(user_id):
    own_path = _user_path(user_id, "cvae_model.pkl")
    if _model_state_exists(own_path, user_id, "cvae_model"):
        return _load_pickle(own_path, CVAEArtModel, epochs=0, user_id=user_id, model_name="cvae_model")

    global_path = _user_path(GLOBAL_MODEL_USER, "cvae_model.pkl")
    if _model_state_exists(global_path, GLOBAL_MODEL_USER, "cvae_model"):
        model = _load_pickle(global_path, CVAEArtModel, epochs=0,
                              user_id=GLOBAL_MODEL_USER, model_name="cvae_model")
        save_cvae_model(user_id, model)
        return model

    return _load_pickle(own_path, CVAEArtModel, epochs=5, n_synthetic_samples=64, user_id=user_id, model_name="cvae_model")


def save_cvae_model(user_id, model):
    _save_pickle(_user_path(user_id, "cvae_model.pkl"), model, user_id=user_id, model_name="cvae_model")


def load_image_art_model(user_id):
    return _load_pickle(_user_path(user_id, "image_art_model.pkl"),
                         GenerativeArtImageModel, n_synthetic_samples=48, epochs=0,
                         user_id=user_id, model_name="image_art_model")


def save_image_art_model(user_id, model):
    _save_pickle(_user_path(user_id, "image_art_model.pkl"), model, user_id=user_id, model_name="image_art_model")


DB_PATH = os.path.join(USER_DIR, "analysis.db")


def _get_db():
    os.makedirs(USER_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS diary_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            text TEXT NOT NULL,
            valence REAL, arousal REAL, dominance REAL,
            clarity REAL, turbulence REAL,
            top_category TEXT,
            corrected_category TEXT,
            reading TEXT, style TEXT,
            palette TEXT,
            logged_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(diary_entries)")}
    if "corrected_category" not in existing_cols:
        conn.execute("ALTER TABLE diary_entries ADD COLUMN corrected_category TEXT")
    return conn


try:
    from supabase import create_client
except ImportError:
    create_client = None


def _get_supabase_client():
    if create_client is None:
        return None
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if url and key:
        return create_client(url, key)
    return None


def load_entry_history(user_id):
    supabase = _get_supabase_client()
    if supabase:
        try:
            res = (supabase.table("diary_entries").select("*").eq("user_id", user_id)
                   .order("logged_at").execute())
            data = res.data or []
            for row in data:
                if isinstance(row.get("palette"), str):
                    row["palette"] = json.loads(row["palette"])
            return data
        except Exception as e:
            print(f"Error fetching from cloud DB: {e}")

    path = _user_path(user_id, "entries.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def append_entry(user_id, entry):
    if "entry_id" not in entry:
        entry["entry_id"] = str(uuid.uuid4())

    history = load_entry_history(user_id)
    history.append(entry)

    supabase = _get_supabase_client()
    cloud_status = "local_only"
    if supabase:
        try:
            db_row = {
                "user_id": user_id,
                "entry_id": entry.get("entry_id"),
                "date": entry.get("date"),
                "text": entry.get("text"),
                "valence": entry.get("valence"),
                "arousal": entry.get("arousal"),
                "dominance": entry.get("dominance"),
                "clarity": entry.get("clarity"),
                "turbulence": entry.get("turbulence"),
                "top_category": entry.get("top_category"),
                "corrected_category": entry.get("corrected_category"),
                "reading": entry.get("reading"),
                "style": entry.get("style"),
                "palette": entry.get("palette")
            }
            supabase.table("diary_entries").insert(db_row).execute()
            cloud_status = "saved"
        except Exception as e:
            print(f"WARNING: Supabase write failed: {e}")
            cloud_status = str(e)

    with open(_user_path(user_id, "entries.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history, cloud_status


def update_entry(user_id, entry, updates):
    entry_id = entry.get("entry_id")
    supabase = _get_supabase_client()
    cloud_status = "local_only"

    if supabase:
        try:
            q = supabase.table("diary_entries").update(updates).eq("user_id", user_id)
            if entry_id:
                q = q.eq("entry_id", entry_id)
            else:
                q = q.eq("date", entry.get("date")).eq("text", entry.get("text"))
            q.execute()
            cloud_status = "saved"
        except Exception as e:
            print(f"WARNING: Supabase update failed: {e}")
            cloud_status = str(e)

    path = _user_path(user_id, "entries.json")
    if os.path.exists(path):
        with open(path) as f:
            local_history = json.load(f)
        for row in local_history:
            matches = (row.get("entry_id") == entry_id if entry_id
                       else row.get("date") == entry.get("date") and row.get("text") == entry.get("text"))
            if matches:
                row.update(updates)
        with open(path, "w") as f:
            json.dump(local_history, f, indent=2)
    return cloud_status


# ==============================================================================
# 8. UNIFIED PIPELINE & ABLATION STUDY
# ==============================================================================
_CVAE_SINGLETON = None
_FORECASTER_SINGLETON = None


def _get_cvae_model():
    global _CVAE_SINGLETON
    if _CVAE_SINGLETON is None:
        _CVAE_SINGLETON = CVAEArtModel(epochs=0)
    return _CVAE_SINGLETON


def _get_forecaster():
    global _FORECASTER_SINGLETON
    if _FORECASTER_SINGLETON is None:
        _FORECASTER_SINGLETON = MoodTemporalForecaster(epochs=0)
    return _FORECASTER_SINGLETON


def run_unified_pipeline(text, history_vad, user_preference_preset=None, ablations=None,
                          clf=None, cvae_model=None):
    ablations = ablations or []
    if clf is None:
        clf = DiaryMoodClassifier()
    res = clf.analyze(text)
    topic = res["top_category"]
    current_vad = [res["valence"], res["arousal"], res["dominance"]]

    if "no_lstm" not in ablations and len(history_vad) >= 3:
        full_seq = history_vad + [current_vad]
        forecaster = _get_forecaster()
        target_vad_arr = evaluate_forecaster_vs_baseline(forecaster, full_seq, verbose=False)
    else:
        target_vad_arr = np.array(current_vad)

    target_vad = {
        "valence": float(target_vad_arr[0]),
        "arousal": float(target_vad_arr[1]),
        "dominance": float(target_vad_arr[2]),
        "clarity": res["clarity"],
        "turbulence": res["turbulence"]
    }

    if "no_cvae" in ablations:
        final_palette = deterministic_palette_hex(
            target_vad["valence"], (target_vad["arousal"] + 1) / 2,
            target_vad["clarity"], target_vad["turbulence"], {}
        )
        render_engine = "Rule-based Fallback (No CVAE)"
    else:
        if cvae_model is None:
            cvae_model = _get_cvae_model()
        if "no_optimiser" not in ablations and user_preference_preset:
            pref_hex = preset_palette(user_preference_preset)
            cvae_model.fine_tune(target_vad, pref_hex, steps=15)
            render_engine = "CVAE + Optimised Preference"
        else:
            render_engine = "Raw CVAE Sample"

        final_palette = cvae_model.sample(target_vad, n_samples=1)[0]

    return {
        "analysis": res,
        "topic": topic,
        "target_vad": target_vad,
        "palette": final_palette,
        "engine": render_engine
    }


# ==============================================================================
# 9. STREAMLIT APP IMPLEMENTATION
# ==============================================================================
if st is not None:
    @st.cache_resource(show_spinner="Loading models into memory...")
    def get_cached_models(user_id):
        clf = load_diary_classifier(user_id)
        cvae = load_cvae_model(user_id)
        forecaster = _get_forecaster()
        img_model = load_image_art_model(user_id)
        return clf, cvae, forecaster, img_model


def _inject_theme_css():
    if st is not None:
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,400;1,500&family=Hanken+Grotesk:wght@300;400;500;600&display=swap');
        :root {
            --accent: #d9b673; --accent-glow: rgba(217, 182, 115, 0.25);
            --ink: #f2ede4; --ink-soft: #b8b0a4; --bg-deep: #0a0a0f; --bg-card: #14131a;
            --bg-card-border: rgba(217, 182, 115, 0.15);
            --serif: 'Cormorant Garamond', Georgia, serif;
            --sans: 'Hanken Grotesk', system-ui, -apple-system, sans-serif;
        }
        .stApp { background: radial-gradient(120% 100% at 20% 0%, #1a1420 0%, var(--bg-deep) 65%); color: var(--ink); font-family: var(--sans); }
        section[data-testid="stSidebar"] { background: var(--bg-card) !important; border-right: 1px solid var(--bg-card-border) !important; }
        h1, h2, h3 { font-family: var(--serif) !important; color: var(--ink) !important; }
        .mood-kicker { font-family: var(--sans); text-transform: uppercase; letter-spacing: 0.18em; font-size: 0.72rem; color: var(--accent); margin-bottom: 6px; }
        .mood-reading { font-family: var(--serif); font-size: 1.35rem; font-style: italic; color: var(--ink); line-height: 1.4; margin: 10px 0; }
        .stTextArea textarea { background: rgba(20, 19, 26, 0.8) !important; color: var(--ink) !important; border: 1px solid var(--bg-card-border) !important; border-radius: 8px !important; font-family: var(--serif) !important; font-size: 1.15rem !important; }
        .stButton > button { font-family: var(--sans); border-radius: 999px !important; border: 1px solid rgba(217,182,115,0.4) !important; background: transparent !important; color: var(--ink) !important; }
        .stButton > button[kind="primary"] { background: var(--accent) !important; color: #100e0a !important; border: none !important; font-weight: 600; }
        div[data-testid="stMetricValue"] { font-family: var(--serif) !important; color: var(--accent) !important; }
        </style>
        """, unsafe_allow_html=True)


def _meditation_rings_html(reading_text=""):
    return f"""
    <style>
    @keyframes moodBreathe {{
        0%   {{ transform: scale(0.65); opacity: 0.35; border-color: rgba(217,182,115,0.3); }}
        40%  {{ transform: scale(1.05); opacity: 0.95; border-color: rgba(217,182,115,0.85); box-shadow: 0 0 30px rgba(217,182,115,0.35); }}
        60%  {{ transform: scale(1.05); opacity: 0.95; border-color: rgba(217,182,115,0.85); box-shadow: 0 0 30px rgba(217,182,115,0.35); }}
        100% {{ transform: scale(0.65); opacity: 0.35; border-color: rgba(217,182,115,0.3); }}
    }}
    .med-container {{ display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 0; background: rgba(20, 19, 26, 0.4); border-radius: 16px; border: 1px solid rgba(217,182,115,0.12); margin-top: 15px; }}
    .med-ring-outer {{ position: relative; width: 180px; height: 180px; border-radius: 50%; border: 1.5px solid rgba(217,182,115,0.6); animation: moodBreathe 11s ease-in-out infinite; }}
    .med-ring-inner {{ position: absolute; top: 18px; left: 18px; width: 140px; height: 140px; border-radius: 50%; border: 1px dashed rgba(217,182,115,0.4); animation: moodBreathe 11s ease-in-out infinite 0.35s; }}
    .med-caption {{ margin-top: 26px; font-family: 'Cormorant Garamond', Georgia, serif; font-style: italic; font-size: 1.25rem; color: #f2ede4; text-align: center; }}
    </style>
    <div class="med-container">
        <div class="med-ring-outer"><div class="med-ring-inner"></div></div>
        <div class="med-caption">"{reading_text}"</div>
    </div>
    """


def _evolution_timeline_svg(history, width=780, height=280):
    if len(history) < 2:
        return None
    pad_x, pad_y = 60, 40
    n = len(history)
    plot_w = width - pad_x * 2
    plot_h = height - pad_y * 2
    mid_y = pad_y + plot_h / 2

    def catmull_rom(points):
        if len(points) < 2:
            return ""
        d = f"M {points[0][0]:.1f} {points[0][1]:.1f}"
        for i in range(len(points) - 1):
            p0 = points[i - 1] if i - 1 >= 0 else points[i]
            p1, p2 = points[i], points[i + 1]
            p3 = points[i + 2] if i + 2 < len(points) else p2
            c1x, c1y = p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6
            c2x, c2y = p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6
            d += f" C {c1x:.1f} {c1y:.1f}, {c2x:.1f} {c2y:.1f}, {p2[0]:.1f} {p2[1]:.1f}"
        return d

    pts = []
    for i, e in enumerate(history):
        x = pad_x + (i / (n - 1)) * plot_w
        v = float(np.clip(e.get("valence", 0.0), -1.0, 1.0))
        y = mid_y - v * (plot_h * 0.42)
        pts.append((x, y))

    path = catmull_rom(pts)
    stops = "".join(
        f'<stop offset="{i/(n-1) if n>1 else 0:.2f}" stop-color="{e.get("palette", ["#d9b673"]*4)[2]}"/>'
        for i, e in enumerate(history)
    )
    dots = "".join(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="4.5" fill="#d9b673" stroke="#0a0a0f" stroke-width="1.5"/>' for p in pts)

    return f"""
    <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;background:rgba(20,19,26,0.5);border-radius:12px;border:1px solid rgba(217,182,115,0.15);padding:10px 0;">
        <defs>
            <linearGradient id="ribbonGrad" x1="0" y1="0" x2="1" y2="0">{stops}</linearGradient>
            <filter id="ribbonGlow" x="-20%" y="-40%" width="140%" height="180%">
                <feGaussianBlur stdDeviation="6" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
        </defs>
        <line x1="{pad_x}" y1="{mid_y:.1f}" x2="{width - pad_x}" y2="{mid_y:.1f}" stroke="rgba(217,182,115,0.12)" stroke-dasharray="4 4" stroke-width="1"/>
        <path d="{path}" fill="none" stroke="url(#ribbonGrad)" stroke-width="7" opacity="0.4" filter="url(#ribbonGlow)"/>
        <path d="{path}" fill="none" stroke="url(#ribbonGrad)" stroke-width="2.5"/>
        {dots}
    </svg>
    """


def run_streamlit_app():
    if st is None:
        return

    st.set_page_config(page_title="Mood Evolution", page_icon="◐", layout="centered")
    _inject_theme_css()

    st.sidebar.markdown('<div class="mood-kicker">Account Sandbox</div>', unsafe_allow_html=True)
    st.sidebar.markdown("### Profile Control")
    user_id = st.sidebar.text_input("User ID", value=st.session_state.get("user_id", "me"))
    st.session_state["user_id"] = user_id

    if st.sidebar.button("Reset User Sandbox"):
        user_dir_path = user_dir(user_id)
        if os.path.exists(user_dir_path):
            shutil.rmtree(user_dir_path)
        st.sidebar.success("Local state cleared. Reload page.")

    clf, cvae_model, forecaster, image_art_model = get_cached_models(user_id)

    tab_today, tab_archive, tab_evolution = st.tabs(["Today", "Archive", "Evolution"])

    with tab_today:
        st.markdown('<div class="mood-kicker">Reflection Entry</div>', unsafe_allow_html=True)
        st.markdown(f"## {dt.date.today().strftime('%B %-d, %Y')}")
        if "draft_text" not in st.session_state:
            st.session_state.draft_text = ""

        text = st.text_area(
            "How are you, today?",
            value=st.session_state.draft_text,
            height=130,
            placeholder="Describe your inner thoughts, subtle feelings, or events...",
            label_visibility="collapsed",
        )
        st.session_state.draft_text = text
        word_count = len(text.split())

        col_reflect, col_regen = st.columns([1, 1])
        reflect_clicked = col_reflect.button("Reflect", disabled=word_count < 3, type="primary", use_container_width=True)
        regenerate_clicked = col_regen.button("Sample Again", disabled="last_result" not in st.session_state, use_container_width=True)

        if reflect_clicked:
            history = load_entry_history(user_id)
            history_vads = [[e["valence"], e["arousal"], e["dominance"]] for e in history if "valence" in e]
            pipe_res = run_unified_pipeline(text, history_vads, clf=clf, cvae_model=cvae_model)
            result = pipe_res["analysis"]

            tokens = re.findall(r"[a-zA-Z']+", text.lower())
            words = [{"w": w, "score": clf.word_score(w)} for w in tokens]
            words = [w for w in words if abs(w["score"]) > 0.03]
            seen, deduped = set(), []
            for w in words:
                if w["w"] not in seen:
                    seen.add(w["w"])
                    deduped.append(w)

            st.session_state.last_result = {
                "text": text,
                "analysis": result,
                "target_vad": pipe_res["target_vad"],
                "cvae_palette": pipe_res["palette"],
                "engine": pipe_res["engine"],
                "words": deduped[:6],
                "art_seed": text,
            }

        if regenerate_clicked and "last_result" in st.session_state:
            r = st.session_state.last_result
            r["cvae_palette"] = cvae_model.sample(r["target_vad"], n_samples=1)[0]
            r["art_seed"] = None

        if "last_result" in st.session_state:
            r = st.session_state.last_result
            result = r["analysis"]

            st.markdown(f'<div class="mood-reading">"{result["reading"]}"</div>', unsafe_allow_html=True)
            st.markdown(
                f'<span style="color:var(--accent);text-transform:uppercase;font-size:0.78rem;letter-spacing:0.14em;">Top Tone: <strong>{result["top_category"]}</strong></span> '
                f'<span style="color:var(--ink-faint);margin:0 8px;">|</span> <span style="color:var(--ink-soft);font-size:0.78rem;">{r["engine"]}</span>',
                unsafe_allow_html=True,
            )

            override_choice = st.selectbox(
                "Palette Selection",
                ["Auto (detected)"] + [CATEGORY_CORRECTION_DEFS[c]["label"] for c in CATEGORIES] + ["Custom color"],
                key="palette_choice",
                label_visibility="collapsed",
            )

            active_palette = r["cvae_palette"]
            override_for_training = None

            if override_choice != "Auto (detected)":
                if override_choice == "Custom color":
                    c_col1, c_col2, c_col3, c_col4 = st.columns(4)
                    c0 = c_col1.color_picker("Base", "#0b0f0f")
                    c1 = c_col2.color_picker("Mid", "#492133")
                    c2 = c_col3.color_picker("Accent", "#b45977")
                    c3 = c_col4.color_picker("Highlight", "#d2c1c1")
                    active_palette = [c0, c1, c2, c3]
                    override_for_training = ("custom", active_palette)
                else:
                    category = [c for c in CATEGORIES if CATEGORY_CORRECTION_DEFS[c]["label"] == override_choice][0]
                    active_palette = category_correction_palette(category)
                    override_for_training = ("preset", category)

            swatch_cols = st.columns(4)
            for i, hexcolor in enumerate(active_palette):
                swatch_cols[i].markdown(
                    f'<div style="background:{hexcolor};height:32px;border-radius:6px;border:1px solid rgba(255,255,255,0.08);box-shadow:0 2px 8px rgba(0,0,0,0.3);"></div>',
                    unsafe_allow_html=True,
                )

            art_mode = st.radio(
                "Art Engine",
                ["Procedural Geometry (fast, stable)", "Generative Canvas (Image CVAE)"],
                horizontal=True,
            )
            style_choice = st.selectbox("Aesthetic Style", STYLE_NAMES, index=0)

            if art_mode.startswith("Generative Canvas"):
                with st.spinner("Generating latent space artwork..."):
                    art = image_art_model.sample(r["target_vad"], style=style_choice, n_samples=1)[0]
            else:
                art = render_abstract_art(active_palette, seed=r["art_seed"] or f"{text}-{id(r)}", style=style_choice)

            meditate_on = st.toggle("🧘 Meditate — Slow breathing motion")
            if meditate_on:
                with st.spinner("Rendering meditation animation..."):
                    if art_mode.startswith("Generative Canvas"):
                        frames = image_art_model.sample_animation(r["target_vad"], style=style_choice, n_frames=24, upscale_to=360)
                    else:
                        frames = render_meditation_gif(active_palette, seed=r["art_seed"] or text, style=style_choice, n_frames=24, size=(640, 400))
                    gif_bytes = frames_to_gif_bytes(frames, duration_ms=90)
                st.image(gif_bytes, use_container_width=True)
                st.markdown(_meditation_rings_html(result["reading"]), unsafe_allow_html=True)
            else:
                st.image(art, use_container_width=True)

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Valence", f"{r['target_vad']['valence']:+.2f}")
            m_col2.metric("Arousal", f"{r['target_vad']['arousal']:+.2f}")
            m_col3.metric("Clarity", f"{r['target_vad']['clarity']:.2f}")

            if st.button("Seal This Day", type="primary", use_container_width=True):
                corrected_category = None
                if override_for_training and override_for_training[0] == "preset":
                    corrected_category = override_for_training[1]

                entry = {
                    "date": dt.date.today().isoformat(),
                    "text": text,
                    "valence": r["target_vad"]["valence"],
                    "arousal": r["target_vad"]["arousal"],
                    "dominance": r["target_vad"]["dominance"],
                    "clarity": r["target_vad"]["clarity"],
                    "turbulence": r["target_vad"]["turbulence"],
                    "top_category": result["top_category"],
                    "reading": result["reading"],
                    "palette": active_palette,
                    "style": style_choice,
                    "corrected_category": corrected_category,
                }
                append_entry(user_id, entry)

                if override_for_training:
                    kind, val = override_for_training
                    if kind == "preset":
                        clf.learn(text, val)
                        save_diary_classifier(user_id, clf)
                    cvae_model.fine_tune(r["target_vad"], active_palette, steps=30)
                    save_cvae_model(user_id, cvae_model)

                st.session_state.draft_text = ""
                del st.session_state["last_result"]
                st.success("Day sealed and preserved.")
                st.rerun()

    with tab_archive:
        history = load_entry_history(user_id)
        if not history:
            st.info("No journal entries sealed yet.")
        else:
            cluster_labels = None
            if len(history) >= 3:
                archive_model = ArchiveClustering(n_clusters=min(3, len(history)))
                assignments = archive_model.fit(history)
                cluster_labels = [archive_model.cluster_labels[a] for a in assignments]

            for i, entry in enumerate(reversed(history)):
                idx = len(history) - 1 - i
                thumb = render_thumbnail(entry["palette"], seed=entry["text"], style=entry.get("style", "cloud"))
                cols = st.columns([1, 3])
                cols[0].image(thumb, use_container_width=True)
                with cols[1]:
                    cluster_str = f" · <em>cluster: {cluster_labels[idx]}</em>" if cluster_labels else ""
                    st.markdown(
                        f'<div class="mood-kicker">{entry["date"]}{cluster_str}</div>'
                        f'<div style="font-family:var(--serif);font-size:1.15rem;color:var(--ink);">{entry["reading"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f'"{entry["text"][:160]}{"..." if len(entry["text"]) > 160 else ""}"')
                    with st.expander("Calibrate or Correct"):
                        correction_choice = st.selectbox(
                            "Corrected Tone",
                            ["No change"] + [CATEGORY_CORRECTION_DEFS[c]["label"] for c in CATEGORIES],
                            key=f"archive_correction_{idx}",
                            label_visibility="collapsed",
                        )
                        if st.button("Apply Correction", key=f"archive_apply_{idx}"):
                            if correction_choice != "No change":
                                category = [c for c in CATEGORIES if CATEGORY_CORRECTION_DEFS[c]["label"] == correction_choice][0]
                                correction_hex = category_correction_palette(category)
                                entry_vad = {
                                    "valence": entry["valence"],
                                    "arousal": entry["arousal"],
                                    "dominance": entry["dominance"],
                                    "clarity": entry["clarity"],
                                    "turbulence": entry["turbulence"]
                                }
                                clf.learn(entry["text"], category)
                                save_diary_classifier(user_id, clf)
                                cvae_model.fine_tune(entry_vad, correction_hex, steps=30)
                                save_cvae_model(user_id, cvae_model)
                                update_entry(user_id, entry, {"corrected_category": category, "palette": correction_hex})
                                st.success("Entry and model weights updated.")
                                st.rerun()
                st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)

    with tab_evolution:
        history = load_entry_history(user_id)
        if len(history) < 2:
            st.info("Log at least 2 entries to activate evolution ribbons and temporal forecasts.")
        else:
            st.markdown('<div class="mood-kicker">Emotional Trajectory</div>', unsafe_allow_html=True)
            st.markdown("### How your inner weather has evolved")
            svg = _evolution_timeline_svg(history)
            if svg:
                st.markdown(svg, unsafe_allow_html=True)

            days = list(range(len(history)))
            valences = [e["valence"] for e in history if "valence" in e]
            history_vads = [[e["valence"], e["arousal"], e["dominance"]] for e in history if "valence" in e]
            forecaster = _get_forecaster()
            next_vad = evaluate_forecaster_vs_baseline(forecaster, history_vads, verbose=False)

            fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="#14131a")
            ax.set_facecolor("#14131a")
            ax.plot(days, valences, "o-", label="Logged Valence", color="#d9b673", linewidth=2)
            ax.plot([len(days)], [next_vad[0]], "o--", label="Forecasted Next Day", color="#b9a3e0", linewidth=2)
            ax.axhline(0, color="#d9b673", alpha=0.2, linestyle="--", linewidth=1)
            ax.tick_params(colors="#b8b0a4", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#d9b673")
                spine.set_alpha(0.15)
            ax.legend(facecolor="#14131a", edgecolor="#d9b673", labelcolor="#f2ede4", fontsize=8)
            


# ==============================================================================
# 10. ENTRYPOINT / EXECUTION ROUTER
# ==============================================================================
if __name__ == "__main__":
    run_streamlit_app()
