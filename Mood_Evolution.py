
import os
import json
import sqlite3
import pickle
from datetime import datetime
import datetime as dt
import hashlib
import math
import random
import itertools
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    import streamlit as st
except ImportError:
    st = None


# ==============================================================================
# SECTION A — DATA & CONSTANTS
# ==============================================================================
CATEGORY_ANCHORS_VAD = {
    "stress":  (-0.55,  0.75, -0.30),
    "anger":   (-0.75,  0.90,  0.50),
    "calm":    ( 0.50, -0.45,  0.40),
    "clarity": ( 0.60,  0.35,  0.50),
    "focus":   ( 0.60,  0.50,  0.60),
    "sad":     (-0.65,  0.15, -0.50),
    "fog":     (-0.42,  0.15, -0.60),
    "heavy":   (-0.58,  0.10, -0.50),
    "hope":    ( 0.60,  0.40,  0.20),
    "joy":     ( 0.78,  0.60,  0.50),
    "light":   ( 0.63,  0.35,  0.30),
}

CATEGORIES = list(CATEGORY_ANCHORS_VAD.keys())

SEED_EXAMPLES = {
    "stress": [
        "Back-to-back meetings and deadlines have my chest tight and my mind racing.",
        "I feel so overwhelmed and anxious, there's too much pressure today.",
        "Tense all day, panic keeps creeping in whenever I check my inbox.",
        "Too busy, too much on my plate, I can't catch my breath.",
    ],
    "anger": [
        "I am furious about how that meeting went, everything feels frustrating.",
        "Rage bubbling under the surface, I snapped at everyone today.",
        "Frustrated and angry, nothing is going the way it should.",
        "Irritated all day, small things kept setting me off.",
    ],
    "calm": [
        "A peaceful morning, slow coffee and gentle light, I feel settled.",
        "Breathing deeply, grounded and still, everything feels safe.",
        "Quiet and at ease, I let a few things go softly today.",
        "Resting today, calm and gentle, nothing urgent to hold onto.",
    ],
    "clarity": [
        "Everything feels clear today, a sharp sense of certainty.",
        "My thoughts are bright and focused, the fog has lifted.",
        "A clean, clear headspace, I can finally see the way forward.",
        "Certain and clear-headed, the confusion from last week is gone.",
    ],
    "focus": [
        "Determined and driven, I feel capable and purposeful today.",
        "Sharp focus all morning, strong and gathered.",
        "Powerful sense of purpose, I know exactly what I need to do.",
        "Disciplined and focused, I got through everything I planned.",
    ],
    "sad": [
        "Empty and lonely today, a slow ache that won't leave.",
        "I feel low and heavy-hearted, close to tears.",
        "A quiet grief sitting with me, everything feels muted.",
        "Down and withdrawn, I just feel sad without a clear reason.",
    ],
    "fog": [
        "Confused and unclear, my thoughts feel blurry and lost.",
        "Everything is foggy today, I can't find a clear thought.",
        "Lost in the shadows of my own head, nothing makes sense.",
        "Unclear and hazy, I can't tell what I'm actually feeling.",
    ],
    "heavy": [
        "Exhausted and drained, I feel stuck carrying something heavy.",
        "Numb and tired, everything feels like too much effort.",
        "Heavy limbs, heavy mind, I can barely move today.",
        "Drained and stuck, even small tasks feel like too much.",
    ],
    "hope": [
        "Something is lifting, a fragile hope is returning.",
        "I feel like I'm slowly healing and becoming lighter.",
        "A quiet hope rising, things feel like they're turning around.",
        "Emerging from a hard stretch, a little more hopeful each day.",
    ],
    "joy": [
        "Grateful and alive, today felt genuinely happy and free.",
        "Bursting with joy, I feel loved and excited about everything.",
        "A beautiful day, I feel light, happy, and full of love.",
        "Excited and joyful, everything today felt easy and good.",
    ],
    "light": [
        "Warm golden light this morning, everything felt gentle and glowing.",
        "A bright shine over the day, warmth breaking through.",
        "Soft warm light, everything feels touched by gold.",
        "Everything glowed a little today, warm and bright.",
    ],
}

_READING_MAP = {
    "stress": "A charged, restless current", "anger": "Heat rising to the surface",
    "calm": "Settling into quiet ground", "clarity": "A clearing — light finding form",
    "focus": "Gathered, deliberate, gilded", "sad": "A slow, low tide",
    "fog": "Soft uncertainty, half-lit", "heavy": "Weight, slowly loosening",
    "hope": "Something becoming", "joy": "Open and luminous", "light": "Warmth breaking through",
}


# ==============================================================================
# SECTION B — EMOTIONAL MAPPER
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


# ==============================================================================
# SECTION C — CURATED PALETTE MAPPER
# ==============================================================================
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

PRESET_TO_CATEGORY = {"calm": "calm", "clarity": "clarity", "warmth": "hope", "fog": "fog", "stress": "stress", "sorrow": "sad"}


def preset_palette(preset_key):
    d = PRESET_DEFS[preset_key]
    return deterministic_palette_hex(d["valence"], d["energy01"], d["clarity"], d["turbulence"], d["theme_scores"])


# ==============================================================================
# SECTION D — DIARY CLASSIFIER
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
        self.vectorizer = CountVectorizer(lowercase=True, stop_words="english")
        self.model = MultinomialNB(alpha=1.0)
        self.mapper = EmotionalMapper(CATEGORY_ANCHORS_VAD)
        self._fit()

    def _fit(self):
        X = self.vectorizer.fit_transform(self.texts)
        self.model.fit(X, self.labels)
        import scipy.sparse as sp
        zero_vec = sp.csr_matrix((1, len(self.vectorizer.vocabulary_)))
        proba = self.model.predict_proba(zero_vec)[0]
        self._baseline_valence = sum(
            p * CATEGORY_ANCHORS_VAD[c][0] for c, p in zip(self.model.classes_, proba)
        )

    def get_state(self):
        return {
            "texts": self.texts, "labels": self.labels,
            "vectorizer": self.vectorizer, "model": self.model,
            "baseline_valence": self._baseline_valence,
        }

    def load_state(self, state):
        self.texts = state["texts"]
        self.labels = state["labels"]
        self.vectorizer = state["vectorizer"]
        self.model = state["model"]
        self._baseline_valence = state["baseline_valence"]
        self.mapper = EmotionalMapper(CATEGORY_ANCHORS_VAD)

    def learn(self, text, category):
        if category not in CATEGORY_ANCHORS_VAD:
            raise ValueError(f"Unknown category: {category}")
        self.texts.append(text)
        self.labels.append(category)
        self._fit()

    def analyze(self, text):
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
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
        classes = list(self.model.classes_)
        raw = sum(p * CATEGORY_ANCHORS_VAD[cat][0] for cat, p in zip(classes, proba))
        return float(raw - self._baseline_valence)

    def training_set_size(self):
        return len(self.texts)


# ==============================================================================
# SECTION E — ART COLOUR MODEL
# ==============================================================================
def _random_theme_scores(rng):
    scores = {c: 0.0 for c in CATEGORIES}
    dominant = rng.choice(CATEGORIES)
    scores[dominant] = rng.uniform(0.3, 1.4)
    for _ in range(rng.integers(0, 3)):
        scores[rng.choice(CATEGORIES)] += rng.uniform(0.05, 0.4)
    return scores


def _theme_scores_from_posterior(posterior, rng, noise=0.15):
    """
    Builds theme_scores CORRELATED with the same posterior used to derive
    the VAD condition -- unlike _random_theme_scores(), which draws an
    independent random dominant category.

    Bug this fixes: deterministic_palette()'s biggest source of colour
    variety is its theme-driven hue selection (anger=red, calm=blue-violet,
    joy/hope=pink, sad=blue, etc.) -- the continuous VAD-driven adjustments
    (hue += valence*12, small saturation/lightness shifts) are comparatively
    tiny. CVAEArtModel and GenerativeArtImageModel were trained with
    _random_theme_scores() drawing an INDEPENDENT theme for every synthetic
    sample, uncorrelated with the VAD condition built from that sample's
    posterior. Since the biggest driver of the target palette was invisible
    to (uncorrelated with) the conditioning input, the MSE-optimal thing to
    learn was the average palette across all independently-drawn themes for
    a given VAD -- a near-constant, muted blend. This is why real diary
    entries with very different classified moods produced visually
    near-identical palettes in the deployed app: the model was trained to
    ignore exactly the signal that would have let it differentiate them.

    This ties theme to VAD the way a real diary entry's classifier posterior
    actually would, using the same posterior already computed for the VAD
    condition in the calling code -- so condition and target are correlated
    the way they are at real inference time.
    """
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
# SECTION F — CVAE ART MODEL & LATENT SPACE INTERPOLATION
# ==============================================================================
COND_DIM = 5
X_DIM = 12
LATENT_DIM = 4


def slerp(val, low, high):
    """Spherical linear interpolation between two latent vectors."""
    low_u = low / np.linalg.norm(low)
    high_u = high / np.linalg.norm(high)
    dot = np.clip(np.dot(low_u, high_u), -1.0, 1.0)
    omega = np.arccos(dot)
    so = np.sin(omega)
    if so == 0 or np.isnan(so):
        return (1.0 - val) * low + val * high
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(X_DIM + COND_DIM, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.mu = nn.Linear(16, LATENT_DIM)
        self.logvar = nn.Linear(16, LATENT_DIM)

    def forward(self, x, c):
        h = self.net(torch.cat([x, c], dim=-1))
        return self.mu(h), self.logvar(h)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + COND_DIM, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, X_DIM), nn.Sigmoid(),
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=-1))


class _CVAENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, c)
        return recon, mu, logvar


def _cvae_loss(recon, x, mu, logvar, beta=0.01):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.item(), kl.item()


class CVAEArtModel:
    def __init__(self, n_synthetic_samples=1000, epochs=200, lr=1e-3, random_state=42):
        torch.manual_seed(random_state)
        self.net = _CVAENet()
        self.mapper = EmotionalMapper()
        self._X, self._C = self._build_synthetic_dataset(n_synthetic_samples, random_state)
        self.n_synthetic_samples = n_synthetic_samples
        self.n_real_corrections = 0
        self._train(epochs, lr, verbose=False)

    def _build_synthetic_dataset(self, n, random_state):
        rng = np.random.default_rng(random_state)
        X, C = [], []
        for _ in range(n):
            concentration = rng.uniform(0.3, 3.0)
            raw = rng.dirichlet(np.ones(len(CATEGORIES)) * concentration)
            posterior = dict(zip(CATEGORIES, raw))
            mapped = self.mapper.map(posterior)
            cond = [mapped["valence"], mapped["arousal"], mapped["dominance"], mapped["clarity"], mapped["turbulence"]]

            theme_scores = _theme_scores_from_posterior(posterior, rng)  # was _random_theme_scores(rng) -- see docstring
            energy01 = (mapped["arousal"] + 1) / 2
            target = deterministic_palette(mapped["valence"], energy01, mapped["clarity"], mapped["turbulence"], theme_scores)

            X.append(target.tolist())
            C.append(cond)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(C, dtype=torch.float32)

    def _train(self, epochs, lr, verbose=True):
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()

        # Loss BEFORE this training pass touches anything — needed so a
        # periodic retrain can actually report "here is what real user
        # corrections changed," not just "training ran."
        with torch.no_grad():
            recon0, mu0, logvar0 = self.net(self._X, self._C)
            loss_before, _, _ = _cvae_loss(recon0, self._X, mu0, logvar0)
        loss_before = loss_before.item()

        last_loss = loss_before
        for epoch in range(epochs):
            opt.zero_grad()
            recon, mu, logvar = self.net(self._X, self._C)
            loss, recon_l, kl_l = _cvae_loss(recon, self._X, mu, logvar)
            loss.backward()
            opt.step()
            last_loss = loss.item()

        if verbose:
            n_examples = self._X.shape[0]
            print(f"CVAEArtModel._train: {epochs} epochs over {n_examples} examples "
                  f"({self.n_real_corrections} real corrections included) | "
                  f"loss before: {loss_before:.4f} -> after: {last_loss:.4f}")

        return loss_before, last_loss

    def maybe_periodic_retrain(self, every=10, epochs=60, lr=1e-3, verbose=True):
        """
        fine_tune() only nudges the net with 30 Adam steps on a single new
        example, and never revisits earlier real corrections together — the
        growing self._X/self._C set otherwise just sits there unused. Every
        `every`-th real correction, this runs a proper multi-epoch pass over
        the FULL accumulated set (original synthetic data + every real
        correction so far), the same way _train() is used in __init__, so
        real corrections actually reinforce each other instead of each being
        a small, easily-overwritten nudge in isolation.
        Returns (ran: bool, loss_before: float|None, loss_after: float|None).
        """
        if self.n_real_corrections > 0 and self.n_real_corrections % every == 0:
            loss_before, loss_after = self._train(epochs, lr, verbose=verbose)
            return True, loss_before, loss_after
        return False, None, None

    def get_state(self):
        return {
            "net_state_dict": self.net.state_dict(),
            "X": self._X, "C": self._C,
            "n_synthetic_samples": self.n_synthetic_samples,
            "n_real_corrections": self.n_real_corrections,
        }

    def load_state(self, state):
        self.net = _CVAENet()
        self.net.load_state_dict(state["net_state_dict"])
        self.mapper = EmotionalMapper()
        self._X = state["X"]
        self._C = state["C"]
        self.n_synthetic_samples = state["n_synthetic_samples"]
        self.n_real_corrections = state["n_real_corrections"]

    def fine_tune(self, mood_vad, chosen_hex_palette, steps=30, lr=5e-4):
        cond = torch.tensor([[mood_vad["valence"], mood_vad["arousal"], mood_vad["dominance"],
                               mood_vad["clarity"], mood_vad["turbulence"]]], dtype=torch.float32)
        target = torch.tensor([[v for hexcolor in chosen_hex_palette for v in self._hex_to_rgb01(hexcolor)]],
                               dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        for _ in range(steps):
            opt.zero_grad()
            recon, mu, logvar = self.net(target, cond)
            loss, _, _ = _cvae_loss(recon, target, mu, logvar)
            loss.backward()
            opt.step()
        self._X = torch.cat([self._X, target], dim=0)
        self._C = torch.cat([self._C, cond], dim=0)
        self.n_real_corrections += 1

    @staticmethod
    def _hex_to_rgb01(hexcolor):
        h = hexcolor.lstrip("#")
        return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]

    @staticmethod
    def _rgb01_to_hex(vec12):
        stops = np.clip(vec12, 0, 1).reshape(4, 3)
        return ["#{:02x}{:02x}{:02x}".format(*(s * 255).astype(int)) for s in stops]

    def sample(self, mood_vad, n_samples=1):
        cond = torch.tensor([[mood_vad["valence"], mood_vad["arousal"], mood_vad["dominance"],
                               mood_vad["clarity"], mood_vad["turbulence"]]] * n_samples, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, LATENT_DIM)
            out = self.net.decoder(z, cond).numpy()
        return [self._rgb01_to_hex(row) for row in out]

    def training_set_size(self):
        return self._X.shape[0]

    def data_summary(self):
        total = self._X.shape[0]
        return {
            "synthetic_examples": self.n_synthetic_samples,
            "real_corrections": self.n_real_corrections,
            "total_training_examples": total,
            "fraction_real": self.n_real_corrections / max(1, total),
        }


def visualize_cvae_latent_interpolation(cvae_model, vad_start, vad_end, steps=5):
    """Generates texture transitions along the CVAE latent manifold."""
    torch.manual_seed(42)
    z1 = torch.randn(1, LATENT_DIM).numpy()[0]
    z2 = torch.randn(1, LATENT_DIM).numpy()[0]

    fig, axes = plt.subplots(1, steps, figsize=(12, 2.2))

    alphas, mean_r, mean_g, mean_b = [], [], [], []
    print("\n--- SLERP interpolation: real captured hex + VAD per step ---")

    for i in range(steps):
        alpha = i / (steps - 1)
        z_interp = torch.tensor([slerp(alpha, z1, z2)], dtype=torch.float32)

        cond_interp = [
            (1 - alpha) * vad_start[k] + alpha * vad_end[k]
            for k in ["valence", "arousal", "dominance", "clarity", "turbulence"]
        ]
        c_tensor = torch.tensor([cond_interp], dtype=torch.float32)

        cvae_model.net.eval()
        with torch.no_grad():
            generated_vec = cvae_model.net.decoder(z_interp, c_tensor).numpy()[0]

        palette_hex = cvae_model._rgb01_to_hex(generated_vec)

        # generated_vec is 12 values = 4 swatches x [R,G,B] in [0,1]. Mean channel
        # across the 4 swatches, at THIS alpha step -- this is the real number
        # that was missing before: nothing previously computed or plotted this.
        swatches = generated_vec.reshape(4, 3)
        alphas.append(alpha)
        mean_r.append(swatches[:, 0].mean())
        mean_g.append(swatches[:, 1].mean())
        mean_b.append(swatches[:, 2].mean())

        print(f"alpha={alpha:.2f}  VAD=[{cond_interp[0]:.2f}, {cond_interp[1]:.2f}, "
              f"{cond_interp[2]:.2f}]  palette={palette_hex}  "
              f"mean_RGB=({mean_r[-1]:.3f}, {mean_g[-1]:.3f}, {mean_b[-1]:.3f})")

        ax = axes[i]
        for idx, hex_color in enumerate(palette_hex):
            ax.add_patch(plt.Rectangle((idx, 0), 1, 1, color=hex_color))
        ax.set_xlim(0, 4); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"α={alpha:.2f}", fontsize=8)

    plt.suptitle("CVAE Continuous Latent Space Trajectory (Mood Transition)", fontsize=10)
    plt.tight_layout()
    plt.show()

    # The chart the original report claimed existed but the code never plotted --
    # real mean R/G/B channel values across alpha, not an ASCII mockup.
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(alphas, mean_r, "o-", color="#c0392b", label="Red channel")
    ax2.plot(alphas, mean_g, "o-", color="#27ae60", label="Green channel")
    ax2.plot(alphas, mean_b, "o-", color="#2980b9", label="Blue channel")
    ax2.set_xlabel("alpha (SLERP step)")
    ax2.set_ylabel("Mean channel value (0-1)")
    ax2.set_title("RGB channel drift across latent trajectory (real, not illustrative)")
    ax2.set_ylim(0, 1)
    ax2.legend()
    plt.tight_layout()
    plt.show()


# ==============================================================================
# SECTION G — ARCHIVE CLUSTERING
# ==============================================================================
class ArchiveClustering:
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.cluster_labels = {}

    def fit(self, entries):
        if len(entries) < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} entries to form {self.n_clusters} clusters."
            )
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


# ==============================================================================
# SECTION H — LSTM TEMPORAL FORECASTER & BASELINE EVALUATION
# ==============================================================================
class MoodLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _generate_synthetic_vad_trajectories(n_sequences=800, seq_len=6, random_state=42):
    """
    Synthetic VAD time series with mood inertia (autoregressive momentum) plus
    noise, so there is an actual temporal pattern for the LSTM to learn —
    unlike raw white noise, which would make training meaningless.
    """
    rng = np.random.default_rng(random_state)
    sequences = []
    for _ in range(n_sequences):
        vad = rng.uniform(-0.5, 0.5, size=3)
        momentum = rng.normal(0, 0.05, size=3)
        seq = [vad.copy()]
        for _ in range(seq_len - 1):
            momentum = 0.7 * momentum + rng.normal(0, 0.08, size=3)
            vad = np.clip(vad + momentum, -1, 1)
            seq.append(vad.copy())
        sequences.append(seq)
    return np.array(sequences, dtype=np.float32)  # (n_sequences, seq_len, 3)


class MoodTemporalForecaster:
    """
    Wraps MoodLSTM with an actual train/val loop, mirroring the persistence
    pattern used by CVAEArtModel (get_state/load_state) so the trained weights
    can be cached to disk instead of retrained on every pipeline call.
    """

    def __init__(self, hidden_dim=16, epochs=150, lr=1e-3, val_fraction=0.15,
                 random_state=42):
        torch.manual_seed(random_state)
        self.net = MoodLSTM(hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.train_losses = []
        self.val_losses = []
        self._train(epochs, lr, val_fraction, random_state)

    def _train(self, epochs, lr, val_fraction, random_state):
        data = _generate_synthetic_vad_trajectories(random_state=random_state)
        n_val = int(len(data) * val_fraction)
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(data))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_train = torch.tensor(data[train_idx][:, :-1, :])
        y_train = torch.tensor(data[train_idx][:, -1, :])
        X_val = torch.tensor(data[val_idx][:, :-1, :])
        y_val = torch.tensor(data[val_idx][:, -1, :])

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.net.train()
            opt.zero_grad()
            pred = self.net(X_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            opt.step()
            self.train_losses.append(loss.item())

            self.net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net(X_val), y_val).item()
            self.val_losses.append(val_loss)

    def predict(self, history_vad_sequence):
        """Predict next VAD state from a real history sequence (min length 3)."""
        self.net.eval()
        x = torch.tensor([history_vad_sequence], dtype=torch.float32)
        with torch.no_grad():
            return self.net(x).squeeze(0).numpy()

    def get_state(self):
        return {"net_state_dict": self.net.state_dict(), "hidden_dim": self.hidden_dim}

    def load_state(self, state):
        self.net = MoodLSTM(hidden_dim=state["hidden_dim"])
        self.net.load_state_dict(state["net_state_dict"])


def evaluate_forecaster_vs_baseline(forecaster, history_vad_sequence, verbose=True):
    """
    Compares the TRAINED forecaster's prediction against a 3-day SMA baseline
    on ONE sequence. This is illustrative, not a benchmark -- a single point
    estimate proves nothing about typical performance. See
    evaluate_forecaster_multi_sequence() for the actual benchmark.
    Falls back to the last observed state (not a random LSTM output) when
    history is too short — a random guess is not a meaningful baseline.
    """
    if len(history_vad_sequence) < 4:
        return np.array(history_vad_sequence[-1])

    data = np.array(history_vad_sequence, dtype=np.float32)
    inputs, target = data[:-1], data[-1]
    pred = forecaster.predict(inputs)
    sma_pred = data[-4:-1].mean(axis=0)

    if verbose:
        print(f"Forecaster MSE: {np.mean((pred - target) ** 2):.4f} | "
              f"SMA MSE: {np.mean((sma_pred - target) ** 2):.4f}  (single sequence -- illustrative only)")
    return pred


def evaluate_forecaster_multi_sequence(forecaster, n_sequences=40, seq_len=6, random_state=999, verbose=True):
    """
    The REAL benchmark. Evaluates the forecaster against the SMA baseline
    across many HELD-OUT sequences, generated with random_state=999 --
    deliberately different from the training random_state (42, see
    MoodTemporalForecaster._train), so this is genuinely unseen data rather
    than a re-draw of the training distribution with the same samples.

    Reports mean +/- std MSE for both methods and the win rate (fraction of
    sequences where the forecaster actually beats SMA) -- a single MSE number
    from one sequence is a point estimate, not evidence of a working model.
    """
    sequences = _generate_synthetic_vad_trajectories(n_sequences=n_sequences, seq_len=seq_len, random_state=random_state)
    forecaster_mses, sma_mses = [], []
    wins = 0

    for seq in sequences:
        data = np.array(seq, dtype=np.float32)
        inputs, target = data[:-1], data[-1]
        pred = forecaster.predict(inputs)
        sma_pred = data[-4:-1].mean(axis=0)
        f_mse = float(np.mean((pred - target) ** 2))
        s_mse = float(np.mean((sma_pred - target) ** 2))
        forecaster_mses.append(f_mse)
        sma_mses.append(s_mse)
        if f_mse < s_mse:
            wins += 1

    f_mean, f_std = float(np.mean(forecaster_mses)), float(np.std(forecaster_mses))
    s_mean, s_std = float(np.mean(sma_mses)), float(np.std(sma_mses))
    win_rate = wins / n_sequences

    if verbose:
        print(f"Held-out sequences: n={n_sequences} (random_state={random_state}, distinct from training seed 42)")
        print(f"Forecaster MSE: {f_mean:.4f} +/- {f_std:.4f}")
        print(f"SMA MSE:        {s_mean:.4f} +/- {s_std:.4f}")
        print(f"Forecaster beat SMA on {wins}/{n_sequences} sequences ({win_rate:.0%})")
        if win_rate < 0.5:
            print("WARNING: the forecaster beats the baseline on FEWER than half the held-out")
            print("sequences. Report this plainly. Do not present this as a successful")
            print("forecasting model -- reframe it as experimental/prototype instead.")
        elif win_rate < 0.65:
            print("NOTE: the forecaster beats the baseline on a majority but not a strong")
            print("majority of sequences. Report the win rate alongside the mean MSE --")
            print("don't lead with the mean alone, it understates the variance.")

    return {
        "forecaster_mses": forecaster_mses, "sma_mses": sma_mses,
        "forecaster_mean": f_mean, "forecaster_std": f_std,
        "sma_mean": s_mean, "sma_std": s_std,
        "win_rate": win_rate, "n": n_sequences,
    }


# ==============================================================================
# SECTION I — ART IMAGE RENDERING
# ==============================================================================
STYLE_NAMES = ["cloud", "silk", "prism", "aurora", "ink", "nebula"]


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


def render_abstract_art(palette_hex, seed=None, size=(640, 400), style="cloud", n_blobs=None, blur_radius=None):
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
    import io
    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=False,
    )
    buf.seek(0)
    return buf.getvalue()


# ==============================================================================
# SECTION J — PER-USER PERSISTENCE
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


def _load_pickle(path, cls, *init_args, **init_kwargs):
    if os.path.exists(path):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj.load_state(state)
        return obj
    obj = cls(*init_args, **init_kwargs)
    _save_pickle(path, obj)
    return obj


def _save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj.get_state(), f)


def load_diary_classifier(user_id):
    return _load_pickle(_user_path(user_id, "diary_classifier.pkl"), DiaryMoodClassifier)


def save_diary_classifier(user_id, clf):
    _save_pickle(_user_path(user_id, "diary_classifier.pkl"), clf)


def load_art_model(user_id):
    return _load_pickle(_user_path(user_id, "art_model.pkl"), ArtColorModel)


def save_art_model(user_id, model):
    _save_pickle(_user_path(user_id, "art_model.pkl"), model)


def load_cvae_model(user_id):
    return _load_pickle(_user_path(user_id, "cvae_model.pkl"), CVAEArtModel, epochs=80)


def save_cvae_model(user_id, model):
    _save_pickle(_user_path(user_id, "cvae_model.pkl"), model)


def load_image_art_model(user_id):
    return _load_pickle(_user_path(user_id, "image_art_model.pkl"),
                         GenerativeArtImageModel, n_synthetic_samples=360, epochs=55)


def save_image_art_model(user_id, model):
    _save_pickle(_user_path(user_id, "image_art_model.pkl"), model)


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
            res = supabase.table("diary_entries").select("*").eq("user_id", user_id).execute()
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
    history = load_entry_history(user_id)
    history.append(entry)

    supabase = _get_supabase_client()
    if supabase:
        try:
            db_row = {
                "user_id": user_id,
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
        except Exception as e:
            print(f"WARNING: Supabase write failed: {e}")

    with open(_user_path(user_id, "entries.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


# ==============================================================================
# SECTION O — GENERATIVE PIXEL-LEVEL CVAE IMAGE MODEL
# ==============================================================================
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
    def __init__(self, n_synthetic_samples=480, epochs=70, lr=1e-3, random_state=42):
        torch.manual_seed(random_state)
        self.net = _ImageCVAENet()
        self.mapper = EmotionalMapper()
        self.n_synthetic_samples = n_synthetic_samples
        self.n_real_examples = 0
        self._X, self._C = self._build_synthetic_dataset(n_synthetic_samples, random_state)
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
            vad = [mapped["valence"], mapped["arousal"], mapped["dominance"], mapped["clarity"], mapped["turbulence"]]
            cond = vad + _style_onehot(style)

            theme_scores = _theme_scores_from_posterior(posterior, rng)  # was _random_theme_scores(rng) -- see docstring
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
        arr = (t.clamp(0, 1).detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
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
# SECTION K — UNIFIED DEPENDENT PIPELINE & ABLATION STUDY
# ==============================================================================
_CVAE_SINGLETON = None
_FORECASTER_SINGLETON = None


def _get_cvae_model():
    global _CVAE_SINGLETON
    if _CVAE_SINGLETON is None:
        _CVAE_SINGLETON = CVAEArtModel(epochs=40)
    return _CVAE_SINGLETON


def _get_forecaster():
    global _FORECASTER_SINGLETON
    if _FORECASTER_SINGLETON is None:
        _FORECASTER_SINGLETON = MoodTemporalForecaster(epochs=150)
    return _FORECASTER_SINGLETON


def run_unified_pipeline(text, history_vad, user_preference_preset=None, ablations=None):
    """
    Executes end-to-end dependency chain:
    Text -> Classifier (Topic) -> Mapper (VAD) -> Forecaster (History) -> CVAE (Visual)

    Forecaster: MoodLSTM trained on synthetic mood-inertia trajectories
    (see MoodTemporalForecaster), not on real user history — history is
    used only at inference time as the input sequence.

    CVAE: a stochastic, VAD-conditioned wrapper around the rule-based
    deterministic_palette() function, optionally personalized via a
    lightweight fine_tune() on one user-chosen preset. It is not learning
    palette structure from real user preference data at scale.
    """
    ablations = ablations or []

    # Step 1: Classification & Topic Extraction
    clf = DiaryMoodClassifier()
    res = clf.analyze(text)
    topic = res["top_category"]

    # Step 2: VAD Mapping
    current_vad = [res["valence"], res["arousal"], res["dominance"]]

    # Step 3: Trained Forecaster History Integration
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

    # Step 4: Palette Optimization & CVAE Generation
    if "no_cvae" in ablations:
        final_palette = deterministic_palette_hex(
            target_vad["valence"], (target_vad["arousal"] + 1) / 2,
            target_vad["clarity"], target_vad["turbulence"], {}
        )
        render_engine = "Rule-based Fallback (No CVAE)"
    else:
        cvae_model = _get_cvae_model()   # cached, not retrained per call
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

def run_ablation_study(sample_text, mock_history):
    """Evaluates output variance across 4 ablation states."""
    print("\n" + "=" * 60)
    print("PIPELINE ABLATION TESTING")
    print("=" * 60)

    configs = {
        "1. Full Pipeline": [],
        "2. W/o CVAE (Static Rules)": ["no_cvae"],
        "3. W/o Palette Optimiser": ["no_optimiser"],
        "4. W/o LSTM Sequence": ["no_lstm"]
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 2.2))

    for ax, (name, flags) in zip(axes, configs.items()):
        out = run_unified_pipeline(sample_text, mock_history, user_preference_preset="warmth", ablations=flags)
        palette = out["palette"]

        for idx, hex_color in enumerate(palette):
            ax.add_patch(plt.Rectangle((idx, 0), 1, 1, color=hex_color))
        ax.set_xlim(0, 4); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\n({out['engine']})", fontsize=7)

    plt.suptitle("Ablation Comparison — Output Visual Palette Variations", fontsize=10)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# SECTION N — STREAMLIT UI
# ==============================================================================
def _inject_theme_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Hanken+Grotesk:wght@400;500;600&display=swap');

    :root {
        --accent: #d9b673;
        --ink: #f2ede4;
        --ink-soft: #b8b0a4;
        --bg-deep: #0a0a0f;
        --bg-panel: #14131a;
        --serif: 'Cormorant Garamond', Georgia, serif;
        --sans: 'Hanken Grotesk', system-ui, sans-serif;
    }

    .stApp {
        background: radial-gradient(120% 100% at 20% 0%, #1a1420 0%, var(--bg-deep) 55%);
        color: var(--ink);
        font-family: var(--sans);
    }
    section[data-testid="stSidebar"] {
        background: var(--bg-panel);
        border-right: 1px solid rgba(217,182,115,0.15);
    }
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: var(--serif) !important;
        font-weight: 500 !important;
        color: var(--ink) !important;
        letter-spacing: 0.01em;
    }
    .mood-kicker {
        font-family: var(--sans);
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.72rem;
        color: var(--accent);
        margin-bottom: 4px;
    }
    .stTextArea textarea, .stTextInput input {
        background: var(--bg-panel) !important;
        color: var(--ink) !important;
        border: 1px solid rgba(217,182,115,0.25) !important;
        font-family: var(--serif) !important;
        font-size: 1.05rem !important;
    }
    .stButton > button {
        font-family: var(--sans);
        border-radius: 999px !important;
        border: 1px solid rgba(217,182,115,0.4) !important;
        background: transparent !important;
        color: var(--ink) !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        color: #16130d !important;
        border: none !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] { font-family: var(--serif); color: var(--accent); }
    hr, .stDivider { border-color: rgba(217,182,115,0.15) !important; }
    </style>
    """, unsafe_allow_html=True)


def _meditation_rings_html(reading_text=""):
    return f"""
    <style>
    @keyframes moodBreathe {{
        0%   {{ transform: scale(0.55); opacity: 0.35; }}
        30%  {{ transform: scale(1.0);  opacity: 0.9; }}
        55%  {{ transform: scale(1.0);  opacity: 0.9; }}
        100% {{ transform: scale(0.55); opacity: 0.35; }}
    }}
    .med-wrap {{
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; padding: 40px 0;
    }}
    .med-ring {{
        position: relative; width: 200px; height: 200px;
        border-radius: 50%; border: 1px solid rgba(217,182,115,0.5);
        animation: moodBreathe 13s ease-in-out infinite;
    }}
    .med-ring-2 {{
        position: absolute; top: 18px; left: 18px;
        width: 164px; height: 164px; border-radius: 50%;
        border: 1px solid rgba(217,182,115,0.3);
        animation: moodBreathe 13s ease-in-out infinite 0.4s;
    }}
    .med-word {{
        margin-top: 28px; font-family: 'Cormorant Garamond', Georgia, serif;
        font-style: italic; font-size: 1.3rem; color: #f2ede4; text-align: center;
    }}
    </style>
    <div class="med-wrap">
        <div class="med-ring"><div class="med-ring-2"></div></div>
        <div class="med-word">{reading_text}</div>
    </div>
    """


def _evolution_timeline_svg(history, width=760, height=300):
    if len(history) < 2:
        return None
    pad_x, mid_y = 60, height / 2
    n = len(history)

    def catmull_rom(points):
        if len(points) < 2:
            return ""
        d = f"M {points[0][0]} {points[0][1]}"
        for i in range(len(points) - 1):
            p0 = points[i - 1] if i - 1 >= 0 else points[i]
            p1, p2 = points[i], points[i + 1]
            p3 = points[i + 2] if i + 2 < len(points) else p2
            c1x, c1y = p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6
            c2x, c2y = p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6
            d += f" C {c1x} {c1y}, {c2x} {c2y}, {p2[0]} {p2[1]}"
        return d

    pts = []
    for i, e in enumerate(history):
        x = pad_x + (i / (n - 1)) * (width - pad_x * 2) if n > 1 else width / 2
        y = mid_y - e["valence"] * 95 - (e["arousal"]) * 15
        pts.append((x, y))
    path = catmull_rom(pts)

    stops = "".join(
        f'<stop offset="{i/(n-1) if n>1 else 0}" stop-color="{e["palette"][2]}"/>'
        for i, e in enumerate(history)
    )
    dots = "".join(f'<circle cx="{p[0]}" cy="{p[1]}" r="5" fill="#d9b673"/>' for p in pts)

    return f"""
    <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
        <defs>
            <linearGradient id="ribbon" x1="0" y1="0" x2="1" y2="0">{stops}</linearGradient>
            <filter id="glow" x="-20%" y="-60%" width="140%" height="220%">
                <feGaussianBlur stdDeviation="7" result="b"/>
                <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
        </defs>
        <path d="{path}" fill="none" stroke="url(#ribbon)" stroke-width="6" opacity="0.35" filter="url(#glow)"/>
        <path d="{path}" fill="none" stroke="url(#ribbon)" stroke-width="2.5"/>
        {dots}
    </svg>
    """


def run_streamlit_app():
    st.set_page_config(page_title="Mood Evolution", page_icon="◐", layout="centered")
    _inject_theme_css()

    st.sidebar.title("Mood Evolution")
    user_id = st.sidebar.text_input("Your name / id", value=st.session_state.get("user_id", "me"))
    st.session_state["user_id"] = user_id

    if st.sidebar.button("Reset this user's data"):
        import shutil
        user_dir_path = user_dir(user_id)
        if os.path.exists(user_dir_path):
            shutil.rmtree(user_dir_path)
        st.sidebar.success("Cleared. Reload the page.")

    clf = load_diary_classifier(user_id)
    cvae_model = load_cvae_model(user_id)
    image_art_model = load_image_art_model(user_id)

    tab_today, tab_archive, tab_evolution = st.tabs(["Today", "Archive", "Evolution"])

    with tab_today:
        st.markdown(f"### {dt.date.today().strftime('%B %-d')}")

        if "draft_text" not in st.session_state:
            st.session_state.draft_text = ""
        text = st.text_area("How are you, today?", value=st.session_state.draft_text, height=120,
                             placeholder="Describe your state of mind…")
        st.session_state.draft_text = text
        word_count = len(text.split())

        col_reflect, col_regen = st.columns([1, 1])
        reflect_clicked = col_reflect.button("Reflect", disabled=word_count < 3, type="primary")
        regenerate_clicked = col_regen.button("Regenerate art (sample again)", disabled="last_result" not in st.session_state)

        if reflect_clicked:
            history = load_entry_history(user_id)
            history_vads = [[e["valence"], e["arousal"], e["dominance"]] for e in history if "valence" in e]

            pipe_res = run_unified_pipeline(text, history_vads)
            result = pipe_res["analysis"]

            import re
            tokens = re.findall(r"[a-zA-Z']+", text.lower())
            words = [{"w": w, "score": clf.word_score(w)} for w in tokens]
            words = [w for w in words if abs(w["score"]) > 0.03]
            seen, deduped = set(), []
            for w in words:
                if w["w"] not in seen:
                    seen.add(w["w"]); deduped.append(w)

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

            st.markdown(f"*{result['reading']}*  —  **{result['top_category']}**  |  `{r['engine']}`")

            override_choice = st.selectbox(
                "Palette", ["Auto (detected)"] + [PRESET_DEFS[k]["label"] for k in PRESET_DEFS] + ["Custom colour"],
                key="palette_choice",
            )
            active_palette = r["cvae_palette"]
            override_for_training = None
            if override_choice != "Auto (detected)":
                if override_choice == "Custom colour":
                    c0 = st.color_picker("Base", "#0b0f0f")
                    c1 = st.color_picker("Mid", "#492133")
                    c2 = st.color_picker("Accent", "#b45977")
                    c3 = st.color_picker("Highlight", "#d2c1c1")
                    active_palette = [c0, c1, c2, c3]
                    override_for_training = ("custom", active_palette)
                else:
                    key = [k for k, v in PRESET_DEFS.items() if v["label"] == override_choice][0]
                    active_palette = preset_palette(key)
                    override_for_training = ("preset", key)

            swatch_cols = st.columns(4)
            for i, hexcolor in enumerate(active_palette):
                swatch_cols[i].markdown(
                    f'<div style="background:{hexcolor};height:36px;border-radius:4px;"></div>',
                    unsafe_allow_html=True,
                )

            if r["words"]:
                chip_html = " ".join(
                    f'<span style="padding:3px 9px;border-radius:12px;margin-right:4px;'
                    f'background:{"#2c4a33" if w["score"]>=0 else "#4a2c2c"};'
                    f'color:{"#a8e0b4" if w["score"]>=0 else "#e0a8a8"};font-size:12px;">{w["w"]}</span>'
                    for w in r["words"]
                )
                st.markdown(chip_html, unsafe_allow_html=True)

            art_mode = st.radio(
                "Art rendering",
                ["Rule-based (fast, stable)", "AI-generated (Conditional VAE over pixels)"],
                horizontal=True,
            )
            style_choice = st.selectbox("Visual style", STYLE_NAMES, index=0)

            if art_mode.startswith("AI-generated"):
                with st.spinner("Generating from trained image model..."):
                    art = image_art_model.sample(r["target_vad"], style=style_choice, n_samples=1)[0]
            else:
                art = render_abstract_art(active_palette, seed=r["art_seed"] or f"{text}-{id(r)}", style=style_choice)

            meditate_on = st.toggle("🧘 Meditate — slow looping motion")
            if meditate_on:
                with st.spinner("Rendering animation..."):
                    if art_mode.startswith("AI-generated"):
                        frames = image_art_model.sample_animation(r["target_vad"], style=style_choice, n_frames=24, upscale_to=360)
                    else:
                        frames = render_meditation_gif(active_palette, seed=r["art_seed"] or text, style=style_choice, n_frames=24, size=(640, 400))
                    gif_bytes = frames_to_gif_bytes(frames, duration_ms=90)
                st.image(gif_bytes, use_container_width=True)
                st.markdown(_meditation_rings_html(result["reading"]), unsafe_allow_html=True)
            else:
                st.image(art, use_container_width=True)

            col_metrics = st.columns(3)
            col_metrics[0].metric("Valence", f"{r['target_vad']['valence']:+.2f}")
            col_metrics[1].metric("Arousal", f"{r['target_vad']['arousal']:+.2f}")
            col_metrics[2].metric("Clarity", f"{r['target_vad']['clarity']:.2f}")

            if st.button("Seal this day", type="primary"):
                corrected_category = None
                if override_for_training:
                    kind_check, value_check = override_for_training
                    if kind_check == "preset":
                        corrected_category = PRESET_TO_CATEGORY.get(value_check)

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
                    kind, value = override_for_training
                    if kind == "preset":
                        category = PRESET_TO_CATEGORY.get(value)
                        if category:
                            clf.learn(text, category)
                            save_diary_classifier(user_id, clf)
                    cvae_model.fine_tune(r["target_vad"], active_palette, steps=30)
                    ran, loss_before, loss_after = cvae_model.maybe_periodic_retrain()
                    if ran:
                        st.toast(f"Periodic retrain: {cvae_model.n_real_corrections} real corrections folded in "
                                 f"together. Loss {loss_before:.4f} -> {loss_after:.4f}.")
                    save_cvae_model(user_id, cvae_model)

                if art_mode.startswith("AI-generated"):
                    image_art_model.fine_tune(r["target_vad"], art, style=style_choice, steps=25)
                    save_image_art_model(user_id, image_art_model)

                st.session_state.draft_text = ""
                del st.session_state["last_result"]
                st.success("Sealed! See Archive and Evolution tabs.")
                st.rerun()

    with tab_archive:
        history = load_entry_history(user_id)
        if not history:
            st.info("No sealed days yet.")
        else:
            cluster_labels = None
            if len(history) >= 3:
                archive_model = ArchiveClustering(n_clusters=min(3, len(history)))
                assignments = archive_model.fit(history)
                cluster_labels = [archive_model.cluster_labels[a] for a in assignments]

            for i, entry in enumerate(reversed(history)):
                idx = len(history) - 1 - i
                cols = st.columns([1, 3])
                thumb = render_thumbnail(entry["palette"], seed=entry["text"], style=entry.get("style", "cloud"))
                cols[0].image(thumb)
                with cols[1]:
                    label = f"**{entry['date']}** — {entry['reading']}"
                    if cluster_labels:
                        label += f"  ·  cluster: _{cluster_labels[idx]}_"
                    st.markdown(label)
                    st.caption(entry["text"][:140] + ("…" if len(entry["text"]) > 140 else ""))

                    with st.expander("Correct this entry"):
                        correction_choice = st.selectbox(
                            "What should this day have been?",
                            ["No change"] + [PRESET_DEFS[k]["label"] for k in PRESET_DEFS] + ["Custom colour"],
                            key=f"archive_correction_{idx}",
                        )
                        corrected_palette = None
                        if correction_choice == "Custom colour":
                            cc0 = st.color_picker("Base", entry["palette"][0], key=f"archive_c0_{idx}")
                            cc1 = st.color_picker("Mid", entry["palette"][1], key=f"archive_c1_{idx}")
                            cc2 = st.color_picker("Accent", entry["palette"][2], key=f"archive_c2_{idx}")
                            cc3 = st.color_picker("Highlight", entry["palette"][3], key=f"archive_c3_{idx}")
                            corrected_palette = [cc0, cc1, cc2, cc3]

                        if st.button("Apply correction", key=f"archive_apply_{idx}"):
                            if correction_choice == "No change":
                                st.warning("Pick a mood preset or a custom colour first.")
                            else:
                                entry_vad = {
                                    "valence": entry["valence"], "arousal": entry["arousal"],
                                    "dominance": entry["dominance"], "clarity": entry["clarity"],
                                    "turbulence": entry["turbulence"],
                                }
                                if correction_choice == "Custom colour":
                                    # Same as a Today-tab "custom" override: Tier 2 (CVAE) only.
                                    cvae_model.fine_tune(entry_vad, corrected_palette, steps=30)
                                    ran, loss_before, loss_after = cvae_model.maybe_periodic_retrain()
                                    save_cvae_model(user_id, cvae_model)
                                    msg = "Tier 2 updated: CVAE fine-tuned on this correction."
                                    if ran:
                                        msg += (f" Periodic retrain also ran ({cvae_model.n_real_corrections} real "
                                                f"corrections so far). Loss {loss_before:.4f} -> {loss_after:.4f}.")
                                    st.success(msg)
                                else:
                                    # Same as a Today-tab "preset" override: Tier 1 + Tier 2.
                                    preset_key = [k for k, v in PRESET_DEFS.items() if v["label"] == correction_choice][0]
                                    category = PRESET_TO_CATEGORY.get(preset_key)
                                    preset_hex = preset_palette(preset_key)
                                    if category:
                                        clf.learn(entry["text"], category)
                                        save_diary_classifier(user_id, clf)
                                    cvae_model.fine_tune(entry_vad, preset_hex, steps=30)
                                    ran, loss_before, loss_after = cvae_model.maybe_periodic_retrain()
                                    save_cvae_model(user_id, cvae_model)
                                    msg = "Tier 1 + Tier 2 updated: classifier and CVAE both retrained on this correction."
                                    if ran:
                                        msg += (f" Periodic retrain also ran ({cvae_model.n_real_corrections} real "
                                                f"corrections so far). Loss {loss_before:.4f} -> {loss_after:.4f}.")
                                    st.success(msg)
                st.divider()

    with tab_evolution:
        history = load_entry_history(user_id)
        if len(history) < 2:
            st.info("Need at least 2 sealed entries to show the evolution timeline and forecast.")
        else:
            st.markdown("### How your inner weather has moved.")
            svg = _evolution_timeline_svg(history)
            if svg:
                st.markdown(svg, unsafe_allow_html=True)

            days = list(range(len(history)))
            valences = [e["valence"] for e in history if "valence" in e]
            history_vads = [[e["valence"], e["arousal"], e["dominance"]] for e in history if "valence" in e]
            forecaster = _get_forecaster()
            next_vad = evaluate_forecaster_vs_baseline(forecaster, history_vads, verbose=False)

            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(days, valences, "o-", label="Logged Valence", color="#d9b673")
            ax.plot([len(days)], [next_vad[0]], "o--", label="LSTM Predicted Next Day", color="#b9a3e0")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.set_xlabel("Day Index")
            ax.set_ylabel("Valence Score")
            ax.legend()
            ax.set_title("VAD Temporal Sequence & LSTM Forecast")
            st.pyplot(fig)


# ==============================================================================
# SECTION P — EXECUTION ENTRY POINT
# ==============================================================================
# ==============================================================================
# SECTION Q -- EVALUATION SUITE (restored): expected-vs-produced, consistency,
# discriminability. Ported from an earlier iteration of this notebook. Real
# numbers, including unflattering ones, are reported as-is -- nothing here is
# tuned to look good.
# ==============================================================================
LABELED_TEST_SET = [
    {"text": "Back-to-back calls all afternoon, I can't switch my brain off.", "expected_category": "stress"},
    {"text": "I slammed the laptop shut, so sick of being interrupted.", "expected_category": "anger"},
    {"text": "Slow Sunday, made tea, nowhere I need to be.", "expected_category": "calm"},
    {"text": "Everything finally makes sense, I can see the whole plan now.", "expected_category": "clarity"},
    {"text": "Got through my whole list today, sharp and on top of it.", "expected_category": "focus"},
    {"text": "Missing people I haven't seen in years, feeling low.", "expected_category": "sad"},
    {"text": "Not sure what I'm even feeling, everything's a bit blurry.", "expected_category": "fog"},
    {"text": "Every task feels like wading through mud today.", "expected_category": "heavy"},
    {"text": "Things are finally starting to look up a little.", "expected_category": "hope"},
    {"text": "Had the best evening, laughing with old friends.", "expected_category": "joy"},
    {"text": "Sat in the garden this morning, everything felt golden.", "expected_category": "light"},
]


def evaluate_classifier(clf, test_set=None):
    """Runs the labeled test set through the classifier, returns per-item
    results and overall accuracy. Prints nothing -- caller decides how to
    display it."""
    test_set = test_set or LABELED_TEST_SET
    results = []
    correct = 0
    for item in test_set:
        result = clf.analyze(item["text"])
        is_correct = result["top_category"] == item["expected_category"]
        correct += is_correct
        results.append({
            "text": item["text"],
            "expected": item["expected_category"],
            "predicted": result["top_category"],
            "correct": is_correct,
            "valence": result["valence"],
            "arousal": result["arousal"],
        })
    accuracy = correct / len(test_set)
    return {"results": results, "accuracy": accuracy, "n": len(test_set), "n_correct": correct}


CONSISTENCY_GROUPS = {
    "stress": [
        "So much on my plate right now, I feel completely overwhelmed.",
        "Deadlines everywhere, I can't keep up, it's too much.",
        "My chest is tight and my mind won't stop racing today.",
    ],
    "calm": [
        "Everything feels quiet and settled this morning.",
        "A peaceful, unhurried kind of day, nothing pressing.",
        "I feel grounded and at ease right now.",
    ],
    "joy": [
        "Genuinely such a happy day, I feel light and grateful.",
        "Had a wonderful time, feeling really good about everything.",
        "Everything felt easy and joyful today.",
    ],
}


def evaluate_consistency(clf, groups=None):
    """For each group, computes the VAD spread (std dev across the group's
    members) -- lower means more consistent. Also computes the average
    pairwise VAD distance BETWEEN groups, for contrast -- should be larger
    than the within-group spread if the model is behaving sensibly."""
    groups = groups or CONSISTENCY_GROUPS
    group_vads = {}
    for label, texts in groups.items():
        vads = []
        for t in texts:
            r = clf.analyze(t)
            vads.append([r["valence"], r["arousal"], r["dominance"]])
        group_vads[label] = np.array(vads)

    within_group_std = {label: float(np.mean(np.std(vads, axis=0))) for label, vads in group_vads.items()}

    centroids = {label: vads.mean(axis=0) for label, vads in group_vads.items()}
    between_group_dists = []
    for a, b in itertools.combinations(centroids.keys(), 2):
        d = float(np.linalg.norm(centroids[a] - centroids[b]))
        between_group_dists.append({"pair": (a, b), "distance": d})

    return {
        "within_group_std": within_group_std,
        "between_group_distances": between_group_dists,
        "mean_within_group_std": float(np.mean(list(within_group_std.values()))),
        "mean_between_group_distance": float(np.mean([d["distance"] for d in between_group_dists])),
    }


DISCRIMINABILITY_MOODS = {
    "stress": "Overwhelmed and anxious, too much happening at once.",
    "calm": "Quiet, settled, nothing urgent, just resting.",
    "joy": "Genuinely happy and grateful, a really good day.",
    "sad": "Low and a bit empty, missing people I care about.",
}


def evaluate_discriminability(clf, art_model, moods=None):
    """For each mood, produces a palette and measures pairwise RGB distance
    between palettes for different moods. Reports actual distances so a
    reader can judge for themselves, rather than taking a bare claim on
    faith."""
    moods = moods or DISCRIMINABILITY_MOODS
    palettes = {}
    for label, text in moods.items():
        r = clf.analyze(text)
        energy01 = (r["arousal"] + 1) / 2
        palette_hex = art_model.predict_palette(r["valence"], energy01, r["clarity"], r["turbulence"], r["posterior"])
        palettes[label] = _hex_list_to_rgb_array(palette_hex)

    distances = []
    for a, b in itertools.combinations(palettes.keys(), 2):
        d = float(np.mean(np.linalg.norm(palettes[a] - palettes[b], axis=1)))
        distances.append({"pair": (a, b), "rgb_distance_0_255": d})

    return {
        "palettes": {k: _rgb_array_to_hex_list(v) for k, v in palettes.items()},
        "pairwise_distances": distances,
        "mean_pairwise_distance": float(np.mean([d["rgb_distance_0_255"] for d in distances])),
    }


def _hex_list_to_rgb_array(hex_list):
    out = []
    for h in hex_list:
        h = h.lstrip("#")
        out.append([int(h[i:i + 2], 16) for i in (0, 2, 4)])
    return np.array(out, dtype=float)


def _rgb_array_to_hex_list(arr):
    return ["#{:02x}{:02x}{:02x}".format(*np.clip(row, 0, 255).astype(int)) for row in arr]


def print_full_report(clf, art_model):
    print("=" * 70)
    print("1. CLASSIFIER: expected vs produced (labeled test set, n={})".format(len(LABELED_TEST_SET)))
    print("=" * 70)
    report = evaluate_classifier(clf)
    for r in report["results"]:
        mark = "v" if r["correct"] else "x"
        print(f"  {mark} expected={r['expected']:8s} predicted={r['predicted']:8s} "
              f"valence={r['valence']:+.2f}  | {r['text'][:50]}")
    print(f"\nAccuracy: {report['n_correct']}/{report['n']} = {report['accuracy']:.0%}")
    print("(Report the number as-is, whatever it is.)")

    print("\n" + "=" * 70)
    print("2. CONSISTENCY: similar inputs -> similar outputs?")
    print("=" * 70)
    cons = evaluate_consistency(clf)
    for label, std in cons["within_group_std"].items():
        print(f"  {label:8s} within-group VAD std: {std:.3f}  (lower = more consistent)")
    print(f"  mean within-group std:    {cons['mean_within_group_std']:.3f}")
    print(f"  mean between-group dist:  {cons['mean_between_group_distance']:.3f}")
    if cons["mean_between_group_distance"] > cons["mean_within_group_std"]:
        print("  -> between-group distance exceeds within-group spread: groups are separable.")
    else:
        print("  -> WARNING: within-group spread exceeds between-group distance -- groups are")
        print("     not cleanly separable at this training set size. Report this, don't hide it.")

    print("\n" + "=" * 70)
    print("3. DISCRIMINABILITY: different moods -> different visuals?")
    print("=" * 70)
    disc = evaluate_discriminability(clf, art_model)
    for label, hexlist in disc["palettes"].items():
        print(f"  {label:8s} -> {hexlist}")
    print()
    for d in disc["pairwise_distances"]:
        print(f"  {d['pair'][0]:8s} vs {d['pair'][1]:8s}: RGB distance = {d['rgb_distance_0_255']:.1f} / ~441 max")
    print(f"  mean pairwise distance: {disc['mean_pairwise_distance']:.1f}")

    return {"classifier": report, "consistency": cons, "discriminability": disc}


# ==============================================================================
# SECTION Q2 -- ADDITIONAL RESTORED DEMOS: rule vs regression vs CVAE
# variation, image-style discriminability, archive clustering + forecaster
# on real (demo) history. Ported from an earlier notebook iteration.
# ==============================================================================
def demo_rule_vs_regression_vs_cvae(art_model, cvae_model):
    """Same mood, three generation methods. Shows the CVAE produces genuine
    sample-to-sample variation (stochastic z) for a FIXED input, unlike the
    deterministic rule and the regression approximation. Real captured hex
    codes -- this is a different question from VAD-conditioning sensitivity
    (see visualize_cvae_latent_interpolation): this tests within-mood
    diversity, not between-mood distinctiveness."""
    mood = dict(valence=-0.5, energy01=0.85, clarity=0.3, turbulence=0.7, theme_scores={"stress": 1.1})
    rule_output = deterministic_palette(**mood)
    rule_hex = ["#{:02x}{:02x}{:02x}".format(*(np.clip(s, 0, 1) * 255).astype(int)) for s in rule_output.reshape(4, 3)]
    regression_output = art_model.predict_palette(**mood)
    mood_vad = {"valence": -0.5, "arousal": 0.7, "dominance": -0.3, "clarity": 0.3, "turbulence": 0.7}
    cvae_samples = cvae_model.sample(mood_vad, n_samples=3)

    print("Curated rule (ground truth):     ", rule_hex)
    print("Regression approximation:        ", regression_output)
    print("CVAE samples (3, same mood):")
    for s in cvae_samples:
        print("  ", s)

    fig, axes = plt.subplots(1, 5, figsize=(11, 1.6))
    titles = ["Rule", "Regression"] + [f"CVAE {i+1}" for i in range(3)]
    for ax, palette, title in zip(axes, [rule_hex, regression_output] + cvae_samples, titles):
        for i, c in enumerate(palette):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c))
        ax.set_xlim(0, 4); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
    plt.suptitle("Same mood: the rule, its regression approximation, and 3 CVAE samples", fontsize=10)
    plt.tight_layout()
    plt.show()


def demo_image_style_discriminability(img_model):
    """Same mood, all 6 styles -- tests whether the pixel CVAE differentiates
    by style, not just by recolouring the same shape. Real pairwise mean
    pixel differences, plus a real fine_tune() before/after data_summary()
    comparison."""
    stress_mood = {"valence": -0.5, "arousal": 0.8, "dominance": -0.2, "clarity": 0.3, "turbulence": 0.7}

    fig, axes = plt.subplots(1, 6, figsize=(15, 2.6))
    imgs_by_style = {}
    for ax, style in zip(axes, STYLE_NAMES):
        img = img_model.sample(stress_mood, style=style, n_samples=1)[0]
        imgs_by_style[style] = np.array(img).astype(float)
        ax.imshow(img); ax.axis('off'); ax.set_title(style, fontsize=9)
    plt.suptitle("Same mood, all 6 AI-generated styles", fontsize=10)
    plt.tight_layout(); plt.show()

    print("Pairwise mean pixel difference across styles (0 = identical):")
    for i, a in enumerate(STYLE_NAMES):
        for b in STYLE_NAMES[i + 1:]:
            diff = np.abs(imgs_by_style[a] - imgs_by_style[b]).mean()
            print(f"  {a:8s} vs {b:8s}: {diff:.1f}")

    print("\n--- fine_tune() demonstration ---")
    kept_pil = img_model.sample(stress_mood, style="prism", n_samples=1)[0]
    print("Before:", img_model.data_summary())
    img_model.fine_tune(stress_mood, kept_pil, style="prism", steps=25)
    print("After: ", img_model.data_summary())


def demo_archive_clustering_and_forecast(clf, forecaster):
    """K-Means clustering on demo diary entries, plus the TRAINED
    MoodTemporalForecaster's prediction from that same short real-feeling
    history. Demonstrated on labeled demo text since this notebook has no
    real user history to draw on -- illustrative only, NOT the forecaster
    benchmark (see evaluate_forecaster_multi_sequence for that)."""
    demo_texts = [
        "Overwhelmed again, meetings all day, chest tight.",
        "Quiet Sunday, tea, feeling settled and calm.",
        "Furious about the deadline change, everyone frustrated.",
        "Clear-headed today, focused and got a lot done.",
        "Slowly feeling more hopeful, things are lifting.",
    ]
    demo_entries = [clf.analyze(t) for t in demo_texts]

    archive = ArchiveClustering(n_clusters=3)
    assignments = archive.fit(demo_entries)
    for text, a in zip(demo_texts, assignments):
        print(f"[cluster {a} -- {archive.cluster_labels[a]}] {text}")

    history_vad = [[e["valence"], e["arousal"], e["dominance"]] for e in demo_entries]
    pred = forecaster.predict(history_vad)
    print(f"\nMoodTemporalForecaster prediction from this 5-entry demo history: "
          f"valence={pred[0]:+.2f}, arousal={pred[1]:+.2f}, dominance={pred[2]:+.2f}")
    print("(Demo data, illustrative only -- NOT the forecaster benchmark.)")


def run_full_pipeline_evaluation():
    """Runs LSTM baseline comparison, Latent Interpolation, and Systematic Ablation Suite."""
    print("=" * 70)
    print("RUNNING COMPREHENSIVE ML PIPELINE EVALUATION")
    print("=" * 70)

    print("\n--- 0. EVALUATION SUITE: expected-vs-produced, consistency, discriminability ---")
    _eval_clf = DiaryMoodClassifier()
    _eval_art_model = ArtColorModel()
    print_full_report(_eval_clf, _eval_art_model)

    print("\n--- 0b. Same mood: rule vs regression vs CVAE variation ---")
    _eval_cvae = CVAEArtModel(epochs=200)
    demo_rule_vs_regression_vs_cvae(_eval_art_model, _eval_cvae)

    print("\n--- 0c. Image-style discriminability (pixel CVAE, all 6 styles) ---")
    _eval_img_model = GenerativeArtImageModel(n_synthetic_samples=480, epochs=70)
    demo_image_style_discriminability(_eval_img_model)

    print("\n--- 0d. Archive clustering + forecaster on demo history ---")
    demo_archive_clustering_and_forecast(_eval_clf, _get_forecaster())

    # 1. Mock Multi-Day VAD History
    mock_vad_history = [
        [-0.55, 0.75, -0.30],  # Day 1: Stress
        [-0.65, 0.15, -0.50],  # Day 2: Sad
        [-0.42, 0.15, -0.60],  # Day 3: Fog
        [0.50, -0.45, 0.40],   # Day 4: Calm
    ]
    test_text = "Back-to-back deadlines have my chest tight, but I am trying to stay grounded."

    # 2. LSTM Sequence vs. Moving Average Baseline
    print("\n--- 1a. Forecaster vs SMA -- single illustrative sequence ---")
    forecaster = _get_forecaster()
    print(f"Forecaster train loss (final): {forecaster.train_losses[-1]:.4f} | "
          f"val loss (final): {forecaster.val_losses[-1]:.4f}")
    evaluate_forecaster_vs_baseline(forecaster, mock_vad_history, verbose=True)

    print("\n--- 1b. Forecaster vs SMA -- REAL BENCHMARK, 40 held-out sequences ---")
    evaluate_forecaster_multi_sequence(forecaster, n_sequences=40, verbose=True)

    # 3. Unified Dependency Pipeline Test
    print("\n--- 2. UNIFIED PIPELINE (Text -> Topic -> VAD -> Forecaster -> CVAE) ---")
    pipeline_out = run_unified_pipeline(test_text, mock_vad_history, user_preference_preset="warmth")
    print(f"Detected Topic     : {pipeline_out['topic']}")
    print(f"Target VAD State   : {pipeline_out['target_vad']}")
    print(f"Synthesized Palette: {pipeline_out['palette']}")
    print(f"Synthesis Engine   : {pipeline_out['engine']}")

    # 4. CVAE Latent-Space Spherical Interpolation (SLERP)
    print("\n--- 3. CVAE LATENT SPACE INTERPOLATION ---")
    cvae = CVAEArtModel(epochs=30)
    start_vad = {"valence": -0.6, "arousal": 0.8, "dominance": -0.4, "clarity": 0.2, "turbulence": 0.8}
    end_vad = {"valence": 0.7, "arousal": -0.3, "dominance": 0.5, "clarity": 0.8, "turbulence": 0.1}
    visualize_cvae_latent_interpolation(cvae, start_vad, end_vad, steps=5)

    # 5. Pipeline Ablation Testing
    print("\n--- 4. SYSTEMATIC ABLATION STUDY ---")
    run_ablation_study(test_text, mock_vad_history)


if __name__ == "__main__":
    if st is not None and getattr(st, "runtime", None) and st.runtime.exists():
        run_streamlit_app()
    else:
        run_full_pipeline_evaluation()
