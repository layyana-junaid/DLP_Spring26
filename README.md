# 🛡️ Phishing Detection System
### Email and URL-Based Detection Framework

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Deployment](https://img.shields.io/badge/Deployment-CPU--Optimized-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Why Two Separate Pipelines?](#-why-two-separate-pipelines)
- [Email Phishing Detection](#-email-phishing-detection)
- [URL Phishing Detection](#-url-phishing-detection)
- [Why Not BERT / Transformers?](#-why-not-bert--transformers)
- [System Architecture](#-system-architecture)
- [Design Decisions](#-design-decisions)
- [Deployment](#-deployment)
- [Future Enhancements](#-future-enhancements)

---

## 🔍 Overview

This project implements a **production-ready phishing detection system** capable of identifying phishing attempts across two distinct surfaces:

- 📧 **Phishing Emails**
- 🔗 **Phishing URLs**

Rather than forcing both into a single model, two **specialized pipelines** were developed — each tailored to the structural and linguistic differences of its data modality.

### Design Philosophy

| Principle | Implementation |
|-----------|---------------|
| Production deployability | CPU-optimized inference, `joblib` export |
| Low latency | Logistic Regression over deep models |
| Scalability | Modular, independently deployable pipelines |
| Interpretability | Transparent feature engineering |
| Cost efficiency | No GPU dependency |

---

## 🤔 Why Two Separate Pipelines?

Phishing emails and URLs represent **fundamentally different data modalities** and require specialized treatment:

| Dimension | Emails | URLs |
|-----------|--------|------|
| Format | Long-form natural language | Short structured strings |
| Key signals | Linguistics, tone, phrasing | Subdomains, symbols, substrings |
| Feature approach | Word-level TF-IDF | Character n-grams + structural features |

A single unified model would fail to capture these distinct characteristics. Separation improves **feature representation**, **model specialization**, and **deployment flexibility**.

---

## 📧 Email Phishing Detection

### Dataset
- Email text with binary labels: `0 = legitimate`, `1 = phishing`
- Dataset exhibited class imbalance (legitimate emails outnumbering phishing samples)

### Preprocessing Pipeline

```
Raw Email → Lowercase → Noise Removal → Normalization → Clean Text
```

Steps include:
- Lowercasing all text
- Removing non-informative characters
- Basic normalization

### Feature Engineering — Word-Level TF-IDF

TF-IDF converts text into a sparse numerical representation, capturing token importance across the corpus. It effectively highlights suspicious patterns such as:

> `urgent` · `verify` · `password` · `account` · `update` · `login`

**Why TF-IDF?**
- Strong performance on text classification tasks
- Interpretable and transparent
- Computationally efficient
- Production-friendly; integrates seamlessly with linear classifiers

### Model — Logistic Regression

```python
LogisticRegression(class_weight="balanced")
```

**Advantages:**
- Handles high-dimensional sparse TF-IDF vectors effectively
- Strong baseline performance for phishing detection
- Interpretable coefficients
- Fast inference suitable for real-time systems

### Threshold Optimization

Rather than using a fixed `0.5` decision threshold, the model:
1. Evaluates F1-score across a range of thresholds
2. Selects the optimal operating point
3. Improves phishing recall while controlling false positives

This aligns the model's behavior with **real-world risk tolerance**.

### Performance

| Metric | Result |
|--------|--------|
| Overall Accuracy | High |
| Phishing Recall | Strong |
| Macro F1 (post-tuning) | Improved |

> Model exported via `joblib` for production deployment.

---

## 🔗 URL Phishing Detection

### Dataset
- **549,000+ labeled URLs**
- Labels: `good` / `bad`
- Purpose-built for phishing URL detection

### Why URLs Need a Different Approach

Unlike emails, URLs are short strings that encode phishing signals through **structural manipulation** and **lexical obfuscation** — not narrative language. A hybrid feature strategy was implemented accordingly.

### URL Normalization

To prevent distribution mismatch between training data and real-world inputs:

```python
# Normalization steps
- Remove scheme (http:// / https://)
- Remove "www" prefix
- Standardize to lowercase
```

### Feature Engineering

#### A) Structural Numeric Features

| Feature | Description |
|---------|-------------|
| URL length | Total character count |
| Domain length | Length of domain segment |
| Number of dots | Subdomain depth indicator |
| Number of hyphens | Common obfuscation signal |
| Number of slashes | Path depth |
| Number of digits | Numeric noise detection |
| IP address presence | Binary flag |
| `@` symbol presence | Binary flag |
| Number of subdomains | Subdomain chain length |

#### B) Character-Level TF-IDF (3–5 grams)

Character n-grams capture suspicious substrings regardless of word boundaries — particularly effective for **brand impersonation** and **lexical manipulation**:

> `login` · `secure` · `verify` · `update` · `paypal` · `microsoft` · `account`

### Hybrid Model Architecture

```
Character TF-IDF Features ──┐
                             ├── Concatenated Sparse Matrix → Logistic Regression → Classification
Structural Numeric Features ─┘
```

Combining both feature types significantly improves detection performance over either alone.

### Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~93% |
| Phishing Recall | ~93% |
| Phishing F1 | 0.89 |
| Macro F1 | 0.92 |

**Detection capabilities:**
- ✅ Brand impersonation
- ✅ IP-based phishing
- ✅ Suspicious subdomain chains
- ✅ Token-based phishing signals

---

## 🚫 Why Not BERT / Transformers?

Transformer-based models (BERT, DistilBERT) were evaluated and deliberately excluded for the following strategic reasons:

### 1. GPU Dependency and Deployment Practicality
Transformer models require GPU acceleration for efficient inference. In many real-world deployment scenarios — edge environments, high-throughput APIs, cost-constrained infrastructure — GPUs are not consistently available.

This project was designed for **CPU-based inference** to ensure:
- Broader deployability across environments
- Lower infrastructure cost
- Reduced and predictable latency
- Simpler production integration

### 2. Complexity vs. Marginal Benefit
For URL detection specifically, character-level TF-IDF already captures lexical phishing patterns effectively. Adding BERT would:
- Significantly increase model complexity
- Increase per-request inference cost
- Offer limited marginal gain on short URL strings

### 3. Operational Efficiency
Classical ML models offer critical operational advantages:

| Property | Classical ML | Transformers |
|----------|-------------|--------------|
| Load time | Instant | Seconds |
| Dependencies | Minimal | Heavy |
| Containerization | Simple | Complex |
| Interpretability | Transparent | Black-box |
| Inference speed | Milliseconds | 100ms+ (CPU) |

In security applications, **interpretability and speed are non-negotiable**.

---

## 🏗️ System Architecture

### Email Pipeline

```
Email Text
    │
    ▼
Text Cleaning & Normalization
    │
    ▼
Word-Level TF-IDF Vectorization
    │
    ▼
Logistic Regression
    │
    ▼
Probability Score → Threshold → Classification
```

### URL Pipeline

```
Raw URL
    │
    ▼
Normalization (scheme/www removal, lowercase)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
Structural Feature Extraction    Character TF-IDF (3–5 grams)
    │                                  │
    └──────────────┬───────────────────┘
                   ▼
        Feature Concatenation (Sparse Matrix)
                   │
                   ▼
           Logistic Regression
                   │
                   ▼
        Probability Score → Threshold → Classification
```

---

## 🧭 Design Decisions

| Decision | Justification |
|----------|--------------|
| Two separate datasets | Emails and URLs are fundamentally different data modalities |
| TF-IDF for emails | Efficient, interpretable NLP baseline with strong classification performance |
| Character TF-IDF for URLs | Captures phishing substrings and brand impersonation patterns |
| Structural URL features | Detects abnormal URL construction and obfuscation tactics |
| Logistic Regression | Fast, interpretable, scalable, stable in production |
| No BERT/Transformers | GPU dependency, deployment cost, and limited marginal gain |

---

## 🚀 Deployment

Both models are production-ready:

```python
import joblib

# Load models
email_model = joblib.load("email_phishing_model.pkl")
url_model   = joblib.load("url_phishing_model.pkl")

# Predict
email_pred = email_model.predict([email_text])
url_pred   = url_model.predict([url_string])
```

| Property | Status |
|----------|--------|
| Export format | `joblib` |
| Inference hardware | CPU (no GPU required) |
| Latency | Low (milliseconds) |
| API compatibility | Flask / FastAPI ready |
| Real-time detection | ✅ |

---

## 🔮 Future Enhancements

- [ ] Ensemble models for improved robustness
- [ ] Integration with reputation APIs (e.g., VirusTotal, Google Safe Browsing)
- [ ] Domain whitelisting and allowlist management
- [ ] Active learning pipeline for model updates
- [ ] Deep learning enhancements for high-resource environments
- [ ] Dashboard for monitoring detection metrics in real time

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
