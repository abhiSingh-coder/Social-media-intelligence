# 📊 Social Media Intelligence System

An AI-powered web application that analyzes social media data using a combination of classical machine learning models and pretrained transformer models (BERT, RoBERTa) to surface trends, sentiment, emotion, misinformation signals, topics, and influencer rankings.

🔗 **Live Demo:** [social-media-intelligence-l9zxemv3htww5vrhwg9ksb.streamlit.app](https://social-media-intelligence-l9zxemv3htww5vrhwg9ksb.streamlit.app/)

---

## 🧠 Overview

This project demonstrates an end-to-end AI/ML pipeline — from raw social media text data to interactive, real-time analysis — covering classical ML, unsupervised learning, transformer-based NLP, and LLM integration within a single Streamlit application.

**Demo Login:**
```
Username: admin
Password: admin123
```

---

## ✨ Features

### 📈 Analytics Modules
- **Trend Analysis** — Detects spikes in post volume over time using rolling averages
- **Hashtag Analysis** — Extracts top hashtags and visualizes co-occurrence as an interactive network graph
- **Keyword Analytics** — TF-IDF based keyword extraction with word cloud visualization

### 🧬 AI / NLP Modules
- **Sentiment Analysis** — Pretrained RoBERTa model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) fine-tuned on tweets
- **Emotion Detection** — Pretrained DistilRoBERTa model classifying text into joy, anger, sadness, fear, surprise, disgust, neutral
- **Fake News Detection** — Custom-trained classifier (TF-IDF + XGBoost) achieving ~99.8% test accuracy on the Kaggle Fake/Real News dataset, compared against Naive Bayes and Logistic Regression baselines
- **Topic Modeling** — Unsupervised topic discovery using Latent Dirichlet Allocation (LDA), manually labeled into human-readable themes

### ✨ Extras
- **Influencer Ranking** — Weighted scoring system combining followers, engagement rate, and posting activity (normalized via min-max scaling)
- **Text Summarization** — Pretrained DistilBART model for condensing long text into short summaries
- **AI Chatbot** — LLM-powered conversational assistant (Google Gemini API) for natural-language queries
- **Content Recommendation** — TF-IDF + cosine similarity engine for finding related posts

### 📊 Dashboard
A unified landing page aggregating live summaries from every module: trending activity, sentiment distribution, emotion breakdown, top keywords (word cloud), top influencers, and top hashtags.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend / App Framework | Streamlit |
| Data Processing | pandas, NumPy |
| Classical ML | scikit-learn, XGBoost |
| NLP / Transformers | Hugging Face Transformers, PyTorch |
| Topic Modeling | scikit-learn (LatentDirichletAllocation) |
| LLM Integration | Google Gemini API |
| Visualization | Plotly, WordCloud, NetworkX, Matplotlib |
| Model Persistence | joblib |

---

## 🤖 Models Used

| Task | Model | Type |
|---|---|---|
| Sentiment Analysis | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Pretrained Transformer (RoBERTa) |
| Emotion Detection | `j-hartmann/emotion-english-distilroberta-base` | Pretrained Transformer (BERT-based) |
| Text Summarization | `sshleifer/distilbart-cnn-12-6` | Pretrained Transformer (BART) |
| Fake News Detection | XGBoost (TF-IDF features) | Custom-trained, compared against Naive Bayes & Logistic Regression |
| Topic Modeling | Latent Dirichlet Allocation | Unsupervised, custom-trained |
| Chatbot | Gemini 2.0 Flash | LLM API |

---

## 📂 Datasets

- **[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)** — 1.6M tweets with sentiment labels (used for Trend, Hashtag, Keyword, Sentiment, Emotion, Topic Modeling, and Recommendation modules)
- **[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)** — Used to train the Fake News Detection classifier

> Note: Influencer metrics (followers, engagement rate) are simulated for demo purposes, since the source dataset does not include real social engagement metadata. Post activity counts are derived from real data.

---

## 📁 Project Structure

```
social-media-intelligence/
│
├── app.py                          # Main entrypoint / landing page
├── pages/                          # Streamlit multi-page app modules
│   ├── 1_Login.py
│   ├── 2_Dashboard.py
│   ├── 3_Trend_Analysis.py
│   ├── 4_Hashtag_Analysis.py
│   ├── 5_Keyword_Analytics.py
│   ├── 6_Sentiment_Analysis.py
│   ├── 7_Emotion_Detection.py
│   ├── 8_Fake_News_Detection.py
│   ├── 9_Topic_Modeling.py
│   ├── 10_Influencer_Ranking.py
│   ├── 11_Summarization.py
│   ├── 12_Chatbot.py
│   └── 13_Recommendation.py
│
├── core/                           # Core logic — model loading, inference, scoring
├── components/                     # Reusable chart/visualization components
├── notebooks/                      # Training & experimentation notebooks
├── trained_models/                 # Saved model artifacts (.pkl)
├── data/
│   ├── raw/                        # Original datasets
│   └── processed/                  # Cleaned/sampled datasets
│
├── requirements.txt
└── README.md
```

---

## 🚀 Running Locally

1. Clone the repository
   ```
   git clone <repo-url>
   cd social-media-intelligence
   ```

2. Create and activate a virtual environment
   ```
   python -m venv venv
   venv\Scripts\activate   # Windows
   source venv/bin/activate # Mac/Linux
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Add your API key (for chatbot) in `.streamlit/secrets.toml`
   ```toml
   GOOGLE_API_KEY = "your-api-key-here"
   ```

5. Run the app
   ```
   streamlit run app.py
   ```

---

## 📌 Notes & Known Limitations

- The Sentiment140 dataset is from 2009 Twitter, and is relatively hashtag-sparse compared to modern social media data — hashtag analysis results reflect this.
- Fake News Detection accuracy (~99.8%) is partly influenced by stylistic/structural patterns specific to the training dataset (e.g. consistent news-wire formatting in real articles) — a known characteristic of this benchmark dataset.
- Influencer Ranking uses simulated engagement metrics, clearly labeled in-app, since the dataset lacks real social metadata.
- Authentication is a single demo account for showcasing the login-gated flow, not a full multi-user system.

---

## 👤 Author

Built as a portfolio/academic project demonstrating full-stack AI application development — combining classical ML, transformer-based NLP, unsupervised learning, and LLM integration in a single deployed system.
