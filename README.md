# ✈️ Airline Tweets Sentiment Analysis using DistilBERT

## 📌 Overview
This project performs **sentiment analysis** on airline-related tweets to classify them into:
- 😊 **Positive** → Shows customer satisfaction, appreciation, or praise for services.
- 😡 **Negative** → Expresses dissatisfaction, complaints, or criticism.
- 😐 **Neutral** → Shares factual, informational, or emotionless statements.

We leverage **DistilBERT**, a lightweight and efficient transformer model, to understand tweet context and detect the emotional tone behind customer feedback.

---

## 🎯 Objective
The goal is to help **airlines**, **travel agencies**, and **data analysts**:
- Understand customer perception
- Track service quality trends
- Identify improvement areas to enhance passenger experience and brand reputation

---

## 📊 Dataset
- **Source:** Airline tweets dataset (CSV format)
- **Features:**
  - `Tweet` — The actual customer tweet text
  - `Sentiment` — Labeled sentiment (Positive, Negative, Neutral)

---

## 🧠 Model
We use the **DistilBERT** model from Hugging Face Transformers, fine-tuned for sentiment classification.

**Why DistilBERT?**
- Lightweight yet high-performing
- Faster inference with minimal accuracy loss
- Excellent for real-time NLP applications

---

## ⚙️ Workflow
1. **Data Preprocessing** 🧹  
   - Text cleaning (hashtags, mentions, URLs removal)  
   - Tokenization  
   - Stopword & special character removal  

2. **Model Training** 🤖  
   - Fine-tune DistilBERT on labeled airline tweets  
   - Optimize hyperparameters for best performance  

3. **Prediction** 📈  
   - Classify new tweets into Positive, Negative, or Neutral sentiment  

---

## 🚀 Installation & Usage
```bash
# Clone repository
git clone https://github.com/yourusername/airline-tweets-sentiment-analysis-distilbert.git
cd airline-tweets-sentiment-analysis-distilbert

# Install dependencies
pip install -r requirements.txt

# Run the script
python sentiment_analysis.ipynb
