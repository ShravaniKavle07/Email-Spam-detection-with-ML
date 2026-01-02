![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%7C%20NLTK-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

# 📧 Email Spam Detection with Machine Learning

## 📌 Project Overview
Spam emails are a major source of phishing attacks, malware distribution, and financial fraud.  
This project implements an **end-to-end Email Spam Detection system** using **Natural Language Processing (NLP)** and **Machine Learning**, capable of classifying emails as **Spam** or **Not Spam (Ham)**.

The project covers the **complete data science lifecycle** — from data preprocessing and model training to deployment using a **Streamlit web application**.

---

## 🚀 Key Features
- End-to-end Machine Learning pipeline
- NLP-based text preprocessing
- TF-IDF feature extraction (unigrams + bigrams)
- Multiple ML model comparison
- High-performance Linear SVM classifier
- Interactive Streamlit web application
- Production-ready model persistence

---

## 🧠 Machine Learning Workflow

Data Collection
      ↓
Text Cleaning & Preprocessing
      ↓
TF-IDF Feature Engineering
      ↓
Model Training (NB, LR, SVM)
      ↓
Model Evaluation
      ↓
Final Model Selection
      ↓
Deployment (Streamlit App)

---

## 🛠️ Tech Stack & Tools

- 🐍 Python  
- 📊 Pandas, NumPy  
- 🧠 Scikit-learn  
- 📝 NLTK  
- 🔎 TF-IDF  
- 🌐 Streamlit  
- 📈 Matplotlib, Seaborn  
- 📦 Joblib  

---

## 📂 Project Structure
Email-Spam-Detection/
│
├── spam_detection.ipynb # Complete ML pipeline
├── spam_app.py # Streamlit web application
├── spam_model.pkl # Trained Linear SVM model
├── tfidf_vectorizer.pkl # TF-IDF feature transformer
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## 📊 Dataset
- **Source:** UCI / Kaggle SMS Spam Collection Dataset  
- **Size:** ~5,500 email messages  
- **Classes:** Spam, Ham (Not Spam)  
- **Challenge:** Imbalanced and noisy text data  

---

## 🧹 Text Preprocessing
- Lowercasing
- Removal of punctuation and numbers
- Stopword removal
- Stemming using Porter Stemmer

These steps reduce noise and improve model performance on textual data.

---

## ⚙️ Feature Engineering
- **TF-IDF Vectorization**
  - Max features: 5000
  - N-grams: Unigram + Bigram
  - Sublinear term frequency scaling

TF-IDF helps capture the importance of words across the email corpus.

---

## 🤖 Models Trained & Evaluation

| Model | Description |
|-----|------------|
| Multinomial Naive Bayes | Fast baseline model |
| Logistic Regression | Interpretable and balanced |
| Linear SVM | Best performance on sparse text |

### ✅ Final Model Selected
**Linear Support Vector Machine (SVM)** due to superior precision and recall on spam detection.

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 🌐 Streamlit Web Application
The trained model is deployed using **Streamlit**, allowing users to:
- Paste email content
- Instantly classify emails as Spam or Not Spam
- View model methodology and explanation

---

## ▶️ How to Run the Project Locally

### 1️⃣ Clone the Repository

git clone https://github.com/your-username/Email-Spam-Detection.git
cd Email-Spam-Detection 

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Run Jupyter Notebook
jupyter notebook Email_Spam_Detection_End_to_End.ipynb

### 4️⃣ Run Streamlit App
streamlit run app.py

---
📌 Use Cases

Email security systems

Phishing detection

Spam filtering services

NLP-based text classification

---
🔮 Future Enhancements

BERT-based deep learning classifier

Confidence/probability score for predictions

Batch email classification (CSV upload)

FastAPI backend integration

Cloud deployment (Streamlit Cloud / AWS)

---
👤 Author

Shravani Kavle
Data Science & AI Enthusiast

⭐ Acknowledgements
UCI Machine Learning Repository

Scikit-learn Documentation

Streamlit Community
