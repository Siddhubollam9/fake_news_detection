---

# 📰 Fake News Detection using Machine Learning

### 🔍 Project Overview

Fake news has become one of the biggest challenges in today’s digital world.
This project aims to build a **machine learning-based Fake News Detection System** that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP).

We collected real-world data, cleaned and processed it using NLP techniques, and trained multiple ML models to predict the authenticity of news.

---

## 🧠 Key Objectives

* Understand how text classification works.
* Learn data preprocessing (cleaning, tokenizing, lemmatization).
* Train multiple ML models to detect fake news.
* Compare model performance using various metrics.
* Build and deploy an interactive Streamlit app.

---

## 📅 Project Timeline

| Day       | Tasks           | Description                                                          |
| --------- | --------------- | -------------------------------------------------------------------- |
| **Day 1** | Setup & Dataset | Installed libraries, downloaded Kaggle Fake News Dataset             |
| **Day 2** | Data Cleaning   | Cleaned text data (stopwords, punctuation, lemmatization)            |
| **Day 3** | Model Training  | Trained Logistic Regression, Naive Bayes, Random Forest              |
| **Day 4** | Evaluation      | Measured accuracy, precision, recall, and visualized results         |
| **Day 5** | Streamlit App   | Built user-friendly app for fake news detection and finalized report |

---

## 🧩 Dataset

We used **Kaggle’s Fake and True News Dataset**.

* **fake.csv** – Contains fake news articles
* **true.csv** – Contains true news articles

👉 Dataset link: [Fake and True News Dataset (Kaggle)](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

---

## ⚙️ Tech Stack

* **Python 3.x**
* **Pandas**, **NumPy**
* **Scikit-learn**
* **NLTK / spaCy**
* **Matplotlib / Seaborn**
* **Streamlit**
* **Pickle / Joblib** (for saving models)

---

## 🚀 Project Workflow

### 1️⃣ Data Preprocessing

* Merged both datasets (fake.csv, true.csv).
* Cleaned text: lowercased, removed punctuation, stopwords, and numbers.
* Tokenized and lemmatized words using **NLTK WordNet Lemmatizer**.
* Converted text into numerical features using **TF-IDF Vectorization**.

### 2️⃣ Model Training

Trained and compared three classical ML models:

* Logistic Regression
* Naive Bayes
* Random Forest

(Optional: LSTM/Neural Network for advanced version)

### 3️⃣ Evaluation

* Evaluated models using **accuracy**, **precision**, **recall**, **F1-score**.
* Visualized **Confusion Matrix** and **ROC Curve**.
* Chose the best-performing model and saved it using **pickle**.

### 4️⃣ Streamlit Web App

Built an interactive app that:

* Takes a news headline or paragraph as input.
* Cleans and preprocesses it.
* Predicts whether it is **Real** or **Fake** in real-time.

---

## 🧾 How to Run Locally

### 🔧 Step 1: Clone the Repository

```
git clone https://github.com/your-username/fake-news-detection-ml.git
cd fake-news-detection-ml
```

### 🔧 Step 2: Install Dependencies

```
pip install -r requirements.txt
```

### 🔧 Step 3: Download NLTK Data

```
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 🔧 Step 4: Run Streamlit App

```
streamlit run app.py
```

Then open the local URL displayed in your terminal.

---

## 🧮 Results

| Model               | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 98.3%    | 98.5%     | 98.2%  | 98.3%    |
| Naive Bayes         | 97.6%    | 97.8%     | 97.4%  | 97.6%    |
| Random Forest       | 96.8%    | 96.5%     | 96.7%  | 96.6%    |

*(Your values may vary slightly depending on random state)*

---

## 🖼️ Screenshots

### 🏠 Streamlit Home Page

<img width="682" height="459" alt="image" src="https://github.com/user-attachments/assets/c3481627-ead6-45c1-8776-2555014b7c5b" />


### 📊 Confusion Matrix Visualization

<img width="452" height="393" alt="image" src="https://github.com/user-attachments/assets/f0d43396-abfc-43df-9846-cd041cc3f795" />
<img width="452" height="393" alt="image" src="https://github.com/user-attachments/assets/b50d2b04-7f35-4b35-ab88-d22592732164" />
<img width="452" height="393" alt="image" src="https://github.com/user-attachments/assets/2495515e-6b16-4a0f-ac3a-46f06e8e8bb2" />



---

## 🧰 Files Structure

```
📂 fake-news-detection-ml
 ┣ 📜 app.py               # Streamlit App
 ┣ 📜 fake.csv             # Fake news dataset
 ┣ 📜 true.csv             # True news dataset
 ┣ 📜 model.pkl            # Saved ML model
 ┣ 📜 vectorizer.pkl       # Saved TF-IDF vectorizer
 ┣ 📜 requirements.txt     # Dependencies
 ┣ 📜 README.md            # Project Documentation
 ┗ 📜 Fake_News_Detection.ipynb  # Jupyter Notebook (model training)
```

---

## 🧠 Learning Outcomes

* Understood **NLP preprocessing pipeline** from raw text to TF-IDF vectors.
* Learned **ML model training and evaluation** on textual data.
* Explored **Streamlit** for ML model deployment.
* Gained **real-world project experience** for resume and interviews.

---

## 📜 Resume Description

> **Implemented a Fake News Detection system using Machine Learning.**
> Collected dataset from Kaggle, applied NLP preprocessing (stopword removal, lemmatization, TF-IDF), trained multiple ML models (Logistic Regression, Naive Bayes, Random Forest), achieved ~98% accuracy, and deployed an interactive web app using Streamlit.

---

## ⭐ Future Improvements

* Integrate **transformer-based models (BERT, RoBERTa)** for better accuracy.
* Add **live news scraping API** for real-time fake news detection.
* Build **dashboard analytics** for fake news trends.

---

## 🏁 Conclusion

This project demonstrates how Natural Language Processing and Machine Learning can be effectively used to combat misinformation.
It serves as a complete end-to-end ML project — from data collection to deployment — ideal for portfolios, hackathons, and interviews.

---
