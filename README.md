# 📰 TruthLens – Fake News Detector

**TruthLens** is an AIML based Fake News Detection System that uses **Natural Language Processing (NLP)** and **Machine Learning** to classify news articles as **Real** or **Fake**.

The application is built with **Python and Streamlit** and provides an interactive interface for analyzing news content.

## 🚀 Features

* 📰 Real/Fake news classification
* 🎯 Prediction confidence score
* 📊 Article statistics
* 🧠 Model explanation
* 📈 Data visualizations
* 🕒 Prediction history
* 📚 Explanation of how the model works
* ⚠️ Fact-checking disclaimer
* 💻 Interactive Streamlit interface

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **NLTK**
* **Matplotlib**
* **TF-IDF Vectorization**
* **Support Vector Machine (SVM)**
* **Joblib**
* **Streamlit**

## 🧠 Machine Learning Approach

The project follows these major steps:

News Article
     ↓
Text Preprocessing
     ↓
Stopword Removal
     ↓
Stemming
     ↓
TF-IDF Vectorization
     ↓
SVM Classification
     ↓
Real / Fake Prediction
     ↓
Confidence Score


### Text Preprocessing

The input text is processed by:

1. Converting text to lowercase
2. Removing non-alphabetic characters
3. Tokenizing the text
4. Removing English stopwords
5. Applying stemming

### Feature Extraction

**TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert processed text into numerical features that can be used by the machine learning model.

### Classification Model

A **Support Vector Machine (SVM)** classifier is trained to distinguish between Real and Fake news.

## 📂 Project Structure

truthlens-fake-news-detector/
│
├── training_model.ipynb
├── app.py
├── model/
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── data/
│   └── combined_dataset.csv
│
├── requirements.txt
└── README.md

> File and folder names can be modified according to the actual project structure.

## ⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/truthlens-fake-news-detector.git

Navigate to the project directory:

cd truthlens-fake-news-detector

Install the required libraries:

pip install -r requirements.txt

## ▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

## 📊 Prediction

TruthLens takes a news article as input and processes it through the trained NLP pipeline.

The system provides:

* **Prediction:** Real or Fake
* **Confidence Score**
* **Article Statistics**
* **Model Insights**

## 🎯 Project Objective

The objective of TruthLens is to demonstrate the practical application of **Machine Learning and NLP for fake news classification**.

It is designed as an educational project to understand the complete ML workflow, from **data preprocessing and feature extraction to model training and deployment**.

## ⚠️ Disclaimer

TruthLens is an **educational machine-learning project**. Its predictions are based on patterns learned from the training data and should **not be considered a definitive fact-checking or news-verification system**.

Always verify important information using reliable sources.

## 🔮 Future Improvements

* Integrate real time news APIs
* Add advanced NLP models such as BERT
* Improve multilingual news detection
* Add explainable AI techniques
* Add real-time fact-checking
* Deploy the application online

## 👨‍💻 Author

**Manvender Singh Rawat**

B.Tech CSE-AIML | Machine Learning & Data Science Enthusiast


⭐ If you find this project useful, consider giving the repository a star!

