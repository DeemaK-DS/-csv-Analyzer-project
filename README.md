#  CSV NLP Analyzer Tool (English & Arabic)

## Project Overview

This project is an **end-to-end NLP analysis and classification tool** built using **Python and Streamlit**.  
It allows users to upload a CSV file containing text data and labels, then automatically performs:

1. Exploratory Data Analysis (EDA)  
2. Text Preprocessing (English & Arabic)  
3. Text Embedding  
4. Model Training & Evaluation  

All steps are executed through an **interactive web interface**, without requiring the user to write code.

---

## How the Project Works (User Flow)

1. The user uploads a **CSV file**.  
2. The user selects:
   - Text column  
   - Label column  
3. The system automatically:
   - Detects the language based on the selected text column  
   - Applies appropriate preprocessing  
   - Generates embeddings  
   - Trains multiple ML models  
   - Evaluates and compares results  
4. Results, models, and intermediate outputs are **saved automatically**.

---

##  Exploratory Data Analysis (EDA)

EDA is performed immediately after uploading the dataset.

###  Visualizations (Plotly)

- **Label Distribution Pie Chart**
  - Shows class balance across labels.
- **Text Length Histogram**
  - Displays text length distribution (in words).

---

##  Text Preprocessing

Preprocessing is applied based on the **detected language** of the selected text column.

###  English Preprocessing:

- Convert text to lowercase  
- Remove punctuation  
- Remove URLs and numbers  
- Optional stopword removal  

###  Arabic Preprocessing:

- Remove links and emojis  
- Remove tashkeel (diacritics)  
- Remove tatweel  
- Remove punctuation and numbers  
- Normalize Arabic letters  
- Optional Arabic stopword removal  

 A preview of **original vs processed text** is shown in the UI.

---

##  Text Embedding

After preprocessing, the processed text is converted into numerical representations.

### Supported Embedding Methods

1. **TF-IDF**
2. **Model2Vec (ARBERTv2)**
   - JadwalAlmaa/model2vec-ARBERTv2
3. **Sentence Transformers**
   - Sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

### Embedding Statistics (Displayed in UI)

- Embedding method  
- Shape (samples × features)  
- Memory usage (MB)  
---

##  Model Training & Evaluation

The generated embeddings are used to train multiple machine learning models.

### Train/Test Split

- **80% Training / 20% Testing**
- Stratified split to preserve class distribution

### Trained Models

1. **K-Nearest Neighbors (KNN)**
2. **Logistic Regression**
3. **Random Forest**

###  Evaluation Metrics

For each model:

- Accuracy  
- Precision 
- Recall
- F1-score  
- Confusion Matrix   
---
##  User Interface (Streamlit)
The entire pipeline is controlled via a **single Streamlit web application**, featuring:

- Column selection  
- Preprocessing options  
- Embedding selection  
- One-click execution (`Run Analysis`)  
- Visual and tabular results  
---
## 🌐 Live Demo

👉 **Try the application here:**  
**https://csv-analyzer-deemak.streamlit.app/**  
