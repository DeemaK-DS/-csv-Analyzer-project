# Weekly Project: Arabic NLP Classification CLI Tool

## 📋 Project Overview

Build a **command-line interface (CLI) tool** to streamline the entire NLP pipeline for text classification. This tool will help you process Arabic text data through the 4 key stages of an NLP project: **EDA → Preprocessing → Embedding → Training**.

The tool is designed to work with **CSV files containing Arabic text** and produce classification models with evaluation metrics.

---

## 🎯 Project Objectives

- Learn how to build scalable CLI applications with multiple commands and subcommands
- Implement the complete NLP pipeline programmatically
- Handle Arabic text-specific challenges (tashkeel, hamza normalization, etc.)
- Integrate multiple libraries (pandas, scikit-learn, sklearn models, transformers)
- Produce publication-ready visualizations and model reports

---

## 📦 Requirements

### Core Dependencies
```
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
click>=8.0.0  # For building CLI
plotly>=5.0.0  # Optional: for interactive plots
requests>=2.26.0
arabic-stop-words>=0.1.0
```

### Optional Dependencies (for bonus features)
```
transformers>=4.20.0  # For BERT, Model2Vec
sentence-transformers>=2.2.0
gensim>=4.1.0  # For Word2Vec, FastText
cameltools>=1.1.0  # For Arabic stemming/lemmatization
datasets>=2.0.0  # For Hugging Face dataset integration
google-generativeai>=0.3.0  # For synthetic data generation
```

---

## 🛠️ Project Structure

```
nlp-cli-tool/
├── main.py                  # CLI entry point
├── commands/
│   ├── eda.py              # EDA commands
│   ├── preprocessing.py    # Preprocessing commands
│   ├── embedding.py        # Embedding commands
│   └── training.py         # Training commands
├── utils/
│   ├── arabic_text.py      # Arabic text utilities
│   ├── visualization.py    # Plotting functions
│   ├── data_handler.py     # CSV loading & processing
│   └── metrics.py          # Model evaluation
├── outputs/
│   ├── models/             # Trained model files
│   ├── embeddings/         # Embedding outputs
│   ├── reports/            # Markdown reports
│   └── visualizations/     # Generated plots
└── requirements.txt
```

---

## 📊 Command Structure

### 1️⃣ **EDA Command** - Exploratory Data Analysis

Analyze your dataset before preprocessing.

```bash
# View class distribution (pie chart)
python main.py eda distribution --csv_path data.csv --label_col class

# View class distribution (bar chart)
python main.py eda distribution --csv_path data.csv --label_col class --plot_type bar

# Generate text length histogram (word count)
python main.py eda histogram --csv_path data.csv --text_col description --unit words

# Generate text length histogram (character count)
python main.py eda histogram --csv_path data.csv --text_col description --unit chars
```

**Expected Output:**
- PNG/HTML visualization saved to `outputs/visualizations/`
- Console summary with statistics (mean, median, std dev)

---

### 2️⃣ **Preprocessing Command** - Clean & Normalize Text

Prepare your text data for embedding and training.

```bash
# Remove Arabic-specific characters (tashkeel, tatweel, tarqeem, links, etc.)
python main.py preprocess remove --csv_path data.csv --text_col description --output cleaned.csv

# Remove stopwords using Arabic stopwords list
python main.py preprocess stopwords --csv_path cleaned.csv --text_col description --output no_stops.csv

# Normalize Arabic text (hamza, alef maqsoura, taa marbouta)
python main.py preprocess replace --csv_path no_stops.csv --text_col description --output normalized.csv

# Chain all preprocessing steps
python main.py preprocess all --csv_path data.csv --text_col description --output final.csv
```

**Subcommands:**

| Subcommand | Purpose | Removes |
|-----------|---------|---------|
| `remove` | Clean unwanted text | Tashkeel (ً،ٌ،ٍ)، Tatweel (ـــ)، Tarqeem (0-9)، URLs، Special chars |
| `stopwords` | Remove common words | Arabic stop words (من، إلى، هذا، etc.) |
| `replace` | Normalize Arabic text | Hamza variants → و/ي/ا, ة → ه, ى → ي |

**Expected Output:**
- Cleaned CSV file with processed text
- Console report showing text statistics (before/after)

---

### 3️⃣ **Embed Command** - Vectorize Text

Convert text into numerical embeddings for model training.

```bash
# TF-IDF Embedding (sklearn)
python main.py embed tfidf --csv_path cleaned.csv --text_col description --max_features 5000 --output tfidf_vectors.pkl

# Model2Vec Embedding (from HuggingFace: JadwalAlmaa/model2vec-ARBERTv2)
python main.py embed model2vec --csv_path cleaned.csv --text_col description --output model2vec_vectors.pkl

# BERT Embedding (bonus)
python main.py embed bert --csv_path cleaned.csv --text_col description --model AraBERT --output bert_vectors.pkl

# Sentence Transformers (bonus)
python main.py embed sentence-transformer --csv_path cleaned.csv --text_col description --model sentence-transformers/distiluse-base-multilingual-cased-v2 --output sent_vectors.pkl
```

**Expected Output:**
- Pickle file or numpy array with embeddings
- Console output showing embedding shape and memory usage

---

### 4️⃣ **Training Command** - Train & Evaluate Models

Train classification models and generate evaluation reports.

```bash
# Train with default models (KNN, Logistic Regression, Random Forest)
python main.py train --csv_path final.csv --input_col embedding --output_col class --test_size 0.2 --models knn lr rf

# Train with all sklearn models
python main.py train --csv_path final.csv --input_col embedding --output_col class --models all

# Train with custom hyperparameters
python main.py train --csv_path final.csv --input_col embedding --output_col class --models "knn:n_neighbors=7" "lr:C=0.5"

# Save best model to disk
python main.py train --csv_path final.csv --input_col embedding --output_col class --save_model best_model.pkl
```

**Expected Output:**
- `outputs/reports/training_report_[timestamp].md` with:
  - Model performance metrics (Accuracy, Precision, Recall, F1)
  - Confusion matrices for each model
  - ROC curves (bonus)
  - Feature importance plots (bonus)
- Trained model file (pickle)

**Report Example:**
```markdown
## Training Report - 2025-01-13

### Dataset Info
- Total samples: 1,000
- Train/Test split: 800/200 (80/20)
- Classes: 3
- Features: 5,000

### Model Performance

#### K-Nearest Neighbors (K=5)
- Accuracy:  0.92
- Precision: 0.90
- Recall:    0.91
- F1-Score:  0.905

[Confusion Matrix visualization]

#### Logistic Regression
- Accuracy:  0.94
- Precision: 0.93
- Recall:    0.94
- F1-Score:  0.935

[Confusion Matrix visualization]

#### Random Forest
- Accuracy:  0.96
- Precision: 0.95
- Recall:    0.96
- F1-Score:  0.955

[Confusion Matrix visualization]

### Best Model: Random Forest ⭐
```

---

## 🎁 Bonus Features (Extra Credit)

Implement any of the following for bonus points:

### 🤖 **Synthetic Data Generation**
```bash
# Generate synthetic Arabic text using a language model
python main.py generate --model gemini --class_name "positive" --count 100 --output synthetic.csv

# Generate using a downloaded model (optional)
python main.py generate --model local --count 100 --output synthetic.csv
```

### 🎯 **Outlier Detection & Removal**
```bash
# Remove statistical outliers from EDA/preprocessing
python main.py eda remove-outliers --csv_path data.csv --text_col description --method iqr --output clean_data.csv
```

### 🌐 **Multilingual Support**
```bash
# Process both Arabic and English
python main.py eda distribution --csv_path data.csv --label_col class --language both

# Multilingual preprocessing
python main.py preprocess all --csv_path data.csv --text_col description --language auto --output final.csv
```

### 📚 **Hugging Face Dataset Integration**
```bash
# Load dataset directly from Hugging Face
python main.py eda distribution --dataset "MARBERT/XNLI" --split "ar" --label_col label

# Use dataset in full pipeline
python main.py train --dataset "ARBML/ArabiCorpus" --text_col text --output_col category
```

### 💾 **One-Line Complete Pipeline**
```bash
# Run all steps in sequence
python main.py pipeline --csv_path data.csv --text_col description --label_col class \
  --preprocessing "remove,stopwords,replace" \
  --embedding tfidf \
  --training "knn,lr,rf" \
  --output results/

# With all options
python main.py pipeline --csv_path data.csv --text_col description --label_col class \
  --preprocessing all --embedding model2vec --training all --save_report --save_models
```

### 🖥️ **GUI Interface** (Streamlit)
```bash
# Launch interactive web interface
python -m streamlit run interface.py

# Dashboard features:
# - Upload CSV file
# - Select columns and preprocessing options
# - Visualize EDA results
# - Configure embeddings
# - Train and compare models
# - Download reports and models
```

### 🔤 **Advanced Embeddings**
```bash
# Word2Vec
python main.py embed word2vec --csv_path cleaned.csv --text_col description --output word2vec_vectors.pkl

# FastText
python main.py embed fasttext --csv_path cleaned.csv --text_col description --output fasttext_vectors.pkl

# Sentence Transformers (pre-trained)
python main.py embed sentence-transformer --csv_path cleaned.csv --text_col description \
  --model sentence-transformers/all-MiniLM-L6-v2 --output sent_vectors.pkl
```

### 🌍 **Multilingual Stopwords**
```bash
# Remove stopwords in any language
python main.py preprocess stopwords --csv_path data.csv --text_col description --language en --output cleaned.csv

python main.py preprocess stopwords --csv_path data.csv --text_col description --language fr --output cleaned.csv

python main.py preprocess stopwords --csv_path data.csv --text_col description --language auto --output cleaned.csv
```

### 🔤 **Stemming & Lemmatization**
```bash
# Apply Arabic stemming (Snowball)
python main.py preprocess stem --csv_path cleaned.csv --text_col description --language ar --stemmer snowball --output stemmed.csv

# Arabic lemmatization (CAMeLTools)
python main.py preprocess lemmatize --csv_path cleaned.csv --text_col description --language ar --output lemmatized.csv

# Choose specific preprocessing
python main.py preprocess remove --csv_path data.csv --text_col description --remove "tashkeel,links" --output partial_clean.csv
```

### ☁️ **Word Clouds**
```bash
# Generate word cloud for each class
python main.py eda wordcloud --csv_path cleaned.csv --text_col description --label_col class --output wordclouds/

# Combined word cloud
python main.py eda wordcloud --csv_path cleaned.csv --text_col description --output wordcloud.png
```

### 📊 **Information Retrieval Version** (Alternative Project)
```bash
# Build a semantic search system instead of classification
python main.py ir-setup --csv_path documents.csv --text_col content --index_type faiss --output index.faiss

# Query the index
python main.py ir-query --query "ما هو الذكاء الاصطناعي؟" --index_path index.faiss --top_k 5
```

---

## 📝 Full Workflow Example

Here's a complete example from start to finish:

```bash
# Step 1: Explore your data
python main.py eda distribution --csv_path data.csv --label_col sentiment

# Step 2: Clean the text
python main.py preprocess remove --csv_path data.csv --text_col description --output step1_cleaned.csv
python main.py preprocess stopwords --csv_path step1_cleaned.csv --text_col description --output step2_nostops.csv
python main.py preprocess replace --csv_path step2_nostops.csv --text_col description --output step3_normalized.csv

# Step 3: Analyze text lengths
python main.py eda histogram --csv_path step3_normalized.csv --text_col description --unit words

# Step 4: Create embeddings
python main.py embed tfidf --csv_path step3_normalized.csv --text_col description --output vectors.pkl

# Step 5: Train and evaluate models
python main.py train --csv_path step3_normalized.csv --input_col vectors.pkl --output_col sentiment --models knn lr rf
```

**Or in one command:**
```bash
python main.py pipeline --csv_path data.csv --text_col description --label_col sentiment \
  --preprocessing "remove,stopwords,replace" \
  --embedding tfidf \
  --training "knn,lr,rf" \
  --save_report
```

---

## 📋 Evaluation Criteria

| Criteria | Points |
|----------|--------|
| **Core Features** | |
| EDA command with both subcommands | 20 |
| Preprocessing command (all 3 subcommands) | 25 |
| Embedding command (at least TF-IDF) | 20 |
| Training command with evaluation metrics | 25 |
| Clean, modular code structure | 10 |
| **Bonus Features** (Choose 3+) | |
| Each bonus feature implemented | +10 each |
| **Total Possible** | **100+ points** |

---

## 💡 Tips & Best Practices

1. **Use Click library** for building professional CLI commands:
   ```python
   import click

   @click.group()
   def cli():
       pass

   @cli.command()
   @click.option('--csv_path', required=True, help='Path to CSV file')
   def eda(csv_path):
       # Implementation
       pass
   ```

2. **Modularize your code**: Keep each command in separate files for maintainability

3. **Add error handling**: Validate inputs and provide helpful error messages

4. **Use logging**: Track what's happening in your pipeline

5. **Cache embeddings**: Don't recompute embeddings unnecessarily

6. **Test locally first**: Before running on full datasets

7. **Document your code**: Add docstrings and comments

---

## 🚀 Submission Checklist

- [ ] CLI tool runs without errors
- [ ] All core commands work (eda, preprocess, embed, train)
- [ ] Visualizations are generated correctly
- [ ] Reports are saved in markdown format
- [ ] Code is well-organized and documented
- [ ] README.md explains how to use the tool
- [ ] Requirements.txt is up to date
- [ ] At least 3 bonus features implemented
- [ ] Tested with Arabic text data

---

## 📚 Resources

- **Click Documentation**: https://click.palletsprojects.com/
- **Scikit-learn**: https://scikit-learn.org/
- **Arabic Stop Words**: https://github.com/mohataher/arabic-stop-words
- **Model2Vec**: https://huggingface.co/JadwalAlmaa/model2vec-ARBERTv2
- **Hugging Face Datasets**: https://huggingface.co/datasets
- **CAMeLTools (Arabic NLP)**: https://camel-tools.readthedocs.io/
- **Streamlit**: https://streamlit.io/

---

## ❓ FAQ

**Q: Can I use a different CLI library instead of Click?**  
A: Yes! Argparse, Typer, or Hydra are also acceptable alternatives.

**Q: What if my CSV doesn't have a label column?**  
A: The EDA command would still work for text analysis. Training would require a label column.

**Q: Can I use pre-trained Arabic models?**  
A: Yes! Model2Vec, BERT variants, and sentence transformers are highly encouraged.

**Q: How large can my dataset be?**  
A: Start with smaller datasets (< 10,000 samples) for testing. Optimize later.

**Q: Do I need to implement all bonus features?**  
A: No! Choose 3+ that interest you most. Quality over quantity.

---

**Good luck! 🚀 Ask questions in class or office hours if you get stuck.**
