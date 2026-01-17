import os
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from model2vec import StaticModel


OUTPUT_DIR = "outputs/embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Utility
def _embedding_stats(embeddings, method_name):
    stats = {
        "Method": method_name,
        "Shape": embeddings.shape,
        "Memory (MB)": round(embeddings.nbytes / (1024 ** 2), 2)
    }

    print(f"[Embedding] {method_name}")
    print(f"Shape        : {stats['Shape']}")
    print(f"Memory Usage : {stats['Memory (MB)']} MB\n")

    return stats


def _save_embeddings(obj, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[Embedding] Saved to {path}")



#  TF-IDF
def tfidf_embedding(texts: pd.Series):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts).toarray()

    stats = _embedding_stats(embeddings, "TF-IDF")

    _save_embeddings(
        {"embeddings": embeddings, "vectorizer": vectorizer},
        "tfidf_embeddings.pkl"
    )

    return embeddings, stats


# Model2Vec (ARBERTv2)
def model2vec_embedding(texts: pd.Series):
    model = StaticModel.from_pretrained(
        "JadwalAlmaa/model2vec-ARBERTv2"
    )

    embeddings = model.encode(texts.tolist())
    embeddings = np.array(embeddings)

    stats = _embedding_stats(embeddings, "Model2Vec (ARBERTv2)")

    _save_embeddings(
        {"embeddings": embeddings, "model": "JadwalAlmaa/model2vec-ARBERTv2"},
        "model2vec_arbertv2_embeddings.pkl"
    )

    return embeddings, stats


# Sentence Transformers
def sentence_transformer_embedding(
    texts: pd.Series,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts.tolist(),
        show_progress_bar=True
    )
    embeddings = np.array(embeddings)

    stats = _embedding_stats(embeddings, "Sentence Transformers")

    _save_embeddings(
        {"embeddings": embeddings, "model": model_name},
        "sentence_transformer_embeddings.pkl"
    )

    return embeddings, stats
