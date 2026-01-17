import pandas as pd
import re
import os
from pathlib import Path
import nltk
from nltk.corpus import stopwords


# Language Detection

def detect_language(text_series, threshold=0.2):
    """
    Detect dominant language in a text column.
    Uses ratio of rows containing Arabic characters.
    """

    arabic_pattern = re.compile(
        r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"
    )

    texts = text_series.dropna().astype(str)

    if len(texts) == 0:
        return "en"

    arabic_count = sum(
        1 for text in texts if arabic_pattern.search(text)
    )

    ratio = arabic_count / len(texts)

    return "ar" if ratio >= threshold else "en"

# Arabic preprocessing
def clean_text_arabic(text: str, options: dict) -> str:
    text = str(text)

    if options.get("remove_links_emojis", False):
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[\U00010000-\U0010ffff]", "", text)

    if options.get("remove_tashkeel", False):
        text = re.sub(r"[ًٌٍَُِّْ]", "", text)

    if options.get("remove_tatweel", False):
        text = text.replace("ـ", "")

    if options.get("remove_tarqeem", False):
        text = re.sub(r"[0-9٠-٩!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]", " ", text)

    if options.get("normalize_letters", False):
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ة", "ه", text)

    return re.sub(r"\s+", " ", text).strip()


BASE_DIR = Path(__file__).resolve().parent
STOPWORDS_PATH = BASE_DIR / "resources" / "list.txt"


def load_arabic_stopwords():
    if not STOPWORDS_PATH.exists():
        return set()
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


ARABIC_STOPWORDS = load_arabic_stopwords()


def remove_arabic_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in ARABIC_STOPWORDS)


# English preprocessing 

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

ENGLISH_STOPWORDS = set(stopwords.words("english"))


def clean_text_english(text: str, options: dict) -> str:
    text = str(text)

    if options.get("remove_urls_numbers", False):
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\d+", "", text)

    if options.get("lowercase", False):
        text = text.lower()

    if options.get("remove_punctuation", False):
        text = re.sub(r"[^\w\s]", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def remove_english_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in ENGLISH_STOPWORDS)



# Preprocessing Pipeline

def run_preprocessing(
    df: pd.DataFrame,
    text_col: str,
    options: dict,
    output_name: str = "cleaned_data.csv"
):
    df = df.copy()
    language = detect_language(df[text_col])

    processed_texts = []

    for text in df[text_col].astype(str):  
        if language == "ar":
            text = clean_text_arabic(text, options)
            if options.get("remove_stopwords", False):
                text = remove_arabic_stopwords(text)
        else:
            text = clean_text_english(text, options)
            if options.get("remove_stopwords", False):
                text = remove_english_stopwords(text)

        processed_texts.append(text)

    df["processed_text"] = processed_texts

    preview_df = df[[text_col, "processed_text"]].head(3)

    metadata = {
        "language": language,
        **options
    }

    save_processed_csv(df, name=output_name)

    return df, preview_df, metadata


# Save Output
def save_processed_csv(df, name="cleaned_data.csv"):
    os.makedirs("outputs/processed", exist_ok=True)
    path = f"outputs/processed/{name}"
    df.to_csv(path, index=False)
    print(f"[Preprocessing] Saved cleaned file to {path}")

def print_before_after_stats(original, processed):
    before = original.dropna().astype(str).apply(lambda x: len(x.split()))
    after = processed.dropna().astype(str).apply(lambda x: len(x.split()))

    print("Preprocessing Report:")
    print(f"Before → mean tokens: {before.mean():.2f}")
    print(f"After  → mean tokens: {after.mean():.2f}")


