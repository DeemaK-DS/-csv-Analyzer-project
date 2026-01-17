import streamlit as st
import pandas as pd

from nlp_tool.eda import (
    label_distribution_plot,
    text_length_histogram
)

from nlp_tool.preprocessing import (
    detect_language,
    run_preprocessing
)

from nlp_tool.embedding import (
    tfidf_embedding,
    model2vec_embedding,
    sentence_transformer_embedding
)

from nlp_tool.training import train_and_evaluate


# App Config
st.set_page_config(
    page_title="analyze Tool",
    layout="wide"
)

st.title("Arabic & English analyze Tool")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"File uploaded successfully ({df.shape[0]} rows)")


    # Column Selection
    st.subheader("Column Selection")

    text_col = st.selectbox("Select Text Column", df.columns)
    label_col = st.selectbox("Select Label Column", df.columns)

    language = detect_language(df[text_col])
    lang_label = "Arabic" if language == "ar" else "English"
    st.info(f"Detected Language (based on `{text_col}`): **{lang_label}**")


    # Configuration
    st.subheader("Configuration")
    col_prep, col_embed = st.columns(2)

    # Preprocessing 
    with col_prep:
        st.markdown("### Preprocessing Options")
        options = {}

        if language == "ar":
            all_selected = st.checkbox("ALL (Arabic)", value=True)

            options["remove_links_emojis"] = all_selected or st.checkbox("Remove links & emojis", True)
            options["remove_tashkeel"] = all_selected or st.checkbox("Remove tashkeel", True)
            options["remove_tatweel"] = all_selected or st.checkbox("Remove tatweel", True)
            options["remove_tarqeem"] = all_selected or st.checkbox("Remove punctuation & numbers", True)
            options["normalize_letters"] = all_selected or st.checkbox("Normalize Arabic letters", True)
            options["remove_stopwords"] = st.checkbox("Remove Arabic stopwords", False)

        else:
            all_selected = st.checkbox("ALL (English)", value=True)

            options["lowercase"] = all_selected or st.checkbox("Convert to lowercase", True)
            options["remove_punctuation"] = all_selected or st.checkbox("Remove punctuation", True)
            options["remove_urls_numbers"] = all_selected or st.checkbox("Remove URLs & numbers", True)
            options["remove_stopwords"] = st.checkbox("Remove English stopwords", False)

    # Embedding 
    with col_embed:
        st.markdown("### Embedding Options")

        embedding_type = st.radio(
            "Select Embedding Method",
            ["TF-IDF", "Model2Vec (ARBERTv2)", "Sentence Transformers"]
        )

        if embedding_type == "TF-IDF":
            st.caption(" Fast, sparse, good lexical baseline")
        elif embedding_type == "Model2Vec (ARBERTv2)":
            st.caption(" Fast dense semantic embeddings (Arabic-optimized)")
        else:
            st.caption(" Deep contextual embeddings (slower, higher quality)")


    # Run Analysis
    if st.button("▶ Run Analysis"):

        #  EDA 
        st.header("Exploratory Data Analysis")
        col1, col2 = st.columns(2)

        with col1:
            pie_fig = label_distribution_plot(df, label_col)
            st.plotly_chart(pie_fig, use_container_width=True)

        with col2:
            hist_fig = text_length_histogram(df, text_col, unit="words")
            st.plotly_chart(hist_fig, use_container_width=True)


        # Preprocessing 
        st.header("Preprocessing")

        processed_df, preview_df, metadata = run_preprocessing(
            df=df,
            text_col=text_col,
            options=options
        )

        preview_df.columns = ["Original Text", "Processed Text"]
        st.dataframe(preview_df, use_container_width=True)


        # Embedding 
        st.header(" Embedding")

        texts = processed_df["processed_text"].dropna()

        with st.spinner(f"Generating {embedding_type} embeddings..."):
            if embedding_type == "TF-IDF":
                embeddings, stats = tfidf_embedding(texts)

            elif embedding_type == "Model2Vec (ARBERTv2)":
                embeddings, stats = model2vec_embedding(texts)

            else:
                embeddings, stats = sentence_transformer_embedding(texts)

        # Embedding stats 
        st.subheader(" Embedding Statistics")

        st.table(pd.DataFrame({
            "Metric": ["Method", "Shape", "Memory Usage (MB)"],
            "Value": [
                stats["Method"],
                str(stats["Shape"]),
                stats["Memory (MB)"]
            ]
        }))


        # Training 
        st.header("Model Training & Evaluation")

        X = embeddings
        y = processed_df[label_col].values

        with st.spinner("Training models and evaluating performance..."):
            report, train_shape, test_shape = train_and_evaluate(X, y)

        #  Dataset info 
        st.subheader(" Dataset Info")
        st.write(f"- Total samples: {len(y)}")
        st.write(f"- Train/Test split: {train_shape[0]}/{test_shape[0]}")
        st.write(f"- Classes: {len(set(y))}")
        st.write(f"- Features: {X.shape[1]}")

        # Best model
        best_model_name = max(
            report.items(),
            key=lambda x: x[1]["metrics"]["F1"]
        )[0]

        st.subheader(f" Best Model: {best_model_name} ")

        # Results 
        for model_name, result in report.items():
            st.markdown(f"### {model_name}")

            metrics = result["metrics"]
            st.table(pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1"],
                "Value": [
                    round(metrics["Accuracy"], 4),
                    round(metrics["Precision"], 4),
                    round(metrics["Recall"], 4),
                    round(metrics["F1"], 4)
                ]
            }))

            st.image(
                result["confusion_matrix"],
                caption=f"{model_name} – Confusion Matrix"
            )

        st.success("Pipeline completed successfully ")
