import os
import pandas as pd
import plotly.express as px
import plotly.io as pio


OUTPUT_DIR = "outputs/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def label_distribution_plot(df: pd.DataFrame, label_col: str):
    """
    Plot label distribution as a pie chart using Plotly
    and save it automatically.
    """
    counts = df[label_col].value_counts().reset_index()
    counts.columns = [label_col, "count"]

    fig = px.pie(
        counts,
        names=label_col,
        values="count",
        title="Label Distribution"
    )

    path = os.path.join(OUTPUT_DIR, "label_distribution.html")
    pio.write_html(fig, file=path, auto_open=False)

    print(f"[EDA] Label distribution saved to {path}")
    return fig


def text_length_histogram(df: pd.DataFrame, text_col: str, unit: str = "words"):
    """
    Plot text length histogram (words or characters),
    print statistics, and save the plot.
    """

    if unit == "words":
        lengths = df[text_col].dropna().astype(str).apply(lambda x: len(x.split()))
        title = "Text Length Distribution (Words)"
        x_label = "Number of Words"
    else:
        lengths = df[text_col].dropna().astype(str).apply(len)
        title = "Text Length Distribution (Characters)"
        x_label = "Number of Characters"


    stats = {
        "Mean": round(lengths.mean(), 2),
        "Median": round(lengths.median(), 2),
        "Std": round(lengths.std(), 2)
    }


    fig = px.histogram(
        lengths,
        nbins=30,
        title=title,
        labels={"value": x_label}
    )

    path = os.path.join(OUTPUT_DIR, f"text_length_histogram_{unit}.html")
    pio.write_html(fig, file=path, auto_open=False)

    print(f"[EDA] Text length histogram saved to {path}")
    return fig



