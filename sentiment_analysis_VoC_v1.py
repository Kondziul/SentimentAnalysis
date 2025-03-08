import streamlit as st
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pickle
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Model Comparison: Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Model Comparison for Sentiment Analysis")
st.markdown("Analyze and compare sentiment analysis models: DistilBERT, TF-IDF + Naive Bayes, and VADER.")

# File upload
filename = st.sidebar.file_uploader("Upload labeled Twitter data:", type=("csv", "xlsx"))

if filename:
    # Load data
    data = pd.read_csv(filename)
    data["text"] = data["text"].astype(str)
    data["Sentiment"] = data["Sentiment"].astype(str)

    # Filter out neutral sentiment for VADER
    filtered_data = data[data["Sentiment"] != "Neutral"]

    # Zamiana NaN na pusty ciąg znaków
    data['text'] = data['text'].fillna('')

    # Upewnienie się, że wszystkie wartości w kolumnie 'text' są typu string
    data['text'] = data['text'].astype(str)

    # Initialize SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # Function to classify sentiment using VADER
    def classify_sentiment_vader(text):
        scores = analyzer.polarity_scores(text)
        compound_score = scores['compound']
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return None  # Skip neutral comments

    # Classify and filter non-neutral comments
    filtered_data["VADER_Predicted"] = filtered_data["text"].apply(classify_sentiment_vader)
    filtered_data = filtered_data[filtered_data["VADER_Predicted"].notnull()]

    # Map Sentiment to binary labels for all models
    label_mapping = {"Positive": 1, "Negative": 0}
    filtered_data["VADER_Predicted"] = filtered_data["VADER_Predicted"].map(label_mapping)
    filtered_data["Sentiment"] = filtered_data["Sentiment"].map(label_mapping)

    # Load TF-IDF + Naive Bayes model
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    tfidf_model = pickle.load(open("naive_bayes_model.pkl", "rb"))

    def classify_sentiment_tfidf(text):
        vectorized_text = vectorizer.transform([text])
        return tfidf_model.predict(vectorized_text)[0]  # najczęściej zwraca "Positive"/"Negative"

    # Apply TF-IDF + Naive Bayes model
    filtered_data["TF-IDF_Predicted"] = filtered_data["text"].apply(classify_sentiment_tfidf)
    # Zmapuj "Positive"/"Negative" na 1/0
    filtered_data["TF-IDF_Predicted"] = filtered_data["TF-IDF_Predicted"].map(label_mapping)

    # Load DistilBERT model
    distilbert_model = DistilBertForSequenceClassification.from_pretrained("./distilbert_model")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def classify_sentiment_distilbert(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = distilbert_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).item()
        return predictions  # powinien zwracać 0 lub 1

    # Apply DistilBERT model
    filtered_data["DistilBERT_Predicted"] = filtered_data["text"].apply(classify_sentiment_distilbert)

    # Evaluate models
    st.subheader("Model Performance")
    models = ["TF-IDF", "DistilBERT", "VADER"]
    results = []
    for model in models:
        y_pred = filtered_data[f"{model}_Predicted"]
        y_true = filtered_data["Sentiment"]
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=["Negative", "Positive"], output_dict=True)
        results.append({
            "Model": model,
            "Accuracy": accuracy,
            "Precision": report["weighted avg"]["precision"],
            "Recall": report["weighted avg"]["recall"],
            "F1-Score": report["weighted avg"]["f1-score"]
        })

    results_df = pd.DataFrame(results)
    st.write(results_df)

    # Visualize comparison
    st.subheader("Model Comparison")
    fig = px.bar(results_df, x="Model", y=["Accuracy", "Precision", "Recall", "F1-Score"],
                 barmode="group", title="Model Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = filtered_data["VADER_Predicted"].value_counts().rename({0: "Negative", 1: "Positive"})
    fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        labels={"x": "Sentiment", "y": "Count"},
        title="Sentiment Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Word cloud generation
    st.subheader("Word Cloud for Sentiments")
    col1, col2 = st.columns(2)
    with col1:
        positive_text = " ".join(filtered_data[filtered_data["VADER_Predicted"] == 1]["text"])
        wordcloud = WordCloud(stopwords=set(STOPWORDS), background_color="white").generate(positive_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()
        st.write("Positive Comments Word Cloud")
    with col2:
        negative_text = " ".join(filtered_data[filtered_data["VADER_Predicted"] == 0]["text"])
        wordcloud = WordCloud(stopwords=set(STOPWORDS), background_color="white", colormap="Reds").generate(negative_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()
        st.write("Negative Comments Word Cloud")

    # Display sample comments
    st.subheader("Sample Comments")
    st.write("### Top Positive Comments")
    for i, row in filtered_data[filtered_data["VADER_Predicted"] == 1].nlargest(5, 'text').iterrows():
        st.write(f"Text: {row['text']}")

    st.write("### Top Negative Comments")
    for i, row in filtered_data[filtered_data["VADER_Predicted"] == 0].nsmallest(5, 'text').iterrows():
        st.write(f"Text: {row['text']}")
