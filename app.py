# app.py

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dataset import get_sentences
import matplotlib.pyplot as plt

# Load model & dataset
model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = get_sentences()
sentence_embeddings = model.encode(sentences)

# Streamlit UI
st.set_page_config(page_title="Semantic Search", layout="centered")
st.title("🔍 Simple Semantic Search App")
st.write("Enter a sentence to find the most similar entries from the dataset.")

# Text input
query = st.text_input("💬 Type your query here and press Enter:")

# Check input
if query.strip() != "":
    # Get query embedding
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]

    # Get top results
    top_k = 3
    top_indices = similarities.argsort()[::-1][:top_k]
    top_sentences = [sentences[i] for i in top_indices]
    top_scores = [round(similarities[i] * 100, 2) for i in top_indices]

    # Show results
    st.subheader("Top Matches:")
    for sent, score in zip(top_sentences, top_scores):
        st.write(f"✅ **{sent}** — *Similarity: {score}%*")

    # Show bar chart
    st.subheader("📊 Similarity Scores")
    fig, ax = plt.subplots()
    ax.barh(top_sentences[::-1], top_scores[::-1], color='skyblue')
    ax.set_xlabel("Similarity (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Top 3 Semantic Matches")
    st.pyplot(fig)

else:
    st.info("⬆️ Enter a sentence to see matching results.")
