# main.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataset import get_sentences

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset and generate embeddings once
sentences = get_sentences()
sentence_embeddings = model.encode(sentences)

def search(query, top_k=3):
    # Embed the user query
    query_embedding = model.encode([query])
    
    # Compute cosine similarities
    similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
    
    # Get top K similar sentences
    top_indices = similarities.argsort()[::-1][:top_k]
    
    # Print top results with similarity scores
    print(f"\nTop {top_k} similar sentences:\n")
    for i in top_indices:
        score_percent = round(similarities[i] * 100, 2)
        print(f"✅ \"{sentences[i]}\" (Similarity: {score_percent}%)")

# Simple loop to accept multiple queries
if __name__ == "__main__":
    print("🔍 Simple Semantic Search (type 'exit' to quit)\n")
    while True:
        query = input("Enter your query: ")
        if query.lower() == 'exit':
            break
        search(query)
