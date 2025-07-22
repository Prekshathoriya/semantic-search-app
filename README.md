# 🔍 Semantic Search App

[![View in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://semantic-search-app-f89m8uraalfkzg5p7y9tyw.streamlit.app/)

An AI-powered semantic search engine that allows users to upload text documents and perform intelligent searches using sentence meaning — not just keywords.

Built with:
- 🧠 Sentence Transformers (BERT-based embeddings)
- 🚀 Streamlit for a simple, interactive UI
- 📄 Python for backend logic

---

## ⚙️ Features

- Upload `.txt` or `.pdf` files
- Enter any search query — not limited to keywords!
- Uses powerful **semantic similarity** for meaningful results
- View top relevant sentences from your documents
- Includes a clear/reset button
- Deployed live via Streamlit Cloud 🚀

---

## 📦 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Prekshathoriya/semantic-search-app.git
cd semantic-search-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
