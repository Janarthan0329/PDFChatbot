# app.py
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pdfplumber
import google.genai as genai
from google.genai import types

# ==========================
# 1. Setup Gemini client
# ==========================
# Use Streamlit Secrets instead of Kaggle Secrets
google_api_key = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=google_api_key)

# ==========================
# 2. Functions
# ==========================
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Simple Vector Store
class SimpleVectorStore:
    def __init__(self, embeddings, chunks):
        self.embeddings = np.asarray(embeddings)
        self.chunks = list(chunks)

    def query(self, query_embedding, top_k=5):
        if self.embeddings.shape[0] == 0:
            return []
        scores = cosine_similarity([query_embedding], self.embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

# Generate LLM answer using Gemini
def generate_answer_gemini(question, top_k_results):
    context = "\n\n".join([chunk for chunk, _ in top_k_results])
    content = f"Here is some context from a document:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
    config = types.GenerateContentConfig(
        temperature=0.3,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[content],
        config=config
    )
    # Return plain text for easy display
    return response.candidates[0].content.parts[0].text

# ==========================
# 3. Streamlit UI
# ==========================
st.title("📄 PDF → Gemini Knowledge Extraction Agent")
st.write("Upload a PDF, ask a question, and get structured answers from Gemini LLM.")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

# Question input
question = st.text_input("Enter your question:")

# Load embedding model once
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# ==========================
# 4. Agent Logic
# ==========================
if uploaded_file and question:
    with st.spinner("Processing PDF and generating answer..."):
        # Extract text
        pdf_text = extract_text_from_pdf(uploaded_file)

        if not pdf_text.strip():
            st.error("PDF is empty or could not be read.")
        else:
            # Split into chunks
            sentences = [s.strip() for s in pdf_text.split("\n") if s.strip()]

            # Create embeddings and vector store
            doc_embeddings = model.encode(sentences, show_progress_bar=False)
            vector_store = SimpleVectorStore(doc_embeddings, sentences)

            # Query vector store
            query_embedding = model.encode([question])
            top_k_results = vector_store.query(query_embedding, top_k=5)

            # Generate answer via Gemini
            gemini_answer_text = generate_answer_gemini(question, top_k_results)

            # Display answer
            st.markdown("**Gemini Answer:**")
            st.markdown(gemini_answer_text)
