import os
import pickle
import numpy as np
from PyPDF2 import PdfReader
import faiss
from google import genai
from google.genai import types
from core.config import settings


# Config
GOOGLE_API_KEY = settings.GOOGLE_API_KEY
client = genai.Client(api_key=GOOGLE_API_KEY)


EMBED_MODEL = "models/text-embedding-004"
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"

def load_pdf(file_path):
    pdf = PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_texts(text_list):
    """Embed texts using Google's text-embedding-004 model"""
    all_vectors = []
    
    # Google API can handle batches, but we'll process one at a time for stability
    for text in text_list:
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        all_vectors.append(result.embeddings[0].values)
    
    return np.array(all_vectors, dtype="float32")

def build_index(pdf_folder="docs"):
    all_chunks = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"Processing: {filename}")
            text = load_pdf(os.path.join(pdf_folder, filename))
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
    
    # Embedding chunks with Google's text-embedding-004
    print("Embedding chunks with Google Generative AI...")
    vectors = embed_texts(all_chunks)
    
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors) #type: ignore
    
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)
    
    # print("✅ Indexing complete! Run `streamlit run app.py` to chat.")

if __name__ == "__main__":
    build_index()