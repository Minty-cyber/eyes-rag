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
EMBED_MODEL = "gemini-embedding-001"  # Remove "models/" prefix
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
    
    # Process in batches for efficiency
    batch_size = 100
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        print(f"Embedding batch {i//batch_size + 1}/{(len(text_list)-1)//batch_size + 1}")
        
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        
        # Extract embeddings from batch result
        for embedding in result.embeddings:
            all_vectors.append(embedding.values)
    
    return np.array(all_vectors, dtype="float32")

def build_index(pdf_folder="docs"):
    all_chunks = []
    
    if not os.path.exists(pdf_folder):
        print(f"❌ Folder '{pdf_folder}' not found. Creating it...")
        os.makedirs(pdf_folder)
        print(f"Please add PDF files to the '{pdf_folder}' folder and run again.")
        return
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"❌ No PDF files found in '{pdf_folder}' folder.")
        return
    
    for filename in pdf_files:
        print(f"Processing: {filename}")
        text = load_pdf(os.path.join(pdf_folder, filename))
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    
    print(f"Total chunks created: {len(all_chunks)}")
    
    # Embedding chunks with Google's text-embedding-004
    print("Embedding chunks with Google Generative AI...")
    vectors = embed_texts(all_chunks)
    
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)  # type: ignore
    
    faiss.write_index(index, INDEX_FILE)
    
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)
    
    print(f"✅ Indexing complete!")
    print(f"   - Index saved to: {INDEX_FILE}")
    print(f"   - Chunks saved to: {CHUNKS_FILE}")
    print(f"   - Total vectors: {len(all_chunks)}")
    print(f"   - Vector dimension: {dim}")

if __name__ == "__main__":
    build_index()