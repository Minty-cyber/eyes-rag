from pydoc import text
from google.genai import types, Client
from core.config import settings
from .doc_ingester import EMBED_MODEL


from groq import Groq
import numpy as np
import faiss
import pickle

INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"


client = Groq(
    api_key= settings.GROQ_API_KEY
)

genai_client = Client(api_key=settings.GOOGLE_API_KEY)

# co = cohere.Client(settings.COHERE_API_KEY)
index = faiss.read_index(INDEX_FILE)
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

def embed_text(texts):
    resp = genai_client.models.embed_content(
        contents=texts,
        model=EMBED_MODEL,
        config=types.EmbedContentConfig(task_type='RETRIEVAL_QUERY')
    )
    vec = np.array(resp.embeddings[0].values, dtype="float32")
    return vec.reshape(1, -1)

def search_index(query, k=10):
    q_vec = embed_text(texts=[query])
    D, I = index.search(q_vec, k)
    return [chunks[i] for i in I[0]]

def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = (
        f"Answer the question based on the context provided. And more importantly, give all your responses in Markdown format."
        "If the question is not related to the context in any way, do NOT attempt to answer. "
        "Instead, strictly reply: 'My knowledge base does not have information about this.' "
        "When answering the questions, go straight to the point. Avoid doing for example: 'Answer: We offer training'. Just give 'We offer training'. NO 'Answer:' first before the answer itself "
        "If it is a normal greeting or some small pleasantries or friendlies, answer cordially and accordinly."
        "Do not try to generate any answer that relates to the context if the question is unrelated.\n\n"
        f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    )
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="openai/gpt-oss-120b",
    )
    return response.choices[0].message.content.strip() #type: ignore