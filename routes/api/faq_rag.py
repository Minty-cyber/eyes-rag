from fastapi import APIRouter, HTTPException
from utils.rag import search_index, generate_answer
from pydantic import BaseModel


class Query(BaseModel):
    question: str
    k: int = 10


router = APIRouter()


@router.get("/")
def home():
    return {"message": "This is the FAQ chat page"}


@router.post("/chat")
def chat(query: Query):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    retrieved_chunks = search_index(query.question, query.k)
    answer = generate_answer(query.question, retrieved_chunks)

    return {
        "question": query.question,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
    }
