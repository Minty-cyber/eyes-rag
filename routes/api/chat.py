from fastapi import APIRouter
from pydantic import BaseModel
from groq import Groq
from core.config import settings

client = Groq(
    api_key=settings.GROQ_API_KEY
)

router = APIRouter()

class Query(BaseModel):
    question: str

@router.get("/chat")
async def home():
    return {"message": "This is the Beauty Chat API"}

@router.post("/beauty")
async def agrichat(query: Query):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are an expert beauty consultant with deep knowledge in skincare, makeup, and haircare. 
                Provide personalized beauty advice while considering:
                - Individual skin types and concerns
                - Product recommendations and ingredients
                - Application techniques and best practices
                - Safety considerations and potential allergies
                - Latest beauty trends and scientific research
                
                Always prioritize safety and suggest professional consultation when appropriate.
                Be clear, specific, and explain the reasoning behind your recommendations."""
            },
            {
                "role": "user",
                "content": query.question
            }
        ],
        model="llama-3.3-70b-versatile",
        max_tokens=1000
        
    )
    
    return {
        "answer": chat_completion.choices[0].message.content,
        "status": "success"
    }

    
    
    
