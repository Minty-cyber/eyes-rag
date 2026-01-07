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

# @router.get("/chat")
# async def home():
#     return {"message": "This is the Beauty Chat API"}

@router.post("/agrichat")
async def agrichat(query: Query):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are an expert agricultural consultant with deep knowledge in crop production, soil science, and sustainable farming practices.

                You must answer **only questions related to agriculture**. If a question is outside the agricultural domain, politely decline and explain that you can only provide assistance on agricultural topics.

                Provide personalized agricultural advice while considering:
                - Crop types, growth stages, and local climate conditions
                - Soil health, fertilization, and irrigation methods
                - Pest, disease, and weed management strategies
                - Use of modern farming tools, technology, and best practices
                - Environmental sustainability and food safety considerations

                Always prioritize sustainable and safe farming practices and suggest consultation with agricultural extension officers or professionals when appropriate.
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

    
    
    
