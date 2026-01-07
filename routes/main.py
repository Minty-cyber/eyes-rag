from fastapi import APIRouter
from routes.api import faq_rag, chat


api_router = APIRouter()


api_router.include_router(faq_rag.router, prefix="/faq", tags=["FAQ"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])



## Admin endpoints
# api_router.include_router(admin_view.router, prefix="/admin", tags="Admin Carts")
