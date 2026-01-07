from fastapi import FastAPI
from routes.main import api_router
from core.config import settings
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
          
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

