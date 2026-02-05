from fastapi import FastAPI
from routes.main import api_router
from core.config import settings
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://eyes-gh.vercel.app",
        "https://eyes-gh-client.vercel.app/",
        "https://eyesgh.com/",
        "https://eyes-rag-124669021559.africa-south1.run.app"
          
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

