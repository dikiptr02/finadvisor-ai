from fastapi import FastAPI

from app.routers import health, retrieve

app = FastAPI(title="RAG Service", version="0.1.0")

app.include_router(health.router)
app.include_router(retrieve.router)