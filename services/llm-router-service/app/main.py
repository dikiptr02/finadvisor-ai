from fastapi import FastAPI

from app.routers import health, generate


app = FastAPI(title="LLM Router Service", version="0.1.0")

app.include_router(health.router)
app.include_router(generate.router)