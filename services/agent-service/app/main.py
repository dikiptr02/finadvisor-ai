from fastapi import FastAPI

from app.routers import agent, health

app = FastAPI(title="Agent Service", version="0.1.0")

app.include_router(agent.router)
app.include_router(health.router)