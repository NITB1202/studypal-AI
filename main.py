from fastapi import FastAPI
from routers.question import router as question_router

app = FastAPI(
    title="AI Planner Service",
    version="1.0",
)

app.include_router(
    question_router,
    prefix="/api",
    tags=["Question"]
)
