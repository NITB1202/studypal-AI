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

@app.get("/awake", tags=["Health"])
async def awake_service():
    """
    Simple endpoint to keep the service awake.
    Can be pinged by cronjob or uptime monitoring.
    """
    return {"status": "alive"}
