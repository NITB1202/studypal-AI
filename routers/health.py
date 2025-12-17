from fastapi import APIRouter

router = APIRouter()

@router.get("/awake")
async def awake_service():
    return {"status": "alive"}

