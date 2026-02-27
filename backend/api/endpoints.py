import logging
from fastapi import APIRouter, HTTPException
from backend.models.schemas import ChatRequest, ChatResponse
from backend.services.agent import process_query

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Accept a natural-language question about the Titanic dataset and return
    the agent's answer, optionally with a URL to a generated visualisation.
    """
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    logger.info("Received query: %s", request.message[:120])

    try:
        result = process_query(request.message)
    except Exception as exc:
        logger.exception("Unexpected error in process_query")
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc

    logger.info(
        "Query answered. has_image=%s",
        result.get("image_url") is not None,
    )
    return ChatResponse(
        response=result["response"],
        image_url=result.get("image_url"),
    )
