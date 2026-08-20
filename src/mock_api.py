import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
import uvicorn

from config import (
    MOCK_API_HOST,
    MOCK_API_PORT,
    MOCK_HISTORICAL_RECORDS,
    MOCK_STREAMING_RECORDS,
)
from synthetic_data import generate_records, balanced_distribution

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HISTORICAL = []
STREAMING = []
RECEIVED = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global HISTORICAL, STREAMING
    logger.info("Generating mock data: %s historical, %s streaming", MOCK_HISTORICAL_RECORDS, MOCK_STREAMING_RECORDS)
    dist = balanced_distribution(benign_ratio=0.6)
    HISTORICAL = generate_records(MOCK_HISTORICAL_RECORDS, dist)
    STREAMING = generate_records(MOCK_STREAMING_RECORDS, dist)
    logger.info("Mock data ready")
    yield


app = FastAPI(title="MCDL Mock Data API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/records")
async def get_records(
    type: str = Query("streaming", enum=["historical", "streaming"]),
    offset: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
):
    source = HISTORICAL if type == "historical" else STREAMING
    end = min(offset + limit, len(source))
    return source[offset:end]


@app.post("/records")
async def post_records(request: Request):
    batch = await request.json()
    RECEIVED.extend(batch)
    logger.info("Received %s tagged records (total received: %s)", len(batch), len(RECEIVED))
    return {"accepted": len(batch), "total_received": len(RECEIVED)}


@app.get("/received")
async def received():
    return {"total_received": len(RECEIVED)}


if __name__ == "__main__":
    uvicorn.run(app, host=MOCK_API_HOST, port=MOCK_API_PORT)
