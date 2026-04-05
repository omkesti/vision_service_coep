import os
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client
from dotenv import load_dotenv

from analysis_service import AnalysisService

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("main")

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

service: AnalysisService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    logger.info("Starting up — loading models...")
    service = AnalysisService()
    logger.info("Service ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    session_id: str
    camera_id: str
    stream_url: str | None = None
    lat: float
    lng: float


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    logger.info("Received: %s", req.model_dump())

    if not req.stream_url:
        raise HTTPException(status_code=400, detail="No stream URL provided")

    try:
        fusion = await asyncio.wait_for(
            service.analyze(req.session_id, req.camera_id, req.stream_url),
            timeout=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Analysis timed out")
    except RuntimeError as e:
        if "download_queue_full" in str(e):
            raise HTTPException(status_code=503, detail="Server busy, try again shortly")
        raise HTTPException(status_code=502, detail=f"Analysis failed: {e}")

    # ── Supabase insert ───────────────────────────────────────────────────────
    risk_out = fusion.risk_model_output or {}
    cls_out  = fusion.classifier_output or {}

    try:
        db_response = supabase.table("incidents").insert({
            "camera_id":     req.camera_id,
            "incident_type": fusion.incident_type,
            "risk_score":    fusion.risk_score,
            "confidence":    fusion.confidence,
            "lat":           req.lat,
            "lng":           req.lng,
            "snapshot_url":  None,
        }).execute()
        logger.info("[%s] Supabase insert OK | id=%s",
                    req.session_id,
                    db_response.data[0].get("id") if db_response.data else "?")
    except Exception as e:
        logger.error("[%s] Supabase insert failed: %s", req.session_id, e)
        raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")

    inserted = db_response.data[0] if db_response.data else {}

    logger.info("[%s] FINAL RESPONSE | incident_type=%s risk_score=%s confidence=%s decision_source=%s",
                req.session_id, fusion.incident_type, fusion.risk_score,
                fusion.confidence, fusion.decision_source)

    return {
        "id":              inserted.get("id"),
        "camera_id":       req.camera_id,
        "incident_type":   fusion.incident_type,
        "risk_score":      fusion.risk_score,
        "confidence":      fusion.confidence,
        "decision_source": fusion.decision_source,
        "lat":             req.lat,
        "lng":             req.lng,
        "snapshot_url":    None,
    }
