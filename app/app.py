# app/main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# import the router (assumes app/routes/analyze.py exists)
from app.routes.analyze import router as analyze_router

# import pipeline to warm up singletons on startup
import app.pipelines.security_pipeline as security_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("email-security-ai")

app = FastAPI(
    title="Email Security AI",
    version="1.0",
    description="Model-1 -> Model-2 -> RAG -> Model-3 email security pipeline"
)

# Allow frontend access during development; tighten in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="")

@app.on_event("startup")
async def startup_event():
    """
    Ensure model singletons are created and log basic info.
    Note: model initialization (importing security_pipeline) may trigger HF model downloads
    on first run — that is expected.
    """
    logger.info("Starting Email Security AI — warming models (may take time on first run).")
    try:
        # Load the newly decoupled pipeline models to memory before serving
        security_pipeline.load_models()
        logger.info("Model singletons initialized and ready for inference.")
    except Exception as e:
        logger.exception("Error while warming models: %s", e)

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "ready": True}