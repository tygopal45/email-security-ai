# app/app.py
"""
FastAPI application entry point.

This wires everything together: it creates the app, warms up the models on
startup, configures CORS, and exposes the routes. The heavy lifting lives in
the pipeline and model modules — this file is just the "front door".
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# The /analyze route lives here.
from app.routes.analyze import router as analyze_router

# Imported so we can warm the model singletons on startup.
import app.pipelines.security_pipeline as security_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("email-security-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook.

    Everything before `yield` runs once at startup; everything after runs at
    shutdown. We load the models here so the first real user doesn't have to
    wait several seconds while they download/initialize.
    """
    logger.info("Starting Email Security AI — warming models (may take time on first run).")
    try:
        security_pipeline.load_models()
        logger.info("Model singletons initialized and ready for inference.")
    except Exception as e:
        # Don't crash the whole app if warm-up fails — load_models() will retry
        # lazily on the first request instead.
        logger.exception("Error while warming models: %s", e)
    yield
    # (Nothing to clean up on shutdown for now.)


app = FastAPI(
    title="Email Security AI",
    version="1.0",
    description="Model-1 -> Model-2 -> RAG -> Model-3 email security pipeline",
    lifespan=lifespan,
)

# CORS: allow any origin so a browser front-end can call us during development.
# Note: the spec forbids wildcard origins together with credentials, so
# credentials must stay False here. Tighten `allow_origins` for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="")


@app.get("/", tags=["health"])
def root():
    """Simple landing endpoint — also handy as a quick 'is it up?' check."""
    return {"status": "ok", "service": "email-security-ai", "docs": "/docs"}


@app.get("/health", tags=["health"])
def health():
    """Health check for load balancers / uptime monitors."""
    return {"status": "ok", "ready": True}
