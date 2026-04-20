"""
FastAPI application.

Point d'entrée de l'API. Charge le modèle au startup,
expose les routes REST et WebSocket.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import stream, transcribe
from api.schemas import HealthResponse
from src.audio.preprocessor import Preprocessor
from src.config import load_config
from src.transcription.postprocessor import Postprocessor
from src.transcription.transcriber import Transcriber

logger = logging.getLogger(__name__)

# État global
_transcriber: Transcriber | None = None
_cfg = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au startup, libère au shutdown."""
    global _transcriber, _cfg

    _cfg = load_config()

    # Logging
    logging.basicConfig(
        level=getattr(logging, _cfg.logging.level),
        format=_cfg.logging.format,
    )

    logger.info("Starting STT API...")

    # Init modèle
    _transcriber = Transcriber(
        model_size=_cfg.transcription.model_size,
        language=_cfg.transcription.language,
        device=_cfg.transcription.device,
        fp16=_cfg.transcription.fp16,
        beam_size=_cfg.transcription.beam_size,
        temperature=_cfg.transcription.temperature,
    )

    preprocessor = Preprocessor(
        vad_enabled=_cfg.preprocessing.vad_enabled,
        vad_threshold=_cfg.preprocessing.vad_threshold,
        normalize=_cfg.preprocessing.normalize,
        target_db=_cfg.preprocessing.target_db,
    )

    postprocessor = Postprocessor(
        remove_filler_words=_cfg.postprocessing.remove_filler_words,
        filler_words=set(_cfg.postprocessing.filler_words),
    )

    # Injection dans les routes
    transcribe.init_route(_transcriber, preprocessor, postprocessor)
    stream.init_route(_transcriber, postprocessor, _cfg.audio.sample_rate)

    logger.info("API ready — model=%s device=%s", _transcriber.model_size, _transcriber.device)
    yield

    logger.info("Shutting down STT API")


app = FastAPI(
    title="Speech-to-Text API",
    description="Transcription audio temps réel avec Whisper",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(transcribe.router)
app.include_router(stream.router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Vérifie que l'API et le modèle sont opérationnels."""
    if _transcriber is None:
        return HealthResponse(status="loading", model="none", device="none")
    return HealthResponse(
        status="ok",
        model=f"whisper-{_transcriber.model_size}",
        device=_transcriber.device,
    )
