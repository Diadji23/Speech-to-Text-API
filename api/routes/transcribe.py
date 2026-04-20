"""
Route de transcription.

POST /transcribe : upload un fichier audio, retourne la transcription.
"""
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import TranscriptionResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Transcription"])

# Le transcriber et preprocessor sont injectés au startup
_transcriber = None
_preprocessor = None
_postprocessor = None


def init_route(transcriber, preprocessor, postprocessor):
    """Injecte les dépendances (appelé au startup de l'app)."""
    global _transcriber, _preprocessor, _postprocessor
    _transcriber = transcriber
    _preprocessor = preprocessor
    _postprocessor = postprocessor


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcrit un fichier audio uploadé.

    Accepte : wav, mp3, flac, ogg, m4a, webm.
    Retourne le texte transcrit avec timestamps et métriques.
    """
    if _transcriber is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Validation du fichier
    allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
    suffix = Path(file.filename or "upload.wav").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {suffix}. Allowed: {allowed}",
        )

    # Sauvegarde temporaire
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info("Received file: %s (%d bytes)", file.filename, len(content))

        # Transcription (Whisper gère le chargement du fichier)
        result = _transcriber.transcribe(tmp_path)

        # Postprocessing
        if _postprocessor:
            result.text = _postprocessor.process(result.text)
            result.segments = _postprocessor.process_segments(result.segments)

        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            segments=[
                {"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in result.segments
            ],
            inference_time=result.inference_time,
            audio_duration=result.audio_duration,
            realtime_factor=result.realtime_factor,
        )

    except Exception as e:
        logger.error("Transcription failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        Path(tmp_path).unlink(missing_ok=True)
