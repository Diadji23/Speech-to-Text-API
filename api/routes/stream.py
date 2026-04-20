"""
Route WebSocket pour la transcription streaming.

WS /ws/stream : reçoit des chunks audio binaires,
retourne la transcription en temps réel.
"""
import logging
import tempfile
import wave
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Streaming"])

_transcriber = None
_postprocessor = None
_sample_rate = 16000


def init_route(transcriber, postprocessor, sample_rate: int = 16000):
    """Injecte les dépendances."""
    global _transcriber, _postprocessor, _sample_rate
    _transcriber = transcriber
    _postprocessor = postprocessor
    _sample_rate = sample_rate


@router.websocket("/ws/stream")
async def stream_transcription(ws: WebSocket):
    """
    WebSocket streaming : reçoit des chunks audio binaires PCM 16-bit,
    transcrit chaque chunk et renvoie le texte en JSON.

    Protocole :
        Client → Server : bytes (PCM 16-bit mono, 16kHz)
        Server → Client : {"text": "...", "is_final": bool}
        Client → Server : "END" (texte) pour terminer
    """
    await ws.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await ws.receive()

            # Message texte = commande
            if "text" in data:
                if data["text"].upper() == "END":
                    logger.info("Client sent END, closing")
                    await ws.close()
                    break
                continue

            # Message binaire = audio chunk
            if "bytes" in data:
                audio_bytes = data["bytes"]

                # Sauvegarder en wav temporaire pour Whisper
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    with wave.open(tmp.name, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(_sample_rate)
                        wf.writeframes(audio_bytes)
                    tmp_path = tmp.name

                try:
                    result = _transcriber.transcribe(tmp_path)
                    text = result.text

                    if _postprocessor:
                        text = _postprocessor.process(text)

                    if text.strip():
                        await ws.send_json({
                            "text": text,
                            "is_final": False,
                        })

                finally:
                    Path(tmp_path).unlink(missing_ok=True)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        await ws.close(code=1011, reason=str(e))
