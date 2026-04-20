"""
Schemas Pydantic pour l'API.

Définit les modèles de requête et réponse pour une
documentation OpenAPI propre et une validation automatique.
"""
from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    """Segment timestampé."""

    start: float = Field(description="Début du segment en secondes")
    end: float = Field(description="Fin du segment en secondes")
    text: str = Field(description="Texte transcrit du segment")


class TranscriptionResponse(BaseModel):
    """Réponse de transcription."""

    text: str = Field(description="Texte transcrit complet")
    language: str = Field(description="Langue détectée (code ISO)")
    segments: list[TranscriptionSegment] = Field(
        default_factory=list, description="Segments avec timestamps"
    )
    inference_time: float = Field(description="Temps d'inférence en secondes")
    audio_duration: float = Field(description="Durée audio en secondes")
    realtime_factor: float = Field(
        description="RTF = inference_time / audio_duration (<1 = temps réel)"
    )


class HealthResponse(BaseModel):
    """Réponse du health check."""

    status: str = "ok"
    model: str = Field(description="Modèle Whisper chargé")
    device: str = Field(description="Device utilisé (cpu/cuda)")


class ErrorResponse(BaseModel):
    """Réponse d'erreur."""

    detail: str = Field(description="Message d'erreur")
