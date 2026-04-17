# Speech-to-Text Pipeline

Pipeline de transcription audio temps réel avec **Whisper** (OpenAI), exposé via une **API FastAPI** avec support REST et WebSocket.

## Architecture

```
Audio (fichier / micro / stream)
        │
        ▼
┌─────────────────────┐
│   Preprocessor      │  ← VAD + normalisation volume
│   (Voice Activity   │
│    Detection)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Transcriber       │  ← Whisper (tiny → large)
│   (Whisper)         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Postprocessor     │  ← Suppression fillers, ponctuation
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Evaluation        │  ← WER, CER, benchmark latence
└────────┬────────────┘
         │
         ▼
    FastAPI (REST + WebSocket)
```

## Quick Start


## Docker

## API Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/health` | Health check + info modèle |
| `POST` | `/transcribe` | Upload fichier audio → transcription |
| `WS` | `/ws/stream` | Streaming temps réel (chunks PCM) |

**Exemple POST /transcribe :**

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav"
```

```json
{
  "text": "Bonjour, comment ça va aujourd'hui ?",
  "language": "fr",
  "segments": [
    {"start": 0.0, "end": 2.1, "text": "Bonjour, comment ça va aujourd'hui ?"}
  ],
  "inference_time": 1.23,
  "audio_duration": 3.5,
  "realtime_factor": 0.35
}
```

## Benchmark

```bash
python main.py --benchmark audio.wav --reference "texte attendu"
```

Produit un rapport avec :
- **Latence** : moyenne, écart-type, min/max sur N runs
- **RTF** (Real-Time Factor) : < 1 = plus rapide que temps réel
- **WER/CER** : Word/Character Error Rate vs texte de référence

## Structure du projet

```
speech-to-text/
├── config/default.yaml          # Hyperparamètres centralisés
├── src/
│   ├── config.py                # Config loader (YAML + dot-notation)
│   ├── audio/
│   │   ├── capture.py           # Enregistrement micro (PyAudio)
│   │   ├── file_loader.py       # Chargement audio (ffmpeg)
│   │   └── preprocessor.py      # VAD + normalisation
│   ├── transcription/
│   │   ├── transcriber.py       # Whisper inference
│   │   └── postprocessor.py     # Nettoyage texte
│   └── evaluation/
│       ├── metrics.py           # WER, CER (Levenshtein)
│       └── benchmark.py         # Benchmark reproductible
├── api/
│   ├── main.py                  # FastAPI app
│   ├── schemas.py               # Pydantic models
│   └── routes/
│       ├── transcribe.py        # POST /transcribe
│       └── stream.py            # WebSocket streaming
├── tests/                       # 40 tests unitaires
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml     # CI : lint + tests
```

## Stack technique

| Composant | Technologie |
|-----------|------------|
| Modèle STT | Whisper (OpenAI) |
| Preprocessing | NumPy (VAD énergie RMS) |
| API | FastAPI + WebSocket |
| Validation | Pydantic v2 |
| Évaluation | WER/CER (Levenshtein from scratch) |
| Tests | pytest (40 tests) |
| CI/CD | GitHub Actions |
| Container | Docker |
| Config | YAML centralisé |

## Configuration

Tous les paramètres sont dans `config/default.yaml` :

```yaml
transcription:
  model_size: "base"     # tiny, base, small, medium, large
  language: "fr"         # null = auto-detect
  device: "auto"         # auto, cpu, cuda

preprocessing:
  vad_enabled: true
  vad_threshold: 0.5
  normalize: true
```

Override en CLI :

```bash
python main.py --file audio.wav --config config/custom.yaml
```

## Compétences démontrées

- **ML Pipeline** : preprocessing → inference → postprocessing → evaluation
- **Audio Processing** : VAD, normalisation, conversion formats (ffmpeg)
- **Transformer Models** : Whisper (encoder-decoder sur spectrogrammes)
- **API Design** : REST + WebSocket streaming, schemas Pydantic
- **Software Engineering** : architecture modulaire, config YAML, tests, CI/CD, Docker
- **Évaluation ML** : métriques standard (WER/CER), benchmark reproductible