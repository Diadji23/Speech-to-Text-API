FROM python:3.11-slim

WORKDIR /app

# FFmpeg pour le chargement audio
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg portaudio19-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py", "--serve"]
