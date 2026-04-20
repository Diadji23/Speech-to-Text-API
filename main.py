"""
Point d'entrée principal.

Permet de lancer le pipeline STT en CLI :
    python main.py --file audio.wav
    python main.py --record 5
    python main.py --serve
"""
import argparse
import logging
import sys

from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Speech-to-Text Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Transcribe an audio file")
    group.add_argument("--record", type=float, help="Record N seconds and transcribe")
    group.add_argument("--serve", action="store_true", help="Start FastAPI server")
    group.add_argument("--benchmark", type=str, help="Benchmark on an audio file")

    parser.add_argument("--reference", type=str, help="Reference text for WER/CER")

    args = parser.parse_args()
    cfg = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format,
    )

    if args.serve:
        import uvicorn
        uvicorn.run(
            "api.main:app",
            host=cfg.api.host,
            port=cfg.api.port,
            reload=False,
        )

    elif args.file:
        from src.audio.file_loader import load_audio
        from src.audio.preprocessor import Preprocessor
        from src.transcription.postprocessor import Postprocessor
        from src.transcription.transcriber import Transcriber

        audio, sr = load_audio(args.file, target_sr=cfg.audio.sample_rate)

        preprocessor = Preprocessor(
            vad_enabled=cfg.preprocessing.vad_enabled,
            vad_threshold=cfg.preprocessing.vad_threshold,
            normalize=cfg.preprocessing.normalize,
            target_db=cfg.preprocessing.target_db,
        )
        audio = preprocessor.process(audio, sr)

        transcriber = Transcriber(
            model_size=cfg.transcription.model_size,
            language=cfg.transcription.language,
            device=cfg.transcription.device,
        )
        result = transcriber.transcribe(audio, sr)

        postprocessor = Postprocessor(
            remove_filler_words=cfg.postprocessing.remove_filler_words,
            filler_words=set(cfg.postprocessing.filler_words),
        )
        text = postprocessor.process(result.text)

        print(f"\n{'=' * 60}")
        print(f"Transcription ({result.language}):")
        print(f"{'=' * 60}")
        print(text)
        print(f"{'=' * 60}")
        print(f"Duration: {result.audio_duration:.2f}s | "
              f"Inference: {result.inference_time:.2f}s | "
              f"RTF: {result.realtime_factor:.2f}x")

    elif args.record:
        from src.audio.capture import AudioCapture
        from src.audio.preprocessor import Preprocessor
        from src.transcription.postprocessor import Postprocessor
        from src.transcription.transcriber import Transcriber

        capture = AudioCapture(sample_rate=cfg.audio.sample_rate)
        audio = capture.record(duration=args.record)
        capture.close()

        preprocessor = Preprocessor(
            vad_enabled=cfg.preprocessing.vad_enabled,
            normalize=cfg.preprocessing.normalize,
        )
        audio = preprocessor.process(audio, cfg.audio.sample_rate)

        transcriber = Transcriber(
            model_size=cfg.transcription.model_size,
            language=cfg.transcription.language,
            device=cfg.transcription.device,
        )
        result = transcriber.transcribe(audio, cfg.audio.sample_rate)

        postprocessor = Postprocessor(
            remove_filler_words=cfg.postprocessing.remove_filler_words,
        )
        print(f"\nTranscription: {postprocessor.process(result.text)}")

    elif args.benchmark:
        from src.audio.file_loader import load_audio
        from src.evaluation.benchmark import Benchmark
        from src.transcription.transcriber import Transcriber

        audio, sr = load_audio(args.benchmark, target_sr=cfg.audio.sample_rate)
        transcriber = Transcriber(
            model_size=cfg.transcription.model_size,
            language=cfg.transcription.language,
            device=cfg.transcription.device,
        )
        bench = Benchmark(transcriber, num_runs=cfg.evaluation.num_benchmark_runs)
        result = bench.run(audio, sr, reference=args.reference)
        print(f"\n{result.summary()}")
        bench.save_report(result, "results/benchmark.json")


if __name__ == "__main__":
    main()
