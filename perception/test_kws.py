#!/usr/bin/env python3
"""
test_kws.py — Quick test for sherpa-onnx KWS keyword spotting.

Usage (run inside perception container):
  python test_kws.py                    # uses default model dir and keyword
  python test_kws.py --mic              # test with live microphone input
  python test_kws.py --wav test.wav     # test with a WAV file

This will:
1. Load the KWS model
2. Read audio (mic or file) and feed to KeywordSpotter
3. Print when keyword is detected
"""

import argparse
import glob
import os
import sys
import time

SAMPLE_RATE = 16000
MODEL_DIR = "/models/sherpa-onnx/kws"
KEYWORD = "x iǎo f àn x iǎo f àn @小范小范"


def find_model(prefix, model_dir, prefer_int8=True):
    pattern = os.path.join(model_dir, f"{prefix}-*.onnx")
    files = glob.glob(pattern)
    if not files:
        return ""
    chunk8 = [f for f in files if "chunk-8" in f]
    cands = chunk8 if chunk8 else files
    if prefer_int8:
        int8f = [f for f in cands if "int8" in f]
        if int8f:
            return int8f[0]
    else:
        fp32f = [f for f in cands if "int8" not in f]
        if fp32f:
            return fp32f[0]
    return cands[0]


def main():
    parser = argparse.ArgumentParser(description="Test KWS keyword spotting")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--keyword", default=KEYWORD)
    parser.add_argument("--mic", action="store_true", help="Use live microphone")
    parser.add_argument("--wav", type=str, help="Path to WAV file to test")
    parser.add_argument("--score", type=float, default=1.5, help="keywords_score")
    parser.add_argument("--threshold", type=float, default=0.1, help="keywords_threshold")
    args = parser.parse_args()

    import sherpa_onnx

    model_dir = args.model_dir
    encoder = find_model("encoder", model_dir, prefer_int8=True)
    decoder = find_model("decoder", model_dir, prefer_int8=False)
    joiner = find_model("joiner", model_dir, prefer_int8=True)
    tokens = os.path.join(model_dir, "tokens.txt")

    print(f"encoder: {encoder}")
    print(f"decoder: {decoder}")
    print(f"joiner:  {joiner}")
    print(f"tokens:  {tokens}")

    if not all([encoder, decoder, joiner, os.path.exists(tokens)]):
        print("ERROR: Model files not found!")
        sys.exit(1)

    # Write keywords file
    kws_file = os.path.join(model_dir, "keywords_test.txt")
    with open(kws_file, 'w', encoding='utf-8') as f:
        f.write(f"{args.keyword}\n")
    print(f"keywords_file: {kws_file}")
    print(f"keyword: {args.keyword}")
    print(f"score={args.score}, threshold={args.threshold}")
    print()

    spotter = sherpa_onnx.KeywordSpotter(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        keywords_file=kws_file,
        num_threads=2,
        provider="cpu",
        keywords_score=args.score,
        keywords_threshold=args.threshold,
    )
    stream = spotter.create_stream()
    print("KWS initialized OK!")

    if args.wav:
        # Test with WAV file
        import wave
        print(f"\nTesting with WAV file: {args.wav}")
        with wave.open(args.wav, 'rb') as wf:
            assert wf.getframerate() == SAMPLE_RATE, f"Expected 16kHz, got {wf.getframerate()}"
            assert wf.getsampwidth() == 2, f"Expected 16-bit, got {wf.getsampwidth()*8}-bit"
            assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"

            chunk_size = 3200  # 200ms at 16kHz
            import struct
            detected = 0
            frames_fed = 0
            while True:
                data = wf.readframes(chunk_size)
                if not data:
                    break
                n = len(data) // 2
                samples = struct.unpack(f'<{n}h', data)
                float_samples = [s / 32768.0 for s in samples]
                stream.accept_waveform(SAMPLE_RATE, float_samples)
                frames_fed += n

                while spotter.is_ready(stream):
                    spotter.decode_stream(stream)

                result = spotter.get_result(stream)
                kw = result.keyword if hasattr(result, 'keyword') else str(result)
                if kw and kw.strip():
                    t = frames_fed / SAMPLE_RATE
                    print(f"  [DETECTED] keyword='{kw.strip()}' at t={t:.2f}s")
                    detected += 1
                    stream = spotter.create_stream()

            print(f"\nDone. Detected {detected} keyword(s) in {frames_fed/SAMPLE_RATE:.1f}s of audio.")

    elif args.mic:
        # Live microphone test
        try:
            import sounddevice as sd
        except ImportError:
            print("ERROR: sounddevice not installed. pip install sounddevice")
            sys.exit(1)

        print("\nListening on microphone... Say the wake word! (Ctrl+C to stop)")
        print("-" * 50)

        import struct
        chunk_duration = 0.2  # 200ms chunks
        chunk_size = int(SAMPLE_RATE * chunk_duration)

        def audio_callback(indata, frames, time_info, status):
            nonlocal stream
            if status:
                print(f"  [status] {status}")
            samples = indata[:, 0].tolist()  # mono
            stream.accept_waveform(SAMPLE_RATE, samples)
            while spotter.is_ready(stream):
                spotter.decode_stream(stream)
            result = spotter.get_result(stream)
            kw = result.keyword if hasattr(result, 'keyword') else str(result)
            if kw and kw.strip():
                print(f"  >>> KEYWORD DETECTED: '{kw.strip()}' at {time.strftime('%H:%M:%S')}")
                stream = spotter.create_stream()

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                            blocksize=chunk_size, callback=audio_callback):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopped.")

    else:
        # Just test initialization
        print("\nKWS model loaded successfully. Use --mic or --wav to test detection.")
        print("Example:")
        print(f"  python {sys.argv[0]} --mic")
        print(f"  python {sys.argv[0]} --wav /path/to/audio.wav")


if __name__ == "__main__":
    main()
