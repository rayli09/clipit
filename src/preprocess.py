def convert_to_mp3(video_path: str, audio_path: str) -> None:
    """Convert video to MP3"""
    from pathlib import Path
    import subprocess
    import os

    video_path = Path(video_path)
    audio_path = Path(audio_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if audio_path.exists():
        # remove existing audio file
        os.remove(audio_path)

    cmd = f'ffmpeg -i "{video_path}" "{audio_path}"'
    subprocess.run(cmd, check=True, capture_output=True, shell=True)


def transcribe_audio(audio_path: str):
    import whisper
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = "large-v3-turbo"

    model = whisper.load_model(model, device=device)
    transcript = model.transcribe(word_timestamps=True, audio=audio_path)
    for segment in transcript["segments"]:
        print(
            "".join(
                f"{word['word']}[{word['start']}/{word['end']}]"
                for word in segment["words"]
            )
        )

    pass


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    video_path = "videos/index_arb.mov"
    audio_path = "videos/index_arb.mp3"
    # 1. to mp3
    convert_to_mp3(video_path, audio_path)
    # 2. transcribe
    transcribe_audio(audio_path)
