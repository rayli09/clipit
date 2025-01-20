import modal
import pathlib
from typing import Optional
import time
import json

# Constants
MODEL_DIR = "/model"
MODEL_NAME = "openai/whisper-large-v3-turbo"
CACHE_DIR = "/cache"

# Set up Modal volume for storing models and temporary files
model_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
storage_volume = modal.NetworkFileSystem.from_name(
    "transcription-storage", create_if_missing=True
)

# Create the Modal image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "huggingface_hub==0.27.0",
        "librosa==0.10.2",
        "soundfile==0.12.1",
        "accelerate==1.2.1",
        "fastapi",
        "python-multipart",
        "ffmpeg-python",
    )
    .apt_install("ffmpeg")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": MODEL_DIR})
)

app = modal.App("transcription-api", image=image)


@app.cls(
    gpu="a10g",
    network_file_systems={CACHE_DIR: storage_volume},
    volumes={MODEL_DIR: model_cache},
)
class TranscriptionModel:
    def __init__(self):
        self.initialized = False

    @modal.enter()
    def initialize(self):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to("cuda")

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.float16,
            device="cuda",
        )
        self.initialized = True

    def process_audio(self, audio_path: str) -> dict:
        import librosa

        # Load and resample audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Transcribe
        result = self.pipeline(
            audio,
            batch_size=8,
            return_timestamps=True,
            # generate_kwargs={"language": "en"},
        )

        return result


@app.function(network_file_systems={CACHE_DIR: storage_volume})
@modal.web_endpoint(method="POST")
async def transcribe(audio_file: modal.File):
    """
    Endpoint for transcribing audio files.
    Accepts multipart form data with an audio file.
    Returns JSON with transcription results.
    """
    import tempfile
    import os

    # Create a temporary file to store the uploaded audio
    with tempfile.NamedTemporaryFile(
        suffix=".mp3", dir=CACHE_DIR, delete=False
    ) as temp_file:
        temp_path = temp_file.name
        with open(temp_path, "wb") as f:
            f.write(audio_file.read())

    try:
        # Initialize model and transcribe
        model = TranscriptionModel()
        result = model.process_audio(temp_path)

        return {
            "status": "success",
            "transcription": result["text"],
            "segments": result["chunks"] if "chunks" in result else [],
            "processing_time": time.time(),
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.function()
@modal.web_endpoint(method="GET")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}
