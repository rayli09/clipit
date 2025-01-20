from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import os
import subprocess
import torch
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""

    pass


class IntroClip(BaseModel):
    """Represents an intro clip with its path and instruction"""

    video_path: str
    instruction: str


class RawClip(BaseModel):
    """Represents a segment of video with timestamps and summary"""

    start_ts: float
    end_ts: float
    summary: str


class SegmentedClips(BaseModel):
    """Container for multiple RawClip instances"""

    clips: List[RawClip]


class VideoProcessor:
    """Main class for handling video processing operations"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.openai_client = self._init_openai_client()

    @staticmethod
    def _init_openai_client() -> OpenAI:
        """Initialize OpenAI client with API key from environment"""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return OpenAI(api_key=api_key)

    def get_intro_clips(
        self, clips_folder: str = "videos/tony_clips"
    ) -> List[IntroClip]:
        """Get list of intro clips with their configurations"""
        configs = {
            0: {"instruction": "Hello. Anyways, ..."},
            1: {
                "instruction": "Hi friends, we're gonna talk about pronouns. Anyways, ..."
            },
            2: {
                "instruction": "I'm a trans man and I'm gay. So you're straight. Anyways, ..."
            },
            3: {
                "instruction": "I feel sorry for your transfriends. I don't have any trans friends."
            },
            4: {
                "instruction": "LGBTQ parent problems. If you have a D, you're a boy. Anyways, ..."
            },
        }

        clips_path = Path(clips_folder)
        if not clips_path.exists():
            raise FileNotFoundError(f"Clips folder not found: {clips_folder}")

        return [
            IntroClip(
                video_path=str(clips_path / f"{k}.mov"), instruction=v["instruction"]
            )
            for k, v in configs.items()
        ]

    def convert_to_mp3(self, input_file: str, output_file: str) -> None:
        """Convert video file to MP3 format"""
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Use -vn to skip video and -acodec libmp3lame for MP3 encoding
            cmd = f'ffmpeg -i "{input_file}" -vn -acodec libmp3lame "{output_file}"'
            subprocess.run(cmd, check=True, capture_output=True, shell=True)
        except subprocess.CalledProcessError as e:
            raise VideoProcessingError(
                f"Failed to convert video to MP3: {e.stderr.decode()}"
            )

    def transcribe_audio(self, audio_path: str) -> Tuple[dict, str]:
        """Transcribe audio using Whisper model"""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3-turbo",
                torch_dtype=torch.int8,
                device=device,
                model_kwargs={
                    "attn_implementation": "flash_attention_2"
                    if is_flash_attn_2_available()
                    else "sdpa"
                },
            )

            outputs = pipe(
                audio_path,
                chunk_length_s=30,
                batch_size=1,
                return_timestamps=True,
            )

            formatted_input = ""
            for chunk in outputs["chunks"]:
                start, end = chunk["timestamp"]
                formatted_input += f"{start},{end},{chunk['text']}\n"

            return outputs, formatted_input

        except Exception as e:
            raise VideoProcessingError(f"Transcription failed: {str(e)}")

    def split_transcript_into_clips(self, formatted_input: str) -> SegmentedClips:
        """Split transcript into video clips using GPT-4"""
        system_message = """
        You are an excellent video editor. You'll be given transcripts of a video, and your job is to segment it into 3-5 clips.
        Transcript is of the form:
        start, end, text
        e.g. 1147.66,1154.16, i think the cost is like five percent to twenty percent even right i think it's highly risky that you
        
        Rules:
        1. Each clip should be about a subset of specific topic. 
        2. Each clip HAS TO BE around 30-40 seconds. DO NOT make short clips < 10 seconds or > 50 seconds.
        3. Do not skip or include multiple clips. Each clip is a consecutive segment of the video.
        4. Add a brief summary for the clip
        5. Skip any clips with meaningless intro or buzz words.
        """

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": formatted_input},
                ],
                response_format=SegmentedClips,
            )

            parsed_response = completion.choices[0].message.parsed
            if not parsed_response:
                raise VideoProcessingError("Failed to parse GPT response")

            return parsed_response

        except Exception as e:
            raise VideoProcessingError(f"Failed to split transcript: {str(e)}")

    def edit_clips(self, video_path: str, clips: SegmentedClips) -> List[str]:
        """Extract clips from video based on timestamps"""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_files = []
        for i, clip in enumerate(clips.clips, 1):
            output_file = (
                self.output_dir / f"clip_{i}_{clip.start_ts:.2f}_{clip.end_ts:.2f}.mov"
            )
            try:
                cmd = f'ffmpeg -i "{video_path}" -ss {clip.start_ts} -to {clip.end_ts} -c copy "{output_file}"'
                subprocess.run(cmd, check=True, capture_output=True, shell=True)
                output_files.append(str(output_file))
                logger.info(f"Created clip {i}: {output_file}")

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create clip {i}: {e.stderr.decode()}")
                continue

        return output_files

    def process_video(self, video_path: str) -> List[str]:
        """Complete pipeline for processing a video"""
        video_path = Path(video_path)

        # Convert video to MP3
        audio_path = os.path.join(self.output_dir / f"{video_path}.mp3")
        logger.warning(f"Converting {video_path} to {audio_path}")
        self.convert_to_mp3(str(video_path), str(audio_path))

        # Transcribe audio
        _, formatted_input = self.transcribe_audio(str(audio_path))

        # Split into clips
        clips = self.split_transcript_into_clips(formatted_input)

        # Edit and save clips
        return self.edit_clips(str(video_path), clips)


def main():
    try:
        processor = VideoProcessor()
        video_path = "videos/index_arb.mov"
        output_files = processor.process_video(video_path)
        logger.info(f"Successfully created {len(output_files)} clips")

    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
