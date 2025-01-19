import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",  # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
    torch_dtype=torch.float16,
    device="cuda:0",  # or mps for Mac devices
    model_kwargs={"attn_implementation": "flash_attention_2"}
    if is_flash_attn_2_available()
    else {"attn_implementation": "sdpa"},
)

outputs = pipe(
    "videos/index_arb.wav",
    chunk_length_s=30,
    batch_size=24,
    return_timestamps=True,
)

outfile = "output/transcription_output.txt"
# Save outputs to a text file
with open(outfile, "w", encoding="utf-8") as f:
    # Save the text
    f.write(outputs["text"])
    f.write("\n\n# Timestamps:\n")
    # Save the chunks with timestamps
    for chunk in outputs["chunks"]:
        f.write(
            f"[{chunk['timestamp'][0]:.2f}s -> {chunk['timestamp'][1]:.2f}s] {chunk['text']}\n"
        )

print("Output saved to {}".format(outfile))


def convert_to_mp3(input_file: str, output_file: str):
    """Convert a video file to an mp3 file."""
    import subprocess

    cmd = f"ffmpeg -i {input_file} -ab 128k -ac 1 -ar 16000 {output_file}"
    subprocess.run(cmd.split(), check=True)
    pass


def main():
    """Planning of the Clipit

    First, we have raw resources. In our case, we have the video files of the recordings.

    Goal: generate clips that are production ready for content creation with our CTAs + advertising.

    Plan:
    1. process the raw video, use ffmpeg to extract to .mp3 file. [done]
    2. use whisper to transcribe the .mp3 and obtain the timestamps + content. At this point, the content is mixed with ZH-CH and English.
    3. Next, use gpt-4o-mini to translate & align each segment to english.



    """
    pass
