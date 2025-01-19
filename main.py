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

# Save outputs to a text file
with open("transcription_output.txt", "w", encoding="utf-8") as f:
    # Save the text
    f.write(outputs["text"])
    f.write("\n\n# Timestamps:\n")
    # Save the chunks with timestamps
    for chunk in outputs["chunks"]:
        f.write(
            f"[{chunk['timestamp'][0]:.2f}s -> {chunk['timestamp'][1]:.2f}s] {chunk['text']}\n"
        )

print("Output saved to transcription_output.txt")
