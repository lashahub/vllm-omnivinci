from pathlib import Path

from transformers import AutoProcessor

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_video


MODEL_ID = "SreyanG-NVIDIA/omnivinci-hf"
ROOT_DIR = Path(__file__).resolve().parents[1]
VIDEO_PATH = (ROOT_DIR / "transformers" / "nvidia.mp4").resolve()
VIDEO_URL = f"file://{VIDEO_PATH}"
PROMPT_TEXT = "Assess the video, followed by a detailed description of it's video and audio contents."
GPU_MEMORY_UTILIZATION = 0.65


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_ID, padding_side="left", use_fast=False)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(VIDEO_PATH)},
                {
                    "type": "text",
                    "text": PROMPT_TEXT,
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        trust_remote_code=False,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        hf_overrides={
            "load_audio_in_video": True,
            "num_video_frames": 128,
            "audio_chunk_length": "max_3600",
        },
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
    )

    video_data = fetch_video(VIDEO_URL)

    outputs = llm.generate(
        [
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "video": [video_data],
                },
            }
        ],
        sampling_params=sampling_params,
    )

    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
