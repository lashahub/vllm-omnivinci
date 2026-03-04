from pathlib import Path

from transformers import AutoProcessor

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_video
from vllm.sampling_params import RepetitionDetectionParams


MODEL_ID = "SreyanG-NVIDIA/omnivinci-hf"
ROOT_DIR = Path(__file__).resolve().parents[1]
VIDEO_PATH = (ROOT_DIR / "transformers" / "nvidia.mp4").resolve()
VIDEO_URL = f"file://{VIDEO_PATH}"
PROMPT_TEXT = "Assess the video, followed by a detailed description of its video and audio contents."
GPU_MEMORY_UTILIZATION = 0.50
NUM_VIDEO_FRAMES = 128


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
    eos_token_id = processor.tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer is missing `eos_token_id`; cannot enforce EOS stopping.")

    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        trust_remote_code=False,
        tokenizer_mode="slow",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        hf_overrides={
            "load_audio_in_video": True,
            "num_video_frames": NUM_VIDEO_FRAMES,
            "audio_chunk_length": "max_3600",
        },
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        ignore_eos=False,
        stop_token_ids=[int(eos_token_id)],
        repetition_detection=RepetitionDetectionParams(
            max_pattern_size=64,
            min_pattern_size=16,
            min_count=3,
        ),
    )

    video_data = fetch_video(
        VIDEO_URL,
        video_io_kwargs={
            "num_frames": NUM_VIDEO_FRAMES,
            "video_backend": "opencv",
            "frame_recovery": True,
        },
    )

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

    output = outputs[0].outputs[0]
    print(output.text)
    print(f"\n[finish_reason={output.finish_reason}]")


if __name__ == "__main__":
    main()
