# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.omnivinci import OmniVinciConfig, OmniVinciProcessor
from transformers.models.omnivinci.media_encoder import (
    BasicImageEncoder,
    BasicSoundEncoder,
    TSPVideoEncoder,
)
from transformers.models.omnivinci.modeling_omnivinci import (
    MultimodalProjector,
    Qwen2AudioTower,
    SiglipVisionTowerDynamicS2,
    SoundMultimodalProjector,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors

from .interfaces import MultiModalEmbeddings, SupportsLoRA, SupportsMultiModal, SupportsPP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

_AUDIO_CHUNK_UNIT_LENGTH = 3000


def _coerce_encoder_config(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _parse_s2_scales(config: OmniVinciConfig) -> list[int]:
    scales = getattr(config, "s2_scales", [384])
    if isinstance(scales, str):
        return sorted(int(s) for s in scales.split(",") if s)
    if isinstance(scales, (list, tuple)):
        return sorted(int(s) for s in scales)
    return [384]


def _token_len(tokenizer, text: str | None) -> int:
    if text is None:
        return 0
    return len(tokenizer(text).input_ids)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if torch.is_tensor(value):
        if value.ndim == 0:
            return [value]
        return [v for v in value]
    return [value]


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [int(v.item()) if torch.is_tensor(v) else int(v) for v in value]
    return [int(value)]


def _extract_sound_feature_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict) and "input_features" in value:
        tensor = value["input_features"]
        if torch.is_tensor(tensor):
            return tensor
    tensor = getattr(value, "input_features", None)
    if torch.is_tensor(tensor):
        return tensor
    raise TypeError(f"Unsupported sound feature payload type: {type(value)}")


def _patch_grid_size(sample_tensor: torch.Tensor, config: OmniVinciConfig) -> int:
    vision_cfg = getattr(config, "vision_tower_cfg", {}) or {}
    patch_size = int(vision_cfg.get("patch_size", 14))
    image_w = int(sample_tensor.shape[-1])
    image_h = int(sample_tensor.shape[-2])
    return max(1, int((image_h // patch_size) * (image_w // patch_size)) ** 0)


def _frame_token_count(sample_tensor: torch.Tensor, config: OmniVinciConfig) -> int:
    vision_cfg = getattr(config, "vision_tower_cfg", {}) or {}
    patch_size = int(vision_cfg.get("patch_size", 14))
    image_w = int(sample_tensor.shape[-1])
    image_h = int(sample_tensor.shape[-2])
    grid_h = max(1, image_h // patch_size)
    grid_w = max(1, image_w // patch_size)
    down_h = math.ceil(grid_h / 2)
    down_w = math.ceil(grid_w / 2)
    return down_h * down_w


def _estimate_image_embed_len(
    config: OmniVinciConfig,
    tiles_tensor: torch.Tensor,
    block_size: tuple[int, int] | None,
    start_len: int,
    end_len: int,
) -> int:
    base = _frame_token_count(tiles_tensor[0], config)

    if block_size is None:
        return base + start_len + end_len

    scales = _parse_s2_scales(config)
    resize_idx = int(getattr(config, "s2_resize_output_to_scale_idx", 0))

    if resize_idx in (-1, len(scales) - 1):
        split_h, split_w = int(block_size[0]), int(block_size[1])
    else:
        split = max(1, int(scales[resize_idx] // scales[0]))
        split_h = split_w = split

    return (split_h * split_w * base) + start_len + end_len


def _estimate_video_embed_len(
    frame_count: int,
    frame_token_count: int,
    video_encoder_cfg: dict[str, Any],
    start_len: int,
    end_len: int,
    sep_len: int,
) -> int:
    if frame_count <= 0:
        return 0

    if "pool_sizes" not in video_encoder_cfg:
        return frame_count * (frame_token_count + start_len + end_len)

    pooled_side = max(1, int(round(math.sqrt(frame_token_count))))
    total = 0
    for pool_size in video_encoder_cfg.get("pool_sizes", []):
        pt, ph, pw = (int(pool_size[0]), int(pool_size[1]), int(pool_size[2]))
        nt = math.ceil(frame_count / max(pt, 1))
        nh = math.ceil(pooled_side / max(ph, 1))
        nw = math.ceil(pooled_side / max(pw, 1))
        total += nt * (nh * nw + start_len + end_len) + sep_len

    return total


def _estimate_sound_embed_len(
    feature: torch.Tensor,
    start_len: int,
    end_len: int,
) -> int:
    seq_len = int(feature.shape[-1])
    num_chunks = max(1, math.ceil(seq_len / _AUDIO_CHUNK_UNIT_LENGTH))

    # Mirrors Qwen2AudioTower.forward_audio_tower_batch + Qwen2AudioEncoder conv output lengths.
    chunk_len = _AUDIO_CHUNK_UNIT_LENGTH
    conv1 = (chunk_len - 1) // 2 + 1
    conv2 = (conv1 - 2) // 2 + 1

    return (num_chunks * conv2) + start_len + end_len


def _estimate_interleaved_video_len(
    video_len: int,
    sound_len: int,
    video_info: dict[str, Any],
    audio_info: dict[str, Any] | None,
    sep_len: int,
) -> int:
    if audio_info is None:
        return video_len

    segment_vis = video_info.get("segment_vis_indices_list") or []
    segment_aud = video_info.get("segment_aud_indices_list") or []
    expected_frame_count = int(video_info.get("expected_frame_count", 0))
    n_stft = int(audio_info.get("new_audio_n_stft_frames", 0))

    if not segment_vis or expected_frame_count <= 0 or n_stft <= 0:
        return video_len

    vis_per_frame = video_len / expected_frame_count
    aud_per_stft = sound_len / n_stft

    vis_end = 0
    aud_end = 0
    total = 0

    for idx, vis_indices in enumerate(segment_vis):
        if vis_indices:
            vis_fea_end = int(math.ceil((vis_indices[-1] + 1) * vis_per_frame))
            vis_fea_end = min(vis_fea_end, video_len)
            total += max(0, vis_fea_end - vis_end)
            vis_end = vis_fea_end

        total += sep_len

        aud_indices = segment_aud[idx] if idx < len(segment_aud) else []
        if aud_indices:
            aud_fea_end = int(math.ceil(aud_indices[-1] * aud_per_stft))
            aud_fea_end = min(aud_fea_end, sound_len)
            total += max(0, aud_fea_end - aud_end)
            aud_end = aud_fea_end

        total += sep_len

    return total


def _count_tiles_for_block_size(
    block_size: tuple[int, int] | None,
    scales: list[int],
) -> int:
    if block_size is None:
        return 1

    base = scales[0]
    n_fixed = sum((scale // base) ** 2 for scale in scales[:-1])
    return n_fixed + int(block_size[0]) * int(block_size[1])


class OmniVinciProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(OmniVinciConfig)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(OmniVinciProcessor, **kwargs)

    def get_default_tok_params(self):
        # Chat templates already contain all control tokens; avoid adding an
        # extra wrapper tokenization layer that can skew generation tails.
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_data_parser(self):
        return MultiModalDataParser(
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None, "audio": None}


class OmniVinciDummyInputsBuilder(BaseDummyInputsBuilder[OmniVinciProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        processor = self.info.get_hf_processor()
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        num_audios = mm_counts.get("audio", 0)
        return (
            processor.image_token * num_images
            + processor.video_token * num_videos
            + processor.sound_token * num_audios
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        _ = seq_len

        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        num_audios = mm_counts.get("audio", 0)

        image_overrides = mm_options.get("image")
        video_overrides = mm_options.get("video")
        audio_overrides = mm_options.get("audio")
        dummy_videos = self._get_dummy_videos(
            width=384,
            height=384,
            num_frames=8,
            num_videos=num_videos,
            overrides=video_overrides,
        )
        dummy_videos_with_metadata = [
            (
                video,
                {
                    "fps": 1.0,
                    "total_num_frames": int(video.shape[0]),
                    "frames_indices": list(range(int(video.shape[0]))),
                },
            )
            for video in dummy_videos
        ]

        return {
            "image": self._get_dummy_images(
                width=384,
                height=384,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "video": dummy_videos_with_metadata,
            "audio": self._get_dummy_audios(
                length=30 * 16000,
                num_audios=num_audios,
                overrides=audio_overrides,
            ),
        }


class OmniVinciMultiModalProcessor(BaseMultiModalProcessor[OmniVinciProcessingInfo]):
    @staticmethod
    def _sync_processor_config_from_model(
        hf_processor: OmniVinciProcessor,
        hf_config: OmniVinciConfig,
    ) -> None:
        """
        Keep multimodal preprocessing behavior aligned with model runtime config.

        In vLLM, `hf_overrides` are applied to the model config; the processor can
        otherwise keep stale values loaded from `preprocessor_config.json`.
        """
        if getattr(hf_processor, "config", None) is None:
            return

        processor_config = hf_processor.config
        for key in (
            "load_audio_in_video",
            "interleaved_vis_aud_in_video",
            "interleaved_video_segment_duration",
            "num_video_frames",
            "audio_chunk_length",
            "audio_sampling_rate",
            "audio_hop_length",
            "audio_mel_bins",
            "image_aspect_ratio",
            "s2_scales",
            "s2_resize_output_to_scale_idx",
            "mm_use_bos_eos_tokens",
        ):
            if hasattr(hf_config, key):
                setattr(processor_config, key, getattr(hf_config, key))

        feature_extractor = getattr(hf_processor, "feature_extractor", None)
        if feature_extractor is not None:
            if hasattr(hf_config, "audio_sampling_rate"):
                feature_extractor.sampling_rate = int(hf_config.audio_sampling_rate)
            if hasattr(hf_config, "audio_hop_length"):
                feature_extractor.hop_length = int(hf_config.audio_hop_length)
            if isinstance(getattr(hf_config, "audio_chunk_length", None), int):
                feature_extractor.chunk_length = int(hf_config.audio_chunk_length)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        _ = tok_kwargs

        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        hf_config = self.info.get_hf_config()
        self._sync_processor_config_from_model(hf_processor, hf_config)

        images = mm_data.get("images")
        videos = mm_data.get("videos")
        audios = mm_data.get("audios")

        outputs = hf_processor(
            text=prompt,
            images=images,
            videos=videos,
            audio=audios,
        )

        input_ids = outputs["input_ids"]
        media = outputs.get("media", {})
        media_config = outputs.get("media_config", {})

        image_tiles = list(media.get("image", []))
        video_items = list(media.get("video", []))
        sound_items = list(media.get("sound", []))

        video_info_nested = media.get("video_info", [])
        video_infos = list(video_info_nested[0]) if video_info_nested else []

        audio_info_nested = media.get("audio_info", [])
        audio_infos = list(audio_info_nested[0]) if audio_info_nested else []

        image_cfg = media_config.get("image", {})
        block_sizes_raw = image_cfg.get("block_sizes", [])
        block_sizes = [tuple(map(int, bs)) for bs in block_sizes_raw]
        scales = _parse_s2_scales(hf_config)

        image_groups: list[tuple[torch.Tensor, tuple[int, int] | None]] = []
        if image_tiles:
            if block_sizes:
                tile_cursor = 0
                for block_size in block_sizes:
                    n_tiles = _count_tiles_for_block_size(block_size, scales)
                    tiles = torch.stack(image_tiles[tile_cursor : tile_cursor + n_tiles], dim=0)
                    tile_cursor += n_tiles
                    image_groups.append((tiles, block_size))
            else:
                for tile in image_tiles:
                    image_groups.append((tile.unsqueeze(0), None))

        sound_cursor = 0
        video_audio_features: list[torch.Tensor | None] = []
        video_audio_infos: list[dict[str, Any] | None] = []

        # Keep sound items in original order to mirror HF consumption.
        sound_input_features: list[torch.Tensor] = []
        sound_meta_infos: list[dict[str, Any] | None] = []
        sound_drop_flags: list[bool] = []

        for video_info in video_infos:
            has_audio = bool(video_info.get("has_audio", False))
            if has_audio and sound_cursor < len(sound_items):
                sound_item = sound_items[sound_cursor]
                sound_feature = _extract_sound_feature_tensor(sound_item)
                audio_info = audio_infos[sound_cursor] if sound_cursor < len(audio_infos) else None
                video_audio_features.append(sound_feature)
                video_audio_infos.append(audio_info)

                sound_input_features.append(sound_feature)
                sound_meta_infos.append(audio_info)
                sound_drop_flags.append(bool(hf_config.interleaved_vis_aud_in_video))
                sound_cursor += 1
            else:
                video_audio_features.append(None)
                video_audio_infos.append(None)

        while sound_cursor < len(sound_items):
            sound_item = sound_items[sound_cursor]
            sound_feature = _extract_sound_feature_tensor(sound_item)
            audio_info = audio_infos[sound_cursor] if sound_cursor < len(audio_infos) else None

            sound_input_features.append(sound_feature)
            sound_meta_infos.append(audio_info)
            sound_drop_flags.append(False)
            sound_cursor += 1

        image_encoder_cfg = _coerce_encoder_config(hf_config.image_encoder)
        video_encoder_cfg = _coerce_encoder_config(hf_config.video_encoder)
        sound_encoder_cfg = _coerce_encoder_config(hf_config.sound_encoder)

        image_start_len = _token_len(hf_processor.tokenizer, image_encoder_cfg.get("start_tokens"))
        image_end_len = _token_len(hf_processor.tokenizer, image_encoder_cfg.get("end_tokens"))
        video_start_len = _token_len(hf_processor.tokenizer, video_encoder_cfg.get("start_tokens"))
        video_end_len = _token_len(hf_processor.tokenizer, video_encoder_cfg.get("end_tokens"))
        video_sep_len = _token_len(hf_processor.tokenizer, video_encoder_cfg.get("sep_tokens"))
        sound_start_len = _token_len(hf_processor.tokenizer, sound_encoder_cfg.get("start_tokens"))
        sound_end_len = _token_len(hf_processor.tokenizer, sound_encoder_cfg.get("end_tokens"))
        interleave_sep_len = _token_len(hf_processor.tokenizer, "\n")

        image_expected_lens: list[int] = []
        for tiles, block_size in image_groups:
            image_expected_lens.append(
                _estimate_image_embed_len(
                    hf_config,
                    tiles,
                    block_size,
                    image_start_len,
                    image_end_len,
                )
            )

        video_expected_lens: list[int] = []
        for idx, video in enumerate(video_items):
            frame_tokens = _frame_token_count(video[0], hf_config)
            base_video_len = _estimate_video_embed_len(
                frame_count=int(video.shape[0]),
                frame_token_count=frame_tokens,
                video_encoder_cfg=video_encoder_cfg,
                start_len=video_start_len,
                end_len=video_end_len,
                sep_len=video_sep_len,
            )

            paired_sound = video_audio_features[idx]
            paired_audio_info = video_audio_infos[idx]
            if (
                paired_sound is not None
                and bool(hf_config.load_audio_in_video)
                and bool(hf_config.interleaved_vis_aud_in_video)
                and idx < len(video_infos)
            ):
                sound_len = _estimate_sound_embed_len(paired_sound, sound_start_len, sound_end_len)
                base_video_len = _estimate_interleaved_video_len(
                    base_video_len,
                    sound_len,
                    video_infos[idx],
                    paired_audio_info,
                    interleave_sep_len,
                )

            video_expected_lens.append(base_video_len)

        sound_expected_lens: list[int] = []
        for feature, drop_flag in zip(sound_input_features, sound_drop_flags):
            if drop_flag:
                sound_expected_lens.append(0)
            else:
                sound_expected_lens.append(
                    _estimate_sound_embed_len(feature, sound_start_len, sound_end_len)
                )

        video_has_audio = []
        video_expected_frame_counts = []
        video_frame_times = []
        video_segment_vis_counts = []
        video_segment_aud_counts = []
        video_audio_start_secs = []
        video_audio_chunk_lengths = []
        video_audio_n_stft_frames = []

        for idx, video_info in enumerate(video_infos):
            has_audio = bool(video_info.get("has_audio", False))
            video_has_audio.append(has_audio)
            video_expected_frame_counts.append(int(video_info.get("expected_frame_count", 0)))

            frame_times = video_info.get("video_frame_times", [])
            frame_times_tensor = torch.tensor(frame_times, dtype=torch.float32)
            video_frame_times.append(frame_times_tensor)

            segment_vis = video_info.get("segment_vis_indices_list") or []
            segment_aud = video_info.get("segment_aud_indices_list") or []
            vis_counts = torch.tensor([len(seg) for seg in segment_vis], dtype=torch.long)
            aud_counts = torch.tensor(
                [int(seg[-1] - seg[0]) if seg else 0 for seg in segment_aud],
                dtype=torch.long,
            )
            video_segment_vis_counts.append(vis_counts)
            video_segment_aud_counts.append(aud_counts)

            audio_info = video_audio_infos[idx] if idx < len(video_audio_infos) else None
            if isinstance(audio_info, dict):
                video_audio_start_secs.append(float(audio_info.get("audio_start_sec", 0.0)))
                video_audio_chunk_lengths.append(int(audio_info.get("new_audio_chunk_length", 0)))
                video_audio_n_stft_frames.append(int(audio_info.get("new_audio_n_stft_frames", 0)))
            else:
                video_audio_start_secs.append(0.0)
                video_audio_chunk_lengths.append(0)
                video_audio_n_stft_frames.append(0)

        sound_audio_start_secs = []
        sound_audio_chunk_lengths = []
        sound_audio_n_stft_frames = []
        for audio_info in sound_meta_infos:
            if isinstance(audio_info, dict):
                sound_audio_start_secs.append(float(audio_info.get("audio_start_sec", 0.0)))
                sound_audio_chunk_lengths.append(int(audio_info.get("new_audio_chunk_length", 0)))
                sound_audio_n_stft_frames.append(int(audio_info.get("new_audio_n_stft_frames", 0)))
            else:
                sound_audio_start_secs.append(0.0)
                sound_audio_chunk_lengths.append(0)
                sound_audio_n_stft_frames.append(0)

        image_block_sizes = (
            torch.tensor([
                [int(bs[0]), int(bs[1])] if bs is not None else [0, 0]
                for _, bs in image_groups
            ], dtype=torch.long)
            if image_groups
            else torch.empty((0, 2), dtype=torch.long)
        )

        data: dict[str, Any] = {
            "input_ids": input_ids,
            "ov_image_tiles": [tiles for tiles, _ in image_groups],
            "ov_image_block_sizes": image_block_sizes,
            "ov_image_expected_lens": torch.tensor(image_expected_lens, dtype=torch.long),
            "ov_video_frames": video_items,
            "ov_video_has_audio": torch.tensor(video_has_audio, dtype=torch.bool),
            "ov_video_expected_frame_counts": torch.tensor(video_expected_frame_counts, dtype=torch.long),
            "ov_video_frame_times": video_frame_times,
            "ov_video_segment_vis_counts": video_segment_vis_counts,
            "ov_video_segment_aud_counts": video_segment_aud_counts,
            "ov_video_audio_features": video_audio_features,
            "ov_video_audio_start_secs": torch.tensor(video_audio_start_secs, dtype=torch.float32),
            "ov_video_audio_chunk_lengths": torch.tensor(video_audio_chunk_lengths, dtype=torch.long),
            "ov_video_audio_n_stft_frames": torch.tensor(video_audio_n_stft_frames, dtype=torch.long),
            "ov_video_expected_lens": torch.tensor(video_expected_lens, dtype=torch.long),
            "ov_sound_features": sound_input_features,
            "ov_sound_audio_start_secs": torch.tensor(sound_audio_start_secs, dtype=torch.float32),
            "ov_sound_audio_chunk_lengths": torch.tensor(sound_audio_chunk_lengths, dtype=torch.long),
            "ov_sound_audio_n_stft_frames": torch.tensor(sound_audio_n_stft_frames, dtype=torch.long),
            "ov_sound_drop_flags": torch.tensor(sound_drop_flags, dtype=torch.bool),
            "ov_sound_expected_lens": torch.tensor(sound_expected_lens, dtype=torch.long),
        }

        return BatchFeature(data)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        _ = hf_processor_mm_kwargs

        fields: dict[str, MultiModalFieldConfig] = {}

        if len(hf_inputs.get("ov_image_expected_lens", [])) > 0:
            fields.update(
                {
                    "ov_image_tiles": MultiModalFieldConfig.batched("image"),
                    "ov_image_block_sizes": MultiModalFieldConfig.batched("image"),
                    "ov_image_expected_lens": MultiModalFieldConfig.batched("image"),
                }
            )

        if len(hf_inputs.get("ov_video_expected_lens", [])) > 0:
            fields.update(
                {
                    "ov_video_frames": MultiModalFieldConfig.batched("video"),
                    "ov_video_has_audio": MultiModalFieldConfig.batched("video"),
                    "ov_video_expected_frame_counts": MultiModalFieldConfig.batched("video"),
                    "ov_video_frame_times": MultiModalFieldConfig.batched("video"),
                    "ov_video_segment_vis_counts": MultiModalFieldConfig.batched("video"),
                    "ov_video_segment_aud_counts": MultiModalFieldConfig.batched("video"),
                    "ov_video_audio_features": MultiModalFieldConfig.batched("video"),
                    "ov_video_audio_start_secs": MultiModalFieldConfig.batched("video"),
                    "ov_video_audio_chunk_lengths": MultiModalFieldConfig.batched("video"),
                    "ov_video_audio_n_stft_frames": MultiModalFieldConfig.batched("video"),
                    "ov_video_expected_lens": MultiModalFieldConfig.batched("video"),
                }
            )

        if len(hf_inputs.get("ov_sound_expected_lens", [])) > 0:
            fields.update(
                {
                    "ov_sound_features": MultiModalFieldConfig.batched("audio"),
                    "ov_sound_audio_start_secs": MultiModalFieldConfig.batched("audio"),
                    "ov_sound_audio_chunk_lengths": MultiModalFieldConfig.batched("audio"),
                    "ov_sound_audio_n_stft_frames": MultiModalFieldConfig.batched("audio"),
                    "ov_sound_drop_flags": MultiModalFieldConfig.batched("audio"),
                    "ov_sound_expected_lens": MultiModalFieldConfig.batched("audio"),
                }
            )

        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        _ = (mm_items, hf_processor_mm_kwargs)

        processor = self.info.get_hf_processor()
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        image_token_id = vocab[processor.image_token]
        video_token_id = vocab[processor.video_token]
        sound_token_id = vocab[processor.sound_token]

        out = out_mm_kwargs.get_data()
        image_lens = _as_int_list(out.get("ov_image_expected_lens"))
        video_lens = _as_int_list(out.get("ov_video_expected_lens"))
        sound_lens = _as_int_list(out.get("ov_sound_expected_lens"))

        sound_drop_raw = out.get("ov_sound_drop_flags")
        if torch.is_tensor(sound_drop_raw):
            sound_drop = [bool(v) for v in sound_drop_raw.tolist()]
        else:
            sound_drop = [bool(v) for v in _as_list(sound_drop_raw)]

        def image_replacement(item_idx: int):
            n = image_lens[item_idx]
            return PromptUpdateDetails.select_token_id([image_token_id] * n, image_token_id)

        def video_replacement(item_idx: int):
            n = video_lens[item_idx]
            return PromptUpdateDetails.select_token_id([video_token_id] * n, video_token_id)

        def sound_replacement(item_idx: int):
            if item_idx < len(sound_drop) and sound_drop[item_idx]:
                return PromptUpdateDetails.from_seq([])

            n = sound_lens[item_idx]
            return PromptUpdateDetails.select_token_id([sound_token_id] * n, sound_token_id)

        updates: list[PromptUpdate] = []
        if image_lens:
            updates.append(
                PromptReplacement(
                    modality="image",
                    target=processor.image_token,
                    replacement=image_replacement,
                )
            )

        if video_lens:
            updates.append(
                PromptReplacement(
                    modality="video",
                    target=processor.video_token,
                    replacement=video_replacement,
                )
            )

        if sound_lens:
            updates.append(
                PromptReplacement(
                    modality="audio",
                    target=processor.sound_token,
                    replacement=sound_replacement,
                )
            )

        return updates


@MULTIMODAL_REGISTRY.register_processor(
    OmniVinciMultiModalProcessor,
    info=OmniVinciProcessingInfo,
    dummy_inputs=OmniVinciDummyInputsBuilder,
)
class OmniVinciForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "llm.": "language_model.",
        }
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        _ = i
        if modality.startswith("image"):
            return "<image>"
        if modality.startswith("video"):
            return "<vila/video>"
        if modality.startswith("audio"):
            return "<sound>"

        raise ValueError("Only image, video and audio modalities are supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: OmniVinciConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_tower = SiglipVisionTowerDynamicS2(config.vision_tower_cfg, config)
            config.mm_hidden_size = self.vision_tower.hidden_size
            self.mm_projector = MultimodalProjector(config)

        with self._mark_tower_model(vllm_config, "audio"):
            self.sound_tower = Qwen2AudioTower(config.sound_tower_cfg, config)
            config.sound_hidden_size = getattr(config, "sound_hidden_size", 1280)
            self.sound_mm_projector = SoundMultimodalProjector(config)

        llm_cfg = getattr(config, "text_config", None)
        if llm_cfg is None:
            llm_spec = dict(getattr(config, "llm_cfg", {}) or {})
            llm_spec.pop("model_type", None)
            llm_cfg = Qwen2Config(**llm_spec)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=llm_cfg,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen2ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        image_encoder_cfg = _coerce_encoder_config(config.image_encoder)
        video_encoder_cfg = _coerce_encoder_config(config.video_encoder)
        sound_encoder_cfg = _coerce_encoder_config(config.sound_encoder)

        image_encoder_cfg.pop("_target_", None)
        video_encoder_cfg.pop("_target_", None)
        sound_encoder_cfg.pop("_target_", None)

        self.encoders = {
            "image": BasicImageEncoder(parent=self, **image_encoder_cfg),
            "video": TSPVideoEncoder(parent=self, **video_encoder_cfg),
            "sound": BasicSoundEncoder(parent=self, **sound_encoder_cfg),
        }

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector=["mm_projector.", "sound_mm_projector."],
            tower_model=["vision_tower.", "sound_tower."],
        )

    @property
    def llm_model_embed_tokens(self):
        return self.language_model.model.embed_tokens

    @property
    def dtype(self) -> torch.dtype:
        return self.llm_model_embed_tokens.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.llm_model_embed_tokens.weight.device

    def embed_text_tokens(self, token_text: str | None) -> torch.Tensor | None:
        if token_text is None:
            return None

        token_ids = torch.tensor(
            self.config.encoder_text_token_ids[token_text],
            device=self.device,
            dtype=torch.long,
        )
        return self.llm_model_embed_tokens(token_ids)

    @staticmethod
    def split_chessboard(x: torch.Tensor, num_split_h: int, num_split_w: int) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        assert height % num_split_h == 0 and width % num_split_w == 0
        split_h = height // num_split_h
        split_w = width // num_split_w

        return torch.cat(
            [
                x[:, :, i * split_h : (i + 1) * split_h, j * split_w : (j + 1) * split_w]
                for i in range(num_split_h)
                for j in range(num_split_w)
            ],
            dim=0,
        )

    @staticmethod
    def merge_chessboard(x: torch.Tensor, num_split_h: int, num_split_w: int) -> torch.Tensor:
        if x.dim() == 3:
            n_tokens = x.shape[1]
            side = int(n_tokens**0.5)
            x = x.view(x.shape[0], side, side, x.shape[2]).permute(0, 3, 1, 2)

        batch = x.shape[0]
        assert batch % (num_split_h * num_split_w) == 0
        chunk = batch // (num_split_h * num_split_w)

        rows = []
        for i in range(num_split_h):
            cols = []
            for j in range(num_split_w):
                idx = i * num_split_w + j
                cols.append(x[idx * chunk : (idx + 1) * chunk])
            rows.append(torch.cat(cols, dim=-1))

        return torch.cat(rows, dim=-2)

    def merge_features_for_dynamic_s2(
        self,
        image_features: torch.Tensor,
        block_sizes: Sequence[tuple[int, int] | None],
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
        scales = self.vision_tower.scales
        resize_idx = self.vision_tower.resize_output_to_scale_idx

        features_per_image: list[torch.Tensor] = []
        new_block_sizes: list[tuple[int, int]] = []
        block_cursor = 0

        for block_size in block_sizes:
            if block_size is None:
                feat = image_features[block_cursor : block_cursor + 1]
                side = int(feat.shape[1] ** 0.5)
                feat = feat.view(1, side, side, feat.shape[2]).permute(0, 3, 1, 2)
                feat = feat.repeat(1, len(scales), 1, 1)
                features_per_image.append(feat)
                new_block_sizes.append((1, 1))
                block_cursor += 1
                continue

            per_scale: list[torch.Tensor] = []
            for scale in scales[:-1]:
                split = scale // scales[0]
                n_blocks = split * split
                per_scale.append(
                    self.merge_chessboard(
                        image_features[block_cursor : block_cursor + n_blocks],
                        num_split_h=split,
                        num_split_w=split,
                    )
                )
                block_cursor += n_blocks

            n_last = int(block_size[0]) * int(block_size[1])
            per_scale.append(
                self.merge_chessboard(
                    image_features[block_cursor : block_cursor + n_last],
                    num_split_h=int(block_size[0]),
                    num_split_w=int(block_size[1]),
                )
            )
            block_cursor += n_last

            output_size = per_scale[resize_idx].shape[-2:]
            merged = torch.cat(
                [
                    nn.functional.interpolate(scale_feat.float(), size=output_size, mode="area").to(scale_feat.dtype)
                    for scale_feat in per_scale
                ],
                dim=1,
            )
            features_per_image.append(merged)

            if resize_idx in (-1, len(scales) - 1):
                new_block_sizes.append((int(block_size[0]), int(block_size[1])))
            else:
                split = scales[resize_idx] // scales[0]
                new_block_sizes.append((split, split))

        assert block_cursor == len(image_features)
        return features_per_image, new_block_sizes

    def encode_images(
        self,
        images: torch.Tensor,
        block_sizes: Sequence[tuple[int, int] | None] | None = None,
        mm_info: dict[str, Any] | None = None,
        num_frames: list[int] | None = None,
    ):
        _ = (mm_info, num_frames)

        if block_sizes is None:
            block_sizes = [None] * len(images)

        image_features = self.vision_tower(images)
        merged_features, new_block_sizes = self.merge_features_for_dynamic_s2(image_features, block_sizes)

        split_features = [
            self.split_chessboard(feat, block_size[0], block_size[1])
            for feat, block_size in zip(merged_features, new_block_sizes)
        ]

        flat_features = torch.cat(
            [feat.permute(0, 2, 3, 1).reshape(feat.shape[0], -1, feat.shape[1]) for feat in split_features],
            dim=0,
        )

        projected = self.mm_projector(flat_features.to(self.device, self.dtype))

        split_sizes = [h * w for h, w in new_block_sizes]
        per_image = list(projected.split(split_sizes, dim=0))
        per_image = [
            self.merge_chessboard(feat, block_size[0], block_size[1])
            for feat, block_size in zip(per_image, new_block_sizes)
        ]
        per_image = [feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1]) for feat in per_image]

        if per_image and all(feat.shape[0] == per_image[0].shape[0] for feat in per_image):
            return torch.stack(per_image, dim=0)
        return per_image

    def encode_video(
        self,
        videos: list[torch.Tensor],
        block_sizes: Sequence[tuple[int, int] | None] | None = None,
        mm_info: dict[str, Any] | None = None,
        num_frames: list[int] | None = None,
    ):
        _ = (mm_info, num_frames)

        if len(videos) == 0:
            return []

        images = torch.cat(videos, dim=0)

        if block_sizes is None:
            block_sizes = [None] * len(images)

        image_features = self.vision_tower(images)
        merged_features, new_block_sizes = self.merge_features_for_dynamic_s2(image_features, block_sizes)

        split_features = [
            self.split_chessboard(feat, block_size[0], block_size[1])
            for feat, block_size in zip(merged_features, new_block_sizes)
        ]

        flat_features = torch.cat(
            [feat.permute(0, 2, 3, 1).reshape(feat.shape[0], -1, feat.shape[1]) for feat in split_features],
            dim=0,
        )

        projected = self.mm_projector(flat_features.to(self.device, self.dtype))

        split_sizes = [h * w for h, w in new_block_sizes]
        per_frame = list(projected.split(split_sizes, dim=0))
        per_frame = [
            self.merge_chessboard(feat, block_size[0], block_size[1])
            for feat, block_size in zip(per_frame, new_block_sizes)
        ]
        per_frame = [feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1]) for feat in per_frame]

        if per_frame and all(feat.shape[0] == per_frame[0].shape[0] for feat in per_frame):
            return torch.stack(per_frame, dim=0)

        return per_frame

    def encode_sound(
        self,
        sounds: list[dict[str, torch.Tensor] | torch.Tensor],
        mm_info: dict[str, Any] | None = None,
    ) -> list[torch.Tensor]:
        _ = mm_info

        audio_features, audio_output_lengths = self.sound_tower(sounds)
        proj_param = next(self.sound_mm_projector.parameters(), None)
        if proj_param is not None and audio_features.dtype != proj_param.dtype:
            audio_features = audio_features.to(proj_param.dtype)

        audio_features = self.sound_mm_projector(audio_features)

        if audio_output_lengths is None:
            return [audio_features]

        split_features: list[torch.Tensor] = []
        start = 0
        for length in audio_output_lengths:
            split_features.append(audio_features[start : start + length])
            start += length

        return split_features

    def _align_embedding_length(
        self,
        embedding: torch.Tensor,
        expected_len: int,
    ) -> torch.Tensor:
        if expected_len < 0:
            return embedding
        if embedding.shape[0] == expected_len:
            return embedding
        if embedding.shape[0] > expected_len:
            return embedding[:expected_len]

        pad = torch.zeros(
            (expected_len - embedding.shape[0], embedding.shape[1]),
            dtype=embedding.dtype,
            device=embedding.device,
        )
        return torch.cat([embedding, pad], dim=0)

    def _interleave_video_audio(
        self,
        video_embedding: torch.Tensor,
        sound_embedding: torch.Tensor,
        video_info: dict[str, Any],
        audio_info: dict[str, Any] | None,
    ) -> torch.Tensor:
        if audio_info is None:
            return video_embedding

        segment_vis = video_info.get("segment_vis_indices_list") or []
        segment_aud = video_info.get("segment_aud_indices_list") or []
        expected_frame_count = int(video_info.get("expected_frame_count", 0))
        n_stft = int(audio_info.get("new_audio_n_stft_frames", 0))

        if not segment_vis or expected_frame_count <= 0 or n_stft <= 0:
            return video_embedding

        sep_embedding = self.encoders["video"].embed_tokens("\n")
        if sep_embedding is None:
            return video_embedding

        vis_per_frame = video_embedding.shape[0] / expected_frame_count
        aud_per_stft = sound_embedding.shape[0] / n_stft

        vis_end = 0
        aud_end = 0
        chunks: list[torch.Tensor] = []

        for idx, vis_indices in enumerate(segment_vis):
            pieces = []
            if vis_indices:
                vis_fea_end = int(math.ceil((vis_indices[-1] + 1) * vis_per_frame))
                vis_fea_end = min(vis_fea_end, video_embedding.shape[0])
                pieces.append(video_embedding[vis_end:vis_fea_end])
                vis_end = vis_fea_end

            pieces.append(sep_embedding)

            aud_indices = segment_aud[idx] if idx < len(segment_aud) else []
            if aud_indices:
                aud_fea_end = int(math.ceil(aud_indices[-1] * aud_per_stft))
                aud_fea_end = min(aud_fea_end, sound_embedding.shape[0])
                pieces.append(sound_embedding[aud_end:aud_fea_end])
                aud_end = aud_fea_end

            pieces.append(sep_embedding)
            chunks.append(torch.cat(pieces, dim=0))

        if not chunks:
            return video_embedding
        return torch.cat(chunks, dim=0)

    def _embed_images(self, **kwargs: object) -> tuple[torch.Tensor, ...] | None:
        image_tiles = kwargs.get("ov_image_tiles")
        if image_tiles is None:
            return None

        block_sizes_raw = kwargs.get("ov_image_block_sizes")
        expected_lens = _as_int_list(kwargs.get("ov_image_expected_lens"))

        tile_items = _as_list(image_tiles)
        block_sizes_tensor = (
            block_sizes_raw
            if torch.is_tensor(block_sizes_raw)
            else torch.tensor(block_sizes_raw, dtype=torch.long)
            if block_sizes_raw is not None
            else torch.empty((0, 2), dtype=torch.long)
        )

        outputs: list[torch.Tensor] = []
        for idx, tiles in enumerate(tile_items):
            if isinstance(tiles, list):
                tiles = torch.stack(tiles, dim=0)
            block_h, block_w = (
                int(block_sizes_tensor[idx, 0].item()),
                int(block_sizes_tensor[idx, 1].item()),
            )
            block_size = None if block_h <= 0 or block_w <= 0 else (block_h, block_w)

            image_list = [tile for tile in tiles]
            config = {"block_sizes": [block_size]} if block_size is not None else {}
            embedding = self.encoders["image"](image_list, config, {})[0]
            embedding = self._align_embedding_length(embedding, expected_lens[idx])
            outputs.append(embedding)

        return tuple(outputs)

    def _embed_videos(self, **kwargs: object) -> tuple[torch.Tensor, ...] | None:
        video_frames = kwargs.get("ov_video_frames")
        if video_frames is None:
            return None

        video_has_audio_raw = kwargs.get("ov_video_has_audio")
        video_expected_frame_counts_raw = kwargs.get("ov_video_expected_frame_counts")
        video_frame_times = _as_list(kwargs.get("ov_video_frame_times"))
        video_segment_vis_counts = _as_list(kwargs.get("ov_video_segment_vis_counts"))
        video_segment_aud_counts = _as_list(kwargs.get("ov_video_segment_aud_counts"))
        video_audio_features = _as_list(kwargs.get("ov_video_audio_features"))
        video_audio_start_secs_raw = kwargs.get("ov_video_audio_start_secs")
        video_audio_chunk_lengths_raw = kwargs.get("ov_video_audio_chunk_lengths")
        video_audio_n_stft_frames_raw = kwargs.get("ov_video_audio_n_stft_frames")
        expected_lens = _as_int_list(kwargs.get("ov_video_expected_lens"))

        if torch.is_tensor(video_has_audio_raw):
            video_has_audio = [bool(v) for v in video_has_audio_raw.tolist()]
        else:
            video_has_audio = [bool(v) for v in _as_list(video_has_audio_raw)]
        video_expected_frame_counts = _as_int_list(video_expected_frame_counts_raw)
        video_audio_start_secs = (
            [float(v) for v in video_audio_start_secs_raw.tolist()]
            if torch.is_tensor(video_audio_start_secs_raw)
            else [float(v) for v in _as_list(video_audio_start_secs_raw)]
        )
        video_audio_chunk_lengths = _as_int_list(video_audio_chunk_lengths_raw)
        video_audio_n_stft_frames = _as_int_list(video_audio_n_stft_frames_raw)

        outputs: list[torch.Tensor] = []
        for idx, frames in enumerate(_as_list(video_frames)):
            if isinstance(frames, list):
                frames = torch.stack(frames, dim=0)

            vis_counts = (
                video_segment_vis_counts[idx].tolist()
                if idx < len(video_segment_vis_counts)
                and torch.is_tensor(video_segment_vis_counts[idx])
                else [int(v) for v in _as_list(video_segment_vis_counts[idx])]
                if idx < len(video_segment_vis_counts)
                else []
            )
            aud_counts = (
                video_segment_aud_counts[idx].tolist()
                if idx < len(video_segment_aud_counts)
                and torch.is_tensor(video_segment_aud_counts[idx])
                else [int(v) for v in _as_list(video_segment_aud_counts[idx])]
                if idx < len(video_segment_aud_counts)
                else []
            )

            segment_vis: list[list[int]] = []
            vis_cursor = 0
            for count in vis_counts:
                segment_vis.append(list(range(vis_cursor, vis_cursor + int(count))))
                vis_cursor += int(count)

            segment_aud: list[list[int]] = []
            aud_cursor = 0
            for count in aud_counts:
                count_i = int(count)
                if count_i > 0:
                    segment_aud.append([aud_cursor, aud_cursor + count_i])
                    aud_cursor += count_i
                else:
                    segment_aud.append([])

            frame_times = (
                video_frame_times[idx].tolist()
                if idx < len(video_frame_times) and torch.is_tensor(video_frame_times[idx])
                else [float(v) for v in _as_list(video_frame_times[idx])]
                if idx < len(video_frame_times)
                else []
            )

            video_info: dict[str, Any] = {
                "has_audio": bool(video_has_audio[idx]) if idx < len(video_has_audio) else False,
                "expected_frame_count": (
                    int(video_expected_frame_counts[idx])
                    if idx < len(video_expected_frame_counts)
                    else int(frames.shape[0])
                ),
                "video_frame_times": frame_times,
                "segment_vis_indices_list": segment_vis,
                "segment_aud_indices_list": segment_aud,
            }

            mm_info = {"video_info": [[video_info]]}
            embedding = self.encoders["video"]([frames], {}, mm_info)[0]

            paired_sound = video_audio_features[idx] if idx < len(video_audio_features) else None
            paired_audio_info = None
            if idx < len(video_audio_n_stft_frames):
                paired_audio_info = {
                    "audio_start_sec": (
                        float(video_audio_start_secs[idx])
                        if idx < len(video_audio_start_secs)
                        else 0.0
                    ),
                    "new_audio_chunk_length": (
                        int(video_audio_chunk_lengths[idx])
                        if idx < len(video_audio_chunk_lengths)
                        else 0
                    ),
                    "new_audio_n_stft_frames": int(video_audio_n_stft_frames[idx]),
                }

            if (
                paired_sound is not None
                and bool(getattr(self.config, "load_audio_in_video", False))
                and bool(getattr(self.config, "interleaved_vis_aud_in_video", False))
                and video_info["has_audio"]
            ):
                paired_sound = _extract_sound_feature_tensor(paired_sound)
                sound_mm_info = {"audio_info": [[paired_audio_info]]}
                sound_embedding = self.encoders["sound"](
                    [{"input_features": paired_sound}],
                    {},
                    sound_mm_info,
                )[0]
                embedding = self._interleave_video_audio(
                    embedding,
                    sound_embedding,
                    video_info,
                    paired_audio_info,
                )

            embedding = self._align_embedding_length(embedding, expected_lens[idx])
            outputs.append(embedding)

        return tuple(outputs)

    def _embed_sounds(self, **kwargs: object) -> tuple[torch.Tensor, ...] | None:
        sound_features = kwargs.get("ov_sound_features")
        if sound_features is None:
            return None

        sound_audio_start_secs_raw = kwargs.get("ov_sound_audio_start_secs")
        sound_audio_chunk_lengths_raw = kwargs.get("ov_sound_audio_chunk_lengths")
        sound_audio_n_stft_frames_raw = kwargs.get("ov_sound_audio_n_stft_frames")
        drop_flags_raw = kwargs.get("ov_sound_drop_flags")
        expected_lens = _as_int_list(kwargs.get("ov_sound_expected_lens"))

        sound_audio_start_secs = (
            [float(v) for v in sound_audio_start_secs_raw.tolist()]
            if torch.is_tensor(sound_audio_start_secs_raw)
            else [float(v) for v in _as_list(sound_audio_start_secs_raw)]
        )
        sound_audio_chunk_lengths = _as_int_list(sound_audio_chunk_lengths_raw)
        sound_audio_n_stft_frames = _as_int_list(sound_audio_n_stft_frames_raw)

        if torch.is_tensor(drop_flags_raw):
            drop_flags = [bool(v) for v in drop_flags_raw.tolist()]
        else:
            drop_flags = [bool(v) for v in _as_list(drop_flags_raw)]

        outputs: list[torch.Tensor] = []
        for idx, feature in enumerate(_as_list(sound_features)):
            if idx < len(drop_flags) and drop_flags[idx]:
                outputs.append(
                    torch.empty((0, self.llm_model_embed_tokens.weight.shape[1]), device=self.device, dtype=self.dtype)
                )
                continue

            feature = _extract_sound_feature_tensor(feature)
            audio_info = {
                "audio_start_sec": (
                    float(sound_audio_start_secs[idx])
                    if idx < len(sound_audio_start_secs)
                    else 0.0
                ),
                "new_audio_chunk_length": (
                    int(sound_audio_chunk_lengths[idx])
                    if idx < len(sound_audio_chunk_lengths)
                    else 0
                ),
                "new_audio_n_stft_frames": (
                    int(sound_audio_n_stft_frames[idx])
                    if idx < len(sound_audio_n_stft_frames)
                    else 0
                ),
            }
            mm_info = {"audio_info": [[audio_info]]}
            embedding = self.encoders["sound"](
                [{"input_features": feature}],
                {},
                mm_info,
            )[0]
            embedding = self._align_embedding_length(embedding, expected_lens[idx])
            outputs.append(embedding)

        return tuple(outputs)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_embeddings = self._embed_images(**kwargs)
        if image_embeddings is not None:
            return image_embeddings

        video_embeddings = self._embed_videos(**kwargs)
        if video_embeddings is not None:
            return video_embeddings

        sound_embeddings = self._embed_sounds(**kwargs)
        if sound_embeddings is not None:
            return sound_embeddings

        return []

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        _ = kwargs
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
