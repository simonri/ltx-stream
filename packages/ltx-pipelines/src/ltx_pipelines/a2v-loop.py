import logging
import tempfile

import numpy as np
import torch
from PIL import Image

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import (
    MultiModalGuider,
    MultiModalGuiderParams,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.loader.fuse_loras import _prepare_deltas
from ltx_core.loader.primitives import LoraStateDictWithStrength
from ltx_core.loader.sft_loader import SafetensorsStateDictLoader
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoPixelShape
from ltx_pipelines.prompt import AUDIO_PATH, IMAGE_PATH, MODELS_ROOT, OUTPUT_PATH, PROMPT
from ltx_pipelines.utils import (
    ModelLedger,
    assert_resolution,
    combined_image_conditionings,
    encode_prompts,
    euler_denoising_loop,
    get_device,
    multi_modal_guider_denoising_func,
    simple_denoising_func,
    trace_step,
)
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import denoise_video_only
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


def compute_lora_deltas(
    model: torch.nn.Module,
    loras: list[LoraPathStrengthAndSDOps],
) -> dict[str, torch.Tensor]:
    """Pre-compute the LoRA weight deltas for every affected parameter.

    Returns a dict mapping parameter names to their delta tensors (on the same
    device/dtype as the model parameters).
    """
    loader = SafetensorsStateDictLoader()
    lora_sd_and_strengths = [
        LoraStateDictWithStrength(loader.load([lora.path], sd_ops=lora.sd_ops), lora.strength)
        for lora in loras
    ]

    deltas: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        delta = _prepare_deltas(lora_sd_and_strengths, name, param.dtype, param.device)
        if delta is not None:
            deltas[name] = delta
    return deltas


def apply_lora_delta(model: torch.nn.Module, deltas: dict[str, torch.Tensor], scale: float = 1.0) -> None:
    """Add (scale=1) or subtract (scale=-1) pre-computed LoRA deltas to model params in-place."""
    params = dict(model.named_parameters())
    for name, delta in deltas.items():
        params[name].data.add_(delta, alpha=scale)


class A2VidPipelineTwoStage:
    """
    Two-stage text/image-to-video generation pipeline.
    Stage 1 generates video at half of the target resolution with CFG guidance (assuming
    full model is used), then Stage 2 upsamples by 2x and refines using a distilled
    LoRA for higher quality output. Supports optional image conditioning via the
    images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.distilled_lora = distilled_lora
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            quantization=quantization,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        initial_image: str,
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        tiling_config: TilingConfig | None = None,
        num_passes: int = 1,
    ) -> tuple[torch.Tensor, Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        segment_duration = num_frames / frame_rate

        # --- One-time setup ---
        with trace_step("encode_prompts"):
            ctx_p, ctx_n = encode_prompts(
                [prompt, negative_prompt],
                self.model_ledger,
            )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, _ = ctx_n.video_encoding, ctx_n.audio_encoding

        audio_shape = AudioLatentShape.from_duration(
            batch=1, duration=segment_duration, channels=8, mel_bins=16
        )

        stage_1_output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate,
        )
        stage_2_output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate,
        )

        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        # --- Load all models once, single transformer with LoRA swapping ---
        with trace_step("load_video_encoder"):
            video_encoder = self.model_ledger.video_encoder()
        with trace_step("load_audio_encoder"):
            audio_encoder = self.model_ledger.audio_encoder()
        with trace_step("load_transformer"):
            transformer = self.model_ledger.transformer()
        with trace_step("compute_lora_deltas"):
            lora_deltas = compute_lora_deltas(transformer.velocity_model, self.distilled_lora)
            logging.info(f"Computed LoRA deltas for {len(lora_deltas)} parameters")
        with trace_step("load_spatial_upsampler"):
            spatial_upsampler = self.model_ledger.spatial_upsampler()
        with trace_step("load_video_decoder"):
            video_decoder = self.model_ledger.video_decoder()

        def first_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=multi_modal_guider_denoising_func(
                    video_guider=MultiModalGuider(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider=MultiModalGuider(
                        params=MultiModalGuiderParams(),
                    ),
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer,
                ),
            )

        def second_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,
                ),
            )

        all_frames: list[torch.Tensor] = []
        prev_last_frame_path: str | None = None

        for pass_idx in range(num_passes):
            logging.info(f"Starting pass {pass_idx + 1}/{num_passes}")

            generator = torch.Generator(device=self.device).manual_seed(seed)
            noiser = GaussianNoiser(generator=generator)
            stepper = EulerDiffusionStep()

            # Build image conditionings for this pass:
            #   - Pass 0: initial image on first frame
            #   - Pass N>0: last frame from previous pass on first frame
            #   - Odd passes: also anchor initial image on last frame
            images: list[ImageConditioningInput] = []
            first_frame_path = initial_image if pass_idx == 0 else prev_last_frame_path
            images.append(ImageConditioningInput(path=first_frame_path, frame_idx=0, strength=1.0))
            if pass_idx % 2 == 1:
                images.append(ImageConditioningInput(
                    path=initial_image, frame_idx=num_frames - 1, strength=1.0,
                ))

            # --- Encode audio for this segment ---
            audio_start = audio_start_time + pass_idx * segment_duration
            with trace_step("encode_audio"):
                decoded_audio = decode_audio_from_file(
                    audio_path, self.device, audio_start,
                    audio_max_duration if audio_max_duration is not None else segment_duration,
                )
                if decoded_audio is not None and decoded_audio.waveform is not None:
                    encoded_audio_latent = vae_encode_audio(decoded_audio, audio_encoder)
                    encoded_audio_latent = encoded_audio_latent[:, :, : audio_shape.frames]
                    if encoded_audio_latent.shape[2] < audio_shape.frames:
                        pad_size = audio_shape.frames - encoded_audio_latent.shape[2]
                        encoded_audio_latent = torch.nn.functional.pad(encoded_audio_latent, (0, 0, 0, pad_size))
                else:
                    encoded_audio_latent = torch.zeros(
                        1, audio_shape.channels, audio_shape.frames, audio_shape.mel_bins,
                        dtype=self.dtype, device=self.device,
                    )

            # --- Stage 1 ---
            with trace_step("stage_1_encode_images"):
                stage_1_conditionings = combined_image_conditionings(
                    images=images,
                    height=stage_1_output_shape.height,
                    width=stage_1_output_shape.width,
                    video_encoder=video_encoder,
                    dtype=self.dtype,
                    device=self.device,
                )

            with trace_step("stage_1_denoise"):
                video_state = denoise_video_only(
                    output_shape=stage_1_output_shape,
                    conditionings=stage_1_conditionings,
                    noiser=noiser,
                    sigmas=sigmas,
                    stepper=stepper,
                    denoising_loop_fn=first_stage_denoising_loop,
                    components=self.pipeline_components,
                    dtype=self.dtype,
                    device=self.device,
                    initial_audio_latent=encoded_audio_latent,
                )

            # --- Stage 2 ---
            with trace_step("stage_2_upsample"):
                upscaled_video_latent = upsample_video(
                    latent=video_state.latent[:1],
                    video_encoder=video_encoder,
                    upsampler=spatial_upsampler,
                )

            with trace_step("stage_2_encode_images"):
                stage_2_conditionings = combined_image_conditionings(
                    images=images,
                    height=stage_2_output_shape.height,
                    width=stage_2_output_shape.width,
                    video_encoder=video_encoder,
                    dtype=self.dtype,
                    device=self.device,
                )

            apply_lora_delta(transformer.velocity_model, lora_deltas, scale=1.0)
            with trace_step("stage_2_denoise"):
                video_state = denoise_video_only(
                    output_shape=stage_2_output_shape,
                    conditionings=stage_2_conditionings,
                    noiser=noiser,
                    sigmas=distilled_sigmas,
                    stepper=stepper,
                    denoising_loop_fn=second_stage_denoising_loop,
                    components=self.pipeline_components,
                    dtype=self.dtype,
                    device=self.device,
                    noise_scale=distilled_sigmas[0],
                    initial_video_latent=upscaled_video_latent,
                    initial_audio_latent=encoded_audio_latent,
                )
            apply_lora_delta(transformer.velocity_model, lora_deltas, scale=-1.0)

            with trace_step("decode_video"):
                decoded_video = vae_decode_video(
                    video_state.latent, video_decoder, tiling_config, generator
                )

            pass_frames = torch.cat(list(decoded_video), dim=0)  # (F, H, W, C)
            all_frames.append(pass_frames)

            # Save last frame for next pass conditioning
            if pass_idx < num_passes - 1:
                last_frame_np = pass_frames[-1].cpu().numpy().astype(np.uint8)
                prev_last_frame_path = tempfile.mktemp(suffix=".png")
                Image.fromarray(last_frame_np).save(prev_last_frame_path)

        combined_video = torch.cat(all_frames, dim=0)

        # Read the full audio span for the final output
        full_audio_duration = num_passes * segment_duration
        full_decoded_audio = decode_audio_from_file(audio_path, self.device, audio_start_time, full_audio_duration)
        if full_decoded_audio is not None and full_decoded_audio.waveform is not None:
            full_audio = Audio(
                waveform=full_decoded_audio.waveform.squeeze(0),
                sampling_rate=full_decoded_audio.sampling_rate,
            )
        else:
            full_audio = Audio(waveform=torch.zeros(1), sampling_rate=44100)

        return combined_video, full_audio



CHECKPOINT_PATH = f"{MODELS_ROOT}/diffusion_models/ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA_PATH = f"{MODELS_ROOT}/loras/ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSAMPLER_PATH = f"{MODELS_ROOT}/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT = f"{MODELS_ROOT}/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized"

NUM_FRAMES = 121
WIDTH = 512 # 704
HEIGHT = 512 # 704
FRAME_RATE = 24.0
NUM_INFERENCE_STEPS = 10
SEED = 42
NUM_PASSES = 3
DISTILLED_LORA_STRENGTH = 0.8


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)

    pipeline = A2VidPipelineTwoStage(
        checkpoint_path=CHECKPOINT_PATH,
        distilled_lora=[
            LoraPathStrengthAndSDOps(DISTILLED_LORA_PATH, DISTILLED_LORA_STRENGTH, LTXV_LORA_COMFY_RENAMING_MAP),
        ],
        spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
        gemma_root=GEMMA_ROOT,
        loras=(),
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(NUM_FRAMES, tiling_config)

    combined_video, full_audio = pipeline(
        prompt=PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        seed=SEED,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        frame_rate=FRAME_RATE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=3.0,
            stg_scale=1.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[29],
        ),
        initial_image=IMAGE_PATH,
        tiling_config=tiling_config,
        audio_path=AUDIO_PATH,
        num_passes=NUM_PASSES,
    )

    encode_video(
        video=combined_video,
        fps=FRAME_RATE,
        audio=full_audio,
        output_path=OUTPUT_PATH,
        video_chunks_number=video_chunks_number * NUM_PASSES,
    )


if __name__ == "__main__":
    main()
