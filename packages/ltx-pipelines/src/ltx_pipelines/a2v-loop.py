import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path

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
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoPixelShape
from ltx_pipelines.utils import (
    ModelLedger,
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    encode_prompts,
    euler_denoising_loop,
    get_device,
    multi_modal_guider_denoising_func,
    simple_denoising_func,
    trace_step,
)
from ltx_pipelines.utils.args import ImageConditioningInput, default_2_stage_arg_parser, detect_checkpoint_path
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES, detect_params
from ltx_pipelines.utils.helpers import denoise_video_only
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()

SHOULD_CLEANUP_MEMORY = False

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
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            quantization=quantization,
        )

        self.stage_2_model_ledger = self.stage_1_model_ledger.with_additional_loras(
            loras=distilled_lora,
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
        images: list[ImageConditioningInput],
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        with trace_step("encode_prompts"):
            ctx_p, ctx_n = encode_prompts(
                [prompt, negative_prompt],
                self.stage_1_model_ledger,
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, _ = ctx_n.video_encoding, ctx_n.audio_encoding

        with trace_step("encode_audio"):
            decoded_audio = decode_audio_from_file(audio_path, self.device, audio_start_time, audio_max_duration)
            encoded_audio_latent = vae_encode_audio(decoded_audio, self.stage_1_model_ledger.audio_encoder())
            audio_shape = AudioLatentShape.from_duration(
                batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16
            )

            encoded_audio_latent = encoded_audio_latent[:, :, : audio_shape.frames]
            # Pad with zeros if the audio is shorter than the target duration
            if encoded_audio_latent.shape[2] < audio_shape.frames:
                pad_size = audio_shape.frames - encoded_audio_latent.shape[2]
                encoded_audio_latent = torch.nn.functional.pad(encoded_audio_latent, (0, 0, 0, pad_size))

        # Stage 1: encode image conditionings with the VAE encoder, then free it
        # before loading the transformer to reduce peak VRAM.
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        with trace_step("stage_1_encode_images"):
            video_encoder = self.stage_1_model_ledger.video_encoder()
            stage_1_conditionings = combined_image_conditionings(
                images=images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
            )
        torch.cuda.synchronize()
        del video_encoder
        if SHOULD_CLEANUP_MEMORY:
            cleanup_memory()

        transformer = self.stage_1_model_ledger.transformer()
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

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
                    transformer=transformer,  # noqa: F821
                ),
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
                dtype=dtype,
                device=self.device,
                initial_audio_latent=encoded_audio_latent,
            )

        torch.cuda.synchronize()
        del transformer
        if SHOULD_CLEANUP_MEMORY:
            cleanup_memory()

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        with trace_step("stage_2_upsample"):
            video_encoder = self.stage_1_model_ledger.video_encoder()
            upscaled_video_latent = upsample_video(
                latent=video_state.latent[:1],
                video_encoder=video_encoder,
                upsampler=self.stage_2_model_ledger.spatial_upsampler(),
            )

        with trace_step("stage_2_encode_images"):
            stage_2_output_shape = VideoPixelShape(
                batch=1, frames=num_frames, width=width, height=height, fps=frame_rate
            )
            stage_2_conditionings = combined_image_conditionings(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
            )
        torch.cuda.synchronize()
        del video_encoder
        if SHOULD_CLEANUP_MEMORY:
            cleanup_memory()

        transformer = self.stage_2_model_ledger.transformer()
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

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
                    transformer=transformer,  # noqa: F821
                ),
            )

        with trace_step("stage_2_denoise"):
            video_state = denoise_video_only(
                output_shape=stage_2_output_shape,
                conditionings=stage_2_conditionings,
                noiser=noiser,
                sigmas=distilled_sigmas,
                stepper=stepper,
                denoising_loop_fn=second_stage_denoising_loop,
                components=self.pipeline_components,
                dtype=dtype,
                device=self.device,
                noise_scale=distilled_sigmas[0],
                initial_video_latent=upscaled_video_latent,
                initial_audio_latent=encoded_audio_latent,
            )

        torch.cuda.synchronize()
        del transformer
        if SHOULD_CLEANUP_MEMORY:
            cleanup_memory()

        with trace_step("decode_video"):
            decoded_video = vae_decode_video(
                video_state.latent, self.stage_2_model_ledger.video_decoder(), tiling_config, generator
            )

        # Return the original input audio instead of VAE-decoded audio to preserve fidelity.
        original_audio = Audio(waveform=decoded_audio.waveform.squeeze(0), sampling_rate=decoded_audio.sampling_rate)

        return decoded_video, original_audio


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    checkpoint_path = detect_checkpoint_path()
    params = detect_params(checkpoint_path)
    parser = default_2_stage_arg_parser(params=params)

    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to the audio file to condition the video generation.",
    )
    parser.add_argument(
        "--audio-start-time",
        type=float,
        default=0.0,
        help="Start time in seconds to read audio from (default: 0.0).",
    )
    parser.add_argument(
        "--audio-max-duration",
        type=float,
        default=None,
        help="Maximum audio duration in seconds. Defaults to video duration (num_frames / frame_rate).",
    )

    args = parser.parse_args()
    pipeline = A2VidPipelineTwoStage(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    num_passes = 3
    segment_duration = args.num_frames / args.frame_rate
    all_frames: list[torch.Tensor] = []
    images = args.images
    audio = None

    for pass_idx in range(num_passes):
        logging.info(f"Starting pass {pass_idx + 1}/{num_passes}")
        audio_start = args.audio_start_time + pass_idx * segment_duration

        video, pass_audio = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.num_inference_steps,
            video_guider_params=MultiModalGuiderParams(
                cfg_scale=args.video_cfg_guidance_scale,
                stg_scale=args.video_stg_guidance_scale,
                rescale_scale=args.video_rescale_scale,
                modality_scale=args.a2v_guidance_scale,
                skip_step=args.video_skip_step,
                stg_blocks=args.video_stg_blocks,
            ),
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=args.enhance_prompt,
            audio_path=args.audio_path,
            audio_start_time=audio_start,
            audio_max_duration=args.audio_max_duration
            if args.audio_max_duration is not None
            else segment_duration,
        )

        # Collect all decoded frames from this pass
        pass_frames = []
        for chunk in video:
            pass_frames.append(chunk)
        pass_frames = torch.cat(pass_frames, dim=0)  # (F, H, W, C)
        all_frames.append(pass_frames)

        if audio is None:
            audio = pass_audio

        # Extract last frame and save as temp image for next pass conditioning
        if pass_idx < num_passes - 1:
            last_frame = pass_frames[-1]  # (H, W, C) uint8
            last_frame_np = last_frame.cpu().numpy().astype(np.uint8)
            tmp_path = tempfile.mktemp(suffix=".png")
            Image.fromarray(last_frame_np).save(tmp_path)
            # Use last frame as first-frame conditioning for next pass
            images = [ImageConditioningInput(path=tmp_path, frame_idx=0, strength=1.0)]

    # Concatenate all passes and encode
    combined_video = torch.cat(all_frames, dim=0)

    # Reconstruct audio from all passes by reading the full duration
    full_audio_duration = num_passes * segment_duration
    decoded_audio = decode_audio_from_file(
        args.audio_path, device, args.audio_start_time, full_audio_duration
    )
    full_audio = Audio(waveform=decoded_audio.waveform.squeeze(0), sampling_rate=decoded_audio.sampling_rate)

    encode_video(
        video=combined_video,
        fps=args.frame_rate,
        audio=full_audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number * num_passes,
    )


if __name__ == "__main__":
    main()
