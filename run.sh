source .venv/bin/activate
export LOGLEVEL=INFO

python -m ltx_pipelines.a2v-loop \
    --checkpoint-path /home/ubuntu/nous/comfyui-data/models/diffusion_models/ltx-2.3-22b-dev.safetensors \
    --distilled-lora /home/ubuntu/nous/comfyui-data/models/loras/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
    --spatial-upsampler-path /home/ubuntu/nous/comfyui-data/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --gemma-root /home/ubuntu/nous/comfyui-data/models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized \
    --audio-path pokimane.mp3 \
    --prompt "Style: realistic - soft cinematic lighting - The girl slightly tilts her head toward the camera while looking forward, her lips moving as she speaks." \
    --output-path output.mp4 \
    --num-inference-steps 10 \
    --num-frames 121 \
    --image content.png 0 1.0 33 \
    --width 704 \
    --height 704 \
    --frame-rate 24.0
