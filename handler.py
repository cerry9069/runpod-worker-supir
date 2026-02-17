"""
RunPod Serverless Handler — SD x4 Image Upscaler
Uses stabilityai/stable-diffusion-x4-upscaler for 4x image enhancement.
Accepts an image URL, upscales it, returns base64 PNG.
"""
import os
import runpod
import torch
import requests
import base64
from io import BytesIO
from PIL import Image

MODEL = None
CACHE_DIR = "/cache/upscaler"


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("[upscaler] Loading SD x4 Upscaler pipeline...")

    from diffusers import StableDiffusionUpscalePipeline

    MODEL = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
    ).to("cuda")

    print("[upscaler] Pipeline loaded.")
    return MODEL


def download_image(url):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def handler(event):
    inp = event.get("input", {})
    image_url = inp.get("image_url")
    if not image_url:
        return {"error": "image_url is required"}

    try:
        pipe = load_model()
        img = download_image(image_url)

        # SD x4 upscaler works best with images around 128x128 to 512x512
        # It upscales to 4x the input size — auto-resize large inputs to fit VRAM
        max_input = inp.get("max_input_size", 512)
        if img.width > max_input or img.height > max_input:
            ratio = min(max_input / img.width, max_input / img.height)
            new_w = int(img.width * ratio)
            new_h = int(img.height * ratio)
            print(f"[upscaler] Resizing input from {img.width}x{img.height} to {new_w}x{new_h}")
            img = img.resize((new_w, new_h), Image.LANCZOS)

        prompt = inp.get("prompt", "high quality, detailed, sharp, realistic skin texture")

        result = pipe(
            prompt=prompt,
            image=img,
            num_inference_steps=inp.get("steps", 20),
            guidance_scale=inp.get("guidance_scale", 7.5),
            noise_level=inp.get("noise_level", 20),
        ).images[0]

        buf = BytesIO()
        result.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "image_base64": img_b64,
            "width": result.width,
            "height": result.height,
        }
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
