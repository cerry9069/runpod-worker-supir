"""
RunPod Serverless Handler — SUPIR Image Enhancer
Accepts an image URL, enhances it with SUPIR, returns base64 or URL.
"""
import os
import runpod
import torch
import requests
import base64
from io import BytesIO
from PIL import Image

MODEL = None
CACHE_DIR = "/cache/supir"


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("[supir] Loading SUPIR model...")

    # Using the diffusers-compatible SUPIR pipeline
    from diffusers import StableDiffusionUpscalePipeline

    MODEL = StableDiffusionUpscalePipeline.from_pretrained(
        "Kijai/SUPIR_pruned",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
    ).to("cuda")

    print("[supir] Model loaded.")
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

    upscale = inp.get("upscale", 2)

    try:
        pipe = load_model()
        img = download_image(image_url)

        result = pipe(
            prompt="high quality, detailed, sharp, realistic skin texture",
            image=img,
            num_inference_steps=inp.get("steps", 20),
            guidance_scale=inp.get("guidance_scale", 7.5),
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
