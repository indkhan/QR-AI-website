import torch
from PIL import Image
import qrcode
from multiprocessing import cpu_count
import requests
import io
import os
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
)
print("1")
qrcode_generator = qrcode.QRCode(
    version=1,
    error_correction=qrcode.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
print("12")
controlnet = ControlNetModel.from_pretrained(
    "DionTimmer/controlnet_qrcode-control_v1p_sd15", torch_dtype=torch.float16
)
print("234")
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda")
print("ok")
pipe.enable_xformers_memory_efficient_attention()
print("done")


import numpy as np
import time
def resize_for_condition_image(input_image: Image.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True),
    "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    "DDIM": lambda config: DDIMScheduler.from_config(config),
    "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
}


def inference(
    qr_code_content: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 10.0,
    controlnet_conditioning_scale: float = 2.0,
    strength: float = 0.8,
    seed: int = -1,
    qrcode_image: Image.Image | None = None,
    sampler="DPM++ Karras SDE",
):
    if prompt is None or prompt == "":
        raise ValueError("Prompt is required")

    if qrcode_image is None and qr_code_content == "":
        raise ValueError("QR Code Image or QR Code Content is required")

    pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)

    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    if qr_code_content != "" or qrcode_image.size == (1, 1):
        print("Generating QR Code from content")
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_code_content)
        qr.make(fit=True)

        qrcode_image = qr.make_image(fill_color="black", back_color="white")
        qrcode_image = resize_for_condition_image(qrcode_image, 768)
    else:
        print("Using QR Code Image")
        qrcode_image = resize_for_condition_image(qrcode_image, 768)

    # Encode QR code image as a tensor (assuming qrcode_image is a PIL image)
    qrcode_image = torch.from_numpy(np.array(qrcode_image)).float() / 255.0
    qrcode_image = qrcode_image.permute(2, 0, 1).unsqueeze(0)

    # Inference
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=qrcode_image,  # Use QR code image as initial image
        control_image=qrcode_image,  # Use QR code image as control image
        width=768,
        height=768,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
        strength=strength,
        num_inference_steps=40,
    )

    return out.images[0]  # Return the generated image


def generate_qr_code_art():
    qr_code_content = input("Enter the content for the QR code (or leave blank to use an image): ")
    prompt = input("Enter a prompt for the art: ")
    negative_prompt = input("Enter any words or phrases to avoid in the art (optional): ")

    try:
        # Handle cases for QR code content or image
        if qr_code_content != "":
            qrcode_image = None
        else:
            qrcode_image_path = input("Enter the path to the QR code image (optional): ")
            if qrcode_image_path:
                qrcode_image = Image.open(qrcode_image_path)

        generated_image = inference(
            qr_code_content=qr_code_content,
            prompt=prompt,
            negative_prompt=negative_prompt,
            qrcode_image=qrcode_image,
        )
        output_filename = f"qr_code_art{int(time.time())}.png"
        generated_image.save(output_filename)
        print("image saved")  # Display the generated image
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    generate_qr_code_art()
