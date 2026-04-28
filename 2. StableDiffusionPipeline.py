#!pip install -q diffusers transformers accelerate safetensors

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Using:", device)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
)

pipe = pipe.to(device)
pipe.enable_attention_slicing()

image = pipe(
    prompt="a realistic motorcycle parked inside a dense green forest, cinematic lighting, highly detailed",
    negative_prompt="blurry, distorted, low quality",
    height=512,
    width=512,
    num_inference_steps=25,
    guidance_scale=7.5
).images[0]

plt.figure(figsize=(7, 7))
plt.imshow(image)
plt.axis("off")
plt.show()
