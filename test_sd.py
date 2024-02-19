import sys

from diffusers import DiffusionPipeline, DDIMScheduler
# from diffusers import StableDiffusionPipeline
from pipeline_stable_diffusion import StableDiffusionPipeline

from matplotlib import pyplot as plt
import torch


if sys.platform == "darwin":
    # MacOS
    device = "mps"
    torch_dtype = torch.float32
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
print("Device:", device)


pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    # "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch_dtype
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

prompt = "a photo of an astronaut riding a horse on mars"
negative_prompt = "jpeg artifacts, low quality"

num_images_per_prompt = 2
num_inference_steps = 25

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
    num_images_per_prompt=num_images_per_prompt,
).images[0]

print(pipe.vae.config.scaling_factor)

plt.figure()
plt.imshow(image)
plt.show()


