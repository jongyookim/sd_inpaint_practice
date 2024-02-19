import sys

from diffusers import DiffusionPipeline, DDIMScheduler
# from diffusers import StableDiffusionInpaintPipeline
from pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline

from matplotlib import pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import torch


if sys.platform == "darwin":
    # MacOS
    device = "mps"
    torch_dtype = torch.float32
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
print("Device:", device)


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    # "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch_dtype
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

prompt = "A cat sitting on a park bench, high resolution"
negative_prompt = "jpeg artifacts, low quality, artifacts"

# image = Image.open("inpaint_rgb.png")
# mask_image = Image.open("inpaint_mask.png")

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

num_images_per_prompt = 1
num_inference_steps = 20

generator = torch.manual_seed(1)

image = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    negative_prompt=negative_prompt,
    strength=1.0,
    num_inference_steps=num_inference_steps,
    num_images_per_prompt=num_images_per_prompt,
    generator=generator,
).images[0]

print(pipe.vae.config.scaling_factor)

plt.figure()
plt.imshow(image)
plt.show()


