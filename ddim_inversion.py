from pnp_pipeline import SDXLDDIMPipeline
import torch

pipe = SDXLDDIMPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")
import PIL

init = PIL.Image.open("example.png")

x = pipe(prompt = "green horse", image = init)
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
