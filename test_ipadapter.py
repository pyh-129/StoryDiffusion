import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os
from ip_adapter import IPAdapterXL
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "sdxl_models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"

# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
# load ip-adapter
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)
# read image prompt
# image = Image.open("/root/autodl-tmp/Proj/StoryDiffusion/examples/Harry2/00002-0-00070-0-Sports4-5.png")
image = Image.open("/root/autodl-tmp/Proj/StoryDiffusion/examples/woman.png")
image.resize((512, 512))

# generate image variations with only image prompt
num_samples = 4
# images = ip_model.generate(pil_image=image, num_samples=num_samples, num_inference_steps=30, seed=420)
# grid = image_grid(images, 1, num_samples)
# grid
output_path = "./res_adapter/res_woman_scale0.6_num1"
os.makedirs(output_path,exist_ok=True)
images = ip_model.generate(pil_image=image, height = 512,width = 512,num_samples=num_samples, num_inference_steps=30, seed=2047,
        prompt="best quality, high quality, standing on the beach", scale=0.6)
for idx, image in enumerate(images):
    image.save(f"{output_path}/generated_image_{idx}.png")
