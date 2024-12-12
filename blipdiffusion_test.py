from diffusers.pipelines import BlipDiffusionPipeline
from diffusers.utils import load_image
import torch

blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
    "Salesforce/blipdiffusion", torch_dtype=torch.float16
).to("cuda")


# cond_subject = "dog"
# tgt_subject = "dog"
# text_prompt_input = "a dog sitting beside a tree"

cond_subject = "robot"
tgt_subject = "robot"
text_prompt_input = "a robot standing"
cond_image = load_image("/root/autodl-tmp/Proj/StoryDiffusion/images/robot.png")

# cond_subject = "a cartoon-character"
# tgt_subject = "a cartoon-character"
# text_prompt_input = "a cartoon-character standing"
# # cond_image = load_image(
# #     "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/dog.jpg"
# # )

# cond_image =  load_image("/root/autodl-tmp/Proj/StoryDiffusion/examples/Harry/00001-0-00029-0-Harley Image.png")
guidance_scale = 7.5
num_inference_steps = 25
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"


output = blip_diffusion_pipe(
    text_prompt_input,
    cond_image,
    cond_subject,
    tgt_subject,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    neg_prompt=negative_prompt,
    height=512,
    width=512,
).images
output[0].save("image3.png")