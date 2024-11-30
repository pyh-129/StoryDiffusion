import torch
from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# 初始化 Accelerator 实例
accelerator = Accelerator()

# 加载并行化模型
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",  # 模型名称，可根据需要替换
    torch_dtype=torch.float16,  # 使用半精度以节省内存
    # device_map="auto"  # 自动分配模型到可用 GPU
)
device = "cuda"
# 将模型移动到 Accelerator 管理的设备
pipe.to(device)

# 配置推理参数
num_steps = 50
guidance_scale = 7.5
prompt = [    "a woman ake up in a cozy, sunlit bed, feeling refreshed.",
    "a woman sit down to a hearty breakfast, enjoying the favorite foods."
    ,"a woman take the road early, heading to the company amidst bustling city traffic.",
    ]
generator = torch.Generator(device=accelerator.device).manual_seed(42)

# 推理并生成图像
with torch.no_grad():
    result = pipe(prompt=prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale, generator=generator)

# 保存生成的图像
for i, image in enumerate(result.images):
    image.save(f"output_image_{i}.png")

# 打印完成消息
print("Image generated and saved as output_image.png")
