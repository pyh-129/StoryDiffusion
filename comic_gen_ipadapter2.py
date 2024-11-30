import gradio as gr
import numpy as np
import torch
import requests
import random
import os
import sys
import pickle
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
from utils.gradio_utils import is_torch2_available
from accelerate import PartialState
from diffusers import DiffusionPipeline
if is_torch2_available():
    from utils.gradio_utils import AttnProcessor2_0 as AttnProcessor
else:
    from utils.gradio_utils import AttnProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
import diffusers
from diffusers import StableDiffusionXLPipeline
from diffusers import DDIMScheduler
import torch.nn.functional as F
from utils.gradio_utils import cal_attn_mask_xl
# from ip_adapter import IPAttnProcessor2_0
# from utils.ip_adapter.attention_processor import  AttnProcessor2_0 as AttnProcessor
import copy
from diffusers.utils import load_image
from utils.utils import get_comic
from utils.style_template import styles
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.pipeline import PhotoMakerStableDiffusionXLPipeline
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
# photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

from accelerate import infer_auto_device_map
from ip_adapter import IPAdapterPlus
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapterPlusXL
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
accelerator = Accelerator(mixed_precision="fp16")

# 全局变量
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"
MAX_SEED = np.iinfo(np.int32).max
global models_dict
use_va = False
models_dict = {
   "Juggernaut": "RunDiffusion/Juggernaut-XL-v8",
   "RealVision": "SG161222/RealVisXL_V4.0",
   "SDXL":"stabilityai/stable-diffusion-xl-base-1.0",
   "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y"
}
torch.cuda.is_available()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Consistent Self-Attention 代码块
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    
#################################################
########Consistent Self-Attention################
#################################################
class SpatialAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size = None, cross_attention_dim=None,id_length = 4,device = "cuda",dtype = torch.float16):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        global total_count,attn_count,cur_step,mask1024,mask4096
        global sa32, sa64
        global write
        global height,width
        if write:
            # print(f"white:{cur_step}")
            self.id_bank[cur_step] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            encoder_hidden_states = torch.cat((self.id_bank[cur_step][0].to(self.device),hidden_states[:1],self.id_bank[cur_step][1].to(self.device),hidden_states[1:]))
        # skip in early step
        if cur_step <5:
            hidden_states = self.__call2__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
        else:   # 256 1024 4096
            random_number = random.random()
            if cur_step <20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                if not write:
                    if hidden_states.shape[1] == (height//32) * (width//32):
                        attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    if hidden_states.shape[1] == (height//32) * (width//32):
                        attention_mask = mask1024[:mask1024.shape[0] // self.total_length * self.id_length,:mask1024.shape[0] // self.total_length * self.id_length]
                    else:
                        attention_mask = mask4096[:mask4096.shape[0] // self.total_length * self.id_length,:mask4096.shape[0] // self.total_length * self.id_length]
                hidden_states = self.__call1__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states,None,attention_mask,temb)
        attn_count +=1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024,mask4096 = cal_attn_mask_xl(self.total_length,self.id_length,sa32,sa64,height,width, device=self.device, dtype= self.dtype)

        return hidden_states
    def __call1__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
        total_batch_size,nums_token,channel = hidden_states.shape
        img_nums = total_batch_size//2
        hidden_states = hidden_states.view(-1,img_nums,nums_token,channel).reshape(-1,img_nums * nums_token,channel)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,nums_token,channel).reshape(-1,(self.id_length+1) * nums_token,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)



        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)


        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        # print(hidden_states.shape)
        return hidden_states
    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channel = (
            hidden_states.shape
        )
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,sequence_length,channel).reshape(-1,(self.id_length+1) * sequence_length,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

# 初始化数据集和Dataloader

    
# def load_image(
#     image: Union[str, PIL.Image.Image], convert_method: Callable[[PIL.Image.Image], PIL.Image.Image] = None
# ) -> PIL.Image.Image:
#     """
#     Loads `image` to a PIL Image.

#     Args:
#         image (`str` or `PIL.Image.Image`):
#             The image to convert to the PIL Image format.
#         convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], optional):
#             A conversion method to apply to the image after loading it. When set to `None` the image will be converted
#             "RGB".

#     Returns:
#         `PIL.Image.Image`:
#             A PIL Image.
#     """
#     if isinstance(image, str):
#         if image.startswith("http://") or image.startswith("https://"):
#             image = PIL.Image.open(requests.get(image, stream=True).raw)
#         elif os.path.isfile(image):
#             image = PIL.Image.open(image)
#         else:
#             raise ValueError(
#                 f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
#             )
#     elif isinstance(image, PIL.Image.Image):
#         image = image
#     else:
#         raise ValueError(
#             "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
#         )

#     image = PIL.ImageOps.exif_transpose(image)

#     if convert_method is not None:
#         image = convert_method(image)
#     else:
#         image = image.convert("RGB")

#     return image


# def set_attention_processor(unet, id_length):
#     attn_procs = {}
#     for name in unet.module.attn_processors.keys():
#         cross_attention_dim = None if name.endswith("attn1.processor") else unet.module.config.cross_attention_dim
#         if name.startswith("mid_block"):
#             hidden_size = unet.module.config.block_out_channels[-1]
#         elif name.startswith("up_blocks"):
#             block_id = int(name[len("up_blocks.")])
#             hidden_size = list(reversed(unet.module.config.block_out_channels))[block_id]
#         elif name.startswith("down_blocks"):
#             block_id = int(name[len("down_blocks.")])
#             hidden_size = unet.module.config.block_out_channels[block_id]
#         if cross_attention_dim is None:
#             if name.startswith("up_blocks"):
#                 attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
#             else:
#                 attn_procs[name] = AttnProcessor()
#         else:
#             attn_procs[name] = AttnProcessor()
#     unet.module.set_attn_processor(attn_procs)
# def set_attention_processor(unet, id_length, is_ipadapter=True):
#     global total_count
#     total_count = 0
#     attn_procs = {}
#     for name in unet.attn_processors.keys():
#         cross_attention_dim = (
#             None
#             if name.endswith("attn1.processor")
#             else unet.config.cross_attention_dim
#         )
#         if name.startswith("mid_block"):
#             hidden_size = unet.config.block_out_channels[-1]
#         elif name.startswith("up_blocks"):
#             block_id = int(name[len("up_blocks.")])
#             hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
#         elif name.startswith("down_blocks"):
#             block_id = int(name[len("down_blocks.")])
#             hidden_size = unet.config.block_out_channels[block_id]
#         if cross_attention_dim is None:
#             # if name.startswith("up_blocks"):
#             #     attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
#             #     total_count += 1
#             # else:
#                 attn_procs[name] = AttnProcessor()
#         else:
#             if is_ipadapter:
#                 attn_procs[name] = IPAttnProcessor2_0(
#                     hidden_size=hidden_size,
#                     cross_attention_dim=cross_attention_dim,
#                     scale=1,
#                     num_tokens = 4,
#                 ).to(unet.device, dtype=torch.float16)
#             else:
#                 attn_procs[name] = AttnProcessor()

#     unet.set_attn_processor(copy.deepcopy(attn_procs))
#     print("Successfully load paired self-attention")
#     print(f"Number of the processor : {total_count}")

# 加载 Stable Diffusion 管道
global attn_count, total_count, id_length, total_length, cur_step, cur_model_type
global write
global sa32, sa64
global height, width
attn_count = 0
total_count = 0
cur_step = 0
id_length = 2
total_length= 3
cur_model_type = ""
# device = "cuda"
global attn_procs, unet
attn_procs = {}
write = False
sa32 = 0.5
sa64 = 0.5
height = 512
width = 512
global pipe
global sd_model_path
# sd_model_path = models_dict["RealVision"]
sd_model_path = models_dict["SDXL"]

# base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
# image_encoder_path = "sdxl_models/image_encoder"
# ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"
base_model_path = "SG161222/RealVisXL_V1.0"
image_encoder_path = "models/models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
# vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
# load SD pipeline
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image


pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
# ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)
ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, id_length = id_length, num_tokens=16)

### define the input ID images
input_folder_name = '/root/autodl-tmp/Proj/StoryDiffusion/examples/lecun'
image_basename_list = os.listdir(input_folder_name)
image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])

input_id_images = []
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))


# prompts = ["comic Cartoon-character img is sitting . graphic illustration, comic art, graphic novel art, vibrant, highly detailed","comic Cartoon-character img is standing . graphic illustration, comic art, graphic novel art, vibrant, highly detailed","comic Cartoon-character img is working . graphic illustration, comic art, graphic novel art, vibrant, highly detailed","comic Cartoon-character img is eating . graphic illustration, comic art, graphic novel art, vibrant, highly detailed"]  # 替换为你的id_prompts列表
# prompts = ["a cartoon-character img is sitting . graphic illustration, comic art, graphic novel art, vibrant, highly detailed","a Cartoon-character img is standing . graphic illustration, comic art, graphic novel art, vibrant, highly detailed","a Cartoon-character img is working . graphic illustration, comic art, graphic novel art, vibrant, highly detailed","a Cartoon-character img is eating . graphic illustration, comic art, graphic novel art, vibrant, highly detailed"] 
# prompts = [
#     "a cartoon-character is standing.",
#     "a cartoon-character is playing the guitar on a stage with colorful lights.",
#     "a cartoon-character is riding a bicycle through a bustling city street.",
#     "a cartoon-character is painting a beautiful sunset on a canvas near the beach.",
#     "a cartoon-character is reading a book under a tree with falling leaves.",
#     "a cartoon-character is eating an ice cream cone on a hot summer day.",
#     # "a cartoon character is cooking in a cozy kitchen with pots and pans around.",
#     # "a cartoon character is flying a kite in a wide-open field with a big smile.",
#     # "a cartoon character is building a sandcastle by the ocean.",
#     # "a cartoon character is playing soccer with friends in a green field.",
#     # "a cartoon character is exploring a forest with a flashlight at dusk.",
#     # "a cartoon character is having a picnic on a checkered blanket with a basket of food.",
#     # "a cartoon character is stargazing through a telescope on a hilltop.",
#     # "a cartoon character is dancing joyfully in a field of flowers.",
#     # "a cartoon character is painting a mural on a large wall in a city square."
# ]
prompts = [
    "standing.",
    "playing the guitar on a stage with colorful lights.",
    "riding a bicycle through a bustling city street.",
    "painting a beautiful sunset on a canvas near the beach.",
    "reading a book under a tree with falling leaves.",
    "eating an ice cream cone on a hot summer day.",
    # "a cartoon character is cooking in a cozy kitchen with pots and pans around.",
    # "a cartoon character is flying a kite in a wide-open field with a big smile.",
    # "a cartoon character is building a sandcastle by the ocean.",
    # "a cartoon character is playing soccer with friends in a green field.",
    # "a cartoon character is exploring a forest with a flashlight at dusk.",
    # "a cartoon character is having a picnic on a checkered blanket with a basket of food.",
    # "a cartoon character is stargazing through a telescope on a hilltop.",
    # "a cartoon character is dancing joyfully in a field of flowers.",
    # "a cartoon character is painting a mural on a large wall in a city square."
]

dataset = PromptDataset(prompts)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

pipe,dataloader = accelerator.prepare(pipe,dataloader)
pipe = pipe.to(accelerator.device)
unet = pipe.unet
# unet.to(accelerator.device)
# 
from accelerate import load_checkpoint_and_dispatch

# model = load_checkpoint_and_dispatch(
#     pipe, checkpoint=checkpoint_file, device_map="auto"
# )

guidance_scale = 5.0
seed = 2047
sa32 = 0.5
sa64 = 0.5

num_steps = 50
general_prompt = "a man with a black suit"
negative_prompt = "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"
prompt_array = ["wake up in the bed",
                "have breakfast",
                "is on the road, go to the company",
                "work in the company",
                "running in the playground",
                "reading book in the home"
                ]
# 插入 PairedAttention
# for name in unet.module.attn_processors.keys():
#     cross_attention_dim = None if name.endswith("attn1.processor") else unet.module.config.cross_attention_dim
#     if name.startswith("mid_block"):
#         hidden_size = unet.module.config.block_out_channels[-1]
#     elif name.startswith("up_blocks"):
#         block_id = int(name[len("up_blocks.")])
#         hidden_size = list(reversed(unet.module.config.block_out_channels))[block_id]
#     elif name.startswith("down_blocks"):
#         block_id = int(name[len("down_blocks.")])
#         hidden_size = unet.module.config.block_out_channels[block_id]
#     if cross_attention_dim is None and (name.startswith("up_blocks")):
#         attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
#         total_count += 1
#     else:
#         attn_procs[name] = AttnProcessor()
# for name in unet.attn_processors.keys():
#     cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
#     if name.startswith("mid_block"):
#         hidden_size = unet.config.block_out_channels[-1]
#     elif name.startswith("up_blocks"):
#         block_id = int(name[len("up_blocks.")])
#         hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
#     elif name.startswith("down_blocks"):
#         block_id = int(name[len("down_blocks.")])
#         hidden_size = unet.config.block_out_channels[block_id]
#     if cross_attention_dim is None and (name.startswith("up_blocks")):
#         attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
#         total_count += 1
#     else:
#         attn_procs[name] = AttnProcessor()

# # unet.module.set_attn_processor(copy.deepcopy(attn_procs))
# unet.set_attn_processor(copy.deepcopy(attn_procs))

# from predict import set_attention_processor
# set_attention_processor(unet,id_length,is_ipadapter=True)
global mask1024, mask4096
mask1024, mask4096 = cal_attn_mask_xl(total_length, id_length, sa32, sa64, height, width, device=accelerator.device, dtype=torch.float16)
def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive) 
def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [p.replace("{prompt}", positive) for positive in positives], n + ' ' + negative
# 设置生成样式
style_name = "Vibrant Cartoon Character"
setup_seed(seed)
generator = torch.Generator(device="cuda").manual_seed(seed)
# prompts = [general_prompt + "," + prompt for prompt in prompt_array]
id_prompts = prompts[:id_length]
real_prompts = prompts[id_length:]
torch.cuda.empty_cache()
write = True
cur_step = 0
attn_count = 0
id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)
# id_images = pipe(id_prompts, num_inference_steps=num_steps, guidance_scale=guidance_scale, height=height, width=width, negative_prompt=negative_prompt, generator=generator).images
print(id_prompts)
style_strength_ratio = 20
start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
if start_merge_step > 30:
    start_merge_step = 30
from diffusers.utils import load_image
input_id_images = [load_image(str("/root/autodl-tmp/Proj/StoryDiffusion/examples/Harry/00001-0-00029-0-Harley Image.png")),load_image("/root/autodl-tmp/Proj/StoryDiffusion/examples/Harry/00013-0-00076-0-Sports2-1.png"),load_image("/root/autodl-tmp/Proj/StoryDiffusion/examples/Harry2/00002-0-00070-0-Sports4-5.png")]
input_id_images = [img.resize((512, 512), Image.LANCZOS) for img in input_id_images]

# input_id_images = [img.to(accelerator.device) for img in input_id_images]
# negative_prompt = "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"
output_path = "./res_t/res_pho2_ipdapter3_scale0.1"
os.makedirs(output_path,exist_ok=True)
# for idx,batch in enumerate(dataloader):
    # print(negative_prompt)
    # print(negative_prompt[0])
print(id_prompts[0])
print("negative")
print(negative_prompt)
id_images = ip_model.generate(pil_image=input_id_images[1],height = height,width = width,prompt = id_prompts[0], num_samples=2, num_inference_steps=50, seed=420,scale = 1, negative_prompt=negative_prompt)

# id_images = pipe(prompt=batch, input_id_images=input_id_images,num_inference_steps=num_steps, guidance_scale=guidance_scale, height=height, width=width, negative_prompt=negative_prompt, generator=generator,num_images_per_prompt=1,start_merge_step=start_merge_step).images

# id_images[0].save(f"generated_image3.png")

for i, img in enumerate(id_images):
    img.save(f"{output_path}/ida_generated_image_{i}.png")
write = False
real_images = []

# print(real_prompts)
for real_prompt in real_prompts:
    cur_step = 0
    real_prompt = apply_style_positive(style_name, real_prompt)
    print(real_prompt)
    real_images.append(ip_model.generate(pil_image=input_id_images[1],height = height,width = width,prompt = real_prompt, num_samples=1, num_inference_steps=50, scale = 0.1, seed=420,negative_prompt=negative_prompt)[0])

# # 保存或处理生成的图片
for idx, image in enumerate(id_images + real_images):
    image.save(f"{output_path}/generated_image_{idx}.png")
