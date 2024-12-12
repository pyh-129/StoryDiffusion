# %load_ext autoreload
# %autoreload 2
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
if is_torch2_available():
    from utils.gradio_utils import \
        AttnProcessor2_0 as AttnProcessor
else:
    from utils.gradio_utils  import AttnProcessor
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import diffusers
from diffusers import StableDiffusionXLPipeline
from diffusers import DDIMScheduler
import torch.nn.functional as F
from utils.gradio_utils import cal_attn_mask_xl
import copy
import os
from diffusers.utils import load_image
from utils.utils import get_comic
from utils.style_template import styles
from accelerate import Accelerator
from accelerate.logging import get_logger
## Global
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"
MAX_SEED = np.iinfo(np.int32).max
global models_dict
use_va = False
models_dict = {
   "Juggernaut":"RunDiffusion/Juggernaut-XL-v8",
   "RealVision":"SG161222/RealVisXL_V4.0" ,
   "SDXL":"stabilityai/stable-diffusion-xl-base-1.0" ,
   "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y"
}



## Inversion
@torch.no_grad()
def invert(
    start_latents,
    prompt,
    pipe,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    
    # device=self.device,
):
    device = pipe.device

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        # cur_step2 = num_inference_steps - i
        
        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
class SpatialAttnProcessor2_1(torch.nn.Module):
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
        global total_count,attn_count,cur_step2,mask1024,mask4096
        global sa32, sa64
        global write
        global height,width
        if write:
            # print(f"white:{cur_step}")
            self.id_bank[cur_step2] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            encoder_hidden_states = torch.cat((self.id_bank[cur_step2][0].to(self.device),hidden_states[:1],self.id_bank[cur_step2][1].to(self.device),hidden_states[1:]))
        # skip in early step
        if cur_step2 <5:
            hidden_states = self.__call2__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
        else:   # 256 1024 4096
            random_number = random.random()
            if cur_step2 <20:
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
            cur_step2 -= 1
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

def set_attention_processor(unet,id_length):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks") :
                attn_procs[name] = SpatialAttnProcessor2_0(id_length = id_length)
            else:    
                attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(attn_procs)
    
    
    
    
    
 
    
global attn_count, total_count, id_length, total_length,cur_step,cur_step2, cur_model_type,id_bank
global write
global  sa32, sa64
global height,width
attn_count = 0
total_count = 0
cur_step = 0
cur_step2=0
id_length = 4
total_length = 5
cur_model_type = ""
device="cuda"
global attn_procs,unet
attn_procs = {}
###
write = False
### strength of consistent self-attention: the larger, the stronger
sa32 = 0.5
sa64 = 0.5
### Res. of the Generated Comics. Please Note: SDXL models may do worse in a low-resolution! 
height = 512
width = 512
accelerator = Accelerator(mixed_precision="fp16")
num_steps = 50
###
global pipe
global sd_model_path








sd_model_path = models_dict["RealVision"] #"SG161222/RealVisXL_V4.0"
### LOAD Stable Diffusion Pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(sd_model_path, torch_dtype=torch.float16,device_map='auto', use_safetensors=True)
# pipe = pipe.to(device)
pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(50)
unet = pipe.unet

### Insert PairedAttention
for name in unet.attn_processors.keys():
    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config.block_out_channels[block_id]
    if cross_attention_dim is None and (name.startswith("up_blocks") ) :
        attn_procs[name] =  SpatialAttnProcessor2_0(id_length = id_length)
        total_count +=1
    else:
        attn_procs[name] = AttnProcessor()
print("successsfully load consistent self-attention")
print(f"number of the processor : {total_count}")
unet.set_attn_processor(copy.deepcopy(attn_procs))
global mask1024,mask4096
mask1024, mask4096 = cal_attn_mask_xl(total_length,id_length,sa32,sa64,height,width,device=device,dtype= torch.float16)



guidance_scale = 5.0
seed = 2047
# seed = 2047
sa32 = 0.5
sa64 = 0.5
# id_length = 4
num_steps = 50
# general_prompt = "a man with a black suit"
general_prompt = "a girl with a shirt"
negative_prompt = "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"
# prompt_array = ["wake up in the bed",
#                 "have breakfast",
#                 "is on the road, go to the company",
#                 "work in the company",
#                 "running in the playground",
#                 "reading book in the home"
#                 ]
# prompt_array = [
#     "Wake up in a cozy, sunlit bed, feeling refreshed.",
#     "Sit down to a hearty breakfast, enjoying your favorite foods.",
#     "Take the road early, heading to the company amidst bustling city traffic.",
#     "Focus on a challenging project at the company's office.",
#     "Run laps in the local playground, feeling the wind and freedom of movement.",
#     "Curl up at home with a captivating book, losing track of time in its pages.",
#     "Stroll through a lively market, picking out fresh produce and artisan goods.",
#     "Meet a friend for a leisurely lunch at a bustling café.",
#     "Spend an afternoon volunteering at a local community center.",
#     "Cook a delicious dinner, experimenting with new recipes and flavors.",
#     "Watch the sunset from a scenic overlook, reflecting on the day's experiences.",
#     "Settle in for an evening of films and relaxation, enjoying some well-earned downtime."
# ]
# prompt_array = [
#     "Once upon a time, in a village nestled between the hills, Lived a boy, with magic in his quills.",
#     "One day, the boy decided to explore the forest near, where whispers of magic and wonder did appear.",
#     "He met fairies who danced on the dew-kissed grass, And shared stories of magic, that filled Harry with awe.",
#     "He followed a path that was winding and long, to a hidden cave where the magic did throng.",
#     "Inside, he found crystals that shone bright and clear, And a pool where the waters were magic to hear.",
#     "In this enchanted forest, every step the boy took was filled with surprise and discovery.",
#     "He encountered a talking stag, with antlers that glittered like the stars.",
#     "The stag spoke of ancient secrets and paths that led to hidden realms.",
#     "Eager to learn more, the boy followed the stag through the twisting underbrush to a glade illuminated by fireflies.",
#     "There, the fireflies swirled around a majestic oak, under which sat an old wise owl.",
#     "The owl, knowing the boy's quest for knowledge, bestowed upon him a golden feather.",
#     "This feather, he explained, held the essence of true magic, capable of unlocking the mysteries of the forest and beyond.",
#     "With the golden feather in his quill, the boy wrote down his adventures.",
#     "Each word he penned sparkled with magic, bringing his stories to life before his eyes.",
#     "Trees whispered their ancient tales, rivers sang of forgotten kingdoms, and the winds carried melodies from distant lands.",
#     "His journey grew ever more magical, as he learned to harness the power of his enchanted quill.",
#     "Writing stories that bridged the world of humans and the mystical forces of nature.",
#     "The boy's tales spread far and wide, inspiring all who heard them to believe in the magic that surrounds us, hidden just beyond the veil of the ordinary."
# ]


# prompt_array = [
#     "Amy was a little girl who lived in a small, quiet town. She had a toy bear named Teddy, who was her best friend and confidant.",
#     "Every day after school, Amy would sit by the window with Teddy, dreaming about all the adventures they could have together.",
#     "One afternoon, Amy decided she wanted to do something special for Teddy, so she carefully crafted a tiny scarf for him using colorful threads.",
#     "When Amy placed the scarf around Teddy's neck, she smiled and imagined that Teddy, like her, was ready for a grand adventure.",
#     "That evening, as the sun set, Amy whispered to Teddy that they were going to have the best adventure ever, and she promised to always keep him by her side.",
#     "The next day, Amy took Teddy outside and showed him the wonders of the garden, telling him all about the birds, the flowers, and the little insects she’d met.",
#     "As they explored, Amy began to realize that every day, with Teddy by her side, even the smallest moments were full of magic and joy.",
#     "From that day on, Amy always made time for her adventures with Teddy, no matter how simple or ordinary they seemed, because with him, every day felt extraordinary."
# ]





# prompt_array = [
#     # "Amy was a little girl who woke up early. She felt tired but excited.",
#     "A little girl woke up early. She felt tired but excited.",
#     "She quickly got dressed. Today was a new day.",
#     "At school, Amy felt nervous. It was her first test.",
#     "Her heart raced. She picked up her pencil.",
#     "As she worked, she felt more confident.",
#     "When the test ended, Amy felt relieved.",
#     "Later, she met her friend. They laughed together.",
#     "As the sun set, Amy felt peaceful.",
#     "At bedtime, she felt proud of herself."
# ]
prompt_array = [
    # "Amy was a little girl who woke up early. She felt tired but excited.",
    "Woke up early from the bed, feeling the warmth of the morning sun on the face.",
    "Swung legs off the bed, feet touching the cool floor, ready to start the day with energy.",
    "Walked to the bathroom, brushing teeth quickly while thinking about the events of the day ahead.",
    "Put on a favorite dress, soft and colorful, feeling its comfort as it fit just right.",
    "At school, sat at the desk, taking out the notebook and preparing for the lessons to come.",
    "Raised a hand to answer a question, speaking softly but clearly, with a quiet confidence.",
    "After class, grabbed the bag and smiled at a friend, feeling the joy of a day well spent.",
    "Ran outside together, the wind against the face, laughing and enjoying the simple pleasure of movement.",
    "Later, sat quietly under a tree, watching the leaves fall gently, feeling peaceful in the moment."
]
prompt_array = [
        "Walked to the kitchen, poured a glass of water, and took a quiet moment to drink it, feeling the coolness refresh the senses.",
    "Sat at the table, opening a book and reading the first few pages with curiosity, absorbed in the new ideas unfolding.",
    "At school, sat at the desk, paying attention to the teacher’s explanation and taking notes, eager to learn and understand more.",
    "Raised a hand to ask a question, wanting to understand the subject more clearly, feeling a sense of engagement with the material.",
    "After class, grabbed the bag and walked outside, feeling the cool breeze on the skin, enjoying the shift from the warmth of the classroom.",
    "Met a friend in the hallway, shared a quick laugh, and talked about the next class, feeling the comfort of familiar company.",
    "Later, sat on a bench outside, watching the sky and feeling a sense of peace in the moment, letting the quiet wash over and clear the mind.",
    "A few moments passed, the clouds drifting lazily by, while thoughts about the day’s events slowly settled into contentment.",
    "Stood up from the bench, stretching arms to the sky, ready to continue the day with a refreshed and calm spirit.",
    "Walked back into the building, feeling the steady rhythm of footsteps, mentally preparing for whatever the next moment would bring."
]

prompt_array = [
    "loved her toy bear, teddy.",
    "every day after school, hugged him tightly and talked to him.",
    "one afternoon, felt sad, wondering if teddy ever felt lonely.",
    "decided to make him a tiny scarf to show how much cared.",
    "when finished, smiled and placed the scarf around teddy's neck.",
    "felt proud. teddy looked even more special now.",
    "that evening, whispered to teddy, 'tomorrow, we'll have an adventure.'",
    "the next day, felt excited. couldn’t wait to take teddy outside.",
    "in the garden, showed teddy all the flowers and insects.",
    "as they played, felt happy, realizing how special every moment was with teddy.",
    "from then on, knew that no matter what, teddy would always be there by side."
]





def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive) 
def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [p.replace("{prompt}", positive) for positive in positives], n + ' ' + negative
### Set the generated Style
style_name = "Photographic"
setup_seed(seed)
generator = torch.Generator(device="cuda").manual_seed(seed)
prompts = [general_prompt+","+prompt for prompt in prompt_array]
print(prompts)
id_prompts = prompts[:id_length]
real_prompts = prompts[id_length:]
torch.cuda.empty_cache()
write = True
cur_step = 0
attn_count = 0
id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)
id_images = pipe(id_prompts, num_inference_steps = num_steps, guidance_scale=guidance_scale,  height = height, width = width,negative_prompt = negative_prompt,generator = generator).images

write = False
output_path = "./res_t/res_real7_teddy"
os.makedirs(output_path,exist_ok=True)
for idx,id_image in enumerate(id_images):
    id_image.save(f'{output_path}/id_image_{idx}.png')
real_images = []
for real_prompt in real_prompts:
    cur_step = 0
    real_prompt = apply_style_positive(style_name, real_prompt)
    real_images.append(pipe(real_prompt,  num_inference_steps=num_steps, guidance_scale=guidance_scale,  height = height, width = width,negative_prompt = negative_prompt,generator = generator).images[0])
for idx, real_image in enumerate(real_images):
    real_image.save(f'{output_path}/real_image_{idx}.png')



