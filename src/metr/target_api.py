#!/usr/bin/env python3
import torch
import torchvision
import argparse

from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline, DDIMScheduler
from .inverse_stable_diffusion import InversableStableDiffusionPipeline

from ._get_noise import get_noise
from ._detect import detect

from PIL import Image
import requests
from io import BytesIO

def main(args):
    model_id = 'stabilityai/stable-diffusion-2-1-base'

    org_name = args.org_name
    model_hash = args.model_hash
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model_id = args.model_id
    # scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    # pipe = InversableStableDiffusionPipeline.from_pretrained(
    #     args.model_id,
    #     scheduler=scheduler,
    #     torch_dtype=torch.float16,
    #     revision='fp16',
    #     )
    # pipe = pipe.to(device)

    # IMPORTANT: We need to make sure to be able to use a normal diffusion pipeline so that people see 
    # the tree-ring-watermark method as general enough
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')
    # or
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe = DiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16,)
    pipe = pipe.to(device)

    shape = (1, 4, 64, 64)
    latents = get_noise(shape, model_hash, org=org_name)

    watermarked_image = pipe(prompt="an astronaut", latents=latents).images[0]

    is_watermarked = detect(watermarked_image, pipe, model_hash, org=org_name)
    print(f'is_watermarked: {is_watermarked}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--org_name', default='Alphonsce')
    parser.add_argument('--model_hash', default='StableWM')
#-----------------
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps

    args = parser.parse_args()
    
    main(args)