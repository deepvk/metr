#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Uncomment the following lines to install the WatermarkAttacker"""
# !git clone https://github.com/XuandongZhao/WatermarkAttacker.git
# %cd WatermarkAttacker
# !pip install -r requirements.txt
# !pip install -e .


# In[2]:

import PIL
import sys
import torch
import os
import glob
import numpy as np

from wm_attacks import ReSDPipeline

from utils import eval_psnr_ssim_msssim, bytearray_to_bits
from wm_attacks.wmattacker_no_saving import DiffWMAttacker, VAEWMAttacker

from tqdm import tqdm

# In[3]:


wm_text = 'test'
device = 'cuda:0'
ori_path = 'images/imgs_no_w/'
output_path = 'images/imgs_w/'
print_width = 50


# In[4]:


os.makedirs(output_path, exist_ok=True)
ori_img_paths = glob.glob(os.path.join(ori_path, '*.*'))
ori_img_paths = sorted([path for path in ori_img_paths if path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])


# In[5]:


# wmarkers = {
#     'DwtDct': InvisibleWatermarker(wm_text, 'dwtDct'),
#     'DwtDctSvd': InvisibleWatermarker(wm_text, 'dwtDctSvd'),
#     'RivaGAN': InvisibleWatermarker(wm_text, 'rivaGan'),
# }

# 
wmarkers = {
    'Tree-Ring': None
}

# In[6]:


pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
pipe.set_progress_bar_config(disable=True)
pipe.to(device)
print('Finished loading model')


# In[7]:


attackers = {
    'diff_attacker_60': DiffWMAttacker(pipe, noise_step=60),
    'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
    'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
    # 'rotation_75': RotateAttacker(degree=75),
    # 'jpeg_attacker_25': JPEGAttacker(quality=25),
    # 'crop_0_75': CropAttacker(crop_size=0.75),
    # 'blur_r_4': GaussianBlurAttacker(kernel_size=4),
    # 'noise_std_0_1': GaussianNoiseAttacker(std=0.1),
    # 'brightness_0_6': BrightnessAttacker(br)
}

# In[8]:


# def add_watermark(wmarker_name, wmarker):
#     print('*' * print_width)
#     print(f'Watermarking with {wmarker_name}')
#     os.makedirs(os.path.join(output_path, wmarker_name + '/noatt'), exist_ok=True)
#     for ori_img_path in ori_img_paths:
#         img_name = os.path.basename(ori_img_path)
#         wmarker.encode(ori_img_path, os.path.join(output_path, wmarker_name + '/noatt', img_name))

# for wmarker_name, wmarker in wmarkers.items():
#     add_watermark(wmarker_name, wmarker)
# print('Finished watermarking')


# In[9]:

os.makedirs(os.path.join(output_path, wmarker_name, attacker_name), exist_ok=True)``

for attacker_name in attackers.keys():
    for ori_img_path in tqdm(ori_img_paths[:1]):
        img_name = os.path.basename(ori_img_path)
        wm_img_path = os.path.join(output_path, "Tree-Ring" + '/noatt', "w_" + img_name)
        wm_img = PIL.Image.open(wm_img_path)
        attacked_wm_img = attackers[attacker_name].attack(wm_img)
    
    attacked_wm_img.save(f"./test_attacks/{attacker_name}_test.png")

print('Finished attacking')


# In[10]:

# wm_results = {}
# for wmarker_name, wmarker in wmarkers.items():
#     print('*' * print_width)
#     print(f'Watermark: {wmarker_name}')
#     wm_successes = []
#     wm_psnr_list = []
#     wm_ssim_list = []
#     wm_msssim_list = []
#     for ori_img_path in ori_img_paths:
#         img_name = os.path.basename(ori_img_path)
#         wm_img_path = os.path.join(output_path, wmarker_name+'/noatt', img_name)
#         wm_psnr, wm_ssim, wm_msssim = eval_psnr_ssim_msssim(ori_img_path, wm_img_path)
#         wm_psnr_list.append(wm_psnr)
#         wm_ssim_list.append(wm_ssim)
#         wm_msssim_list.append(wm_msssim)
#     wm_results[wmarker_name] = {}
#     wm_results[wmarker_name]['wm_psnr'] = np.array(wm_psnr_list).mean()
#     wm_results[wmarker_name]['wm_ssim'] = np.array(wm_ssim_list).mean()
#     wm_results[wmarker_name]['wm_msssim'] = np.array(wm_msssim_list).mean()

# print('Finished evaluating watermarking')


# # In[11]:


# wm_results


# # In[12]:


# detect_wm_results = {}
# for wmarker_name, wmarker in wmarkers.items():
#     print('*' * print_width)
#     print(f'Watermark: {wmarker_name}')
#     bit_accs = []
#     wm_successes = []
#     for ori_img_path in ori_img_paths:
#         img_name = os.path.basename(ori_img_path)
#         wm_img_path = os.path.join(output_path, wmarker_name+'/noatt', img_name)
#         wm_text = wmarkers[wmarker_name].decode(wm_img_path)
#         try:
#             if type(wm_text) == bytes:
#                 a = bytearray_to_bits('test'.encode('utf-8'))
#                 b = bytearray_to_bits(wm_text)
#             elif type(wm_text) == str:
#                 a = bytearray_to_bits('test'.encode('utf-8'))
#                 b = bytearray_to_bits(wm_text.encode('utf-8'))
#             bit_acc = (np.array(a) ==  np.array(b)).mean()
#             bit_accs.append(bit_acc)
#             if bit_acc > 24/32:
#                 wm_successes.append(img_name)
#         except:
#             print('#' * print_width)
#             print(f'failed to decode {wm_text}', type(wm_text), len(wm_text))
#             pass
#     detect_wm_results[wmarker_name] = {}
#     detect_wm_results[wmarker_name]['bit_acc'] = np.array(bit_accs).mean()
#     detect_wm_results[wmarker_name]['wm_success'] = len(wm_successes) / len(ori_img_paths)
# print('Finished evaluating watermarking')


# # In[13]:


# detect_wm_results


# # In[14]:


# detect_att_results = {}
# for wmarker_name, wmarker in wmarkers.items():
#     print('*' * print_width)
#     print(f'Watermark: {wmarker_name}')
#     detect_att_results[wmarker_name] = {}
#     for attacker_name, attacker in attackers.items():
#         print(f'Attacker: {attacker_name}')
#         bit_accs = []
#         wm_successes = []
#         for ori_img_path in ori_img_paths:
#             img_name = os.path.basename(ori_img_path)
#             att_img_path = os.path.join(output_path, wmarker_name, attacker_name, img_name)
#             att_text = wmarkers[wmarker_name].decode(att_img_path)
#             try:
#                 if type(att_text) == bytes:
#                     a = bytearray_to_bits('test'.encode('utf-8'))
#                     b = bytearray_to_bits(att_text)
#                 elif type(att_text) == str:
#                     a = bytearray_to_bits('test'.encode('utf-8'))
#                     b = bytearray_to_bits(att_text.encode('utf-8'))
#                 bit_acc = (np.array(a) ==  np.array(b)).mean()
#                 bit_accs.append(bit_acc)
#                 if bit_acc > 24/32:
#                     wm_successes.append(img_name)
#             except:
#                 print('#' * print_width)
#                 print(f'failed to decode {wm_text}', type(wm_text), len(wm_text))
#                 pass
#         detect_att_results[wmarker_name][attacker_name] = {}
#         detect_att_results[wmarker_name][attacker_name]['bit_acc'] = np.array(bit_accs).mean()
#         detect_att_results[wmarker_name][attacker_name]['wm_success'] = len(wm_successes) / len(ori_img_paths)


# # In[15]:


# detect_att_results


# # In[16]:


# from IPython.display import Image


# # In[18]:


# img_id = '000000000711.png'
# Image(filename='examples/ori_imgs/'+img_id) # original image


# # In[19]:


# Image(filename='examples/wm_imgs/DwtDct/noatt/'+img_id) # watermarked image


# # In[20]:


# Image(filename='examples/wm_imgs/DwtDct/diff_attacker_60/'+img_id) # diffusion attacker

