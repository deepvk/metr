import argparse
import copy
import glob
import json
import math
import os
import sys

import numpy as np
import PIL
import torch
import wandb
from diffusers import DPMSolverMultistepScheduler
from PIL import Image, ImageFile
from pytorch_msssim import ssim
from tqdm import tqdm
from wm_attacks import ReSDPipeline
from wm_attacks.wmattacker_with_saving import DiffWMAttacker, VAEWMAttacker

from .inverse_stable_diffusion import InversableStableDiffusionPipeline
from .io_utils import *
from .optim_utils import *
from .pytorch_fid.fid_score import *
from .stable_sig.utils_model import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return 100
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.0).item()


def compute_ssim(a, b):
    return ssim(a, b, data_range=1.0).item()


def eval_psnr_ssim_msssim(ori_img_path, new_img_path):
    ori_img = Image.open(ori_img_path).convert("RGB")
    new_img = Image.open(new_img_path).convert("RGB")
    if ori_img.size != new_img.size:
        new_img = new_img.resize(ori_img.size)
    ori_x = transforms.ToTensor()(ori_img).unsqueeze(0)
    new_x = transforms.ToTensor()(new_img).unsqueeze(0)
    return compute_psnr(ori_x, new_x), compute_ssim(ori_x, new_x)


def main(args):
    table = None
    if args.with_tracking:
        wandb.init(project=args.project_name, name=args.run_name, tags=["tree_ring_watermark_fid"])
        wandb.config.update(args)
        table = wandb.Table(columns=["gen_no_w", "gen_w", "prompt"])

    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision="fp16",
    )
    pipe = pipe.to(device)

    if args.use_stable_sig:
        pipe = change_pipe_vae_decoder(pipe, weights_path=args.decoder_state_dict_path, args=args)
        print("VAE CHANGED!")

    # hard coding for now
    with open(args.prompt_file) as f:
        dataset = json.load(f)
        image_files = dataset["images"]
        dataset = dataset["annotations"]
        prompt_key = "caption"

    no_w_dir = args.image_folder + "/no_w_gen"
    w_dir = args.image_folder + "/w_gen"
    if args.attack_type == "diff":
        att_w_dir = args.image_folder + f"/diff_{args.diff_attack_steps}"
    if args.attack_type == "vae":
        att_w_dir = args.image_folder + f"/{args.vae_attack_name}_{args.vae_attack_quality}"
    os.makedirs(no_w_dir, exist_ok=True)
    os.makedirs(w_dir, exist_ok=True)

    # ground-truth patch
    if not args.use_random_msgs:
        gt_patch = get_watermarking_pattern(pipe, args, device)

    if args.run_generation:
        for i in tqdm(range(args.start, args.end)):
            seed = i + args.gen_seed

            current_prompt = dataset[i][prompt_key]

            if args.use_random_msgs:
                msg_key = torch.randint(0, 2, (1, args.w_radius), dtype=torch.float32, device="cpu")
                msg_str = "".join([str(int(ii)) for ii in msg_key.tolist()[0]])

            if args.use_random_msgs:
                gt_patch = get_watermarking_pattern(pipe, args, device, message=msg_str)

            ### generation
            # generation without watermarking
            set_random_seed(seed)
            init_latents_no_w = pipe.get_random_latents()

            if args.run_no_w:
                outputs_no_w = pipe(
                    current_prompt,
                    num_images_per_prompt=args.num_images,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    height=args.image_length,
                    width=args.image_length,
                    latents=init_latents_no_w,
                )
                orig_image_no_w = outputs_no_w.images[0]
            else:
                orig_image_no_w = None

            # generation with watermarking
            if init_latents_no_w is None:
                set_random_seed(seed)
                init_latents_w = pipe.get_random_latents()
            else:
                init_latents_w = copy.deepcopy(init_latents_no_w)

            # get watermarking mask
            watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

            # inject watermark
            init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)

            outputs_w = pipe(
                current_prompt,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=init_latents_w,
            )
            orig_image_w = outputs_w.images[0]

            # distortion
            orig_image_no_w_auged, orig_image_w_auged = image_distortion(orig_image_no_w, orig_image_w, seed, args)
            orig_image_no_w, orig_image_w = orig_image_no_w_auged, orig_image_w_auged

            if args.with_tracking:
                if i < args.max_num_log_image:
                    if args.run_no_w:
                        table.add_data(wandb.Image(orig_image_no_w), wandb.Image(orig_image_w), current_prompt)
                    else:
                        table.add_data(None, wandb.Image(orig_image_w), current_prompt)
                else:
                    table.add_data(None, None, current_prompt)

            image_file_name = image_files[i]["file_name"]
            if args.run_no_w:
                orig_image_no_w.save(f"{no_w_dir}/{image_file_name}")
            orig_image_w.save(f"{w_dir}/{image_file_name}")

    ### calculate fid
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()

    num_workers = min(num_cpus, 8) if num_cpus is not None else 0

    ori_img_paths = glob.glob(os.path.join(no_w_dir, "*.*"))
    ori_img_paths = sorted(
        [path for path in ori_img_paths if path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    )

    if args.use_attack:
        if args.attack_type == "diff":
            attack_pipe = ReSDPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16"
            )
            attack_pipe.set_progress_bar_config(disable=True)
            attack_pipe.to(device)
            attacker = DiffWMAttacker(attack_pipe, noise_step=args.diff_attack_steps, batch_size=5, captions={})
        if args.attack_type == "vae":
            attacker = VAEWMAttacker(args.vae_attack_name, quality=args.vae_attack_quality, metric="mse", device=device)

        wm_img_paths = []
        att_img_paths = []
        os.makedirs(os.path.join(att_w_dir), exist_ok=True)
        for ori_img_path in ori_img_paths:
            img_name = os.path.basename(ori_img_path)
            wm_img_paths.append(os.path.join(w_dir, img_name))
            att_img_paths.append(os.path.join(att_w_dir, img_name))
        # using attacker on whole folder: we are using from with_saving file
        attacker.attack(wm_img_paths, att_img_paths)

    # fid for no_w
    target_folder = args.gt_folder
    if args.target_clean_generated:
        target_folder = no_w_dir

    if args.run_no_w:
        fid_value_no_w = calculate_fid_given_paths([target_folder, no_w_dir], 50, device, 2048, num_workers)
    else:
        fid_value_no_w = None

    # fid for w
    fid_value_w = calculate_fid_given_paths([target_folder, w_dir], 50, device, 2048, num_workers)

    # fid for att_w
    if args.use_attack:
        fid_value_w_att = calculate_fid_given_paths([target_folder, att_w_dir], 50, device, 2048, num_workers)

    # psnr and ssim
    if args.additional_metrics:
        # no_w_dir - это orig_path
        # w_dir - это wm_path
        # att_w_dir - это att_path
        clean_psnr_list = []
        clean_ssim_list = []

        wm_psnr_list = []
        wm_ssim_list = []

        att_psnr_list = []
        att_ssim_list = []

        for ori_img_path in tqdm(ori_img_paths):
            img_name = os.path.basename(ori_img_path)
            wm_img_path = os.path.join(w_dir, img_name)
            att_img_path = os.path.join(att_w_dir, img_name)
            target_image_path = os.path.join(target_folder, img_name)

            clean_psnr, clean_ssim = eval_psnr_ssim_msssim(target_image_path, ori_img_path)
            clean_psnr_list.append(clean_psnr)
            clean_ssim_list.append(clean_ssim)

            wm_psnr, wm_ssim = eval_psnr_ssim_msssim(target_image_path, wm_img_path)
            wm_psnr_list.append(wm_psnr)
            wm_ssim_list.append(wm_ssim)

            if args.use_attack:
                att_psnr, att_ssim = eval_psnr_ssim_msssim(target_image_path, att_img_path)
                att_psnr_list.append(att_psnr)
                att_ssim_list.append(att_ssim)

        clean_psnr = np.array(clean_psnr_list).mean()
        clean_ssim = np.array(clean_ssim_list).mean()
        wm_psnr = np.array(wm_psnr_list).mean()
        wm_ssim = np.array(wm_ssim_list).mean()
        if args.use_attack:
            att_psnr = np.array(att_psnr_list).mean()
            att_ssim = np.array(att_ssim_list).mean()

    if args.with_tracking:
        wandb.log({"Table": table})
        metrics_table = {"fid_no_w": fid_value_no_w, "fid_w": fid_value_w}
        if args.use_attack:
            metrics_table["fid_att_w"] = fid_value_w_att
        if args.additional_metrics:
            metrics_table["psnr_no_w"] = clean_psnr
            metrics_table["ssim_no_w"] = clean_ssim
            metrics_table["psnr_w"] = wm_psnr
            metrics_table["ssim_w"] = wm_ssim
        if args.use_attack:
            metrics_table["psnr_att_w"] = att_psnr
            metrics_table["ssim_att_w"] = att_ssim
        wandb.log(metrics_table)

    print(f"fid_no_w: {fid_value_no_w}, fid_w: {fid_value_w}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion watermark")
    parser.add_argument("--project_name", default="watermark_attacks")
    parser.add_argument("--run_name", default="test")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=10, type=int)
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--num_images", default=1, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=40, type=int)
    parser.add_argument("--max_num_log_image", default=100, type=int)
    parser.add_argument("--run_no_w", action="store_true")
    parser.add_argument("--gen_seed", default=0, type=int)

    parser.add_argument("--prompt_file", default="fid_outputs/coco/meta_data.json")
    parser.add_argument("--gt_folder", default="fid_outputs/coco/ground_truth")
    parser.add_argument("--image_folder", default="fid_outputs/coco/fid_run")
    # Compute metrics with gen_no_w and gen_w:
    parser.add_argument("--target_clean_generated", action="store_true")

    parser.add_argument("--run_generation", action="store_true")
    parser.add_argument("--additional_metrics", action="store_true")

    # watermark
    parser.add_argument("--w_seed", default=999999, type=int)
    parser.add_argument("--w_channel", default=3, type=int)
    parser.add_argument("--w_pattern", default="ring")
    parser.add_argument("--w_mask_shape", default="circle")
    parser.add_argument("--w_radius", default=10, type=int)
    parser.add_argument("--w_measurement", default="l1_complex")
    parser.add_argument("--w_injection", default="complex")
    parser.add_argument("--w_pattern_const", default=0, type=float)

    # VAE or Diff attack
    parser.add_argument("--use_attack", action="store_true")
    parser.add_argument("--attack_type", default="diff")
    parser.add_argument("--use_attack_prompt", action="store_true")
    parser.add_argument("--diff_attack_steps", default=60, type=int)
    parser.add_argument("--vae_attack_name", default="cheng2020-anchor")
    parser.add_argument("--vae_attack_quality", default=3, type=int)

    # Message encryption (for testing: putting the same message on each image, but they can be different):
    parser.add_argument("--msg_type", default="rand", help="Can be: rand or binary or decimal")
    parser.add_argument("--msg", default="1110101101")
    parser.add_argument("--use_random_msgs", action="store_true", help="Generate random message each step of cycle")
    parser.add_argument("--msg_scaler", default=100, type=int, help="Scaling coefficient of message")

    # METR++:
    parser.add_argument("--use_stable_sig", action="store_true")
    parser.add_argument("--decoder_state_dict_path", default="finetune_ldm_decoder/ldm_decoder_checkpoint_000.pth")
    parser.add_argument("--stable_sig_full_model_config", default="v2-inference.yaml")
    parser.add_argument("--stable_sig_full_model_ckpt", default="v2-1_512-ema-pruned.ckpt")

    # for image distortion
    parser.add_argument("--r_degree", default=None, type=float)
    parser.add_argument("--jpeg_ratio", default=None, type=int)
    parser.add_argument("--crop_scale", default=None, type=float)
    parser.add_argument("--crop_ratio", default=None, type=float)
    parser.add_argument("--gaussian_blur_r", default=None, type=int)
    parser.add_argument("--gaussian_std", default=None, type=float)
    parser.add_argument("--brightness_factor", default=None, type=float)
    parser.add_argument("--rand_aug", default=0, type=int)

    args = parser.parse_args()

    main(args)
