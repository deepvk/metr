from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import PIL
import sys
import torch
import os
import glob
import numpy as np

from wm_attacks import ReSDPipeline

from wm_attacks.wmattacker_no_saving import DiffWMAttacker, VAEWMAttacker

# ------------

import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

from .inverse_stable_diffusion import InversableStableDiffusionPipeline

from diffusers import DPMSolverMultistepScheduler

from .pytorch_fid.fid_score import *
from .open_clip import create_model_and_transforms, get_tokenizer

from .optim_utils import *
from .io_utils import *
from .stable_sig.utils_model import *

def main(args):
    if args.save_locally:
        if not os.path.exists(args.local_path) and not os.path.exists(args.local_path + f"/imgs_no_w/"):
            os.makedirs(args.local_path)
            os.makedirs(args.local_path + f"/imgs_no_w/")
            os.makedirs(args.local_path + f"/imgs_w/")

    table = None
    if args.with_tracking:
        wandb.init(project=args.project_name, name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)
        if args.use_attack:
            columns = ['gen_no_w', 'no_w_clip_score', 'gen_w', 'w_clip_score', 'att_gen_w', 'prompt', 'no_w_metric', 'w_metric']
        else:
            columns = ['gen_no_w', 'no_w_clip_score', 'gen_w', 'w_clip_score', 'prompt', 'no_w_metric', 'w_metric']

        if args.use_random_msgs:
            columns.append('message')
        table = wandb.Table(columns=columns)
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe = pipe.to(device)

    if not args.no_stable_sig:
        pipe = change_pipe_vae_decoder(pipe, weights_path=args.decoder_state_dict_path)
        print("VAE CHANGED!")

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # ground-truth patch
    if not args.use_random_msgs:
        gt_patch = get_watermarking_pattern(pipe, args, device)

    results = []
    clip_scores = []
    clip_scores_w = []
    no_w_metrics = []
    w_metrics = []

    bit_accs = []
    words_right = 0

    if args.use_attack:
        if args.attack_type == "diff":
            attack_pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
            attack_pipe.set_progress_bar_config(disable=True)
            attack_pipe.to(device)
            attacker = DiffWMAttacker(attack_pipe, noise_step=args.diff_attack_steps)
        if args.attack_type == "vae":
            attacker = VAEWMAttacker(args.vae_attack_name, quality=args.vae_attack_quality, metric='mse', device=device)

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        
        current_prompt = dataset[i][prompt_key]
        if args.given_prompt:
            current_prompt = args.given_prompt

        if args.use_random_msgs:
            msg_key = torch.randint(0, 2, (1, args.w_radius), dtype=torch.float32, device="cpu")
            msg_str = "".join([ str(int(ii)) for ii in msg_key.tolist()[0]])

        if args.use_random_msgs:
            gt_patch = get_watermarking_pattern(pipe, args, device, message=msg_str)
        
        ### generation
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = pipe.get_random_latents()
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
        
        # generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

        # inject watermark into latents
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

        ### test watermark
        # distortion
        orig_image_no_w_auged, orig_image_w_auged = image_distortion(orig_image_no_w, orig_image_w, seed, args)

        ### VAE or diffusion attack:
        # move this code block upper if want to combine simple distortions with attacks
        if args.use_attack:
            if args.use_attack_prompt and args.attack_type == "diff":
                att_img_w = attacker.attack(orig_image_w, prompt=current_prompt)
            else:
                att_img_w = attacker.attack(orig_image_w)
            orig_image_w_auged = att_img_w

        # reverse img without watermarking
        img_no_w = transform_img(orig_image_no_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        # image latents means: z - latents in Stable diff, i.e. image passed through vae
        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)

        # forward_diffusion здесь - это получение шума по картинке
        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # reverse img with watermarking: on attacked image
        img_w = transform_img(orig_image_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # eval
        no_w_metric, w_metric = eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args)
        
        if args.save_rev_lat:
            rev_lat_path = f"{args.path_rev_lat}/s_{args.msg_scaler}_r_{args.w_radius}"
            if not os.path.exists(rev_lat_path):
                os.makedirs(rev_lat_path)
            torch.save(reversed_latents_w.to(torch.complex64), f"{rev_lat_path}/rev_lat.pt")

        # detect msg
        if args.msg_type == "binary":
            correct_bits_tmp = 0
            if args.use_random_msgs:
                true_msg_str = msg_str
            else:
                true_msg_str = args.msg

            true_msg = np.array(list(map(lambda x: 1 if x == "1" else 0, list(true_msg_str))))
            pred_msg = np.array(detect_msg(reversed_latents_w, args))
            print(f"true msg: {true_msg}; pred_msg: {pred_msg}")

            correct_bits_tmp = np.equal(true_msg, pred_msg).sum()

            bit_accs.append(correct_bits_tmp / args.w_radius)

            if correct_bits_tmp == args.w_radius:
                words_right += 1


        if args.reference_model is not None:
            sims = measure_similarity([orig_image_no_w, orig_image_w], current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
            w_no_sim = sims[0].item()
            w_sim = sims[1].item()
        else:
            w_no_sim = 0
            w_sim = 0

        results.append({
            'no_w_metric': no_w_metric, 'w_metric': w_metric, 'w_no_sim': w_no_sim, 'w_sim': w_sim,
        })

        no_w_metrics.append(-no_w_metric)
        w_metrics.append(-w_metric)

        if args.save_locally:
            orig_image_no_w_auged.save(args.local_path + f"/imgs_no_w/img{i}.png")
            orig_image_w_auged.save(args.local_path + f"/imgs_w/w_img{i}.png")

        if args.with_tracking:
            if (args.reference_model is not None) and (i < args.max_num_log_image):
                # log images when we use reference_model
                if args.use_attack:
                    data_to_add = [wandb.Image(orig_image_no_w), w_no_sim, wandb.Image(orig_image_w), w_sim, wandb.Image(att_img_w), current_prompt, no_w_metric, w_metric]
                else:
                    data_to_add = [wandb.Image(orig_image_no_w), w_no_sim, wandb.Image(orig_image_w), w_sim, current_prompt, no_w_metric, w_metric]
            else:
                if args.use_attack:
                    data_to_add = [None, w_no_sim, None, w_sim, None, current_prompt, no_w_metric, w_metric]
                else:
                    data_to_add = [None, w_no_sim, None, w_sim, current_prompt, no_w_metric, w_metric]

            if args.use_random_msgs:
                data_to_add.append(msg_str)

            table.add_data(*data_to_add)

            clip_scores.append(w_no_sim)
            clip_scores_w.append(w_sim)

        # roc
        if (i - args.start + 1) % args.freq_log == 0 and (i - args.start) >= args.freq_log - 1:
            preds = no_w_metrics + w_metrics
            t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

            fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            acc = np.max(1 - (fpr + (1 - tpr))/2)
            low = tpr[np.where(fpr<.01)[0][-1]]

            if args.with_tracking:
                wandb.log({'Table': table})
                if (i - args.start) > 0:
                    metrics_dict = {
                        'clip_score_mean': mean(clip_scores), 'clip_score_std': stdev(clip_scores),
                        'w_clip_score_mean': mean(clip_scores_w), 'w_clip_score_std': stdev(clip_scores_w),
                        'auc': auc, 'acc':acc, 'TPR@1%FPR': low,
                        'w_det_dist_mean': -mean(w_metrics), 'w_det_dist_std': stdev(w_metrics),
                        'no_w_det_dist_mean': -mean(no_w_metrics), 'no_w_det_dist_std': stdev(no_w_metrics),
                    }
                    if args.msg_type == "binary":
                        metrics_dict["Bit_acc"] = mean(bit_accs)
                        metrics_dict["Word_acc"] = words_right / (i + 1)

                if (i - args.start) > 0:
                    wandb.log(metrics_dict)
    
            print(f'clip_score_mean: {mean(clip_scores)}')
            print(f'w_clip_score_mean: {mean(clip_scores_w)}')
            print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--project_name', default='watermark_attacks')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')

    # logs and metrics:
    parser.add_argument('--freq_log', default=20, type=int)
    parser.add_argument('--save_locally', action='store_true')
    parser.add_argument('--local_path', default='/data/varlamov_a_data/dima/images')

    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=40, type=int)
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

    # VAE or Diff attack
    parser.add_argument('--use_attack', action='store_true')
    parser.add_argument('--attack_type', default='diff')
    parser.add_argument('--use_attack_prompt', action='store_true')
    parser.add_argument('--diff_attack_steps', default=60, type=int)
    parser.add_argument('--vae_attack_name', default='cheng2020-anchor')
    parser.add_argument('--vae_attack_quality', default=3, type=int)

    # Stable-Tree
    parser.add_argument('--decoder_state_dict_path', default='/data/varlamov_a_data/tree-ring-watermark/ldm_decoders/sd2_decoder.pth')
    parser.add_argument('--no_stable_sig', action='store_true')

    # Message encryption (for testing: putting the same message on each image, but they can be different):
    parser.add_argument('--msg_type', default='rand', help="Can be: rand or binary or decimal")
    parser.add_argument('--msg', default='1110101101')
    parser.add_argument('--use_random_msgs', action='store_true', help="Generate random message each step of cycle")
    parser.add_argument('--msg_scaler', default=100, type=int, help="Scaling coefficient of message")

    # For testing
    parser.add_argument('--given_prompt', default=None, type=str)
    parser.add_argument('--save_rev_lat', action='store_true', help="Flag to save reversed latents")
    parser.add_argument('--path_rev_lat', default=None, type=str)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)