import copy
import json
import random
from typing import Any, Mapping

import numpy as np
import scipy
import torch
from datasets import load_dataset
from PIL import Image, ImageFilter
from torchvision import transforms


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


def image_distortion(img1, img2, seed, args):
    if args.r_degree is not None:
        img1 = transforms.RandomRotation((args.r_degree, args.r_degree))(img1)
        img2 = transforms.RandomRotation((args.r_degree, args.r_degree))(img2)

    if args.jpeg_ratio is not None:
        img1.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img1 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")
        img2.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img2 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")

    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(
            img1.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio)
        )(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(
            img2.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio)
        )(img2)

    if args.gaussian_blur_r is not None:
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img1 = transforms.ColorJitter(brightness=args.brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor)(img2)

    return img1, img2


# for one prompt to multiple images
def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)


def get_dataset(args):
    if "laion" in args.dataset:
        dataset = load_dataset(args.dataset)["train"]
        prompt_key = "TEXT"
    elif "coco" in args.dataset:
        with open("fid_outputs/coco/meta_data.json") as f:
            dataset = json.load(f)
            dataset = dataset["annotations"]
            prompt_key = "caption"
    else:
        dataset = load_dataset(args.dataset)["test"]
        prompt_key = "Prompt"

    return dataset, prompt_key


def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0) ** 2 + (y - y0) ** 2) <= r**2


def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == "circle":
        np_mask = circle_mask(init_latents_w.shape[-1], r=args.w_radius)
        torch_mask = torch.tensor(np_mask).to(device)

        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == "square":
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_mask[
                :,
                :,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
            ] = True
        else:
            watermarking_mask[
                :,
                args.w_channel,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
            ] = True
    elif args.w_mask_shape == "no":
        pass
    else:
        raise NotImplementedError(f"w_mask_shape: {args.w_mask_shape}")

    return watermarking_mask


class MsgError(Exception):
    "Raised, when len(args.msg) != args.w_radius"
    pass


def encrypt_message(gt_init, args, device, message):
    """
    Inserts given message into Fourier space of gaussian noise
    """
    if args.use_random_msgs and (not message or len(message) != args.w_radius):
        raise MsgError("Message argument not passed or its length is not equal to radius ")

    if len(args.msg) != args.w_radius and args.msg_type != "rand" and not args.use_random_msgs:
        raise MsgError("Message length is not equal to watermark radius")

    gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
    message_mat = np.ones([4, args.w_radius])

    if args.msg_type == "rand":
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)

            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    elif args.msg_type == "binary":
        if not args.use_random_msgs:
            print(f"NOT USING RANDOM MESSAGE, INSERTING: {args.msg}")
            message_mat[args.w_channel] = list(
                map(lambda x: args.msg_scaler if x == "1" else -args.msg_scaler, list(args.msg))
            )
        else:
            print(f"USING RANDOM MESSAGE, INSERTING: {message}")
            message_mat[args.w_channel] = list(
                map(lambda x: args.msg_scaler if x == "1" else -args.msg_scaler, list(message))
            )
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)

            for j in range(gt_patch.shape[1]):  # итерация по каналам
                gt_patch[:, j, tmp_mask] = message_mat[j][i - 1]

    elif args.msg_type == "decimal":
        pass

    return gt_patch


def get_watermarking_pattern(pipe, args, device, shape=None, message=None):
    """
    Creates elements of gt_patch array
    """
    set_random_seed(args.w_seed)
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)
    else:
        gt_init = pipe.get_random_latents()

    if "seed_ring" in args.w_pattern:
        gt_patch = gt_init

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)

            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif "seed_zeros" in args.w_pattern:
        gt_patch = gt_init * 0
    elif "seed_rand" in args.w_pattern:
        gt_patch = gt_init
    elif "rand" in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif "zeros" in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif "const" in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const
    elif "ring" in args.w_pattern:
        gt_patch = encrypt_message(gt_init, args, device, message)

    return gt_patch


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    """
    Injects gt_patch elements into watermarking_mask indexes
    """
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    if args.w_injection == "complex":
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif args.w_injection == "seed":
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f"w_injection: {args.w_injection}")

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    """
    Compares values on watermarking_mask indexes in fourier space of image with gt_patch
    """
    if "complex" in args.w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = gt_patch
    elif "seed" in args.w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        NotImplementedError(f"w_measurement: {args.w_measurement}")

    if "l1" in args.w_measurement:
        no_w_metric = (
            torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        )
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        NotImplementedError(f"w_measurement: {args.w_measurement}")

    return no_w_metric, w_metric


def detect_msg(reversed_latents_w, args):
    """
    Get predicted message from reversed_latents
    """
    pred_msg = []
    r = args.w_radius
    channel = args.w_channel

    if "complex" in args.w_measurement:
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
    elif "seed" in args.w_measurement:
        reversed_latents_w_fft = reversed_latents_w
    else:
        NotImplementedError(f"w_measurement: {args.w_measurement}")

    for i in range(r, 0, -1):
        # Getting the edges of circles:
        if r > 1:
            tmp_mask = (
                circle_mask(reversed_latents_w.shape[-1], r=i).astype(int)
                - circle_mask(reversed_latents_w.shape[-1], r=i - 1).astype(int)
            ).astype(bool)
        else:
            tmp_mask = circle_mask(reversed_latents_w.shape[-1], r=i)

        pred_circle_tmp_value = reversed_latents_w_fft[:, channel, tmp_mask].real.mean()

        pred_msg.append((pred_circle_tmp_value > 0).to(int).item())

    return pred_msg[::-1]  # Prediction is done from the biggest cirlce


def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    # assume it's Fourier space wm
    reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[
        watermarking_mask
    ].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[
        watermarking_mask
    ].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])

    # no_w
    reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    sigma_no_w = reversed_latents_no_w_fft.std()
    lambda_no_w = (target_patch**2 / sigma_no_w**2).sum().item()
    x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch**2 / sigma_w**2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w


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
