# METR watermark :tractor:

## About
This is the implementation of METR watermark. We propose an attack resistant watermark to inject large amount of unique messages without image quality reduction.

## Setup:

**We ran all experiments on python==3.10**:

```bash
conda create -n metr python==3.10
```

Clone repository:
```bash
git clone https://github.com/deepvk/metr.git
```

Install dependencies:
```bash
cd metr
pip install -r requirements.txt 
```

Login to your wandb and create accelerate config:
```bash
WANDB_KEY ="your/wandb/key"
wandb login $WANDB_KEY

accelerate config default
```

## Running METR watermark:

### METR Detection metrics to wandb:

#### Generate images with random message:

To save images locally include additional argument `--save_locally` and provide path with `--local_path /path/to/save`.

```bash
accelerate launch -m metr.run_metr \
  --project_name metr_detection \
  --model_id stabilityai/stable-diffusion-2-1-base \
  --run_name no_attack --w_channel 3 --w_pattern ring \
  --start 0 --end 1000 \
  --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
  --with_tracking \
  --msg_type binary \
  --use_random_msgs \
  --save_locally \
  --w_radius 10 \
  --msg_scaler 100 \
  --no_stable_sig \
  --local_path generated_images
```

#### Generate image with fixed message and fixed prompt:

```bash
accelerate launch -m metr.run_metr \
  --project_name fixed_msg \
  --model_id stabilityai/stable-diffusion-2-1-base \
  --run_name fixed_msg --w_channel 3 --w_pattern ring \
  --start 0 --end 1 \
  --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
  --with_tracking \
  --freq_log 1 \
  --w_radius 10 \
  --msg_type binary \
  --msg_scaler 100 \
  --msg 1010101010 \
  --given_prompt "sci-fi bernese mountain dog"
```

#### Perform diffusion model attack on METR:

```bash
accelerate launch -m metr.run_metr \
  --project_name metr_diff_att \
  --run_name diff_150 --w_channel 3 --w_pattern ring \
  --start 0 --end 1000 \
  --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
  --with_tracking \
  --w_radius 10 \
  --msg_type binary \
  --use_random_msgs \
  --msg_scaler 100 \
  --no_stable_sig \
  --use_attack \
  --attack_type diff \
  --diff_attack_steps 150
```

### Evaluate FID for METR
#### Download dataset and extract:

```bash
wget -O fid_outputs.zip https://huggingface.co/datasets/Alphonsce/MSCOCO_zip/resolve/main/fid_outputs.zip?download=true
unzip fid_outputs.zip
```
[Google drive version if HF is not working](https://drive.google.com/drive/u/1/folders/1v0xj-8Yx8vZ_4qGsC5EU5FJBvEWTFmE3)

#### FID on ground-truth images (FID gt):

Argument `--image_folder` is where images are saved.

```bash
accelerate launch -m metr.run_metr_fid \
  --project_name fid_gen \
  --model_id stabilityai/stable-diffusion-2-1-base \
  --run_name no_attack --w_channel 3 --w_pattern ring \
  --start 0 --end 5000 \
  --with_tracking \
  --w_radius 10 \
  --run_generation \
  --additional_metrics \
  --run_no_w \
  --image_folder fid_eval_gt_metr \
  --msg_type binary \
  --use_random_msgs \
  --msg_scaler 100
```

#### FID on generated images (FID gen):
To evaluate FID on generated images, add argument `--target_clean_generated`:

```bash
accelerate launch -m metr.run_metr_fid \
  --project_name fid_gen \
  --model_id stabilityai/stable-diffusion-2-1-base \
  --run_name no_attack --w_channel 3 --w_pattern ring \
  --start 0 --end 5000 \
  --with_tracking \
  --w_radius 10 \
  --run_generation \
  --additional_metrics \
  --run_no_w \
  --image_folder fid_eval_gen_metr \
  --msg_type binary \
  --use_random_msgs \
  --msg_scaler 100 \
  --target_clean_generated
```

## Running METR++ watermark
We forked Stable Signature repository to adjust it to be comparable with METR. It can be found [here](https://github.com/Alphonsce/stable_signature/).

Install weights for WM extractor for Stable Signature (taken from [official Stable-Signature repository](https://github.com/facebookresearch/stable_signature) )
```bash
wget https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt
```

Install full model checkpoint and config to train VAE decoder of it:

```bash
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt
wget https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference.yaml
```

You can download our pre-trained for message `111010110101000001010111010011010100010000100111` VAE decoder from [here](https://huggingface.co/Alphonsce/St_Sig_vae_decoder/resolve/main/ldm_decoder_checkpoint_000.pth?download=true).

### Fine-tune VAE decoder to given ID:

In example down below we fine-tune VAE decoder on samples from MSCOCO dataset and evaluate on images previously generated with METR watermark in `generated_images` folder. Pretrained VAE decoder weights will be saved in `finetune_ldm_decoder/ldm_decoder_checkpoint_000.pth` by default. You can change the name of checkpoint with `--checkpoint_name` argument.

```bash
TRAIN_DIR=fid_outputs/coco/ground_truth
VAL_DIR=generated_images/imgs_w

  accelerate launch -m metr.finetune_ldm_decoder --num_keys 1 \
  --ldm_config v2-inference.yaml \
  --ldm_ckpt v2-1_512-ema-pruned.ckpt \
  --msg_decoder_path dec_48b_whit.torchscript.pt \
  --train_dir $TRAIN_DIR \
  --val_dir $VAL_DIR \
  --with_tracking \
  --project_name finetune_ldm_decoder \
  --run_name test \
  --output_dir finetune_ldm_decoder \
  --batch_size 4 \
  --steps 100 \
  --num_val_imgs 200 \
  --num_keys 1 \
  --not_rand_key \
  --key_str 111010110101000001010111010011010100010000100111
```

### Generate images with METR++ watermark and evaluate METR part of it:
To generate images with METR++ watermark, just remove `--no_stable_sig` argument and provide a path to tuned VAE decoder: `--decoder_state_dict_path /path/to/decoder/weights`:
```bash
VAE_DECODER_PATH=finetune_ldm_decoder/ldm_decoder_checkpoint_000.pth

accelerate launch -m metr.run_metr \
  --project_name metr_detection \
  --model_id stabilityai/stable-diffusion-2-1-base \
  --run_name no_attack --w_channel 3 --w_pattern ring \
  --stable_sig_full_model_config v2-inference.yaml \
  --stable_sig_full_model_ckpt v2-1_512-ema-pruned.ckpt \
  --start 0 --end 1000 \
  --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
  --with_tracking \
  --msg_type binary \
  --use_random_msgs \
  --w_radius 10 \
  --msg_scaler 100 \
  --save_locally \
  --local_path metr_pp_generated_images \
  --decoder_state_dict_path $VAE_DECODER_PATH
```

### Evaluate Stable Signature part of METR++
Evaluation is performed on a folder of generated images, you need to pass folder into `--` with images generated with Stable-Signature watermark.
```bash
EVAL_FOLDER=metr_pp_generated_images/imgs_w

accelerate launch -m metr.metr_pp_eval_stable_sig \
  --project_name eval_st_sig \
  --with_tracking \
  --run_name test \
  --eval_imgs False --eval_bits True \
  --img_dir $EVAL_FOLDER \
  --output_dir eval_st_sig_logs \
  --msg_decoder_path dec_48b_whit.torchscript.pt \
  --attack_mode none \
  --key_str 111010110101000001010111010011010100010000100111
```

### Evaluate FID for METR++:
To evaluate FID for images with METR++ watermark pass `--use_stable_sig` argument.
```bash
accelerate launch -m metr.run_metr_fid \
  --project_name fid_gen \
  --model_id stabilityai/stable-diffusion-2-1-base \
  
  --run_name no_attack --w_channel 3 --w_pattern ring \
  --start 0 --end 5000 \
  --with_tracking \
  --w_radius 10 \
  --run_generation \
  --additional_metrics \
  --run_no_w \
  --image_folder fid_eval_gt_metr_pp \
  --msg_type binary \
  --use_random_msgs \
  --msg_scaler 100 \
  --use_stable_sig
```

## Reproducing experiments from paper:

### Diffusion and VAE attack on METR:

#### Diffusion attack:
```bash
bash ./scripts/.sh
```
#### VAE attack:
```bash

```

---
# References:

## Tree-Ring watermark:

### [Repository link](https://github.com/YuxinWenRick/tree-ring-watermark)

### [Paper link](https://arxiv.org/abs/2305.20030)

## Stable Signature:

### [Repository link](https://github.com/facebookresearch/stable_signature)

### [Paper link](https://arxiv.org/abs/2303.15435)

## Generative Model watermark attacker:

### [Repository link](https://github.com/XuandongZhao/WatermarkAttacker)

### [Paper link](https://arxiv.org/abs/2306.01953)
