# METR watermark.

## About
This is the implementation of METR watermark. We propose an attack resistant watermark to inject large amount of unique messages without image quality reduction.

## Setup:

**We ran all experiments on python==3.10**

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

All METR related code was made in our fork of Tree-Ring repository, which is
[here](https://github.com/Alphonsce/tree-ring-watermark).

### METR Detection metrics to wandb:

#### Generate images with random message:

To save images locally include additional argument "--save_locally" and provide path with "--local_path /path/to/save".


```bash
accelerate launch -m tree_ring_watermark.run_stable_tree \
  --project_name metr_detection \
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
accelerate launch -m tree_ring_watermark.run_stable_tree \
  --project_name fixed_msg \
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

### Evaluate FID for METR
#### Download dataset and extract:

```bash
wget -O fid_outputs.zip https://huggingface.co/datasets/Alphonsce/MSCOCO_zip/resolve/main/fid_outputs.zip?download=true
unzip fid_outputs.zip
```
[Google drive version if HF is not working](https://drive.google.com/drive/u/1/folders/1v0xj-8Yx8vZ_4qGsC5EU5FJBvEWTFmE3)

#### FID on ground-truth images (FID gt):

```bash

```

#### FID on generated images (FID gen):

```bash
accelerate launch -m tree_ring_watermark.run_tree_ring_watermark_fid \
  --project_name fid_gen \
  --run_name no_attack --w_channel 3 --w_pattern ring \
  --start 0 --end 5000 \
  --with_tracking \
  --w_radius 10 \
  --run_generation \
  --additional_metrics \
  --run_no_w \
  --image_folder fid_eval_gen \
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

### Fine-tune VAE decoder to given ID:

In example down below we fine-tune VAE decoder on samples from MSCOCO dataset and evaluate on images previously generated with METR watermark.

```bash

```

Generate images with METR++ watermark and evaluate METR part of it:
```bash

```

Evaluate Stable Signature part of METR++
```bash

```

## Reproducing experiments from paper:

Go to scripts directory:

```bash
cd metr/scripts
```

Diffusion and VAE attack on METR:

```bash
bash .sh
```

```bash

```