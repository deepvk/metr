# Reproducing experiments from paper:

## Diffusion and VAE attack on METR:

### Diffusion attack:
```bash
bash scripts/metr_generative_att/diff_att_metr.sh
```
### VAE attack:
```bash
bash scripts/metr_generative_att/vae_2018_att_metr.sh
```

## All attacks on METR:
### Detection:
```bash 
bash scripts/metr_all_att/detect_metr_default_vae.sh
```
### Image Quality:
```bash
# FID gt
bash scripts/metr_all_att/fid_gt_all_att_metr.sh
# FID gen
bash scripts/metr_all_att/fid_gen_all_att_metr.sh
```

## METR Image quality dispersion because of different message:
```bash
# FID gt:
bash scripts/fid_message_dispersion/gt.sh
# FID gen:
bash scripts/fid_message_dispersion/gen.sh
```

## All attacks on METR++:
### METR with fine-tuned VAE decoder:
```bash 
bash scripts/metr_pp_all_att/detect_metr_changed_vae.sh
```

### Stable-Signature part of METR++:
```bash
# Generate images:
bash scripts/metr_pp_all_att/stable_sig_attacks/generate.sh
# Evaluate:
bash scripts/metr_pp_all_att/stable_sig_attacks/eval.sh
```

### Image quality:
```bash
# FID gt:
scripts/metr_all_att/fid_gt_all_att_metr.sh
# FID gen:
scripts/metr_all_att/fid_gen_all_att_metr.sh
```

## Stable-Signature tuned on images sampled from different distributions:

To run this, one need to have folder called `generated_images`, that was created using script provided in [section](#metr-detection-metrics-to-wandb).

Create mock val directory named `val_mock` with 1 image to run following scripts without error:
```bash
mkdir val_mock
cp fid_outputs/coco/ground_truth/000000000139.jpg val_mock
```

Run all fine-tunings, generations and evaluations:

```bash
# N-N
bash scripts/stable_sig_different_distr/N-N.sh
# N-METR
bash scripts/stable_sig_different_distr/N-METR.sh
# METR-N
bash scripts/stable_sig_different_distr/METR-N.sh
# METR-METR
bash scripts/stable_sig_different_distr/METR-METR.sh
```
