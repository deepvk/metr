# METR watermark.

## About
This is the implementation of METR watermark. We propose an attack resistant watermark to inject large amount of unique messages without image quality

## Setup:

Clone repository:
```bash
git clone https://github.com/deepvk/metr.git
```

Install dependencies:
```bash
cd metr
pip install -r requirements.txt 
```

## Running METR watermark:
Generate images with random messages and evaluate detection metrics:

```bash

```

Evaluate FID metric. You can download MSCOCO-5000 dataset from: link

```bash

```

## Running METR++ watermark
Code related to Stable Signature can be found here

Fine-tune VAE decoder to given ID:

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
