# GaussLoRA

Minimal training repository for ViT fine-tuning with **LoRA** and **Fast Gaussian Splatting** adapters.

Derived from `inrlora`, trimmed to only the adapters and training logic needed for LoRA + Gaussian splatting experiments.

## Structure

```
gausslora/
├── adapters.py          # LoraAdapter, FastGaussianSplattingAdapter, LayerWithAdapter
├── main.py              # Training pipeline (no adapter restart/reset)
├── config.yaml          # Experiment definitions
├── run_experiments.py   # Queue runner for all config experiments
├── ds.py                # Dataset loading
├── muon.py              # Optional Muon optimizer
├── plots/               # Output folder for plots
├── gifs/                # Gaussian evolution GIFs
└── saved_adapters/      # Saved checkpoints
```

## Supported adapters

- `LoraAdapter`
- `FastGaussianSplattingAdapter`

## Quick start

```bash
conda env create -f environment.yaml
conda activate gausslora

python main.py --dataset stanfordcars --exp_index 0 --device cuda:0
python run_experiments.py
```

## Config experiments

- `Lora rank 4`
- `GAUSSIAN SPLATTING` (FastGaussianSplatting, 81 splats)
- `Lora r4 + GS`
- `Lora r3 + GS`