# MMD-GAN Experiment Suite

This repository now provides reproducible scripts for the three roadmap experiments from **Demystifying MMD GANs**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All scripts automatically use `cuda` if available, otherwise they fall back to `cpu`.

## 1) Toy experiment: `make_moons`

Runs MMD-GAN with/without gradient penalty and compares `linear`, `rbf`, `rq` kernels.

```bash
python3 scripts/run_toy_moons.py --mode grid --steps 3000
```

Outputs:
- sample plots + witness heatmaps per configuration
- checkpoints
- `summary.json` with final MMD values

## 2) Bias illustration

Controlled example for fixed-critic vs learned-critic discrepancy.

```bash
python3 scripts/run_bias_illustration.py --steps 2000 --delta 0.0
```

`delta=0.0` is the key setting to show critic-learning bias when the two distributions are actually equal.

Outputs:
- `bias_curves.png`
- `summary.json` (includes learned vs fixed train/holdout gap)

## 3) CIFAR-10 large-scale experiment

### MMD-GAN (RQ default)

```bash
python3 scripts/run_cifar10.py --method mmd --kernel-list rq --steps 30000
```

### Optional kernel comparison

```bash
python3 scripts/run_cifar10.py --method mmd --kernel-list rq,rbf,linear --steps 30000
```

### Optional WGAN-GP baseline

```bash
python3 scripts/run_cifar10.py --method mmd --kernel-list rq --run-baseline-wgan --steps 30000
```

Outputs:
- periodic sample grids
- checkpoints
- run summaries with FID/KID when `torchmetrics` is available

## Notes

- MMD kernels follow paper-style mixtures:
  - RBF: `sigma in {2,5,10,20,40,80}`
  - RQ: `alpha in {0.2,0.5,1,2,5}` plus optional linear term
- Default MMD gradient penalty scale is `1.0` (as recommended in the paper’s experiments).
- CIFAR metrics use `torchmetrics` (`FID` + `KID`) when installed; otherwise metrics are skipped gracefully with a warning.
