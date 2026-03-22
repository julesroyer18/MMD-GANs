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

### Bias illustration v2

This second version is the smaller gradient-sweep experiment: one scalar generator parameter `psi`,
one critic trained once and frozen, then a per-`psi` critic re-fit from finite samples.

```bash
python3 scripts/run_bias_illustration_v2.py
```

Outputs:
- `gradient_bias_plot.png`
- `summary.json` with mean/std gradient curves over seeds
- `deliverables.md` containing the requested schematic table and theory paragraph

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

## 4) Detailed Run Plan For The New Protocol Scripts

This section lists the concrete commands to launch the newer protocol-oriented experiments added alongside the original scripts.
These new runners do **not** replace the previous ones:

- `scripts/run_toy_moons_v2.py`
- `scripts/run_bias_illustration_v3.py`
- `scripts/run_cifar10_v2.py`

Recommended execution order:

1. run the `make_moons` suite first
2. run the bias illustration once the toy setup is stable
3. launch CIFAR-10 only after the toy experiments are validated
4. add the WGAN-GP baseline last

### A. Install Dependencies

Before launching any of the protocol scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want FID/KID on CIFAR-10, make sure `torchmetrics` and its image dependencies are available in the same environment.

### B. `make_moons` Protocol Runs

The full course-project suite runs `linear`, `rbf`, and `rq`, each with and without witness GP, over 5 seeds:

```bash
python3 scripts/run_toy_moons_v2.py \
  --mode suite \
  --steps 20000 \
  --batch-size 256 \
  --critic-steps 5 \
  --lr-g 1e-3 \
  --lr-c 1e-3 \
  --z-dim 4 \
  --hidden-dim 64 \
  --feat-dim 8 \
  --train-size 10000 \
  --val-size 2000 \
  --noise 0.08 \
  --seeds 0,1,2,3,4
```

This produces:

- one run directory per `(kernel, gp, seed)`
- per-run sample checkpoints
- witness heatmaps
- aggregate comparison plots at the root of the run folder

If you want to rerun only one configuration, use `--mode single`. Example: pure RQ with witness GP:

```bash
python3 scripts/run_toy_moons_v2.py \
  --mode single \
  --kernel rq \
  --gp \
  --steps 20000 \
  --seeds 0,1,2,3,4
```

If you want the optional `RQ + linear` variant after the pure RQ runs are stable:

```bash
python3 scripts/run_toy_moons_v2.py \
  --mode single \
  --kernel rq_linear \
  --gp \
  --steps 20000 \
  --seeds 0,1,2,3,4
```

Useful lower-cost smoke test before the full suite:

```bash
python3 scripts/run_toy_moons_v2.py \
  --mode suite \
  --steps 2000 \
  --seeds 0
```

### C. Bias Illustration Protocol Runs

The recommended run uses the 1D Gaussian location family, fixed train/holdout sets per seed and per `delta`, a frozen critic baseline, and a learned critic for each `delta`:

```bash
python3 scripts/run_bias_illustration_v3.py \
  --delta-grid=-2,-1.5,-1,-0.5,0,0.5,1,1.5,2 \
  --num-seeds 20 \
  --train-size 512 \
  --holdout-size 512 \
  --kernel rq \
  --critic-hidden-dim 64 \
  --critic-feat-dim 8 \
  --critic-steps 1500 \
  --critic-lr 3e-4 \
  --activation-penalty 1e-3 \
  --fixed-critic-mode trained \
  --fixed-critic-delta0 1.0
```

This produces:

- `objective_vs_delta.png`
- `gradient_vs_delta.png`
- `gap_vs_delta.png`
- `summary.json`
- `deliverables.md`

If you want a random frozen baseline instead of a critic trained once at `delta0`:

```bash
python3 scripts/run_bias_illustration_v3.py \
  --fixed-critic-mode random \
  --kernel rq \
  --num-seeds 20
```

If autograd becomes unstable for any reason, switch to finite differences:

```bash
python3 scripts/run_bias_illustration_v3.py \
  --gradient-mode finite-diff \
  --finite-diff-eps 1e-2
```

Useful lower-cost smoke test:

```bash
python3 scripts/run_bias_illustration_v3.py \
  --num-seeds 3 \
  --critic-steps 200 \
  --train-size 128 \
  --holdout-size 128
```

### D. CIFAR-10 Protocol Runs

The default protocol runner launches the three priority MMD-GAN experiments:

1. small critic, `rq*`
2. large critic, `rq*`
3. small critic, `rbf`

Recommended course-project launch:

```bash
python3 scripts/run_cifar10_v2.py \
  --suite course \
  --steps 30000 \
  --critic-steps 5 \
  --batch-size 64 \
  --lr-g 1e-4 \
  --lr-c 1e-4 \
  --feature-dim 16 \
  --activation-penalty 1.0 \
  --checkpoint-metric-samples 10000 \
  --final-metric-samples 25000
```

This creates one subdirectory per run:

- `mmd_small_rq_star`
- `mmd_large_rq_star`
- `mmd_small_rbf`

Each run logs:

- sample grids every 5,000 steps
- checkpoints every 5,000 steps
- training losses and feature norms
- KID/FID when available
- a per-run `summary.json`

To include the WGAN-GP baseline in the same launch:

```bash
python3 scripts/run_cifar10_v2.py \
  --suite paper \
  --steps 30000 \
  --critic-steps 5 \
  --batch-size 64 \
  --lr-g 1e-4 \
  --lr-c 1e-4
```

Equivalent explicit baseline toggle:

```bash
python3 scripts/run_cifar10_v2.py \
  --suite course \
  --include-wgan \
  --steps 30000
```

For a quick infrastructure test before the longer run:

```bash
python3 scripts/run_cifar10_v2.py \
  --suite course \
  --steps 500 \
  --metrics-every 0 \
  --sample-every 250 \
  --save-every 250
```

### E. Suggested Full Project Schedule

Minimal robust sequence:

1. `make_moons` smoke test
2. full `make_moons` suite
3. bias illustration smoke test
4. full bias illustration
5. CIFAR-10 course suite at `30k` steps
6. optional WGAN-GP baseline

Concrete sequence:

```bash
python3 scripts/run_toy_moons_v2.py --mode suite --steps 2000 --seeds 0
python3 scripts/run_toy_moons_v2.py --mode suite --steps 20000 --seeds 0,1,2,3,4
python3 scripts/run_bias_illustration_v3.py --num-seeds 3 --critic-steps 200 --train-size 128 --holdout-size 128
python3 scripts/run_bias_illustration_v3.py --num-seeds 20 --critic-steps 1500 --train-size 512 --holdout-size 512
python3 scripts/run_cifar10_v2.py --suite course --steps 30000
python3 scripts/run_cifar10_v2.py --suite paper --steps 30000
```

### F. Where To Look After Each Run

- `summary_all.json` at the run root for the multi-run launchers
- per-run `summary.json` inside each experiment subdirectory
- sample images inside the run subdirectories
- `deliverables.md` for the bias illustration write-up
- `comparison_matrix.md` for the CIFAR-10 protocol launcher
