# Vision Transformer wave functions for the kagome Heisenberg antiferromagnet

This repository contains the code for the paper [*Gapless Nature of Spin Liquid in the S=1/2 Kagome Heisenberg Antiferromagnet with Neural-Network Quantum States*](https://arxiv.org/abs/XXXX.XXXXX). It trains variational ground states of the spin-1/2 Heisenberg antiferromagnet on the kagome lattice with a Vision Transformer (ViT) wave function.

The code is built on [`jax`](https://github.com/jax-ml/jax), [`flax`](https://github.com/google/flax) and [`tachys`](https://tachys.org).

## Content of the repository

- `train_kagome.py`: the kagome lattice, the Heisenberg Hamiltonian and the variational optimization.
- `vit_kagome.py`: the Vision Transformer wave function and its spin-inversion symmetrization.
- `vmc_config.yaml`: the configuration file, with the options defined in `config.py`.

## Usage

Install `tachys` (Python 3.12 or newer):

```bash
pip install "tachys[cuda12,wandb]==0.1.7"
```

For CUDA 13 drivers replace `cuda12` by `cuda13`.

The simulation is configured in `vmc_config.yaml`, and every option can also be overridden from the command line. A short test run:

```bash
python train_kagome.py --transformer.num_layers=2 --vmc.N_mc=1024 --optimizer.N_opt=200
```

On several GPUs, start one process per GPU, e.g. `srun python train_kagome.py` with as many tasks per node as GPUs per node.

## Output

The energy, variance, V-score and acceptance of every step are printed during the run.

Runs are logged with [Weights & Biases](https://wandb.ai) in offline mode. Each run gets a local folder `wandb/offline-run-<date>-<id>/files/` containing

- `checkpoints/`, the parameters, optimizer state and Markov chains (every 200 steps by default)
- `history.json`, the energy per site, variance per site, V-score, acceptance and learning rate of every step, written at the end of the run

A run can be uploaded later with `wandb sync wandb/offline-run-<date>-<id>`, or logged live with `--wandb.mode=online`. With `--no_wandb.log` no checkpoints are written and `history.json` goes to the current folder.
