from config import cfg, VARIABLES

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tachys.lattice.lattice import Lattice
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_hamiltonian
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.lattice.bond_exchange import BondExchange
from tachys.wavefunction import WaveFunction
from tachys.optimizer import MARCH
from tachys.ground_state_training import train
from tachys.checkpoint import load_checkpoint, get_last_step
from tachys.parallel import rank, MASTER

from vit_kagome import ViT_phase, SpinFlipExpSum


def kagome_nk(extent):
    L1, L2=extent
    return Lattice.create(a1=[0.5, np.sqrt(3.0)/2], a2=[1.0, 0.0], shape=(L2, L1),
                          basis=[[0.5, 0.0], [0.25, np.sqrt(3.0)/4], [0.75, np.sqrt(3.0)/4]])


def kagome_nk_nn_bonds(J=1.0):
    return [((0, 0), J, 0, 1), ((0, 0), J, 0, 2), ((0, 0), J, 1, 2),
            ((1, 0), J, 2, 0), ((0, 1), J, 2, 1), ((-1, 1), J, 0, 1)]


def lr_schedule(step):
    alpha_lr=cfg.optimizer.eta_final/cfg.optimizer.eta
    frac=step/cfg.optimizer.N_opt
    return cfg.optimizer.eta*(alpha_lr + (1 - alpha_lr)*0.5*(1 + math.cos(math.pi*frac)))


lattice=kagome_nk(extent=(cfg.lattice.L, cfg.lattice.L))
NUM_SITES=lattice.Ns

hamiltonian=heisenberg_hamiltonian(lattice, nn=kagome_nk_nn_bonds(cfg.hamiltonian.J))

key_master=jax.random.key(cfg.seed)
key_for_model_params, key_for_sampler=jax.random.split(key_master)
key_init_config, key_train, _=jax.random.split(key_for_sampler, 3)

initial_spins=init_config_fixed_magn(key_init_config, NUM_SITES, sz=cfg.hamiltonian.sz, N_mc=cfg.vmc.N_mc)
state=SpinState(spins=initial_spins, lattice=lattice)

action=BondExchange.create(lattice, max_dist=cfg.vmc.d_max, Nbands=cfg.vmc.nbands)

vit_module=ViT_phase(num_layers=cfg.transformer.num_layers, d_model=cfg.transformer.d_model, n_heads=cfg.transformer.heads,
                     patch_size=cfg.transformer.b, transl_invariant=cfg.transformer.transl_invariant,
                     spatial_attention=cfg.transformer.spatial_attention, use_phase=cfg.transformer.use_phase,
                     real_projection=cfg.transformer.real_projection)

if cfg.transformer.s_flip:
    vit_module=SpinFlipExpSum(module=vit_module, s_flip=cfg.transformer.s_flip)

params=vit_module.init(key_for_model_params, initial_spins[:1].astype(jnp.float64))


def apply_fn(p, s):
    return vit_module.apply(p, s.spins.astype(jnp.float64))


wf=WaveFunction(params=params, apply_fn=apply_fn)

optimizer=MARCH(diag_shift=cfg.optimizer.diag_shift, mode=cfg.optimizer.mode, mu=cfg.optimizer.momentum, beta=cfg.optimizer.beta)

opt_state, start_step=None, 0
if cfg.checkpoint.input:
    params, opt_state, state, key_train=load_checkpoint(cfg.checkpoint.input, state, optimizer.init(wf.params), params_template=wf.params)
    wf=wf.replace(params=params)
    start_step=get_last_step(cfg.checkpoint.input)

wandb_run=None
if cfg.wandb.log and rank==MASTER:
    import wandb
    wandb_run=wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=cfg.wandb.run_name, mode=cfg.wandb.mode,
                         config={**VARIABLES, "checkpoint_every": cfg.checkpoint.every, "checkpoint_keep": cfg.checkpoint.keep})

key_train, state, wf, opt_state, history=train(key_train, hamiltonian, state, wf, optimizer, action, cfg.optimizer.N_opt-start_step,
                                               lr_schedule, cfg.vmc.N_mc, wandb_run=wandb_run, nsweeps=cfg.vmc.nsweeps,
                                               opt_state=opt_state, start_step=start_step)

if rank==MASTER:
    out=Path(wandb_run.dir) if wandb_run is not None else Path(".")
    log={"step": list(range(start_step, start_step+len(history["energy"]))), **{k: [float(x) for x in v] for k, v in history.items()}}
    (out/"history.json").write_text(json.dumps(log))

if wandb_run is not None:
    wandb_run.finish()
