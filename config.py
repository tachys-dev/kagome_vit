import json
from pathlib import Path
from typing import Literal, Optional

from jsonargparse import ArgumentParser, ActionConfigFile, ActionYesNo

from jax import config
config.update("jax_enable_x64", True)

parser=ArgumentParser(default_config_files=[str(Path(__file__).parent/"vmc_config.yaml")])
parser.add_argument('--config', action=ActionConfigFile)

parser.add_argument('--seed', type=int)
parser.add_argument('--device', type=str)

parser.add_argument('--wandb.log', action=ActionYesNo)
parser.add_argument('--wandb.entity', type=Optional[str])
parser.add_argument('--wandb.project', type=str)
parser.add_argument('--wandb.name', type=str)
parser.add_argument('--wandb.mode', type=Literal['offline', 'online'])

parser.add_argument('--checkpoint.input', type=str)
parser.add_argument('--checkpoint.every', type=int)
parser.add_argument('--checkpoint.keep', type=int)

parser.add_argument('--lattice.L', type=int)

parser.add_argument('--hamiltonian.J', type=float)
parser.add_argument('--hamiltonian.sz', type=int)

parser.add_argument('--transformer.num_layers', type=int)
parser.add_argument('--transformer.heads', type=int)
parser.add_argument('--transformer.f', type=int)
parser.add_argument('--transformer.b', type=int)
parser.add_argument('--transformer.transl_invariant', action=ActionYesNo)
parser.add_argument('--transformer.spatial_attention', action=ActionYesNo)
parser.add_argument('--transformer.use_phase', action=ActionYesNo)
parser.add_argument('--transformer.real_projection', action=ActionYesNo)
parser.add_argument('--transformer.s_flip', type=int)

parser.add_argument('--vmc.N_mc', type=int)
parser.add_argument('--vmc.nsweeps', type=int)
parser.add_argument('--vmc.d_max', type=int)
parser.add_argument('--vmc.nbands', type=int)

parser.add_argument('--optimizer.N_opt', type=int)
parser.add_argument('--optimizer.eta', type=float)
parser.add_argument('--optimizer.eta_final', type=float)
parser.add_argument('--optimizer.diag_shift', type=float)
parser.add_argument('--optimizer.momentum', type=float)
parser.add_argument('--optimizer.beta', type=float)
parser.add_argument('--optimizer.mode', type=str)

cfg=parser.parse_args()
VARIABLES=json.loads(parser.dump(cfg, format="json"))

cfg.transformer.d_model=cfg.transformer.f*cfg.transformer.heads
cfg.wandb.run_name=f"L={cfg.lattice.L}_s_flip={cfg.transformer.s_flip}_nl={cfg.transformer.num_layers}_h={cfg.transformer.heads}_f={cfg.transformer.f}"
if cfg.wandb.name:
    cfg.wandb.run_name+=f"_{cfg.wandb.name}"

if cfg.device=='cpu':
    config.update('jax_platform_name', 'cpu')

assert cfg.transformer.s_flip in (0, 1, -1)
assert cfg.transformer.s_flip==0 or cfg.hamiltonian.sz==0
