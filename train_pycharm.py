import ignite.distributed as idist
import hydra
from omegaconf import DictConfig, OmegaConf
import os

import torch

from models.bts.trainer import training as bts
from models.bts.trainer_overfit import training as bts_overfit

## connection to the cluster server and debugging
import pydevd_pycharm   ## for external debugging
sv = 51    ## atcremers(sv) server allocation for debug server in Pycharm IDE
port_ = 58023
if   sv == 51:    debug_sv = '131.159.18.70'
elif sv == 59:    debug_sv = '131.159.18.113'
elif sv == 85:
    debug_sv = '131.159.18.198'
    port_ = 58024

import pydevd_pycharm
pydevd_pycharm.settrace(debug_sv, port=port_, stdoutToServer=True, stderrToServer=True)  ## IDE host name of the machine where the IDE is running

@hydra.main(version_base=None, config_path="configs", config_name="exp_kitti_360_DFT")
def main(config: DictConfig):

    OmegaConf.set_struct(config, False)

    os.environ["NCCL_DEBUG"] = "INFO"
    # torch.autograd.set_detect_anomaly(True)

    backend = config.get("backend", None)
    nproc_per_node = config.get("nproc_per_node", None)
    with_amp = config.get("with_amp", False)
    spawn_kwargs = {}

    spawn_kwargs["nproc_per_node"] = nproc_per_node
    if backend == "xla-tpu" and with_amp:
        raise RuntimeError("The value of with_amp should be False if backend is xla")
    ## the script will use the "bts_overfit" training function that's been imported from models.bts.trainer_overfit
    training = globals()[config["model"]]
    ## A distributed training context is created and the training function is run:
    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
        parallel.run(training, config)
    ## idist.Parallel creates a context for distributed training.
    ## The parallel.run method starts the distributed computation.
if __name__ == "__main__":
    main()