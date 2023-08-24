import ignite.distributed as idist
import hydra
from omegaconf import DictConfig, OmegaConf
import os

import torch

from models.bts.trainer import training as bts
from models.bts.trainer_overfit import training as bts_overfit


import debugpy

@hydra.main(version_base=None, config_path="configs", config_name="exp_kitti_360_DFT_slurm")
def main(config: DictConfig):
    ## connecting to the cluster's remote server to debug. Note: remember to port forward to the port as the port 58022 is taken by atcremers PC
    sv, port_ = config.get("sv_", 58), config.get("port_", 22)  ## atcremers(sv) server allocation for debug server in Pycharm IDE
    debugpy.listen(port_)  ## IDE host name of the machine where the IDE is running
    print("__Waiting for debugger attach")
    debugpy.wait_for_client()
    debugpy.breakpoint()
    print('__break on this line')


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
    
    training = globals()[config["model"]]   ## the script will use the "bts_overfit" training function that's been imported from models.bts.trainer_overfit
    
    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:   ## A distributed training context is created and the training function is run
        parallel.run(training, config)
        
if __name__ == "__main__":
    main()