from dataclasses import dataclass

from hydra_plugins.hydra_submitit_launcher.submitit_launcher import BaseQueueConf, BaseSubmititLauncher


@dataclass
class InProcessQueueConf(BaseQueueConf):
    _target_: str = "hydra_plugins.hydra_process_launcher.InProcessLauncher"


class InProcessLauncher(BaseSubmititLauncher):
    _EXECUTOR = "debug"
