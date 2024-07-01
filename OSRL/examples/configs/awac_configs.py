from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class AWACTrainConfig:
    # wandb params
    project: str = "OSRL-AWAC"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "AWAC"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    # training params
    # task: str = "OfflineMetadrive-easydense-v0"
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    dataset: str = None
    seed: int = 0
    device: str = "cuda:0"
    threads: int = 4
    reward_scale: float = 0.1
    cost_scale: float = 1
    cost_limit: int = 10
    actor_lr: float = 0.0003
    critic_lr: float = 0.0003
    episode_len: int = 300
    batch_size: int = 512
    #batch_size: int = 256
    update_steps: int = 300_000
    num_workers: int = 8
    # model params
    hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    gamma: float = 0.99
    tau: float = 0.005
    awac_lambda: float = 2.0
    exp_adv_max: float = 100.0
    num_q: int = 1
    #start_update_policy_step: int = 0
    episode_len: int = 300
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500
    # eval_every: int = 25

@dataclass
class AWACCarCircleConfig(AWACTrainConfig):
    pass

# Add more configurations for different tasks if needed

@dataclass
class AWACCarCircleConfig(AWACTrainConfig):
    pass


@dataclass
class AWACAntRunConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class AWACDroneRunConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class AWACDroneCircleConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class AWACCarRunConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class AWACAntCircleConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class AWACBallRunConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class AWACBallCircleConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class AWACCarButton1Config(AWACTrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class AWACCarButton2Config(AWACTrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class AWACCarCircle1Config(AWACTrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class AWACCarCircle2Config(AWACTrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class AWACCarGoal1Config(AWACTrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class AWACCarGoal2Config(AWACTrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class AWACCarPush1Config(AWACTrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class AWACCarPush2Config(AWACTrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class AWACPointButton1Config(AWACTrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class AWACPointButton2Config(AWACTrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class AWACPointCircle1Config(AWACTrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class AWACPointCircle2Config(AWACTrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class AWACPointGoal1Config(AWACTrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class AWACPointGoal2Config(AWACTrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class AWACPointPush1Config(AWACTrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class AWACPointPush2Config(AWACTrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class AWACAntVelocityConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class AWACHalfCheetahVelocityConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class AWACHopperVelocityConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class AWACSwimmerVelocityConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class AWACWalker2dVelocityConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class AWACEasySparseConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class AWACEasyMeanConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class AWACEasyDenseConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class AWACMediumSparseConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class AWACMediumMeanConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class AWACMediumDenseConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class AWACHardSparseConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class AWACHardMeanConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class AWACHardDenseConfig(AWACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


AWAC_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": AWACCarCircleConfig,
    "OfflineAntRun-v0": AWACAntRunConfig,
    "OfflineDroneRun-v0": AWACDroneRunConfig,
    "OfflineDroneCircle-v0": AWACDroneCircleConfig,
    "OfflineCarRun-v0": AWACCarRunConfig,
    "OfflineAntCircle-v0": AWACAntCircleConfig,
    "OfflineBallCircle-v0": AWACBallCircleConfig,
    "OfflineBallRun-v0": AWACBallRunConfig,
    # safety_gymnasium: car
    "OfflineCarButton1Gymnasium-v0": AWACCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": AWACCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": AWACCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": AWACCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": AWACCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": AWACCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": AWACCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": AWACCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": AWACPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": AWACPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": AWACPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": AWACPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": AWACPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": AWACPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": AWACPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": AWACPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": AWACAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": AWACHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": AWACHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": AWACSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": AWACWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": AWACEasySparseConfig,
    "OfflineMetadrive-easymean-v0": AWACEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": AWACEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": AWACMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": AWACMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": AWACMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": AWACHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": AWACHardMeanConfig,
    "OfflineMetadrive-harddense-v0": AWACHardDenseConfig
}
