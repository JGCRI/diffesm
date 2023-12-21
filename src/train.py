from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from diffusers import DDPMScheduler

from data.climate_dataset import ClimateDataset
from trainers.unet_trainer import UNetTrainer
from models.video_net import UNetModel3D


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    # Create accelerator object and set RNG seed
    accelerator = Accelerator(**cfg.trainer.accelerator)
    set_seed(cfg.seed)

    # Logger works with distributed processes
    logger = get_logger(__name__, log_level="INFO")

    # Init logger
    logger.info(f"Instantiating datasets <{cfg.data.shared._target_}>")
    train_cfg = dict(cfg.data.shared) | dict(cfg.data.train)
    val_cfg = dict(cfg.data.shared) | dict(cfg.data.val)

    # Avoid race conditions when loading data
    with accelerator.main_process_first():
        train_set: ClimateDataset = instantiate(
            train_cfg, data_dir=cfg.paths.data_dir, _recursive_=False
        )
        val_set: ClimateDataset = instantiate(
            val_cfg, data_dir=cfg.paths.data_dir, _recursive_=False
        )

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: UNetModel3D = instantiate(cfg.model)

    logger.info(f"Instantiating scheduler <{cfg.scheduler._target_}>")
    scheduler: DDPMScheduler = instantiate(cfg.scheduler)

    # Start experiment tracking
    logger.info("Initializing Wandb")
    accelerator.init_trackers(
        cfg.wandb.project_name,
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        init_kwargs={"wandb": cfg.wandb.init_kwargs},
    )

    logger.info(f"Instantiating Trainer <{cfg.trainer._target_}>")
    trainer: UNetTrainer = instantiate(
        cfg.trainer,
        train_set,
        val_set,
        model=model,
        accelerator=accelerator,
        scheduler=scheduler,
    )

    logger.info("Starting training")
    trainer.train()


if __name__ == "__main__":
    main()
