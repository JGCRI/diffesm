from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from diffusers import AutoencoderKL
import torch

from data.climate_dataset import ClimateDataset
from trainers.autoencoder_trainer import AutoEncoderTrainer


@hydra.main(
    version_base=None, config_path="../configs", config_name="pretrain_first_stage.yaml"
)
def main(cfg: DictConfig) -> None:
    # Create accelerator object and set RNG seed
    accelerator: Accelerator = instantiate(cfg.accelerator)
    set_seed(cfg.seed)

    # Logger works with distributed processes
    logger = get_logger(__name__, log_level="INFO")

    # Init logger
    logger.info(f"Instantiating datasets <{cfg.data.shared._target_}>")
    train_cfg = dict(cfg.data.shared) | dict(cfg.data.train)
    val_cfg = dict(cfg.data.shared) | dict(cfg.data.val)

    # Avoid race conditions when loading data
    with accelerator.main_process_first():
        train_set: ClimateDataset = instantiate(train_cfg, _recursive_=False)
        val_set: ClimateDataset = instantiate(val_cfg, _recursive_=False)

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: AutoencoderKL = torch.compile(instantiate(cfg.model))

    # Start experiment tracking
    logger.info("Initializing Wandb")
    accelerator.init_trackers(
        cfg.wandb.project_name,
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        init_kwargs={"wandb": cfg.wandb.init_kwargs},
    )

    logger.info(f"Instantiating Trainer <{cfg.trainer._target_}>")
    trainer: AutoEncoderTrainer = instantiate(
        cfg.trainer,
        train_set,
        val_set,
        autoencoder=model,
        accelerator=accelerator,
    )

    logger.info("Starting training")
    trainer.train()


if __name__ == "__main__":
    main()
