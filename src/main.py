import logging

import hydra
import lightning as L
import rootutils
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig

# Sets the project root directory to the real root directory
# This will set the PYTHONPATH root directory (to make accessing modules easier)
# And will also set the PROJECT_ROOT environment variable
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # Set seed for random number generators
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Init logger
    logging.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = instantiate(cfg.data)

    # logging.info(f"Instantiating model <{cfg.model._target_}>")
    # model = DNN(data_module.image_size, data_module.num_classes)
    # classifier = Classifier(model)
    # logger = WandbLogger(name='mnist', project='pytorch-lightning')

    # trainer = pl.Trainer(max_epochs=5, logger=logger)

    # Train the model
    # trainer.fit(classifier, datamodule=data_module)


if __name__ == "__main__":
    main()
