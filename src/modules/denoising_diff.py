import lightning as L

from src.modules.diffusers.gaussian import GaussianDiffusion


class LitModel(L.LightningModule):
    def __init__(self, diffuser: GaussianDiffusion):
        super().__init__()
