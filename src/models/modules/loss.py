from typing import Optional

import lpips
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.modules.discriminator import PatchGANDiscriminator


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


class PerceptualAdversarialLoss(nn.Module):
    def __init__(
        self,
        discriminator: PatchGANDiscriminator,
        disc_start: int,
        logvar_init: float = 0.0,
        kl_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.disc_factor = disc_factor
        self.disc_weight = disc_weight
        self.disc_start = disc_start
        self.discriminator = discriminator

        self.loss = nn.L1Loss(reduction="none")
        self.perceptual_loss = lpips.LPIPS(net="vgg").eval()

        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight

    def forward(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        posteriors: Tensor,
        global_step: int,
        weights: Optional[float] = None,
        last_layer: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict[str, float]]:
        # Standard L1 Loss - not reduced per pixel, rather per image
        rec_loss = self.loss(inputs.contiguous(), reconstructions.contiguous())

        # Use a pretrained VGG to get a perceptual loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        # self.logvar is a learned parameter to control the importance of rec loss
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar

        # Weight the NLL Loss by the weights if provided
        weighted_nll_loss = nll_loss if weights is None else nll_loss * weights

        # Sum to get a per-image loss and divide by batch size
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # Calculate KL Divergence between posterior and the normal distribtuon
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # Generator Update
        logits_fake = self.discriminator(reconstructions.contiguous())
        gen_loss = -torch.mean(logits_fake)

        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)

        if disc_factor > 0:
            try:
                disc_weight = self.calculate_adaptive_weight(
                    nll_loss, gen_loss, last_layer=last_layer
                )
            except RuntimeError:
                assert not self.training
                disc_weight = torch.tensor([0.0], device=next(self.parameters()).device)
        else:
            disc_weight = torch.tensor([0.0], device=next(self.parameters()).device)

        # Combine the losses
        ae_loss = (
            weighted_nll_loss + self.kl_weight * kl_loss + gen_loss * disc_factor * disc_weight
        )

        # Now calculate discriminator loss
        logits_real = self.discriminator(inputs.contiguous().detach())
        logits_fake = self.discriminator(reconstructions.contiguous().detach())

        disc_loss = disc_factor * hinge_d_loss(logits_real, logits_fake)

        # Return a dictionary with all of the losses
        loss_dict = {
            "Reconstruction Loss": rec_loss.mean(),
            "KL Loss": kl_loss.mean(),
            "Total Loss": ae_loss.mean(),
            "NLL Loss": nll_loss.mean(),
            "Logvar": self.logvar,
            "Disc Loss": disc_loss.mean(),
            "Disc Weight": disc_weight,
            "Logits Real": logits_real.mean(),
            "Logits Fake": logits_fake.mean(),
        }
        return ae_loss, disc_loss, loss_dict
