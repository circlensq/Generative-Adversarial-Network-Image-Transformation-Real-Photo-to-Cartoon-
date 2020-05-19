import torch
import torch.nn as nn
import torch.autograd as ag
import numpy as np

from itertools import combinations, product

class AdversarialLoss(nn.Module):
    def __init__(self, type, device, true_label=1., fake_label=0.):
        super(AdversarialLoss, self).__init__()
        self.device = device
        self.type = type
        self.true_label = torch.tensor(true_label, device=device)
        self.fake_label = torch.tensor(fake_label, device=device)
        if type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif type == "lsgan":
            self.loss = nn.MSELoss()
        elif type in ["wgan", "wgangp", "hinge"]:
            self.loss = None

    def forward(self, x, bool):
        if self.type in ["vanilla", "lsgan"]:
            if bool:
                y = self.true_label.expand_as(x)
                return self.loss(x, y)
            else:
                y = self.fake_label.expand_as(x)
                return self.loss(x, y)
        elif self.type in ["wgan", "wgangp"]:
            return - x.mean() if bool else x.mean()
        elif self.type == "hinge":
            relu = nn.ReLU(True)
            return relu(1. - x).mean() if bool else relu(1. + x).mean()


@torch.enable_grad()
def compute_gp(critic, real, fake):
    alpha = torch.rand((real.size(0), 1, 1, 1), device=real.device)
    alpha = alpha.expand(real.size())
    alpha.requires_grad_(True)
    interpol = alpha * real + (1 - alpha) * fake
    interpol_critic = critic(interpol)
    gradients = ag.grad(outputs=interpol_critic, inputs=interpol,
                        grad_outputs=torch.ones(interpol_critic.size(),
                                                device=real.device),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return ((gradients_norm - 1) ** 2).mean()

def tv_loss_reg(img, REGULARIZATION=1e-6):
    reg_loss = REGULARIZATION * (
                torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + 
                torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])))

    return reg_loss