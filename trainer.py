import torch
import torch.nn as nn
from networks.default import Generator, Discriminator
from losses import AdversarialLoss
from losses import compute_gp, tv_loss_reg
from torch.optim import Adam
from utils import initialize_weights
import torchvision.models as models

import os
import copy

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class TravelGAN(nn.Module):
    def __init__(self, hparams, device="cpu", ):
        super(TravelGAN, self).__init__()
        # Parameters
        self.hparams = hparams
        self.device = device
         
        # Modules
        self.gen_ab = Generator(**hparams["gen"])
        self.gen_ba = Generator(**hparams["gen"])
       
        self.dis_a = Discriminator(**hparams["dis"])
        self.dis_b = Discriminator(**hparams["dis"])

        # Loss coefficients
        self.lambda_adv = hparams["lambda_adv"]
        self.lambda_gp = hparams["lambda_gp"]
        self.type = hparams["type"]

        # Learning rates
        self.lr_dis = hparams["lr_dis"]
        self.lr_gen = hparams["lr_gen"]

        # Optimizers
        dis_params = list(self.dis_a.parameters()) + \
            list(self.dis_b.parameters())
        gen_params = list(self.gen_ab.parameters()) + \
            list(self.gen_ba.parameters())
        self.dis_optim = Adam([p for p in dis_params],
                            lr=self.lr_dis, betas=(0.5, 0.9))
        self.gen_optim = Adam([p for p in gen_params],
                              lr=self.lr_gen, betas=(0.5, 0.9))

        # Losses
        self.adv_loss = AdversarialLoss(self.type, device)
        if self.type == "wgangp":
            self.gp = compute_gp

        self.total_variation_loss = tv_loss_reg
        # Initialization
        self.apply(initialize_weights)
        self.set_to(device)

    def forward(self, x_a, x_b):
        self.eval()
        return self.gen_ab(x_a), self.gen_ba(x_b)
    
    def transformToCartoon(self, x_a):
        self.eval()
        return self.gen_ab(x_a)

    def transformToReal(self, x_b):
        self.eval()
        return self.gen_ba(x_b)

    def dis_update(self, x_a, x_b):
        self.dis_optim.zero_grad()
        x_ab = self.gen_ab(x_a).detach()
        x_ba = self.gen_ba(x_b).detach()
       
        adv_loss = self.adv_loss(self.dis_a(x_a), True) + \
            self.adv_loss(self.dis_b(x_b), True) + \
            self.adv_loss(self.dis_b(x_ab), False) + \
            self.adv_loss(self.dis_a(x_ba), False)
        dis_loss = self.lambda_adv * adv_loss
        if self.type == "wgangp":
            gp = self.gp(self.dis_a, x_a, x_ba) + \
                self.gp(self.dis_b, x_b, x_ab)
            dis_loss += self.lambda_gp * gp
        dis_loss.backward()
        self.dis_optim.step()
        return dis_loss.item()

    def gen_update(self, x_a, x_b):
        self.gen_optim.zero_grad()
        
        x_ab = self.gen_ab(x_a)
        x_ba = self.gen_ba(x_b)
     
        adv_loss = self.adv_loss(self.dis_b(x_ab), True) + \
            self.adv_loss(self.dis_a(x_ba), True)
        
        total_variation_loss = self.total_variation_loss(x_ab) + self.total_variation_loss(x_ba)

        gen_loss = self.lambda_adv * adv_loss + \
            total_variation_loss + \
            con_loss
        
        gen_loss.backward(retain_graph=True)
        self.gen_optim.step()
        
        return x_ab, x_ba, gen_loss.item()

    def resume(self, file):
        state_dict = torch.load(file, map_location=self.device)
        self.load_state_dict(state_dict)

    def save(self, checkpoint_dir, epoch):
        file = 'model_{}.pt'.format(epoch + 1)
        file = os.path.join(checkpoint_dir, file)
        torch.save(self.state_dict(), file)

    def set_to(self, device):
        self.device = device
        self.to(device)
        print("Model loaded on device : {}".format(device))
