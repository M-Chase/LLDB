import logging

import torch

from models import modules as M
from models.modules.free_lunch_utils import register_free_upblock2d
logger = logging.getLogger("base")

# Generator
def define_G(opt):
    opt_net = opt["network_G"]
    which_model = opt_net["which_model_G"]
    setting = opt_net["setting"]
    netG = getattr(M, which_model)(**setting)
    register_free_upblock2d(netG)
    print(netG)
    return netG


# Discriminator
def define_D(opt):
    opt_net = opt["network_D"]
    setting = opt_net["setting"]
    netD = getattr(M, which_model)(**setting)
    return netD


# Perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt["gpu_ids"]
    device = torch.device("cuda" if gpu_ids else "cpu")
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = M.VGGFeatureExtractor(
        feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, device=device
    )
    netF.eval()  # No need to train
    return netF
