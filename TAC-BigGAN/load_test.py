import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

from BigGAN import Generator, Discriminator
import utils


os.environ["CUDA_VISIBLE_DEVICES"]= "3"

device = 'cuda'
device = 'cpu'
strict = True

parser = utils.prepare_parser()
config = vars(parser.parse_args())


config['resolution'] = utils.imsize_dict[config['dataset']]
config['n_classes'] = utils.nclass_dict[config['dataset']]
config['G_activation'] = utils.activation_dict[config['G_nl']]
config['D_activation'] = utils.activation_dict[config['D_nl']]
G = Generator(**config)
# D = Discriminator()

print(os.getcwd())
G.load_state_dict(torch.load('../Biggan_result/weights/Twin_AC_AC_weight1.0_BigGAN_C100_seed0_Gch64_Dch64_bs100_nDs2_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema/G.pth'), strict=strict)



# D.load_state_dict(torch.load('../Biggan_result/weights/Twin_AC_AC_weight1.0_BigGAN_C100_seed0_Gch64_Dch64_bs100_nDs2_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema/D.pth'), strict=strict)
