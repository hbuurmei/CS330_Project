import argparse
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

from load_data import TrajectoryDataset
from model import DynamicsModel


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dynamics_model', help='What model to train')
parser.add_argument('--mode', default='last', help='Fine-tuning mode')  # {shared, head, all}
args = parser.parse_args()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Set seed for reproducibility
seed = 0
torch.manual_seed(seed)

# Data that shall be used for fine-tuning
ft_quad_ids = [10]
dataset = TrajectoryDataset(ft_quad_ids)
train_data, val_data, test_data = random_split(dataset, [0.1, 0.1, 0.8])


# Fine tune function
def fine_tune(model, mode):
    # Get the parameters to tune
    num_shared = model.num_shared
    if mode == 'shared':
        ft_params = 0
    elif mode == 'head':
        ft_params = 0
    elif mode == 'all':
        ft_params = list(model.parameters())
    else:
        raise ValueError(f"Unrecognized fine-tuning mode {mode}")


# Load model we want to fine-tune
dynamics_model = DynamicsModel()
dynamics_model.to(device)
model_name = 'dynamics_model'
dynamics_model.load_state_dict(torch.load(f'models/{model_name}.pt'))
print(dynamics_model)
print(dynamics_model.parameters())


