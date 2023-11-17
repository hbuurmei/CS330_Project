import argparse
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

from load_data import TrajectoryDataset
from model import DynamicsModel
from train_model import train


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fine_tune', help='fine-tune model', action=argparse.BooleanOptionalAction)
parser.add_argument('--scratch', help='train model from scratch', action=argparse.BooleanOptionalAction)
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
n_train = int(0.005 * len(dataset))
n_val = int(0.1 * len(dataset))
n_test = len(dataset) - n_train - n_val
train_data, val_data, test_data = random_split(dataset, [n_train, n_val, n_test])
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


# Get parameters to be fine-tuned function
def parameters_to_fine_tune(model, mode):
    # Get the parameters to tune
    num_shared = model.num_shared
    if mode == 'shared':
        ft_params = list(model.shared_layers.parameters())
    elif mode == 'head':
        ft_params = list(model.force_layers.parameters()) + list(model.torque_layers.parameters())
    elif mode == 'all':
        ft_params = list(model.parameters())
    else:
        raise ValueError(f"Unrecognized fine-tuning mode {mode}")
    
    return ft_params


# Model that we train from scratch for comparison
dynamics_model_scratch = DynamicsModel()
dynamics_model_scratch.to(device)
model_scratch_name = 'dynamics_model_scratch'

# Create an optimizer and learning rate scheduler
optimizer = optim.Adam(dynamics_model_scratch.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

# Training loop
num_epochs = int(1e2)
n_save = 1e2
n_log = 1e1
if args.scratch:
    train_losses = []
    val_losses = []
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_loss, val_loss = train(train_loader, val_loader, dynamics_model_scratch, optimizer, device)
        train_losses.append(train_loss.cpu())
        val_losses.append(val_loss.cpu())
        if epoch % n_save == 0 or epoch == num_epochs - 1:
            torch.save(dynamics_model_scratch.state_dict(), f'models/{model_scratch_name}.pt')
        if epoch % n_log == 0:
            print(f"Epoch {epoch}/{num_epochs}, Training Loss: {train_loss:.5e}, Validation Loss: {val_loss:.5e}")
    
    print("Training from scratch Complete!")
    np.save(f'results/{model_scratch_name}_train_losses.npy', np.array(train_losses))
    np.save(f'results/{model_scratch_name}_val_losses.npy', np.array(val_losses))
else:
    dynamics_model_scratch.load_state_dict(torch.load(f'models/{model_scratch_name}.pt'))


# Load model we want to fine-tune
dynamics_model_ft = DynamicsModel()
dynamics_model_ft.to(device)
model_name = 'dynamics_model'
dynamics_model_ft.load_state_dict(torch.load(f'models/{model_name}.pt'))
model_ft_name = 'dynamics_model_ft'

# Parameters to fine tune
mode = args.mode
ft_params = parameters_to_fine_tune(dynamics_model_ft, mode)

# Create an optimizer and learning rate scheduler
optimizer = optim.Adam(ft_params, lr=1e-3)
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

# Training loop
num_epochs = int(1e2)
n_save = 1e2
n_log = 1e1
if args.fine_tune:
    train_losses = []
    val_losses = []
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_loss, val_loss = train(train_loader, val_loader, dynamics_model_ft, optimizer, device)
        train_losses.append(train_loss.cpu())
        val_losses.append(val_loss.cpu())
        if epoch % n_save == 0 or epoch == num_epochs - 1:
            torch.save(dynamics_model_ft.state_dict(), f'models/{model_ft_name}_{mode}.pt')
        if epoch % n_log == 0:
            print(f"Epoch {epoch}/{num_epochs}, Training Loss: {train_loss:.5e}, Validation Loss: {val_loss:.5e}")
    
    print("Fine-tuning Complete!")
    np.save(f'results/{model_ft_name}_{mode}_train_losses.npy', np.array(train_losses))
    np.save(f'results/{model_ft_name}_{mode}_val_losses.npy', np.array(val_losses))
else:
    dynamics_model_ft.load_state_dict(torch.load(f'models/{model_ft_name}_{mode}.pt'))


# Evaluate models (scratch vs fine-tuned)
def evaluate_model(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    loss = 0
    with torch.no_grad():
        for test_traj in loader:
            states = test_traj['states'].to(dtype=torch.float32, device=device)
            controls = test_traj['controls'].to(dtype=torch.float32, device=device)
            inputs = torch.cat((states, controls), dim=1)
            forces = test_traj['forces'].to(dtype=torch.float32, device=device)
            torques = test_traj['torques'].to(dtype=torch.float32, device=device)
            pred_forces, pred_torques = model(inputs)
            loss += criterion(pred_forces, forces) + criterion(pred_torques, torques)
    
    return loss / len(loader)


print(f"Scratch model test MSE: {evaluate_model(dynamics_model_scratch, test_loader, device):.5e}")
print(f"Fine-tuned model test MSE: {evaluate_model(dynamics_model_ft, test_loader, device):.5e}")

# Plot losses during training for both models
train_losses_scratch = np.load(f'results/{model_scratch_name}_train_losses.npy')
val_losses_scratch = np.load(f'results/{model_scratch_name}_val_losses.npy')
train_losses_ft = np.load(f'results/{model_ft_name}_{mode}_train_losses.npy')
val_losses_ft = np.load(f'results/{model_ft_name}_{mode}_val_losses.npy')

plt.figure()
plt.plot(train_losses_scratch, label='Scratch')
plt.plot(train_losses_ft, label='Fine-tuned')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'figures/ft_training_losses.png')
