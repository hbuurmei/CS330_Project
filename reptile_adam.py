import numpy as np
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from load_data import TrajectoryDataset
from model import DynamicsModel


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--meta_train', help='meta-train Reptile', action=argparse.BooleanOptionalAction)
parser.add_argument('--train_eval', help='train Reptile on evaluation set', action=argparse.BooleanOptionalAction)
parser.add_argument('--n_quads', type=int, default=10, help='Number of quadrotors to train on')
parser.add_argument('--inner_stepsize', type=float, default=0.02, help='Stepsize in inner SGD')
parser.add_argument('--inner_epochs', type=int, default=2, help='Number of epochs in inner SGD')
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
rng = np.random.RandomState(seed)
torch.manual_seed(seed)

# Define data splits
n_quads = args.n_quads
num_train = 8
num_val = 1
num_test = n_quads - num_train - num_val

# Load data
batch_size = 16
train_quad_ids = list(range(1, num_train + 1))
train_loaders = []
for train_quad_id in train_quad_ids:
    train_data = TrajectoryDataset([train_quad_id])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_loaders.append(train_loader)
val_quad_ids = list(range(num_train + 1, num_train + num_val + 1))
val_data = TrajectoryDataset(val_quad_ids)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
test_quad_ids = list(range(num_train + num_val + 1, n_quads + 1))
test_data = TrajectoryDataset(test_quad_ids)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Create model
dynamics_model_reptile = DynamicsModel()
dynamics_model_reptile.to(device)
model_name = 'dynamics_model_reptile_meta_adam'

# Create optimizer
optimizer = optim.Adam(dynamics_model_reptile.parameters(), lr=1e-4)


def train_inner_reptile(train_loader, val_loader, model, device, args):

    model.train()
    loss = nn.MSELoss()
    total_train_loss = 0.0
    n_train_batches = len(train_loader)
    for _ in range(args.inner_epochs):
        for batch in train_loader:
            # Extract batch data
            states = batch['states'].to(dtype=torch.float32, device=device)
            controls = batch['controls'].to(dtype=torch.float32, device=device)
            forces = batch['forces'].to(dtype=torch.float32, device=device)
            torques = batch['torques'].to(dtype=torch.float32, device=device)
            inputs = torch.cat([states, controls], dim=1)
            pred_forces, pred_torques = model(inputs)
            # Compute batch loss as mse of predicted vs true forces and torques
            batch_loss = loss(pred_forces, forces) + loss(pred_torques, torques)
            # Gradient step
            model.zero_grad()
            batch_loss.backward()
            # Apply gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            for param in model.parameters():
                param.data -= args.inner_stepsize * param.grad.data

            total_train_loss += batch_loss

    # Average the training losses
    total_train_loss /= n_train_batches

    # Validation
    model.eval()
    total_val_loss = 0.0
    n_val = len(val_loader)
    with torch.no_grad():
        for val_traj in val_loader:
            states = val_traj['states'].to(dtype=torch.float32, device=device)
            controls = val_traj['controls'].to(dtype=torch.float32, device=device)
            forces = val_traj['forces'].to(dtype=torch.float32, device=device)
            torques = val_traj['torques'].to(dtype=torch.float32, device=device)
            inputs = torch.cat([states, controls], dim=1)
            pred_forces, pred_torques = model(inputs)
            traj_loss = loss(pred_forces, forces) + loss(pred_torques, torques)
            total_val_loss += traj_loss

    # Average the validation losses
    total_val_loss /= n_val

    return total_train_loss.detach(), total_val_loss.detach()


# Reptile (outer) training loop
if args.meta_train:
    num_it = int(10e3)  # approximately same number as epochs as in fine-tuning
    n_save = 1e3
    n_log = 1e1
    train_losses = []
    val_losses = []
    for it in range(num_it):
        weights_before = deepcopy(dynamics_model_reptile.state_dict())
        # Sample a task from the training set
        train_loader = rng.choice(train_loaders)
        # Train on task for args.inner_epochs
        train_loss, val_loss = train_inner_reptile(train_loader, val_loader, dynamics_model_reptile, device, args)
        train_losses.append(train_loss.cpu())
        val_losses.append(val_loss.cpu())

        # For the outer optimization, we use the meta-gradient as (weights_before - weights_after)
        # which is essentially interpolating between current weights and trained weights from this task
        weights_after = dynamics_model_reptile.state_dict()
        # Use Adam with (weights_after - weights_before) as gradient
        optimizer.zero_grad()
        for name, param in dynamics_model_reptile.named_parameters():
            param.grad = weights_after[name] - weights_before[name]
        optimizer.step()

        if it % n_save == 0 or it == num_it - 1:
            torch.save(dynamics_model_reptile.state_dict(), f'models/{model_name}.pt')
        if it % n_log == 0:
            print(f"it {it}/{num_it}, Training Loss: {train_loss:.5e}, Validation Loss: {val_loss:.5e}")

    print("Reptile meta-training complete!")
    np.save(f'results/{model_name}_train_losses.npy', np.array(train_losses))
    np.save(f'results/{model_name}_val_losses.npy', np.array(val_losses))
else:
    dynamics_model_reptile.load_state_dict(torch.load(f'models/{model_name}.pt'))

model_name = 'dynamics_model_reptile_adam'
test_quad_ids = [10]
dataset = TrajectoryDataset(test_quad_ids)
n_train = int(0.1 * len(dataset))
n_val = int(0.1 * len(dataset))
n_test = len(dataset) - n_train - n_val
train_data, val_data, test_data = random_split(dataset, [n_train, n_val, n_test])
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Train Reptile on part of evaluation set
if args.train_eval:
    num_epochs = int(1e3)
    n_save = 1e2
    n_log = 1e1
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_loss, val_loss = train_inner_reptile(train_loader, val_loader, dynamics_model_reptile, device, args)

        if epoch % n_save == 0 or epoch == num_epochs - 1:
            torch.save(dynamics_model_reptile.state_dict(), f'models/{model_name}_test_train.pt')
        if epoch % n_log == 0:
            print(f"Epoch {epoch}/{num_epochs}, Training Loss: {train_loss:.5e}, Validation Loss: {val_loss:.5e}")
        
    print("Reptile training complete!")
    np.save(f'results/{model_name}_train_losses.npy', np.array(train_losses))
    np.save(f'results/{model_name}_val_losses.npy', np.array(val_losses))
else:
    dynamics_model_reptile.load_state_dict(torch.load(f'models/{model_name}_test_train.pt'))


def evaluate_model(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    loss = 0
    with torch.no_grad():
        i = 0
        for test_traj in loader:
            i += 1
            states = test_traj['states'].to(dtype=torch.float32, device=device)
            controls = test_traj['controls'].to(dtype=torch.float32, device=device)
            inputs = torch.cat((states, controls), dim=1)
            forces = test_traj['forces'].to(dtype=torch.float32, device=device)
            torques = test_traj['torques'].to(dtype=torch.float32, device=device)
            pred_forces, pred_torques = model(inputs)
            loss += criterion(pred_forces, forces) + criterion(pred_torques, torques)
    return loss / len(loader)


print(f"Reptile model test MSE: {evaluate_model(dynamics_model_reptile, test_loader, device):.5e}")

# Plot losses during training for both models
train_losses_reptile = np.load(f'results/{model_name}_train_losses.npy')
val_losses_reptile = np.load(f'results/{model_name}_val_losses.npy')

plt.figure()
plt.plot(train_losses_reptile, label='Train')
plt.plot(val_losses_reptile, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'figures/reptile_adam_losses.png')
