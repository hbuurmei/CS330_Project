import argparse
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from load_data import TrajectoryDataset
from model import DynamicsModel


# Training function
def train(train_loader, val_loader, model, optimizer, device):
    
    model.train()
    loss = nn.MSELoss()
    total_train_loss = 0.0
    n_train_batches = len(train_loader)
    for batch in train_loader:
        # Extract batch data
        states = batch['states'].to(dtype=torch.float32, device=device)
        controls = batch['controls'].to(dtype=torch.float32, device=device)
        forces = batch['forces'].to(dtype=torch.float32, device=device)
        torques = batch['torques'].to(dtype=torch.float32, device=device)
        inputs = torch.cat([states, controls], dim=1)
        pred_forces, pred_torques = model(inputs)
        # Compute batch loss as mse of predicted vs true forces and torques (equal weights for now)
        batch_loss = loss(pred_forces, forces) + loss(pred_torques, torques)
        # Gradient step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

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


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_quads', type=int, default=10, help='Number of quadrotors to train on')
    parser.add_argument('--train', help='train model', action=argparse.BooleanOptionalAction)
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

    # Define data splits
    n_quads = args.n_quads
    num_train = 8
    num_val = 1
    num_test = n_quads - num_train - num_val

    # Load data
    batch_size = 16
    train_quad_ids = list(range(1, num_train + 1))
    train_data = TrajectoryDataset(train_quad_ids)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_quad_ids = list(range(num_train + 1, num_train + num_val + 1))
    val_data = TrajectoryDataset(val_quad_ids)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_quad_ids = list(range(num_train + num_val + 1, n_quads + 1))
    test_data = TrajectoryDataset(test_quad_ids)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Create model
    dynamics_model = DynamicsModel()
    dynamics_model.to(device)
    model_name = 'dynamics_model'

    # Create an optimizer and learning rate scheduler
    optimizer = optim.Adam(dynamics_model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.95)

    # Training loop
    num_epochs = int(1e2)
    n_save = 1e2
    n_log = 1e1
    if args.train:
        train_losses = []
        val_losses = []
        for epoch in tqdm.tqdm(range(num_epochs)):
            train_loss, val_loss = train(train_loader, val_loader, dynamics_model, optimizer, device)
            train_losses.append(train_loss.cpu())
            val_losses.append(val_loss.cpu())
            if epoch % n_save == 0 or epoch == num_epochs - 1:
                torch.save(dynamics_model.state_dict(), f'models/{model_name}.pt')
            if epoch % n_log == 0:
                print(f"Epoch {epoch}/{num_epochs}, Training Loss: {train_loss:.5e}, Validation Loss: {val_loss:.5e}")
        
        print("Training Complete!")
        np.save(f'results/{model_name}_train_losses.npy', np.array(train_losses))
        np.save(f'results/{model_name}_val_losses.npy', np.array(val_losses))
    else:
        dynamics_model.load_state_dict(torch.load(f'models/{model_name}.pt'))
