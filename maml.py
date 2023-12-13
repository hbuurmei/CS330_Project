import numpy as np
import argparse
import copy
import matplotlib.pyplot as plt

import torch
from torch import nn, autograd
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from load_data import TrajectoryDataset
from model import DynamicsModel


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--meta_train', help='meta-train MAML', action=argparse.BooleanOptionalAction)
parser.add_argument('--train_eval', help='train MAML on evaluation set', action=argparse.BooleanOptionalAction)
parser.add_argument('--n_quads', type=int, default=10, help='Number of quadrotors to train on')
parser.add_argument('--inner_lr', type=float, default=0.02, help='Learning rate in inner optimization')
parser.add_argument('--inner_steps', type=int, default=1, help='Number of steps in inner optimization')
parser.add_argument('--outer_lr', type=float, default=0.001, help='Learning rate for outer optimization')
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
task_train_datasets = []
for train_quad_id in train_quad_ids:
    task_train_data = TrajectoryDataset([train_quad_id])
    task_train_datasets.append(task_train_data)
val_quad_ids = list(range(num_train + 1, num_train + num_val + 1))
val_data = TrajectoryDataset(val_quad_ids)
# val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
test_quad_ids = list(range(num_train + num_val + 1, n_quads + 1))
test_data = TrajectoryDataset(test_quad_ids)
# test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Create model
dynamics_model_maml = DynamicsModel()
dynamics_model_maml.to(device)
model_name = 'dynamics_model_maml_meta'

# Create optimizer
optimizer = optim.Adam(dynamics_model_maml.parameters(), lr=1e-4)


# def train_inner_maml(train_loader, dynamics_model_maml, device, args, train=True):

#     create_graph = True if train else False  # we do not want to create graph of derivatives when evaluating

#     parameters = {
#             k: torch.clone(v).requires_grad_(True)
#             for k, v in dynamics_model_maml.named_parameters()
#         }

#     loss = nn.MSELoss()
#     for _ in range(args.inner_steps):
#         inner_model = DynamicsModel(custom_parameters=parameters).to(device)
#         total_loss = 0.0
#         for sample in train_loader:
#             states = sample['states'].to(dtype=torch.float32, device=device)
#             controls = sample['controls'].to(dtype=torch.float32, device=device)
#             inputs = torch.cat((states, controls), dim=1)
#             forces = sample['forces'].to(dtype=torch.float32, device=device)
#             torques = sample['torques'].to(dtype=torch.float32, device=device)
#             pred_forces, pred_torques = inner_model(inputs)
#             sample_loss = loss(pred_forces, forces) + loss(pred_torques, torques)
#             total_loss += sample_loss
#         total_loss /= len(train_loader)
#         print(total_loss)

#         # We calculate the gradients of the loss w.r.t. the (current) parameters
#         gradients = autograd.grad(total_loss, parameters.values(), create_graph=create_graph)
#         # Then we do simple gradient descent on the parameters using the gradients
#         # Note that each parameter has its own learning rate defined in self._inner_lrs
#         parameters = {k: v - args.inner_lr * g for k, v, g in zip(parameters.keys(), parameters.values(), gradients)}

#     ### END CODE HERE ###
#     return parameters, total_loss


def train_inner_maml(train_loader, dynamics_model_maml, device, args, train=True):

    create_graph = True if train else False  # we do not want to create graph of derivatives when evaluating

    # Create a copy of the initial model parameters
    initial_parameters = copy.deepcopy(dynamics_model_maml.state_dict())
    
    loss = nn.MSELoss()
    for _ in range(args.inner_steps):
        # Restore the initial model parameters
        dynamics_model_maml.load_state_dict(initial_parameters)

        total_loss = 0.0
        for sample in train_loader:
            states = sample['states'].to(dtype=torch.float32, device=device)
            controls = sample['controls'].to(dtype=torch.float32, device=device)
            inputs = torch.cat((states, controls), dim=1)
            forces = sample['forces'].to(dtype=torch.float32, device=device)
            torques = sample['torques'].to(dtype=torch.float32, device=device)
            
            pred_forces, pred_torques = dynamics_model_maml(inputs)
            sample_loss = loss(pred_forces, forces) + loss(pred_torques, torques)
            total_loss += sample_loss
        
        total_loss /= len(train_loader)
        print(total_loss)

        # Calculate the gradients of the loss w.r.t. the model parameters
        gradients = autograd.grad(total_loss, dynamics_model_maml.parameters(), create_graph=create_graph)
        
        # Perform a gradient descent step on the model parameters
        for param, grad in zip(dynamics_model_maml.parameters(), gradients):
            param.data -= args.inner_lr * grad

    return dynamics_model_maml, total_loss



# maml (outer) training loop
if args.meta_train:
    num_epochs = int(1e2)  # approximately same number as epochs as in fine-tuning
    n_save = 1e2
    n_log = 1e1
    outer_losses = []
    for epoch in range(num_epochs):
        loss = nn.MSELoss()
        outer_loss = 0.0
        for task_train_dataset in task_train_datasets:
            # Split up task data into training and testing
            n_train = int(0.7 * len(task_train_dataset))
            n_test = len(task_train_dataset) - n_train
            train_data, test_data = random_split(task_train_dataset, [n_train, n_test])
            train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
            dynamics_model_maml, _ = train_inner_maml(train_loader, dynamics_model_maml, device, args)
            outer_loss_task = 0.0
            for sample in test_loader:
                states = sample['states'].to(dtype=torch.float32, device=device)
                controls = sample['controls'].to(dtype=torch.float32, device=device)
                inputs = torch.cat((states, controls), dim=1)
                forces = sample['forces'].to(dtype=torch.float32, device=device)
                torques = sample['torques'].to(dtype=torch.float32, device=device)
                pred_forces, pred_torques = dynamics_model_maml(inputs)
                outer_loss_task += loss(pred_forces, forces) + loss(pred_torques, torques)
            outer_loss_task /= len(test_loader)

            outer_loss += outer_loss_task
        outer_loss /= len(task_train_datasets)
        print('outer_loss', outer_loss)
        outer_losses.append(outer_loss)
        optimizer.zero_grad()
        outer_loss.backward()
        optimizer.step()

        if epoch % n_save == 0 or epoch == num_epochs - 1:
            torch.save(dynamics_model_maml.state_dict(), f'models/{model_name}.pt')
        if epoch % n_log == 0:
            print(f"Epoch {epoch}/{num_epochs}, Outer Loss: {outer_loss:.5e}")
        
    print("maml meta-training complete!")
    np.save(f'results/{model_name}_outer_losses.npy', np.array(outer_losses))
else:
    dynamics_model_maml.load_state_dict(torch.load(f'models/{model_name}.pt'))

model_name = 'dynamics_model_maml'
test_quad_ids = [10]
dataset = TrajectoryDataset(test_quad_ids)
n_train = int(0.1 * len(dataset))
n_val = int(0.1 * len(dataset))
n_test = len(dataset) - n_train - n_val
train_data, val_data, test_data = random_split(dataset, [n_train, n_val, n_test])
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Train maml on part of evaluation set
if args.train_eval:
    num_epochs = int(1e3)
    n_save = 1e2
    n_log = 1e1
    train_losses = []
    model_parameters = dynamics_model_maml.state_dict()
    for epoch in range(num_epochs):
        parameters, train_loss = train_inner_maml(train_loader, model_parameters, device, args, train=False)
        model_parameters = parameters
        train_losses.append(train_loss.cpu())

        if epoch % n_save == 0 or epoch == num_epochs - 1:
            torch.save(dynamics_model_maml.state_dict(), f'models/{model_name}_test_train.pt')
        if epoch % n_log == 0:
            print(f"Epoch {epoch}/{num_epochs}, Training Loss: {train_loss:.5e}")
        
    print("MAML training complete!")
    np.save(f'results/{model_name}_train_losses.npy', np.array(train_losses))
else:
    dynamics_model_maml.load_state_dict(torch.load(f'models/{model_name}_test_train.pt'))


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


print(f"MAML model test MSE: {evaluate_model(dynamics_model_maml, test_loader, device):.5e}")

# Plot losses during training for both models
train_losses_maml = np.load(f'results/{model_name}_train_losses.npy')

plt.figure()
plt.plot(train_losses_maml, label='Train')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'figures/maml_train_losses.png')
