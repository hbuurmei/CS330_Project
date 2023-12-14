import torch
import torch.nn as nn


# Model that returns the forces and torques as two separate heads, with n_shared shared layers followed by n_head layers
class DynamicsModel(nn.Module):
    def __init__(self,
                 num_inputs: int = 17,
                 num_shared: int = 3,
                 num_sep: int = 2,
                 num_neurons: list = [256, 128, 64, 32, 16],
                 num_outputs: int = 3,
                 act: nn.Module = nn.ReLU()):
        super(DynamicsModel, self).__init__()

        assert len(num_neurons) == num_shared + num_sep, "Number of (total) layers should match the length of the 'num_neurons' list."

        # TODO: remove this (unnecessary)
        # Save the number of shared layers for fine-tuning later
        self.register_buffer('num_shared', torch.tensor([num_shared]))

        # Define shared layers
        shared_layers = []
        # Input layer
        shared_layers.append(nn.Linear(num_inputs, num_neurons[0]))
        shared_layers.append(act)
        # Hidden layers
        for i in range(num_shared - 1):
            shared_layers.append(nn.Linear(num_neurons[i], num_neurons[i + 1]))
            shared_layers.append(act)
        self.shared_layers = nn.ModuleList(shared_layers)

        # Define force layers
        force_layers = []
        for j in range(num_sep):
            force_layers.append(nn.Linear(num_neurons[num_shared + j - 1], num_neurons[num_shared + j]))
            force_layers.append(act)
        force_layers.append(nn.Linear(num_neurons[num_shared + num_sep - 1], num_outputs))
        self.force_layers = nn.ModuleList(force_layers)

        # Define torque layers
        torque_layers = []
        for j in range(num_sep):
            torque_layers.append(nn.Linear(num_neurons[num_shared + j - 1], num_neurons[num_shared + j]))
            torque_layers.append(act)
        torque_layers.append(nn.Linear(num_neurons[num_shared + num_sep - 1], num_outputs))
        self.torque_layers = nn.ModuleList(torque_layers)

    def forward(self, x):
        # Shared layers
        for layer in self.shared_layers:
            x = layer(x)

        # Split the path for force and torque
        force_x = x
        torque_x = x

        # Force head
        for layer in self.force_layers:
            force_x = layer(force_x)
        forces = force_x

        # Torque head
        for layer in self.torque_layers:
            torque_x = layer(torque_x)
        torques = torque_x

        return forces, torques
