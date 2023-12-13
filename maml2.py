import random
import tqdm
import argparse
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import DataLoader, random_split

from load_data import TrajectoryDataset


NUM_INPUTS = 17
NUM_SHARED = 3
NUM_SEP = 2
NUM_NEURONS = [256, 128, 64, 32, 16]
NUM_OUTPUTS = 3
ACT = nn.ReLU()


class MAML:
    """Trains a MAML."""

    def __init__(
            self,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            device
    ):
        meta_parameters = {}

        self.device = device

        meta_parameters = {}
        # Shared layer parameters
        in_features = NUM_INPUTS
        for i in range(NUM_SHARED):
            meta_parameters[f'linear{i}_weight'] = nn.init.xavier_uniform_(
                torch.empty(
                    NUM_NEURONS[i],
                    in_features,
                    requires_grad=True
                )
            )
            meta_parameters[f'linear{i}_bias'] = nn.init.zeros_(
                torch.empty(
                    NUM_NEURONS[i],
                    requires_grad=True
                )
            )
            in_features = NUM_NEURONS[i]
        # Force and torque layer parameters
        for j in range(NUM_SEP):
            meta_parameters[f'force_linear{j}_weight'] = nn.init.xavier_uniform_(
                torch.empty(
                    NUM_NEURONS[NUM_SHARED + j],
                    in_features,
                    requires_grad=True
                )
            )
            meta_parameters[f'force_linear{j}_bias'] = nn.init.zeros_(
                torch.empty(
                    NUM_NEURONS[NUM_SHARED + j],
                    requires_grad=True
                )
            )
            meta_parameters[f'torque_linear{j}_weight'] = nn.init.xavier_uniform_(
                torch.empty(
                    NUM_NEURONS[NUM_SHARED + j],
                    in_features,
                    requires_grad=True
                )
            )
            meta_parameters[f'torque_linear{j}_bias'] = nn.init.zeros_(
                torch.empty(
                    NUM_NEURONS[NUM_SHARED + j],
                    requires_grad=True
                )
            )
            in_features = NUM_NEURONS[NUM_SHARED + j]
        # Output layer parameters
        meta_parameters['output_weight'] = nn.init.xavier_uniform_(
            torch.empty(
                NUM_OUTPUTS,
                in_features,
                requires_grad=True
            )
        )
        meta_parameters['output_bias'] = nn.init.zeros_(
            torch.empty(
                NUM_OUTPUTS,
                requires_grad=True
            )
        )

        self._meta_parameters = meta_parameters
        self._num_inner_steps = num_inner_steps
        self._inner_lrs = {
            k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs)
            for k in self._meta_parameters.keys()
        }
        self._outer_lr = outer_lr

        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()),
            lr=self._outer_lr
        )

    def _forward(self, input, parameters):
        """Computes predicted forces and torques."""

        x = input

        # Shared layers
        for i in range(NUM_SHARED):
            x = F.linear(
                x,
                parameters[f'linear{i}_weight'].to(self.device),
                parameters[f'linear{i}_bias'].to(self.device)
            )
            x = ACT(x)

        # Split the path for force and torque
        force_x = x
        torque_x = x

        # Force head
        for j in range(NUM_SEP):
            force_x = F.linear(
                force_x,
                parameters[f'force_linear{j}_weight'].to(self.device),
                parameters[f'force_linear{j}_bias'].to(self.device)
            )
            force_x = ACT(force_x)
        forces = F.linear(
            force_x,
            parameters['output_weight'].to(self.device),
            parameters['output_bias'].to(self.device)
        )
        
        # Torque head
        for j in range(NUM_SEP):
            torque_x = F.linear(
                torque_x,
                parameters[f'torque_linear{j}_weight'].to(self.device),
                parameters[f'torque_linear{j}_bias'].to(self.device)
            )
            torque_x = ACT(torque_x)
        torques = F.linear(
            torque_x,
            parameters['output_weight'].to(self.device),
            parameters['output_bias'].to(self.device)
        )

        return forces, torques

    def _inner_loop(self, train_loader, train):
        """Computes the adapted network parameters via the MAML inner loop."""

        parameters = {
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }        
        create_graph = True if train else False  # we do not want to create graph of derivatives when evaluating
        loss = nn.MSELoss()
        for _ in range(self._num_inner_steps):
            total_loss = 0.0
            for sample in train_loader:
                states = sample['states'].to(dtype=torch.float32, device=self.device)
                controls = sample['controls'].to(dtype=torch.float32, device=self.device)
                inputs = torch.cat((states, controls), dim=1)
                forces = sample['forces'].to(dtype=torch.float32, device=self.device)
                torques = sample['torques'].to(dtype=torch.float32, device=self.device)

                pred_forces, pred_torques = self._forward(inputs, parameters)
                sample_loss = loss(pred_forces, forces) + loss(pred_torques, torques)
                total_loss += sample_loss
            
            support_loss_i = total_loss / len(train_loader)
            # We calculate the gradients of the support loss w.r.t. the (current) parameters
            gradients = autograd.grad(support_loss_i, parameters.values(), create_graph=create_graph)
            # Then we do simple gradient descent on the parameters using the gradients
            # Note that each parameter has its own learning rate defined in self._inner_lrs
            parameters = {k: v - self._inner_lrs[k] * g for k, v, g in zip(parameters.keys(), parameters.values(), gradients)}

        return parameters

    def _outer_step(self, task_batch, train):
        """Computes the MAML loss and metrics on a batch of tasks."""

        outer_loss_batch = []
        for task_dataset in task_batch:
            n_train = int(0.7 * len(task_dataset))
            n_test = len(task_dataset) - n_train
            train_data, test_data = random_split(task_dataset, [n_train, n_test])
            train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

            parameters_L = self._inner_loop(train_loader, train)  # we get \phi_L and do not need the gradients
            
            loss = nn.MSELoss()
            outer_loss_task = 0.0
            for sample in test_loader:
                states = sample['states'].to(dtype=torch.float32, device=self.device)
                controls = sample['controls'].to(dtype=torch.float32, device=self.device)
                inputs = torch.cat((states, controls), dim=1)
                forces = sample['forces'].to(dtype=torch.float32, device=self.device)
                torques = sample['torques'].to(dtype=torch.float32, device=self.device)
                
                pred_forces, pred_torques = self._forward(inputs, parameters_L)
                sample_loss = loss(pred_forces, forces) + loss(pred_torques, torques)
                outer_loss_task += sample_loss
            outer_loss_task /= len(test_loader)
            outer_loss_batch.append(outer_loss_task)
            
        outer_loss = torch.mean(torch.stack(outer_loss_batch))

        return outer_loss

    def train(self, meta_train_datasets, iterations, batch_size=16):
        """Train the MAML."""

        for i_step in tqdm.tqdm(range(iterations)):
            task_batch = random.sample(meta_train_datasets, batch_size)
            self._optimizer.zero_grad()
            outer_loss = (
                self._outer_step(task_batch, train=True)
            )
            outer_loss.backward()
            self._optimizer.step()

            # Save model
            if i_step % 10 == 0:
                torch.save(self._meta_parameters, f'models/dynamics_model_maml_base.pt')


    def test(self, train_data, test_data):
        """Evaluate the MAML on test tasks."""
        outer_loss = self._outer_step(train_data, train=False)

        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        criterion = nn.MSELoss()
        loss = 0
        with torch.no_grad():
            i = 0
            for test_traj in test_loader:
                i += 1
                states = test_traj['states'].to(dtype=torch.float32, device=self.device)
                controls = test_traj['controls'].to(dtype=torch.float32, device=self.device)
                inputs = torch.cat((states, controls), dim=1)
                forces = test_traj['forces'].to(dtype=torch.float32, device=self.device)
                torques = test_traj['torques'].to(dtype=torch.float32, device=self.device)
                pred_forces, pred_torques = self._forward(inputs, self._meta_parameters)
                loss += criterion(pred_forces, forces) + criterion(pred_torques, torques)
        test_loss = loss / len(test_loader)
        return test_loss


def main(args):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    maml = MAML(
        args.num_inner_steps,
        args.inner_lr,
        args.learn_inner_lrs,
        args.outer_lr,
        device
    )

    n_quads = args.n_quads
    num_train = 8
    num_val = 1
    num_test = n_quads - num_train - num_val

    if not args.test:
        train_quad_ids = list(range(1, num_train + 1))
        task_train_datasets = []
        for train_quad_id in train_quad_ids:
            task_train_data = TrajectoryDataset([train_quad_id])
            task_train_datasets.append(task_train_data)
        task_train_datasets *= args.num_train_iterations
        maml.train(task_train_datasets, args.num_train_iterations, args.batch_size)
    else:
        # Load the pretrained model
        maml._meta_parameters = torch.load('models/dynamics_model_maml_base.pt')
        test_quad_ids = list(range(num_train + num_val + 1, n_quads + 1))
        test_data = TrajectoryDataset(test_quad_ids)
        n_train = int(0.1 * len(test_data))
        n_val = int(0.1 * len(test_data))
        n_test = len(test_data) - n_train - n_val
        train_data, _, test_data = random_split(test_data, [n_train, n_val, n_test])
        test_loss = maml.test(train_data, test_data)
        print(f"MAML model test MSE: {test_loss:.5e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument('--n_quads', type=int, default=10,
                        help='number of quads to use')
    parser.add_argument('--num_inner_steps', type=int, default=2,
                        help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=0.4,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--learn_inner_lrs', default=False, action='store_true',
                        help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='outer-loop learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=100,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')

    args = parser.parse_args()

    main(args)
