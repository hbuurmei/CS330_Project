"""
Script for loading data.
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, quad_ids):
        self.load_trajectories(quad_ids)

    def load_trajectories(self, quad_ids):
        """Load all trajectories from the specified quadrotor IDs."""
        _quad_ids, _times, _states, _controls, _forces, _torques = [], [], [], [], [], []
        for quad_id in quad_ids:
            trajectory = torch.from_numpy(np.loadtxt(f'quad_sim_data/quad{quad_id}/trajectory.csv', delimiter=','))
            quad_ids_tensor = torch.full((trajectory.shape[0],), quad_id, dtype=torch.int64)  # Creating a tensor of quad_id
            _quad_ids.append(quad_ids_tensor)
            _times.append(trajectory[:, 0])
            _states.append(trajectory[:, 1:14])
            _controls.append(trajectory[:, 14:18])
            _forces.append(trajectory[:, 18:21])
            _torques.append(trajectory[:, 21:24])
        self._quad_ids = torch.cat(_quad_ids)
        self._times = torch.cat(_times)
        self._states = torch.cat(_states)
        self._controls = torch.cat(_controls)
        self._forces = torch.cat(_forces)
        self._torques = torch.cat(_torques)

    def __len__(self):
        """Return the number of trajectories in the dataset."""
        return len(self._times)

    def __getitem__(self, idx):
        """Return the idx-th sample in the dataset."""
        sample = {
            'quad_ids': self._quad_ids[idx],
            'times': self._times[idx],
            'states': self._states[idx],
            'controls': self._controls[idx],
            'forces': self._forces[idx],
            'torques': self._torques[idx]
        }
        return sample
