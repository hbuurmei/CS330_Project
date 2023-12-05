import numpy as np
import torch
import argparse
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy

from load_data import TrajectoryDataset
from model import DynamicsModel
from train_model import train


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train', help='train Reptile', action=argparse.BooleanOptionalAction)
parser.add_argument('--plot', help='plot performance', action=argparse.BooleanOptionalAction)
parser.add_argument('--n_quads', type=int, default=10, help='Number of quadrotors to train on')
parser.add_argument('--inner_stepsize', type=float, default=0.02, help='Stepsize in inner SGD')
parser.add_argument('--inner_epochs', type=int, default=1, help='Number of epochs in inner SGD')
parser.add_argument('--outer_stepsize', type=float, default=0.1, help='Stepsize of outer optimization, i.e., meta-optimization')
parser.add_argument('--n_iterations', type=int, default=30000, help='Number of outer updates; each iteration we sample one task and update on it')
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


def totorch(x):
    return ag.Variable(torch.Tensor(x))

def train_on_batch(x, y):
    x = totorch(x)
    y = totorch(y)
    model.zero_grad()
    ypred = model(x)
    loss = (ypred - y).pow(2).mean()
    loss.backward()
    for param in model.parameters():
        param.data -= innerstepsize * param.grad.data

def predict(x):
    x = totorch(x)
    return model(x).data.numpy()

# Choose a fixed task and minibatch for visualization
f_plot = gen_task()
xtrain_plot = x_all[rng.choice(len(x_all), size=ntrain)]

# Reptile training loop
for iteration in range(niterations):
    weights_before = deepcopy(model.state_dict())
    # Generate task
    f = gen_task()
    y_all = f(x_all)
    # Do SGD on this task
    inds = rng.permutation(len(x_all))
    for _ in range(innerepochs):
        for start in range(0, len(x_all), ntrain):
            mbinds = inds[start:start+ntrain]
            train_on_batch(x_all[mbinds], y_all[mbinds])
    # Interpolate between current weights and trained weights from this task
    # I.e. (weights_before - weights_after) is the meta-gradient
    weights_after = model.state_dict()
    outerstepsize = outerstepsize0 * (1 - iteration / niterations) # linear schedule
    model.load_state_dict({name : 
        weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize 
        for name in weights_before})

    # Periodically plot the results on a particular task and minibatch
    if plot and iteration==0 or (iteration+1) % 1000 == 0:
        plt.cla()
        f = f_plot
        weights_before = deepcopy(model.state_dict()) # save snapshot before evaluation
        plt.plot(x_all, predict(x_all), label="pred after 0", color=(0,0,1))
        for inneriter in range(32):
            train_on_batch(xtrain_plot, f(xtrain_plot))
            if (inneriter+1) % 8 == 0:
                frac = (inneriter+1) / 32
                plt.plot(x_all, predict(x_all), label="pred after %i"%(inneriter+1), color=(frac, 0, 1-frac))
        plt.plot(x_all, f(x_all), label="true", color=(0,1,0))
        lossval = np.square(predict(x_all) - f(x_all)).mean()
        plt.plot(xtrain_plot, f(xtrain_plot), "x", label="train", color="k")
        plt.ylim(-4,4)
        plt.legend(loc="lower right")
        plt.pause(0.01)
        model.load_state_dict(weights_before) # restore from snapshot
        print(f"-----------------------------")
        print(f"iteration               {iteration+1}")
        print(f"loss on plotted curve   {lossval:.3f}")