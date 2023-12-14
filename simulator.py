import os
import json
import argparse
import tqdm
import numpy as np
import torch
import quaternion
import matplotlib.pyplot as plt

from model import DynamicsModel


# Define models that we evaluate
models = ['scratch', 'ft_head', 'reptile', 'reptile_adam']


def rigid_body_dynamics(x, F, tau, measured_params, constants):
    """
    Rigid-body dynamics.
    """
    # Unpack values from params
    m = measured_params['m']
    Ixx = measured_params['Ixx']
    Iyy = measured_params['Iyy']
    Izz = measured_params['Izz']
    g = constants['g']

    # Extract states
    r = x[0:3]
    v = x[3:6]
    q = np.quaternion(*x[6:10])
    om = x[10:13]

    # Create inertia matrix
    I = np.diag([Ixx, Iyy, Izz])

    # Calculate state derivatives
    r_dot = v
    v_dot = np.array([0, 0, -g]) + quaternion.as_rotation_matrix(q) @ F / m
    q_dot = 0.5 * q * np.quaternion(0, *om)
    q_dot = quaternion.as_float_array(q_dot)
    om_dot = np.linalg.solve(I, tau - np.cross(om, I @ om))

    # Pack state derivatives
    x_dot = np.concatenate([r_dot, v_dot, q_dot, om_dot])
    return x_dot


def integrate_velocity_exp_map(q, om, dt):
    """
    Integrate angular rate to find quaternion using the exponential map.
    """
    # Calculate the quaternion exponential
    omega_magnitude = np.linalg.norm(om)
    if omega_magnitude > 1e-10:  # to avoid division by zero
        # print(omega_magnitude)
        q_exp = [np.cos(0.5 * omega_magnitude * dt), 
                    *(np.sin(0.5 * omega_magnitude * dt) * om / omega_magnitude)]
    else:
        q_exp = [1, 0, 0, 0]  # identity quaternion for small angular velocities
    
    # Update the quaternion
    q = np.quaternion(*q_exp) * np.quaternion(*q)
    q = quaternion.as_float_array(q)

    # Normalize quaternion (theoretically not necessary, but numerically recommended)
    q /=  np.linalg.norm(q)
    return q


def quad_actuation(u, quad_params):
    """
    Mapping from control inputs to force and torques for a 3D quadrotor.
    """
    # Unpack values from params
    L = quad_params['L']
    kF = quad_params['kF']
    kM = quad_params['kM']
    thetas = quad_params['phis']
    phis = quad_params['phis']

    # Control inputs to force is just the sum
    F = np.array([
        np.sum([kF * u * np.sin(thetas[i]) * np.cos(phis[i]) for i in range(len(u))]),
        np.sum([kF * u * np.sin(thetas[i]) * np.sin(phis[i]) for i in range(len(u))]),
        np.sum([kF * u * np.cos(thetas[i]) for i in range(len(u))])
        ])

    # Control inputs to torques
    tau = np.array([
        L * kF * (u[3] * np.cos(phis[3]) - u[1] * np.cos(phis[1])),
        L * kF * (u[2] * np.cos(phis[2]) - u[0] * np.cos(phis[0])),
        kM * (u[0] * np.cos(phis[0]) - u[1] * np.cos(phis[1]) + u[2] * np.cos(phis[2]) - u[3] * np.cos(phis[3]))
        ])

    return F, tau


def generate_quads(n_quads):
    """
    Generate random quadrotor parameters and save as JSON files.
    """
    for quad_idx in range(n_quads):
        # Sample measured parameters
        m = np.random.uniform(0.5, 1.5)
        Ixx = np.random.uniform(1, 5)
        Iyy = np.random.uniform(1, 5)
        Izz = np.random.uniform(1, 5)
        measured_params = {'m': m, 'Ixx': Ixx, 'Iyy': Iyy, 'Izz': Izz}
        
        # Sample parameters
        L = np.random.uniform(0.1, 0.2)
        kF = np.random.uniform(0.1, 0.3)
        kM = np.random.uniform(0.01, 0.05)
        phis = [np.random.uniform(-1, 1) * np.pi for _ in range(4)]  # offset angles w.r.t. vertical (for each propeller)
        thetas = [np.random.uniform(-10, 10) * np.pi/180 for _ in range(4)]  # offset angles in horizontal plane (for each propeller)
        quad_params = {'L': L, 'kF': kF, 'kM': kM, 'thetas': thetas, 'phis': phis}

        # Create directory if it doesn't exist
        if not os.path.exists('quad_sim_data'):
            os.mkdir('quad_sim_data')

        if not os.path.exists(f'quad_sim_data/quad{quad_idx+1}'):
            os.mkdir(f'quad_sim_data/quad{quad_idx+1}')

        # Save parameters as JSON
        with open(f'quad_sim_data/quad{quad_idx+1}/measured_params.json', 'w') as f:
            json.dump(measured_params, f)
        with open(f'quad_sim_data/quad{quad_idx+1}/quad_params.json', 'w') as f:
            json.dump(quad_params, f)


def run_quad_sim(measured_params, quad_params, constants, dt, T):
    """
    Run a simulation of a quadrotor.
    """
    # Proportional constant for altitude control
    Kp_om = 1.0

    # Initialize trajectory (time, states, control inputs)
    N = int(T / dt)+1
    trajectory = np.zeros((N, 24))
    trajectory[:, 0] = np.linspace(0, T, N)

    # Define initial conditions
    r0 = np.array([0, 0, 10])
    v0 = np.array([0, 0, 0])
    q0 = np.quaternion(1, 0, 0, 0)
    om0 = np.array([0, 0, 0])
    x0 = np.concatenate([r0, v0, quaternion.as_float_array(q0), om0])

    # Weight
    w = measured_params['m'] * constants['g']
    u_hover = np.array([w / (4 * quad_params['kF'])] * 4)
    u_eps = 0.05 * u_hover  # 5% of hover force

    # Simulate dynamics
    trajectory[0, 1:14] = x0
    for t_idx in range(N-1):
        # Calculate angular velocity error correction
        desired_om = np.array([0, 0, 0])
        current_om = trajectory[t_idx, 11:14]
        om_error = desired_om - current_om
        om_correction = Kp_om * om_error

        # Control inputs with stochastic term
        u = u_hover + np.random.uniform(-1, 1, size=4) * u_eps
        u[3] = om_correction[0] + u[1]
        u[2] = om_correction[1] + u[0]
        trajectory[t_idx, 14:18] = u

        # Get force and torques
        F, tau = quad_actuation(u, quad_params)
        trajectory[t_idx, 18:21] = F
        trajectory[t_idx, 21:24] = tau

        # Simulate dynamics
        x_dot = rigid_body_dynamics(trajectory[t_idx, 1:14], F, tau, measured_params, constants)
        trajectory[t_idx+1, 1:7] = trajectory[t_idx, 1:7] + x_dot[0:6] * dt  # integrate position and velocity
        trajectory[t_idx+1, 7:11] = integrate_velocity_exp_map(trajectory[t_idx, 7:11], trajectory[t_idx, 11:14], dt)  # integrate quaternion
        trajectory[t_idx+1, 11:14] = trajectory[t_idx, 11:14] + x_dot[10:13] * dt  # integrate angular velocity

    return trajectory


def simulate_model(model, gt_trajectory, measured_params, constants, dt, T):
    # Get initial condition from ground truth trajectory
    x0 = gt_trajectory[0, 1:14]

    # Simulate dynamics using ground truth control inputs
    N = int(T / dt / 10)+1
    trajectory = np.zeros((N, 24))
    trajectory[:, 0] = np.linspace(0, T, N)
    trajectory[0, 1:14] = x0
    for t_idx in range(N-1):
        # Get control inputs from ground truth trajectory
        controls = gt_trajectory[t_idx, 14:18]

        # Get force and torques
        states = torch.from_numpy(trajectory[t_idx, 1:14]).float()
        controls = torch.from_numpy(controls).float()
        inputs = torch.cat([states, controls])
        F, tau = model(inputs)
        F = F.detach().numpy()
        tau = tau.detach().numpy()

        # Simulate dynamics
        x_dot = rigid_body_dynamics(trajectory[t_idx, 1:14], F, tau, measured_params, constants)
        trajectory[t_idx+1, 1:7] = trajectory[t_idx, 1:7] + x_dot[0:6] * dt  # integrate position and velocity
        trajectory[t_idx+1, 7:11] = integrate_velocity_exp_map(trajectory[t_idx, 7:11], trajectory[t_idx, 11:14], dt)  # integrate quaternion
        trajectory[t_idx+1, 11:14] = trajectory[t_idx, 11:14] + x_dot[10:13] * dt  # integrate angular velocity

    return trajectory


def plot_trajectories(gt_trajectory, trajectories):
    """
    Plot the trajectories for the evaluated models.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for model_i, trajectory in enumerate(trajectories):
        # Extract relevant states
        r = trajectory[:, 1:4]
        
        # Plot 3D trajectory
        ax.plot(r[:, 0], r[:, 1], r[:, 2], label=models[model_i])

    # Plot ground truth trajectory
    # First cut length of ground truth to be the same as the generated trajectories
    gt_trajectory = gt_trajectory[:trajectories[0].shape[0], :]
    r_gt = gt_trajectory[:, 1:4]
    ax.plot(r_gt[:, 0], r_gt[:, 1], r_gt[:, 2], label='Ground Truth')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()
    plt.savefig('figures/trajectories.png')


def plot_errors(gt_trajectory, trajectories):
    """
    Plot the errors for the evaluated models.
    """
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8))

    # Plot ground truth trajectory
    # First cut length of ground truth to be the same as the generated trajectories
    gt_trajectory = gt_trajectory[:trajectories[0].shape[0], :]
    r_gt = gt_trajectory[:, 1:4]
    v_gt = gt_trajectory[:, 4:7]
    q_gt = gt_trajectory[:, 7:11]
    om_gt = gt_trajectory[:, 11:14]

    for model_i, trajectory in enumerate(trajectories):
        # Extract relevant states
        r = trajectory[:, 1:4]
        v = trajectory[:, 4:7]
        q = trajectory[:, 7:11]
        om = trajectory[:, 11:14]

        # Plot errors
        axs[0].plot(np.linalg.norm(r - r_gt, axis=1), label=models[model_i])
        axs[1].plot(np.linalg.norm(v - v_gt, axis=1), label=models[model_i])
        axs[2].plot(np.linalg.norm(q - q_gt, axis=1), label=models[model_i])
        axs[3].plot(np.linalg.norm(om - om_gt, axis=1), label=models[model_i])

    axs[0].set_ylabel('Position Error [m]')
    axs[1].set_ylabel('Velocity Error [m/s]')
    axs[2].set_ylabel('Quaternion Error [-]')
    axs[3].set_ylabel('Ang. Velocity Error [rad/s]')
    axs[3].set_xlabel('Timestep')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    plt.tight_layout()
    plt.savefig('figures/trajectory_errors.png')


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_quads', type=int, default=10, help='Number of (random) quadrotors to simulate')
    parser.add_argument('--dt', type=int, default=0.01, help='Time steps for simulation')
    parser.add_argument('--T', type=int, default=10, help='Time to simulate')
    parser.add_argument('--eval', default=False, action='store_true', help='Generate training data or evaluate models')
    args = parser.parse_args()

    # Define constants
    constants = {'g': 9.81}

    if not args.eval:
        # Generate random quadrotor parameters
        generate_quads(args.n_quads)

        for quad_idx in tqdm.tqdm(range(args.n_quads)):
            # Load parameters
            with open(f'quad_sim_data/quad{quad_idx+1}/measured_params.json', 'r') as f:
                measured_params = json.load(f)
            with open(f'quad_sim_data/quad{quad_idx+1}/quad_params.json', 'r') as f:
                quad_params = json.load(f)
            trajectory = run_quad_sim(measured_params, quad_params, constants, args.dt, args.T)
            np.savetxt(f'quad_sim_data/quad{quad_idx+1}/trajectory.csv', trajectory, delimiter=',')
    else:
        test_quad_ids = [8]
        for quad_idx in test_quad_ids:
            # Load measured parameters
            with open(f'quad_sim_data/quad{quad_idx}/measured_params.json', 'r') as f:
                measured_params = json.load(f)
            
            # Load ground truth trajectory
            gt_trajectory = np.loadtxt(f'quad_sim_data/quad{quad_idx}/trajectory.csv', delimiter=',')

            # Load each model to generate trajectories
            trajectories = []
            for model in models:
                dynamics_model = DynamicsModel()
                dynamics_model.load_state_dict(torch.load(f'models/dynamics_model_{model}.pt'))
                trajectory = simulate_model(dynamics_model, gt_trajectory, measured_params, constants, args.dt, args.T)
                trajectories.append(trajectory)
            print("Generated all evaluation trajectories!")

            # Plot trajectories
            plot_trajectories(gt_trajectory, trajectories)

            # Plot state errors
            plot_errors(gt_trajectory, trajectories)
