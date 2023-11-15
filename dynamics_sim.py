import os
import json
import argparse
import tqdm
import numpy as np
import quaternion
import matplotlib.pyplot as plt


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


def quad_actuation(u, quad_params):
    """
    Mapping from control inputs to force and torques for a 3D quadrotor.
    """
    # Unpack values from params
    L = quad_params['L']
    kF = quad_params['kF']
    kM = quad_params['kM']
    phis = quad_params['phis']

    # Control inputs to force is just the sum
    F = np.array([0, 0, kF * np.sum(u * np.cos(np.array(phis)))])

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
        Ixx = np.random.uniform(0.001, 0.005)
        Iyy = np.random.uniform(0.001, 0.005)
        Izz = np.random.uniform(0.001, 0.005)
        measured_params = {'m': m, 'Ixx': Ixx, 'Iyy': Iyy, 'Izz': Izz}
        
        # Sample parameters
        L = np.random.uniform(0.1, 0.2)
        kF = np.random.uniform(0.1, 0.3)
        kM = np.random.uniform(0.05, 0.25)
        phis = [np.random.uniform(-1, 1) * np.pi/180 for _ in range(4)]  # offset angles w.r.t. vertical (for each propeller)
        quad_params = {'L': L, 'kF': kF, 'kM': kM, 'phis': phis}

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
    Kp = 1.0

    # Initialize trajectory (time, states, control inputs)
    N = int(T / dt)+1
    trajectory = np.zeros((N, 18))
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
        # Calculate altitude error correction
        desired_altitude = 10
        current_altitude = trajectory[t_idx, 3]
        error = desired_altitude - current_altitude
        correction = Kp * error

        # Control inputs with stochastic term
        u = u_hover + np.array([correction] * 4) + np.random.uniform(-1, 1, size=4) * u_eps
        trajectory[t_idx, 14:] = u

        # Get force and torques
        F, tau = quad_actuation(u, quad_params)

        # Simulate dynamics
        x_dot = rigid_body_dynamics(trajectory[t_idx, 1:14], F, tau, measured_params, constants)
        trajectory[t_idx+1, 1:14] = trajectory[t_idx, 1:14] + x_dot * dt
        trajectory[t_idx+1, 7:11] = trajectory[t_idx+1, 7:11] / np.linalg.norm(trajectory[t_idx+1, 7:11])  # normalize quaternions

    return trajectory


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_quads', type=int, default=10, help='Number of (random) quadrotors to simulate')
    parser.add_argument('--dt', type=int, default=0.002, help='Time steps for simulation')
    parser.add_argument('--T', type=int, default=10, help='Time to simulate')
    args = parser.parse_args()

    # Generate random quadrotor parameters
    generate_quads(args.n_quads)

    # Define constants
    constants = {'g': 9.81}

    for quad_idx in tqdm.tqdm(range(args.n_quads)):
        # Load parameters
        with open(f'quad_sim_data/quad{quad_idx+1}/measured_params.json', 'r') as f:
            measured_params = json.load(f)
        with open(f'quad_sim_data/quad{quad_idx+1}/quad_params.json', 'r') as f:
            quad_params = json.load(f)
        trajectory = run_quad_sim(measured_params, quad_params, constants, args.dt, args.T)
        np.savetxt(f'quad_sim_data/quad{quad_idx+1}/trajectory.csv', trajectory, delimiter=',')
