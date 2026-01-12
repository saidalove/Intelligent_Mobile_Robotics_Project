"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""

import numpy as np
import matplotlib.pyplot as plt

class QuinticTrajectoryGenerator:
    """
    Generates a smooth trajectory passing through given waypoints using quintic polynomial interpolation.
    """

    def __init__(self, path_points, total_time=None, dt=0.01):
        """
        path_points: N x 3 numpy array of waypoints [x, y, z]
        total_time: total trajectory duration in seconds (if None, estimate from distance)
        dt: time step for trajectory sampling
        """
        self.path_points = np.array(path_points)
        self.N = len(path_points)
        self.dt = dt

        # estimate total time if not provided
        if total_time is None:
            # use distance-based heuristic: assume 1 m/s average speed
            dist = np.sum(np.linalg.norm(np.diff(self.path_points, axis=0), axis=1))
            self.total_time = max(dist, 1.0)  # at least 1 second
        else:
            self.total_time = total_time

        # time allocation for each segment proportionally to segment length
        segment_lengths = np.linalg.norm(np.diff(self.path_points, axis=0), axis=1)
        self.segment_times = (segment_lengths / np.sum(segment_lengths)) * self.total_time

    def _compute_quintic_coeffs(self, p0, pf, v0=0, vf=0, a0=0, af=0, T=1.0):
        """
        Compute quintic polynomial coefficients for one segment.
        p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        """
        # boundary conditions
        # [p0, pf, v0, vf, a0, af] at t=0 and t=T
        A = np.array([
            [0,      0,       0,    0,    0,   1],
            [T**5,   T**4,   T**3,  T**2,  T,  1],
            [0,      0,       0,    0,    1,  0],
            [5*T**4, 4*T**3, 3*T**2,2*T,  1,  0],
            [0,      0,       0,    2,    0,  0],
            [20*T**3,12*T**2,6*T,   2,    0,  0]
        ])
        b = np.array([p0, pf, v0, vf, a0, af])
        coeffs = np.linalg.solve(A, b)
        return coeffs  # returns [a5, a4, a3, a2, a1, a0]

    def _evaluate_quintic(self, coeffs, t):
        """
        Evaluate quintic polynomial at time t
        """
        a5, a4, a3, a2, a1, a0 = coeffs
        return a5*t**5 + a4*t**4 + a3*t**3 + a2*t**2 + a1*t + a0

    def generate_trajectory(self):
        """
        Generate continuous trajectory for x(t), y(t), z(t)
        Returns: t_samples, x_traj, y_traj, z_traj
        """
        t_samples = [0.0]
        x_traj = [self.path_points[0,0]]
        y_traj = [self.path_points[0,1]]
        z_traj = [self.path_points[0,2]]

        t_current = 0.0

        # iterate over segments
        for i in range(self.N-1):
            p0 = self.path_points[i]
            pf = self.path_points[i+1]
            T = self.segment_times[i]

            # compute quintic coefficients for x, y, z
            coeffs_x = self._compute_quintic_coeffs(p0[0], pf[0], T=T)
            coeffs_y = self._compute_quintic_coeffs(p0[1], pf[1], T=T)
            coeffs_z = self._compute_quintic_coeffs(p0[2], pf[2], T=T)

            num_steps = int(np.ceil(T / self.dt))
            for s in range(1, num_steps+1):
                t = s * self.dt
                if t > T:
                    t = T  # clamp last step

                x_traj.append(self._evaluate_quintic(coeffs_x, t))
                y_traj.append(self._evaluate_quintic(coeffs_y, t))
                z_traj.append(self._evaluate_quintic(coeffs_z, t))

                t_samples.append(t_current + t)

            t_current += T

        return np.array(t_samples), np.array(x_traj), np.array(y_traj), np.array(z_traj)

    def plot_trajectory(self):
        """
        Plot x(t), y(t), z(t) with the path points overlaid.
        """
        t, x, y, z = self.generate_trajectory()

        fig, axs = plt.subplots(3,1,figsize=(8,6), sharex=True)

        axs[0].plot(t, x, label='x(t)')
        axs[0].scatter(np.cumsum([0]+list(self.segment_times)), self.path_points[:,0], color='red', label='waypoints')
        axs[0].set_ylabel('x (m)')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(t, y, label='y(t)')
        axs[1].scatter(np.cumsum([0]+list(self.segment_times)), self.path_points[:,1], color='red', label='waypoints')
        axs[1].set_ylabel('y (m)')
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(t, z, label='z(t)')
        axs[2].scatter(np.cumsum([0]+list(self.segment_times)), self.path_points[:,2], color='red', label='waypoints')
        axs[2].set_xlabel('time (s)')
        axs[2].set_ylabel('z (m)')
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        return fig
