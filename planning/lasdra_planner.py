# lasdra_planner.py

import numpy as np
from scipy.linalg import logm
from math import pi, cos, sin
from project_math.Tinv import Tinv
from project_math.getLogisticSigmoidPosition import getLogisticSigmoidPosition
from planning.closed_loop_inverse_kinematics import ClosedLoopInverseKinematics
from scipy.spatial.transform import Rotation as R

class LasdraPlanner:
    def __init__(self, params):
        self.dof = params['LASDRA']['dof']
        self.q_buffer = []
        self.q_buffer_max = 4
        self.Tcset = [None] * len(params['ODAR'])
        self.Tcset_prev = [None] * len(params['ODAR'])
        self.T_nc_to_ef = Tinv(
            np.block([[np.eye(3), params['ODAR'][-1]['joint_to_com'].reshape(3,1)],
                      [np.zeros((1,3)), 1]])
        ) @ np.block([
            [np.eye(3), (params['ODAR'][-1]['length'] * np.array([[1], [0], [0]]))],
            [np.zeros((1,3)), 1]
        ])
        self.clik = ClosedLoopInverseKinematics(params)

    def set_initial_joint_position(self, q0):
        self.update_joint_position(q0)
        self.update_link_SE3()

    def get_clik_solution(self, time, trajectory_mode='circular'):
        T_d = self.select_end_effector_trajectory(time, trajectory_mode)
        if not self.q_buffer or np.all(self.q_buffer[-1] == 0):
            q_eps = 1e-3 * (-1)**np.arange(1, self.dof + 1)
            q_opt = self.clik.solve(T_d, q_eps)
        else:
            q_opt = self.clik.solve(T_d, self.q_buffer[-1])
        self.update_joint_position(q_opt)
        return q_opt

    def select_end_effector_trajectory(self, time, mode='circular'):
        nlinks = len(self.Tcset)
        R_mat = np.eye(3)
        if mode == 'circular':
            r = 1.2
            t_first_approach = 5
            omega = 0.1
            t_d = 2.5
            if self.time_in_range(time, 0, t_first_approach):
                padding = 5e-3
                p = np.array([nlinks - padding, 0, 0]) + \
                    getLogisticSigmoidPosition(np.array([-1 + padding, r, 0]).reshape(-1, 1),
                                               t_first_approach,
                                               np.array([time]),
                                               t_d).flatten()
            else:
                a = 5
                v = 2 * omega * pi
                delta_t = time - t_first_approach
                if delta_t < t_d:
                    theta = (0.25 * t_d * np.log(np.cosh(a * (2 * delta_t / t_d - 1)) / np.cosh(a)) / a
                             + 0.5 * delta_t) * v
                else:
                    theta = 0.5 * t_d * v + (delta_t - t_d) * v
                p = np.array([nlinks - 1, r * cos(theta), r * sin(theta)])
        elif mode == 'const_straight':
            p = np.array([nlinks, 0, 0])
        elif mode == 'const_swing':
            p = np.array([nlinks - 0.5, 0, 0])
        elif mode == 'debug':
            R_mat = R.from_euler('xzy', [10, (np.random.rand() - 0.5) * 10, 38], degrees=True).as_matrix()
            p = 0.001 * np.array([3021, 1325, 3110])
        else:
            raise ValueError("Unknown trajectory mode")
        T_EF = np.block([[R_mat, p.reshape(3,1)], [np.zeros((1,3)), 1]])
        return T_EF

    def select_joint_trajectory(self, time, mode='dummy'):
        q_log = np.zeros(self.dof)
        if mode == 'straight':
            pass
        elif mode == 'IROS':
            pitch = 5 * pi / 180
            q_log[1] = -pitch
            q_log[3::4] = 2 * pitch
            q_log[5::4] = -2 * pitch
        elif mode == 'sinusoidal':
            pitch = 8 * pi / 180
            theta = 0.25 * pi * time
            q_log[1] = -pitch * sin(theta)
            q_log[3::4] = 2 * pitch * sin(theta)
            q_log[5::4] = -2 * pitch * sin(theta)
        elif mode == 'dummy':
            pitch = 45 * pi / 180
            q_log[1] = -pitch
            q_log[3::4] = 2 * pitch
            q_log[5::4] = -2 * pitch
        else:
            raise ValueError("Unknown trajectory mode")
        self.update_joint_position(q_log)
        return q_log

    def get_joint_velocity(self, q_buffer, dt):
        n = len(q_buffer)
        if n == 1:
            coeff = 0
        elif n == 2:
            coeff = np.array([-1, 1])
        elif n == 3:
            coeff = np.array([0.5, -2, 1.5])
        else:
            coeff = np.array([-1/3, 1.5, -3, 11/6])
        return np.dot(np.stack(q_buffer, axis=1), coeff) / dt

    def get_joint_acceleration(self, q_buffer, dt):
        n = len(q_buffer)
        if n == 1:
            coeff = 0
        elif n == 2:
            coeff = np.array([0, 0])
        elif n == 3:
            coeff = np.array([1, -2, 1])
        else:
            coeff = np.array([-1, 4, -5, 2])
        return np.dot(np.stack(q_buffer, axis=1), coeff) / (dt**2)

    def get_link_twist(self, dt):
        twist = []
        if len(self.q_buffer) < 2:
            for _ in range(len(self.Tcset)):
                twist.append(np.zeros(6))
        else:
            for T_prev, T_now in zip(self.Tcset_prev, self.Tcset):
                R1 = T_prev[:3, :3]
                R2 = T_now[:3, :3]
                angular = self.unskew(logm(R1.T @ R2)) / dt
                linear = (T_now[:3, 3] - T_prev[:3, 3]) / dt
                twist.append(np.concatenate([angular, linear]))
        return twist

    def update_joint_position(self, q):
        if len(self.q_buffer) < self.q_buffer_max:
            self.q_buffer.append(q)
        else:
            self.q_buffer = self.q_buffer[1:] + [q]

    def update_link_SE3(self):
        T_list = self.clik.get_link_GCSE3(self.q_buffer[-1])
        self.Tcset_prev = self.Tcset
        self.Tcset = T_list
        self.Tef = self.Tcset[-1] @ self.T_nc_to_ef
        return self.Tcset

    def get_end_effector_SE3(self):
        return self.Tef

    def unskew(self, R):
        return np.array([R[2,1], R[0,2], R[1,0]])

    def time_in_range(self, time, start, end):
        return start <= time <= end
