# /Users/dnbn/code/transformer_fault_diagnosis/control/impedance_controller.py

import numpy as np

class ImpedanceControllerMaximal:
    def __init__(self, param):
        self.mass = param.mass
        self.inertia_matrix = param.inertia_matrix
        self.Kp = param.position_PID.p
        self.Ki = param.position_PID.i
        self.Ka = param.position_PID.a
        self.Kd = param.position_PID.d
        self.Kr = param.orientation_PID.p
        self.Kri = param.orientation_PID.i
        self.Kra = param.orientation_PID.a
        self.Kw = param.orientation_PID.d
        self.e_x_max = param.position_PID.p_length_max
        self.e_i_max = param.position_PID.i_force_max / np.diag(self.Ki)
        self.e_ri_max = param.orientation_PID.i_torque_max / np.diag(self.Kri)

        self.g = np.array([0, 0, -9.81])
        self.pose_desired = {'linear': np.zeros(3), 'angular': np.eye(3)}
        self.position_desired_buffer = []
        self.position_desired_buffer_max = 4
        self.pose_buffer = {'linear': [], 'angular': []}
        self.pose_buffer_max = 4
        self.twist_desired = {'linear': np.zeros(3), 'angular': np.zeros(3)}
        self.twist_previous = {'linear': np.zeros(3), 'angular': np.zeros(3)}
        self.num_diff_damp_coeff = 0.97
        self.e_i = np.zeros(3)
        self.e_ri = np.zeros(3)

    def set_desired_pose(self, pose_desired):
        self.pose_desired['linear'] = pose_desired[:3, 3]
        self.pose_desired['angular'] = pose_desired[:3, :3]
        if len(self.position_desired_buffer) < self.position_desired_buffer_max:
            self.position_desired_buffer.append(pose_desired[:3, 3])
        else:
            self.position_desired_buffer.pop(0)
            self.position_desired_buffer.append(pose_desired[:3, 3])

    def set_desired_twist(self, twist_desired):
        self.twist_desired['angular'] = twist_desired[:3]
        self.twist_desired['linear'] = twist_desired[3:]

    def set_desired_state(self, pose_desired, twist_desired):
        self.set_desired_pose(pose_desired)
        self.set_desired_twist(twist_desired)

    def get_control_output(self, dt, pose, twist=None):
        xs = pose[:3, 3]
        Rb = pose[:3, :3]
        self.update_se3_pose(pose)

        if twist is None:
            if len(self.pose_buffer['linear']) < 2:
                v_s = np.zeros(3)
                w_b = np.zeros(3)
            else:
                v_s = self.get_linear_velocity(self.pose_buffer['linear'], dt)
                w_b = self.get_body_angular_velocity(self.pose_buffer['angular'][-2], Rb, dt)
                w_b = self.get_damped_value(self.twist_previous['angular'], w_b, self.num_diff_damp_coeff)
        else:
            v_s = twist['linear']
            w_b = twist['angular']

        self.twist_previous['linear'] = v_s
        self.twist_previous['angular'] = w_b

        e_x, e_v = self.compute_position_error(xs, v_s, dt)
        e_r, e_w = self.compute_orientation_error(Rb, w_b, dt)
        wrench = self.get_control_output_using_gain(e_x, self.e_i, e_v, e_r, self.e_ri, e_w, w_b)

        # 중력 보상 + feedforward 가속도
        wrench['force'] -= self.mass * self.g
        wrench['force'] += self.mass * self.get_linear_acceleration(self.position_desired_buffer, dt)
        return wrench

    def compute_position_error(self, position, velocity, dt):
        e_x = position - self.pose_desired['linear']
        e_v = velocity - self.twist_desired['linear']
        self.e_i += e_x * dt
        self.e_i = self.apply_anti_windup(self.e_i, self.e_i_max, -self.e_i_max, self.Ka)
        return e_x, e_v

    def compute_orientation_error(self, orientation, angular_velocity, dt):
        e_r = 0.5 * self.unskew(self.pose_desired['angular'].T @ orientation - orientation.T @ self.pose_desired['angular'])
        e_w = angular_velocity - orientation.T @ self.pose_desired['angular'] @ self.twist_desired['angular']
        self.e_ri += e_r * dt
        self.e_ri = self.apply_anti_windup(self.e_ri, self.e_ri_max, -self.e_ri_max, self.Kra)
        return e_r, e_w

    def get_control_output_using_gain(self, e_x, e_i, e_v, e_r, e_ri, e_w, w_b):
        if np.linalg.norm(e_x) > self.e_x_max:
            e_x = e_x * self.e_x_max / np.linalg.norm(e_x)
        force = -self.Kp @ e_x - self.Ki @ self.bound(e_i, -self.e_i_max, self.e_i_max) - self.Kd @ e_v
        torque = np.cross(w_b, self.inertia_matrix @ w_b) - self.Kr @ e_r - self.Kri @ self.bound(e_ri, -self.e_ri_max, self.e_ri_max) - self.Kw @ e_w
        return {'force': force, 'torque': torque}

    def apply_anti_windup(self, e_i, e_i_max, e_i_min, K_a):
        over_max = np.maximum(e_i - e_i_max, 0)
        e_i -= K_a @ over_max
        under_min = np.minimum(e_i - e_i_min, 0)
        e_i -= K_a @ under_min
        return e_i

    def update_se3_pose(self, T):
        R = T[:3, :3]
        p = T[:3, 3]
        if len(self.pose_buffer['linear']) < self.pose_buffer_max:
            self.pose_buffer['linear'].append(p)
            self.pose_buffer['angular'].append(R)
        else:
            self.pose_buffer['linear'].pop(0)
            self.pose_buffer['linear'].append(p)
            self.pose_buffer['angular'].pop(0)
            self.pose_buffer['angular'].append(R)

    def get_linear_velocity(self, position_buffer, dt):
        n = len(position_buffer)
        if n == 2:
            coeffs = np.array([-1, 1])
        elif n == 3:
            coeffs = np.array([0.5, -2, 1.5])
        else:
            coeffs = np.array([-1/3, 1.5, -3, 11/6])
        pos_mat = np.vstack(position_buffer).T  # shape (3,n)
        return (pos_mat @ coeffs) / dt

    def get_linear_acceleration(self, position_buffer, dt):
        n = len(position_buffer)
        if n == 1:
            return np.zeros(3)
        elif n == 2:
            return np.zeros(3)
        elif n == 3:
            coeffs = np.array([1, -2, 1])
        else:
            coeffs = np.array([-1, 4, -5, 2])
        pos_mat = np.vstack(position_buffer).T  # shape (3,n)
        return (pos_mat @ coeffs) / (dt**2)

    def get_body_angular_velocity(self, R1, R2, dt):
        return 0.5 * self.unskew(R1.T @ R2 - R2.T @ R1) / dt

    @staticmethod
    def unskew(M):
        # 표준 unskew 연산 (skew-symmetric → 벡터)
        return np.array([
            M[2, 1] - M[1, 2],
            M[0, 2] - M[2, 0],
            M[1, 0] - M[0, 1]
        ])

    @staticmethod
    def bound(q, q_min, q_max):
        return np.maximum(q_min, np.minimum(q_max, q))

    @staticmethod
    def get_damped_value(v_prev, v, damping):
        return (1 - damping) * v_prev + damping * v
