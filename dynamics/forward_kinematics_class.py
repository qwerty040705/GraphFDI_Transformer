import numpy as np
from project_math.Adinv import Adinv
from project_math.Ad import Ad

class ForwardKinematics:
    def compute_end_effector_analytic_jacobian(self, q):
        T_EF, J_b = self.compute_body_jacobian(q)
        Ad_T = Ad(T_EF)           
        J_a = Ad_T @ J_b          
        return J_a
    
    def __init__(self, param):
        self.ODAR = param['ODAR']
        self.dof = param['LASDRA']['dof']

        Aset = param['LASDRA']['body_joint_screw_axes']
        self.Aset = self._to_6_by_dof(Aset) 

        self.T_joint_to_com_set = []
        self.T_joint_to_gc_set = []
        self.T_joint_to_joint_set = []
        self._init_fixed_transforms(self.ODAR)

        self.Tjset = [np.eye(4) for _ in range(self.dof)]


    @staticmethod
    def _to_6_by_dof(A_cell):
        A = np.asarray(A_cell)
        if A.ndim == 2:
            if A.shape[0] == 6:
                return A
            if A.shape[1] == 6:
                return A.T
        if A.ndim == 1 and (A.size % 6 == 0):
            dof = A.size // 6
            return A.reshape(dof, 6).T
        if isinstance(A_cell, (list, tuple)):
            cols = []
            for a in A_cell:
                arr = np.asarray(a).reshape(-1)
                if arr.size != 6:
                    raise ValueError(f"[FK] Each screw axis must be 6x1; got shape {arr.shape}")
                cols.append(arr)
            return np.stack(cols, axis=1)
        raise ValueError(f"[FK] Unsupported screw axes shape: {A_cell if isinstance(A_cell, (list, tuple)) else A.shape}")


    def _init_fixed_transforms(self, odar_array):
        for odar in odar_array:
            Tjc = np.eye(4)
            Tjc[:3, 3] = odar.joint_to_com.reshape(3,)
            self.T_joint_to_com_set.append(Tjc)

            Tjg = np.eye(4)
            Tjg[:3, 3] = 0.5 * odar.length * np.array([1.0, 0.0, 0.0])
            self.T_joint_to_gc_set.append(Tjg)

            Tjj = np.eye(4)
            Tjj[:3, 3] = odar.length * np.array([1.0, 0.0, 0.0])
            self.T_joint_to_joint_set.append(Tjj)

    @staticmethod
    def _skew(w):
        return np.array([[0, -w[2], w[1]],
                         [w[2], 0, -w[0]],
                         [-w[1], w[0], 0]])

    @classmethod
    def _exp_se3(cls, xi, theta):
        xi = np.asarray(xi).reshape(6,)
        w, v = xi[:3], xi[3:]
        th = float(theta)

        if np.linalg.norm(w) < 1e-12:  # prismatic
            R = np.eye(3)
            p = v * th
        else:
            w_hat = cls._skew(w)
            th2 = th * th
            R = np.eye(3) + np.sin(th) * w_hat + (1 - np.cos(th)) * (w_hat @ w_hat)
            V = np.eye(3) * th + (1 - np.cos(th)) * w_hat + (th - np.sin(th)) * (w_hat @ w_hat)
            p = V @ v

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T


    def compute_inter_joint_transition(self, q):
        q_flat = np.asarray(q).reshape(-1)  # (dof,)
        Tij = [None] * self.dof

        jointcnt = 0
        for iodar, odar in enumerate(self.ODAR):
            nj = len(odar.body_joint_screw_axes)
            for ijoint in range(nj):
                theta = float(q_flat[jointcnt])
                xi = self.Aset[:, jointcnt]  # (6,)
                Texp = self._exp_se3(xi, theta)

                if iodar != 0 and ijoint == 0:
                    Texp = self.T_joint_to_joint_set[iodar - 1] @ Texp

                Tij[jointcnt] = Texp
                jointcnt += 1

        return Tij

    def compute_CoM_frame(self, q):
        Tijset = self.compute_inter_joint_transition(q)

        self.Tjset[0] = Tijset[0]
        for i in range(1, self.dof):
            self.Tjset[i] = self.Tjset[i-1] @ Tijset[i]

        T_list = []
        jointcnt = 0
        for iodar in range(len(self.ODAR)):
            nj = len(self.ODAR[iodar].body_joint_screw_axes)
            last_joint_idx = jointcnt + nj - 1
            T_link_last = self.Tjset[last_joint_idx]
            T_list.append(T_link_last @ self.T_joint_to_com_set[iodar])
            jointcnt += nj
        return T_list

    def compute_GC_frame(self, q):
        Tijset = self.compute_inter_joint_transition(q)

        self.Tjset[0] = Tijset[0]
        for i in range(1, self.dof):
            self.Tjset[i] = self.Tjset[i-1] @ Tijset[i]

        T_list = []
        jointcnt = 0
        for iodar in range(len(self.ODAR)):
            nj = len(self.ODAR[iodar].body_joint_screw_axes)
            last_joint_idx = jointcnt + nj - 1
            T_link_last = self.Tjset[last_joint_idx]
            T_list.append(T_link_last @ self.T_joint_to_gc_set[iodar])
            jointcnt += nj
        return T_list

    def compute_end_effector_frame(self, q):
        Tij = self.compute_inter_joint_transition(q)
        T_EF = np.eye(4)
        for i in range(self.dof):
            T_EF = T_EF @ Tij[i]
        T_EF = T_EF @ self.T_joint_to_joint_set[-1]
        return T_EF


    def compute_body_jacobian(self, q):
        q_flat = np.asarray(q).reshape(-1)
        d = self.dof

        T_ie = self.T_joint_to_joint_set[-1].copy()

        J_b = np.zeros((6, d))
        for i in reversed(range(d)):
            xi = self.Aset[:, i]
            J_b[:, i] = (Adinv(T_ie) @ xi.reshape(6, 1)).reshape(6,)
            T_ie = self._exp_se3(xi, float(q_flat[i])) @ T_ie

        T_EF = T_ie 
        return T_EF, J_b

    def compute_end_effector_body_jacobian(self, q):
        _, J_b = self.compute_body_jacobian(q)
        return J_b
