# control/selective_mapping.py

import numpy as np

class SelectiveMapping:
    """
    MATLAB selective_mapping_class 대응 Python 버전
    ODAR 객체를 직접 입력으로 받음.
    """
    def __init__(self, odar):
        self.B = np.asarray(odar.B, dtype=float)  # [force; torque]
        self.Bnsv = odar.Bnsv
        self.eps0 = None
        self.eps1 = None

    def get_adjusted_thrust(self, wrench_d: dict) -> np.ndarray:
        """
        wrench_d = {'force': (3,), 'torque': (3,)}
        내부 임시: B^+ w + nullspace shaping (간단 버전)
        """
        w = self.convert_wrench_to_matrix(wrench_d)  # [force; torque]
        lambda_ = np.linalg.pinv(self.B, rcond=1e-9) @ w

        lambda_alpha = self.optimize_in_null_space(lambda_)
        lambda_beta  = self.avoid_dead_zones(lambda_alpha)
        lambda_gamma = self.smooth_set_points(lambda_beta)
        return lambda_gamma

    def optimize_in_null_space(self, lambda_: np.ndarray) -> np.ndarray:
        u = np.asarray(self.Bnsv['upper'], dtype=float).reshape(-1)
        l = np.asarray(self.Bnsv['lower'], dtype=float).reshape(-1)
        upper_prod = lambda_[:4] * u[:4]
        adjust_factor = 0.5 * (np.max(upper_prod) + np.min(upper_prod))
        lambda_ = lambda_ - adjust_factor * u

        lower_prod = lambda_[4:] * u[4:]
        k_lower = 0.5 * (np.max(lower_prod) + np.min(lower_prod))
        lambda_ = lambda_ - k_lower * l
        return lambda_

    def avoid_dead_zones(self, lambda_: np.ndarray) -> np.ndarray:
        # TODO: eps0 사용 보강
        return lambda_

    def smooth_set_points(self, lambda_: np.ndarray) -> np.ndarray:
        # TODO: eps1 사용 보강
        return lambda_

    @staticmethod
    def convert_wrench_to_matrix(wrench: dict) -> np.ndarray:
        # [force; torque]
        return np.concatenate((np.asarray(wrench['force'], dtype=float),
                               np.asarray(wrench['torque'], dtype=float)), axis=0)
