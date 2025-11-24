import numpy as np
from typing import Tuple, Optional, Union as _Union

def inject_faults_fast(lambda_arr: np.ndarray,
                       fault_time: _Union[int, float, None] = None,
                       epsilon_scale: float = 0.0,
                       fault_lambda: float = 1.0,  # ← 이제 의미 없음(호환용 파라미터 유지)
                       seed: Optional[int] = None,
                       return_labels: bool = False
                       ) -> _Union[
                            Tuple[np.ndarray, np.ndarray],
                            Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]
                       ]:
    """
    고장 주입기 (fast, **항상 모터 1개 고정**):
      - 입력 1D (M,): (out, type_matrix)
      - 입력 2D (T,M):
          return_labels=False →
              (out, type_matrix)
          return_labels=True  →
              (out, type_matrix, label_TxM, t0, idx, which_fault_mask, onset_idx)

    정의:
      * type_matrix: (N,8) (N=M//8) 링크×모터 라벨(시간축 없음, 0=고장, 1=정상)
      * label_TxM : (T,M) 시간축 라벨(1=정상, 0=고장), t>=t0 & m∈idx → 0
      * t0: 고장 시작 프레임 인덱스(int)
      * idx: 고장 모터의 컬럼 인덱스 배열(shape (1,))  # 항상 1개
      * which_fault_mask: (M,) 모터별 고장 존재 여부(0/1)
      * onset_idx: (M,) 모터별 고장 시작 프레임(없으면 -1)

    정책:
      - 항상 정확히 **1개 모터만** 고장.
      - 고장 이전 구간엔 노이즈를 주지 않음.
      - 건강한 모터에는 전 구간 노이즈 주지 않음.
      - 고장난 모터는 고장 시점 이후에만 '노이즈만' 남기고 베이스 제거.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(lambda_arr)
    eps = 1e-9

    if arr.ndim == 1:
        # ---------------- 1D: 한 시점 ----------------
        M = arr.shape[0]
        if M % 8 != 0:
            raise ValueError("width must be multiple of 8")
        N = M // 8

        # 고장 모터 수 k → 항상 1
        k = 1

        # 고장 모터 인덱스 1개 선택
        idx = rng.choice(M, size=k, replace=False)
        idx = np.sort(idx)

        fault_mask = np.zeros(M, dtype=bool)
        fault_mask[idx] = True

        # (N,8) 타입 라벨
        type_matrix = np.ones((N, 8), dtype=int)
        link_idx, motor_idx = np.divmod(idx, 8)
        type_matrix[link_idx, motor_idx] = 0

        # 출력: 기본적으로 원신호 유지, 고장 모터는 노이즈만 남김
        out = arr.copy()
        ref = np.abs(arr).astype(float)
        ref[ref < eps] = 1.0
        noise = rng.normal(0.0, epsilon_scale, size=M) * ref
        out[fault_mask] = noise[fault_mask]

        return out, type_matrix

    elif arr.ndim == 2:
        # ---------------- 2D: 시계열 ----------------
        T, M = arr.shape
        if M % 8 != 0:
            raise ValueError("width must be multiple of 8")
        N = M // 8

        # 고장 시작 프레임 t0
        if fault_time is None:
            lo = int(T * 0.2)
            hi = max(int(T * 0.8), lo + 1)  # hi는 제외 상한
            t0 = int(rng.integers(low=lo, high=hi))
        elif isinstance(fault_time, float) and 0.0 < fault_time < 1.0:
            t0 = int(T * fault_time)
        else:
            t0 = int(fault_time)
        t0 = max(0, min(T, t0))  # 0 ≤ t0 ≤ T

        # 고장 모터 수 k → 항상 1
        k = 1

        # 고장 모터 인덱스 1개 선택
        idx = rng.choice(M, size=k, replace=False)
        idx = np.sort(idx)

        fault_mask = np.zeros(M, dtype=bool)
        fault_mask[idx] = True

        # (N,8) 타입 라벨(시간축 없음)
        type_matrix = np.ones((N, 8), dtype=int)
        link_idx, motor_idx = np.divmod(idx, 8)
        type_matrix[link_idx, motor_idx] = 0

        # 시간축 라벨 (T,M): 기본 1, t>=t0 & m in idx → 0
        label_TxM = np.ones((T, M), dtype=int)
        if t0 < T:
            label_TxM[t0:, fault_mask] = 0

        # 메타
        which_fault_mask = fault_mask.astype(np.int32)      # (M,) 0/1
        onset_idx = np.full(M, -1, dtype=np.int32)          # (M,)
        if t0 < T:
            onset_idx[fault_mask] = t0

        # 출력: 기본적으로 원신호 유지
        out = arr.copy()

        if t0 < T:
            # t0 이후 고장 모터는 베이스 제거 후 노이즈만
            ref = np.median(np.abs(arr), axis=0).astype(float)
            ref[ref < eps] = 1.0
            noise = rng.normal(0.0, epsilon_scale, size=(T, M)) * ref[None, :]
            out[t0:, fault_mask] = noise[t0:, fault_mask]
        # (고장 전/건강 모터는 out == arr 유지)

        if return_labels:
            return out, type_matrix, label_TxM, int(t0), idx, which_fault_mask, onset_idx
        else:
            return out, type_matrix

    else:
        raise ValueError("lambda_arr must be 1D or 2D")
