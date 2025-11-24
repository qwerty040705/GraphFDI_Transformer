from copy import deepcopy
from typing import Any, List, Sequence, Union, Optional, Dict
import numpy as np

def _maybe_call(x: Any):
    return x() if callable(x) else x

def _fetch(obj: Any, candidates: Sequence[str], required: bool = True):
    for name in candidates:
        # dict 키
        if isinstance(obj, dict) and name in obj:
            return _maybe_call(obj[name])
        # 속성
        if hasattr(obj, name):
            return _maybe_call(getattr(obj, name))
        # get_ 접두 메서드(예: body_joint_screw_axes -> get_body_joint_screw_axes)
        gname = name if name.startswith("get_") else f"get_{name}"
        if hasattr(obj, gname):
            fn = getattr(obj, gname)
            if callable(fn):
                return fn()
    if required:
        avail = list(obj.keys()) if isinstance(obj, dict) else dir(obj)
        raise AttributeError(f"ODAR entry does not provide any of {candidates}. Available: {avail[:50]}...")
    return None

def _as_list_of_6vec(axes: Union[np.ndarray, List, tuple]) -> List[np.ndarray]:
    if axes is None:
        return []
    if isinstance(axes, np.ndarray):
        a = axes
        if a.ndim == 2:
            if a.shape[0] == 6:  # (6, n)
                return [a[:, i].reshape(6,) for i in range(a.shape[1])]
            if a.shape[1] == 6:  # (n, 6)
                return [a[i, :].reshape(6,) for i in range(a.shape[0])]
        if a.ndim == 1 and a.shape[0] == 6:
            return [a.reshape(6,)]
        raise ValueError(f"Unsupported screw axis ndarray shape: {a.shape}")
    out = []
    for v in axes:
        v = np.asarray(v).reshape(-1)
        if v.size != 6:
            raise ValueError(f"screw axis must be 6D, got {v.size}")
        out.append(v.astype(float))
    return out

def _as_list_of_6x6(inertias: Union[np.ndarray, List, tuple]) -> List[np.ndarray]:
    """다양한 형태의 관성 텐서를 [ (6,6), (6,6), ... ] 리스트로 통일."""
    if inertias is None:
        return []
    if isinstance(inertias, np.ndarray):
        I = inertias
        if I.ndim == 3 and I.shape[1:] == (6, 6):
            return [I[i].astype(float) for i in range(I.shape[0])]
        if I.ndim == 2 and I.shape == (6, 6):
            return [I.astype(float)]
        raise ValueError(f"Unsupported inertia ndarray shape: {I.shape}")
    out = []
    for M in inertias:
        M = np.asarray(M)
        if M.shape == (6, 6):
            out.append(M.astype(float))
        else:
            m = M.reshape(-1)
            if m.size == 36:
                out.append(m.reshape(6, 6).astype(float))
            else:
                raise ValueError(f"inertia must be 6x6, got {M.shape} / size={m.size}")
    return out

def parameters_model(mode: int = 0, params_prev: Optional[Dict] = None, *, quiet: bool = True) -> Dict:
    if params_prev is None:
        raise ValueError("parameters_model(): params_prev must be provided (build it via get_parameters(nlinks)).")

    params = deepcopy(params_prev)

    if 'ODAR' not in params or len(params['ODAR']) == 0:
        raise ValueError("parameters_model(): params['ODAR'] is empty.")

    link_count = int(params['LASDRA'].get('total_link_number', len(params['ODAR'])))
    if link_count < 1:
        raise ValueError("total_link_number must be >= 1")

    if link_count > len(params['ODAR']):
        if not quiet:
            print(f"[WARN] Requested link_count={link_count}, but ODAR has {len(params['ODAR'])}. Using {len(params['ODAR'])}.")
        link_count = len(params['ODAR'])

    params['ODAR'] = list(params['ODAR'])[:link_count]
    params['LASDRA']['total_link_number'] = link_count  # 재기록

    screw_axes_all: List[np.ndarray] = []
    inertia_all: List[np.ndarray] = []

    for i, odar in enumerate(params['ODAR']):
        axes = _fetch(odar, ["body_joint_screw_axes", "joint_screw_axes", "screw_axes"], required=True)
        axes_list = _as_list_of_6vec(axes)
        if len(axes_list) == 0:
            raise ValueError(f"ODAR[{i}] has no screw axes.")
        screw_axes_all.extend(axes_list)

        inertias = _fetch(odar, ["joint_inertia_tensor", "inertia_tensor", "inertia_matrix", "inertia_tensors"], required=False)
        if inertias is not None:
            inertia_list = _as_list_of_6x6(inertias)
            if len(inertia_list) and len(inertia_list) != len(axes_list):
                n = min(len(inertia_list), len(axes_list))
                inertia_all.extend(inertia_list[:n])
            else:
                inertia_all.extend(inertia_list)

    params['LASDRA']['body_joint_screw_axes'] = screw_axes_all  # List[(6,)]
    params['LASDRA']['inertia_matrix'] = inertia_all            # List[(6,6)]
    params['LASDRA']['dof'] = len(screw_axes_all)

    if mode == 0 or mode is None:
        if not quiet:
            print("perfect model parameter is used")
        return params

    nlinks = params['LASDRA']['total_link_number']
    odars = params['ODAR']

    if mode == 1:
        for odar in odars:
            odar.length -= 0.02

    elif mode == 2:
        for odar in odars:
            odar.length += 0.02

    elif mode == 3:
        for odar in odars:
            odar.length -= 0.02
            odar.mass += 0.1

    elif mode == 777:
        if not quiet:
            print("random mass and a bit biased")
        for i in range(nlinks):
            odars[i].mass   += (np.random.rand() - 0.5) * 0.3
            odars[i].length += (np.random.rand() - 0.5) * 0.005 + 0.002
        odars[-1].joint_to_com[0] -= 0.05
        odars[0].joint_to_com[0]  += 0.05

    elif mode == 1107:
        if not quiet:
            print("masses are biased to base")
        if nlinks >= 4:
            odars[-1].joint_to_com[0] -= 0.05
            odars[3].mass -= 0.16
            odars[2].mass += 0.31
            odars[1].mass -= 0.31
            odars[0].joint_to_com[0] += 0.05
            odars[0].mass += 0.43

    else:
        if not quiet:
            print("default param error are used")
        for odar in odars:
            odar.length += 0.02
            odar.mass   += 0.2

    return params
