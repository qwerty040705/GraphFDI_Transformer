from __future__ import annotations
import argparse, csv, glob, math, os, time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm  # 진행바 라이브러리

# data_generate_new.py와 동일한 모델 파라미터 구성을 위해 import
from parameters import get_parameters
from parameters_model import parameters_model

# ------------------------- CUDA-friendly defaults -------------------------

def setup_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_arg)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print(f"Using CUDA: {name} | capability={cap}")
        except Exception:
            print("Using CUDA")
    return device

# ------------------------- Small numeric helpers -------------------------

def so3_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    tr = np.clip(tr, -1.0, 3.0)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = math.acos(cos_theta)

    if theta < 1e-8:
        w = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) * 0.5
        return w.astype(np.float64)

    if abs(math.pi - theta) < 1e-4:
        A = (R + np.eye(3)) / 2.0
        axis = np.array([
            math.sqrt(max(A[0,0], 0.0)),
            math.sqrt(max(A[1,1], 0.0)),
            math.sqrt(max(A[2,2], 0.0)),
        ], dtype=np.float64)
        axis[0] = math.copysign(axis[0], R[2,1] - R[1,2])
        axis[1] = math.copysign(axis[1], R[0,2] - R[2,0])
        axis[2] = math.copysign(axis[2], R[1,0] - R[0,1])
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis /= n
        return (axis * theta).astype(np.float64)

    s = max(math.sin(theta), 1e-8)
    wvee = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], dtype=np.float64)
    w = wvee * (0.5 * theta / s)
    return w

# ------------------------- Window labeling utilities -------------------------

@dataclass
class WindowLabels:
    onset: int
    link_class: int
    motor_class: int

# ------------------------- Dataset with Frame Cache -------------------------

class MotorGeom:
    def __init__(self, path: Optional[str], motors_per_link: int = 8):
        self.available = False
        self.mpl = motors_per_link
        self.links: Optional[int] = None

        if path is None or not os.path.exists(path):
            if path is not None:
                print(f"[WARN] motor_geom not found at '{path}'. Proceeding without motor nodes.")
            return

        data = np.load(path)
        self.rotvec = np.asarray(data['rotvec_m2l'])
        self.r_mL   = np.asarray(data['r_mL'])
        self.f_dir  = np.asarray(data['f_dir'])
        self.arm    = np.asarray(data['arm'])

        L = int(self.rotvec.shape[0])
        self.links = L
        self.available = True
        print(f"[INFO] motor_geom loaded: {path} (L={L}, motor nodes enabled)")

class LASDRAFDIDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 shard_ids: List[int],
                 T_win: int = 128,
                 stride: int = 64,
                 motors_per_link: int = 8,
                 normalize: bool = True,
                 motor_geom: Optional[MotorGeom] = None,
                 cache_windows: bool = False,
                 cache_dir: str = "./cache_frames",
                 cache_dtype: str = "fp32"):
        super().__init__()
        self.data_dir = data_dir
        self.T_win = T_win
        self.stride = stride
        self.mpl = motors_per_link
        self.normalize = normalize
        self.geom = motor_geom if (motor_geom and motor_geom.available) else None
        self.cache_windows = cache_windows
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        assert cache_dtype in ("fp32","fp16")
        self.cache_dtype = np.float32 if cache_dtype=="fp32" else np.float16

        self.samples: List[Dict[str, np.ndarray]] = []
        for sid in shard_ids:
            path = os.path.join(data_dir, f"fault_dataset_shard_{sid:05d}.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            data = np.load(path)
            pack = {
                'sid'             : sid,
                'desired_link_cum': data['desired_link_cum'],
                'actual_link_cum' : data['actual_link_cum'],
                'label'           : data['label'],
                'onset_idx'       : data['onset_idx'],
                'dt'              : float(data['dt']),
                'link_wrench'     : data['link_wrench'],
                'fault_mask'      : data['fault_mask'],
                'link_count'      : int(data['link_count']),
            }
            pack['S'] = pack['desired_link_cum'].shape[0]
            pack['T'] = pack['desired_link_cum'].shape[1]
            self.samples.append(pack)

        if len(self.samples) == 0:
            raise RuntimeError("No shard loaded.")

        self.L = int(self.samples[0]['link_count'])
        print(f"[INFO] Detected link_count={self.L} from shard.")

        # 1. B_links 생성
        self.B_links = self._build_B_links(self.L)

        # 2. [최적화] 벡터화를 위한 정적 데이터 미리 계산
        if self.geom is not None:
            norms = np.linalg.norm(self.B_links, axis=1, keepdims=True)
            self.B_links_norm = self.B_links / (norms + 1e-8)

            L = self.L
            mpl = self.mpl
            N = L + L * mpl
            self.static_edge_template = np.zeros((N, N, 10), dtype=np.float32)
            
            for li in range(L):
                for mj in range(mpl):
                    m_global = L + li*mpl + mj
                    rv  = self.geom.rotvec[li, mj]
                    rml = self.geom.r_mL[li, mj]
                    fdr = self.geom.f_dir[li, mj]
                    arm = self.geom.arm[li, mj]
                    
                    em = np.zeros((10,), dtype=np.float32)
                    em[0:3] = rv; em[3:6] = rml; em[6:9] = fdr; em[9] = arm
                    
                    self.static_edge_template[li, m_global, :] = em
                    self.static_edge_template[m_global, li, :] = em
        else:
            self.B_links_norm = None
            self.static_edge_template = None

        # 3. Frame Cache 로드
        self.caches: List[Dict[str, np.ndarray]] = []
        for pack in self.samples:
            cache = self._ensure_frame_cache(pack)
            self.caches.append(cache)

        # 4. Window Index
        self.index: List[Tuple[int, int, int]] = []
        for sh_idx, pack in enumerate(self.samples):
            S, T = pack['S'], pack['T']
            for s in range(S):
                t0 = 0
                while t0 + self.T_win <= T:
                    self.index.append((sh_idx, s, t0))
                    t0 += self.stride

        # 5. Norm Stats
        self._mu = None
        self._std = None
        if self.normalize:
            self._fit_norm_stats(max_windows=4000)

    def _build_B_links(self, link_count: int) -> np.ndarray:
        base_param = get_parameters(link_count)
        base_param['ODAR'] = base_param['ODAR'][:link_count]
        screw_axes, inertias = [], []
        for odar in base_param['ODAR']:
            screw_axes.extend(odar.body_joint_screw_axes)
            inertias.extend(odar.joint_inertia_tensor)
        base_param['LASDRA'].update(
            body_joint_screw_axes=screw_axes,
            inertia_matrix=inertias,
            dof=len(screw_axes)
        )
        model_param = parameters_model(mode=0, params_prev=base_param)
        eye_perm = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [np.eye(3), np.zeros((3, 3))]
        ])
        B_links = []
        for odar in model_param['ODAR']:
            B_raw = np.asarray(odar.B, dtype=float)
            B_perm = eye_perm @ B_raw
            B_links.append(B_perm.astype(np.float32))
        B_links = np.stack(B_links, axis=0)
        return B_links

    def _cache_path(self, sid: int) -> str:
        return os.path.join(self.cache_dir, f"frames_{sid:05d}.npz")

    def _ensure_frame_cache(self, pack: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        sid = pack['sid']
        path = self._cache_path(sid)
        if (not self.cache_windows) and (not os.path.exists(path)):
            print(f"[INFO] frame cache disabled; will compute on-the-fly for shard {sid}")
            return {'node_link': None, 'edge_link': None}

        if os.path.exists(path):
            z = np.load(path, mmap_mode='r')
            return {'node_link': z['node_link'], 'edge_link': z['edge_link']}

        print(f"[CACHE] Building frame cache for shard {sid} ...")
        S, T, L = pack['S'], pack['T'], self.L
        node_link = np.zeros((S, T, L, 9), dtype=self.cache_dtype)
        num_dir_edges = max(0, 2 * (L - 1))
        edge_link = np.zeros((S, T, num_dir_edges, 6), dtype=self.cache_dtype)

        d_cum = pack['desired_link_cum']
        a_cum = pack['actual_link_cum']

        for s in range(S):
            for t in range(T):
                for k in range(L):
                    R_des = d_cum[s,t,k,:3,:3]; R_act = a_cum[s,t,k,:3,:3]
                    p_des = d_cum[s,t,k,:3,3];  p_act = a_cum[s,t,k,:3,3]
                    R_err = R_des.T @ R_act
                    node_link[s,t,k,0:3] = so3_log(R_err)
                    node_link[s,t,k,3:6] = (p_act - p_des)

                e_idx = 0
                for i in range(L - 1):
                    j = i + 1
                    Rij = a_cum[s,t,i,:3,:3].T @ a_cum[s,t,j,:3,:3]
                    edge_link[s,t,e_idx,0:3] = so3_log(Rij)
                    edge_link[s,t,e_idx,3:6] = a_cum[s,t,j,:3,3] - a_cum[s,t,i,:3,3]
                    e_idx += 1
                    Rji = a_cum[s,t,j,:3,:3].T @ a_cum[s,t,i,:3,:3]
                    edge_link[s,t,e_idx,0:3] = so3_log(Rji)
                    edge_link[s,t,e_idx,3:6] = a_cum[s,t,i,:3,3] - a_cum[s,t,j,:3,3]
                    e_idx += 1
        
        node_link = np.nan_to_num(node_link, nan=0.0, posinf=1e6, neginf=-1e6)
        edge_link = np.nan_to_num(edge_link, nan=0.0, posinf=1e6, neginf=-1e6)
        np.savez_compressed(path, node_link=node_link.astype(self.cache_dtype),
                            edge_link=edge_link.astype(self.cache_dtype))
        print(f"[CACHE] Saved: {path}")
        z = np.load(path, mmap_mode='r')
        return {'node_link': z['node_link'], 'edge_link': z['edge_link']}

    def _fit_norm_stats(self, max_windows: int = 4000):
        feats = []
        count = 0
        for (sh_idx, s, t0) in self.index:
            nl = self.caches[sh_idx]['node_link']
            if nl is not None:
                node_link = nl[s, t0:t0+self.T_win]
            else:
                node_link = self._compute_node_link_onfly(sh_idx, s, t0)
            node_link = np.nan_to_num(node_link, nan=0.0, posinf=1e6, neginf=-1e6)
            feats.append(node_link.astype(np.float32))
            count += 1
            if count >= max_windows:
                break
        X = np.concatenate(feats, axis=0)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        mu = X.mean(axis=(0,1))
        std = X.std(axis=(0,1))
        std = np.where(std < 1e-6, 1.0, std)
        self._mu = mu.astype(np.float32)
        self._std = std.astype(np.float32)

    def _compute_node_link_onfly(self, sh_idx: int, s: int, t0: int):
        pack = self.samples[sh_idx]
        T = self.T_win; L = self.L
        node_link = np.zeros((T, L, 9), dtype=np.float32)
        d_cum = pack['desired_link_cum'][s, t0:t0+T]
        a_cum = pack['actual_link_cum' ][s, t0:t0+T]
        for t in range(T):
            for k in range(L):
                R_des = d_cum[t, k, :3, :3]
                R_act = a_cum[t, k, :3, :3]
                p_des = d_cum[t, k, :3, 3]
                p_act = a_cum[t, k, :3, 3]
                R_err = R_des.T @ R_act
                node_link[t, k, 0:3] = so3_log(R_err)
                node_link[t, k, 3:6] = (p_act - p_des)
        return np.nan_to_num(node_link, nan=0.0, posinf=1e6, neginf=-1e6)

    def __len__(self) -> int:
        return len(self.index)

    def _extract_window(self, sh_idx: int, s: int, t0: int):
        L, T = self.L, self.T_win

        # 1) node_link from cache
        nl = self.caches[sh_idx]['node_link']
        if nl is not None:
            node_link = nl[s, t0:t0+T].astype(np.float32)
        else:
            node_link = self._compute_node_link_onfly(sh_idx, s, t0)

        # 2) dW inside window (vectorized)
        dW = np.zeros((T, L, 3), dtype=np.float32)
        dW[1:-1] = 0.5 * (node_link[2:, :, 0:3] - node_link[:-2, :, 0:3])
        dW[0]    = node_link[1, :, 0:3] - node_link[0, :, 0:3]
        dW[-1]   = node_link[-1, :, 0:3] - node_link[-2, :, 0:3]
        node_link[:, :, 6:9] = dW

        # 2-1) [최적화] link_wrench 기반 motor similarity (Einsum 사용)
        motor_sim = None
        if self.geom is not None and 'link_wrench' in self.samples[sh_idx]:
            w_win = self.samples[sh_idx]['link_wrench'][s, t0:t0+T].astype(np.float32)
            w_norm_val = np.linalg.norm(w_win, axis=-1, keepdims=True)
            w_n = w_win / (w_norm_val + 1e-8)
            # Similarity calculation: w_n(T,L,6) @ B_norm(L,6,8) -> (T,L,8)
            motor_sim = np.einsum('tlc,lcm->tlm', w_n, self.B_links_norm)

        # 3) Construct node_feat
        Dn = 10
        if self.geom is None:
            N = L
            node_feat = np.zeros((T, N, Dn), dtype=np.float32)
            node_feat[:, :L, 0:9] = node_link
        else:
            N = L + L * self.mpl
            node_feat = np.zeros((T, N, Dn), dtype=np.float32)
            node_feat[:, :L, 0:9] = node_link
            if motor_sim is not None:
                node_feat[:, L:, 9] = motor_sim.reshape(T, -1)

        # 4) Edge features
        De = 16
        edge_feat = np.zeros((T, N, N, De), dtype=np.float32)

        # Dynamic link-link (from cache)
        el = self.caches[sh_idx]['edge_link']
        if el is not None:
            ed = el[s, t0:t0+T].astype(np.float32)
            e_idx = 0
            for i in range(L - 1):
                j = i + 1
                edge_feat[:, i, j, 0:6] = ed[:, e_idx]; e_idx += 1
                edge_feat[:, j, i, 0:6] = ed[:, e_idx]; e_idx += 1

        # [최적화] Static link-motor edges (Template Broadcasting)
        if self.static_edge_template is not None:
            edge_feat[:, :, :, 6:16] = self.static_edge_template[np.newaxis, :, :, :]

        # 5) Attention mask
        attn_mask = np.zeros((N, N), dtype=bool)
        np.fill_diagonal(attn_mask, True)
        for i in range(L - 1):
            attn_mask[i, i+1] = True
            attn_mask[i+1, i] = True
        if self.geom is not None:
             for li in range(L):
                m_start = L + li * self.mpl
                m_end   = L + (li + 1) * self.mpl
                attn_mask[li, m_start:m_end] = True
                attn_mask[m_start:m_end, li] = True

        # 6) Labels
        label_full = self.samples[sh_idx]['label'][s]
        t_cur = t0 + T - 1
        cur_labels = label_full[t_cur]
        faulty_mask = (cur_labels == 0)
        onset = 1 if np.any(faulty_mask) else 0

        if onset:
            fm_reshaped = faulty_mask.reshape(L, self.mpl)
            per_link_fault = fm_reshaped.any(axis=1)
            link_idx = int(np.argmax(per_link_fault))
            link_class = link_idx + 1
            faulty_motors = np.where(fm_reshaped[link_idx])[0]
            motor_local = int(faulty_motors[0]) if faulty_motors.size > 0 else 0
        else:
            link_class = 0
            motor_local = 0

        node_feat = np.nan_to_num(node_feat, nan=0.0, posinf=1e6, neginf=-1e6)
        edge_feat = np.nan_to_num(edge_feat, nan=0.0, posinf=1e6, neginf=-1e6)

        labels = WindowLabels(onset=onset, link_class=link_class, motor_class=motor_local)
        return node_feat, edge_feat, attn_mask, labels

    def __getitem__(self, idx: int):
        sh_idx, s, t0 = self.index[idx]
        node_feat, edge_feat, attn_mask, labels = self._extract_window(sh_idx, s, t0)

        if self._mu is not None:
            L = self.L
            node_feat[:, :L, 0:9] = (node_feat[:, :L, 0:9] - self._mu) / self._std

        node_feat = np.nan_to_num(node_feat, nan=0.0, posinf=1e6, neginf=-1e6)
        edge_feat = np.nan_to_num(edge_feat, nan=0.0, posinf=1e6, neginf=-1e6)

        x = torch.from_numpy(node_feat).float()
        e = torch.from_numpy(edge_feat).float()
        m = torch.from_numpy(attn_mask).bool()
        y_onset = torch.tensor(labels.onset, dtype=torch.float32)
        y_link  = torch.tensor(labels.link_class, dtype=torch.long)
        y_motor = torch.tensor(labels.motor_class, dtype=torch.long)
        return x, e, m, y_onset, y_link, y_motor

# ------------------------- Model: Edge-Biased Graph Transformer -------------------------

class EdgeEncoder(nn.Module):
    def __init__(self, edge_dim: int, heads: int, hidden: int = 64, clamp: float = 5.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, heads)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.clamp = clamp

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        out = self.net(E)                 # (B,N,N,H)
        out = out.permute(0,3,1,2)        # (B,H,N,N)
        return out.clamp(-self.clamp, self.clamp)

class EdgeBiasedMHA(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads
        self.dk = d_model // heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, X: torch.Tensor, B_ij: torch.Tensor, mask: Optional[torch.Tensor]):
        B, N, D = X.shape
        H, dk = self.h, self.dk
        q = self.q(X).view(B,N,H,dk).transpose(1,2)
        k = self.k(X).view(B,N,H,dk).transpose(1,2)
        v = self.v(X).view(B,N,H,dk).transpose(1,2)
        scores = (q @ k.transpose(-1,-2)) / math.sqrt(dk)  # (B,H,N,N)
        scores = scores + B_ij
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1,2).contiguous().view(B,N,D)
        return self.o(out)

class GraphTFBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, edge_dim: int, dropout: float):
        super().__init__()
        self.edge_enc = EdgeEncoder(edge_dim, heads)
        self.attn = EdgeBiasedMHA(d_model, heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, E: torch.Tensor, mask: Optional[torch.Tensor]):
        B_ij = self.edge_enc(E)          # (B,H,N,N)
        Y = self.attn(X, B_ij, mask)
        X = self.norm1(X + self.drop(Y))
        Y = self.ff(X)
        X = self.norm2(X + self.drop(Y))
        return X

class SpatialGraphTF(nn.Module):
    def __init__(self, d_in: int, d_model: int, heads: int, edge_dim: int, depth: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([
            GraphTFBlock(d_model, heads, edge_dim, dropout) for _ in range(depth)
        ])

    def forward(self, X: torch.Tensor, E: torch.Tensor, mask: Optional[torch.Tensor]):
        # X: (B,T,N,D_in), E: (B,T,N,N,De), mask: (B,1,N,N)
        B, T, N, D_in = X.shape
        De = E.size(-1)

        H = self.inp(X)  # (B,T,N,d_model)
        H_flat = H.view(B * T, N, -1)                # (B*T,N,D)
        E_flat = E.view(B * T, N, N, De)             # (B*T,N,N,De)

        if mask is not None:
            mask_bt = mask.unsqueeze(1).expand(B, T, 1, N, N)
            M_flat = mask_bt.contiguous().view(B * T, 1, N, N)
        else:
            M_flat = None

        for blk in self.blocks:
            H_flat = blk(H_flat, E_flat, M_flat)     # (B*T,N,D)

        H = H_flat.view(B, T, N, -1)
        return H

# ------------------------- Temporal encoders -------------------------

class TemporalConv(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 256, d_out: int = 256, kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        pad = kernel - 1
        self.net = nn.Sequential(
            nn.Conv1d(d_in, d_hidden, kernel_size=kernel, padding=pad),
            nn.GELU(),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=kernel, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(d_hidden, d_out)
        self.pad = pad

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B, T, N, D = H.shape
        H = H.permute(0,2,1,3).contiguous().view(B*N, T, D).transpose(1,2)
        Y = self.net(H)[:, :, :-self.pad]  
        Y = Y.mean(dim=-1)
        Y = self.proj(Y)
        return Y.view(B, N, -1)

class TemporalTransformer(nn.Module):
    def __init__(self, d_in: int, n_layers: int = 3, n_heads: int = 6, d_ff: int = 1024, dropout: float = 0.1, pool: str = 'last'):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_in, nhead=n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = pool

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        m = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return m  

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B, T, N, D = H.shape
        H = H.permute(0,2,1,3).contiguous().view(B*N, T, D)
        mask = self._causal_mask(T, H.device)
        Z = self.enc(H, mask=mask)
        pooled = Z[:, -1, :]  
        return pooled.view(B, N, D)

# ------------------------- Heads -------------------------

class Heads(nn.Module):
    def __init__(self, d_repr: int, links: int = 3, motors_per_link: int = 8):
        super().__init__()
        self.links = links
        self.l1 = nn.Sequential(
            nn.Linear(d_repr*links, d_repr),
            nn.GELU(),
            nn.Linear(d_repr, 1)
        )
        self.l2 = nn.Sequential(
            nn.Linear(d_repr*links, d_repr),
            nn.GELU(),
            nn.Linear(d_repr, links+1)
        )
        self.m_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_repr, d_repr//2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_repr//2, motors_per_link)
            )
            for _ in range(links)
        ])

    def forward(self, Z: torch.Tensor):
        B, N, D = Z.shape
        L = self.links
        flat = Z[:, :L, :].reshape(B, L*D)
        onset_logit = self.l1(flat).squeeze(-1)
        link_logits = self.l2(flat)
        motor_logits = [head(Z[:, li, :]) for li, head in enumerate(self.m_heads)]
        return onset_logit, link_logits, motor_logits

class GraphFDIModel(nn.Module):
    def __init__(self, d_node_in: int, d_edge_in: int, d_model: int = 256, heads: int = 4, depth: int = 3,
                 temporal: str = 'transformer', d_temporal: int = 256, links: int = 3,
                 motors_per_link: int = 8, dropout: float = 0.1):
        super().__init__()
        self.spatial = SpatialGraphTF(d_node_in, d_model, heads, d_edge_in, depth, dropout)
        if temporal == 'transformer':
            self.temporal = TemporalTransformer(d_in=d_model, n_layers=3, n_heads=heads,
                                                d_ff=4*d_model, dropout=dropout, pool='last')
            d_repr = d_model
        else:
            self.temporal = TemporalConv(d_in=d_model, d_hidden=d_model, d_out=d_temporal,
                                         kernel=5, dropout=dropout)
            d_repr = d_temporal
        self.heads = Heads(d_repr, links=links, motors_per_link=motors_per_link)

    def forward(self, X: torch.Tensor, E: torch.Tensor, mask: Optional[torch.Tensor]):
        H = self.spatial(X, E, mask)
        Z = self.temporal(H)
        return self.heads(Z)

# ------------------------- Post-processing (K-of-N with one-way latch) -------------------------

def kofn_hysteresis(probs: np.ndarray, K: int = 3, N: int = 5, up: float = 0.6, down: float = 0.4) -> np.ndarray:
    T = len(probs)
    buf = np.zeros(N, dtype=np.float32)
    out = np.zeros(T, dtype=np.int32)
    latched = 0

    for t in range(T):
        if latched == 0:
            buf[t % N] = 1.0 if probs[t] >= up else 0.0
            vote = int(buf.sum() >= K)
            if vote == 1:
                latched = 1
        out[t] = latched
    return out

# ------------------------- Training / Eval -------------------------

def collate_fn(batch):
    Xs, Es, Ms, y1s, y2s, y3s = zip(*batch)
    X = torch.stack(Xs, dim=0).float()
    E = torch.stack(Es, dim=0).float()
    M = torch.stack(Ms, dim=0).unsqueeze(1).bool()
    y1 = torch.stack(y1s, dim=0).float()
    y2 = torch.stack(y2s, dim=0).long()
    y3 = torch.stack(y3s, dim=0).long()
    return X, E, M, y1, y2, y3

@dataclass
class TrainConfig:
    data_dir: str
    shard_first: int
    shard_last: int
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = 'auto'
    num_workers: int = 4
    seed: int = 42
    T_win: int = 128
    stride: int = 64
    d_model: int = 256
    heads: int = 4
    depth: int = 3
    dropout: float = 0.1
    temporal: str = 'transformer'
    motor_geom: Optional[str] = None
    apply_post: bool = False
    post_k: int = 3
    post_n: int = 5
    post_up: float = 0.6
    post_down: float = 0.4
    export_onset_series: int = 0
    log_csv: str = 'train_log.csv'
    eval_test_each_epoch: int = 1
    save_every: int = 1
    cache_windows: int = 0
    cache_dir: str = "./cache_frames"
    cache_dtype: str = "fp32"
    amp: int = 1
    compile: int = 0
    lambda_l1: float = 1.0
    lambda_l2: float = 1.0
    lambda_l3: float = 1.0
    motor_freeze_epochs: int = 6
    label_smoothing: float = 0.05

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.device = setup_device(cfg.device)

        shard_ids = list(range(cfg.shard_first, cfg.shard_last+1))
        geom = MotorGeom(cfg.motor_geom) if cfg.motor_geom else None
        full = LASDRAFDIDataset(cfg.data_dir, shard_ids, T_win=cfg.T_win, stride=cfg.stride,
                                motors_per_link=8, normalize=True, motor_geom=geom,
                                cache_windows=bool(cfg.cache_windows),
                                cache_dir=cfg.cache_dir, cache_dtype=cfg.cache_dtype)
        n = len(full)

        # ----------------- 체크포인트 저장 경로 설정 -----------------
        self.links = full.L

        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.save_dir_all = os.path.join(base_dir, f"Link{self.links}_All")
        os.makedirs(self.save_dir_all, exist_ok=True)

        self.save_dir_last = os.path.join(base_dir, f"GraphFDI_Transformer_L{self.links}")
        os.makedirs(self.save_dir_last, exist_ok=True)
        # -----------------------------------------------------------

        idx = np.arange(n); np.random.shuffle(idx)
        n_train = int(0.8*n); n_val = int(0.1*n)

        train_indices = idx[:n_train].tolist()
        val_indices   = idx[n_train:n_train+n_val].tolist()
        test_indices  = idx[n_train+n_val:].tolist()

        self.train_ds = torch.utils.data.Subset(full, train_indices)
        self.val_ds   = torch.utils.data.Subset(full, val_indices)
        self.test_ds  = torch.utils.data.Subset(full, test_indices)

        print("[INFO] Precomputing train sample weights (Optimized)...")
        train_weights: List[float] = []
        T_win = full.T_win
        
        for ds_idx in train_indices:
            sh_idx, s, t0 = full.index[ds_idx]
            pack = full.samples[sh_idx]
            
            label_full = pack["label"][s]
            t_cur = t0 + T_win - 1
            if t_cur >= len(label_full): t_cur = len(label_full) - 1
            cur_labels = label_full[t_cur]
            
            faulty = (cur_labels == 0).any()
            onset = 1 if faulty else 0
            w = 1.0 + 3.0 * onset
            train_weights.append(w)

        train_sampler = WeightedRandomSampler(weights=train_weights,
                                              num_samples=len(train_indices),
                                              replacement=True)

        pin = (self.device.type == 'cuda')
        pers = (cfg.num_workers > 0)
        
        self.train_loader = DataLoader(self.train_ds, batch_size=cfg.batch_size,
                                       sampler=train_sampler,
                                       num_workers=cfg.num_workers, collate_fn=collate_fn,
                                       pin_memory=pin, persistent_workers=pers,
                                       prefetch_factor=2 if pers else None)
        self.val_loader   = DataLoader(self.val_ds, batch_size=cfg.batch_size, shuffle=False,
                                       num_workers=cfg.num_workers, collate_fn=collate_fn,
                                       pin_memory=pin, persistent_workers=pers,
                                       prefetch_factor=2 if pers else None)
        self.test_loader  = DataLoader(self.test_ds, batch_size=cfg.batch_size, shuffle=False,
                                       num_workers=cfg.num_workers, collate_fn=collate_fn,
                                       pin_memory=pin, persistent_workers=pers,
                                       prefetch_factor=2 if pers else None)

        d_node_in = 10
        d_edge_in = 16
        self.model = GraphFDIModel(d_node_in=d_node_in, d_edge_in=d_edge_in, d_model=cfg.d_model,
                                   heads=cfg.heads, depth=cfg.depth, temporal=cfg.temporal,
                                   d_temporal=cfg.d_model, links=full.L, motors_per_link=8,
                                   dropout=cfg.dropout).to(self.device)
        if cfg.compile:
            try:
                self.model = torch.compile(self.model)
                print("[INFO] torch.compile enabled")
            except Exception as e:
                print(f"[WARN] torch.compile failed: {e}")

        self.log_vars = nn.Parameter(torch.zeros(3, device=self.device))

        self.opt = torch.optim.AdamW(
            list(self.model.parameters()) + [self.log_vars],
            lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max(cfg.epochs, 1))

        self.bce = nn.BCEWithLogitsLoss()
        self.ce  = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        self.use_amp = (self.device.type == 'cuda') and bool(cfg.amp)
        try:
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
            self.autocast_ctor = lambda: torch.amp.autocast('cuda', enabled=self.use_amp)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
            self.autocast_ctor = lambda: torch.cuda.amp.autocast(enabled=self.use_amp)

        with open(cfg.log_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch','split','loss','l1_loss','l2_loss','l3_loss',
                        'l1_acc','l2_acc','l3_acc','overall_acc','count_l3','lr'])

        self.best_val = float('inf')

    def _write_csv(self, epoch, split, loss, l1_loss, l2_loss, l3_loss,
                   l1_acc, l2_acc, l3_acc, overall_acc, cnt3, lr):
        with open(self.cfg.log_csv, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([epoch, split, f"{loss:.6f}", f"{l1_loss:.6f}", f"{l2_loss:.6f}",
                        f"{l3_loss:.6f}", f"{l1_acc:.6f}", f"{l2_acc:.6f}",
                        f"{l3_acc:.6f}", f"{overall_acc:.6f}", cnt3, f"{lr:.8f}"])

    @torch.no_grad()
    def eval_one_epoch(self, loader, split='val'):
        self.model.eval()
        tot = 0
        sum_loss = sum_l1 = sum_l2 = sum_l3 = 0.0
        corr_onset = corr_link  = corr_motor = 0
        overall_correct = 0
        counted_motor = 0
        all_onset_probs = []

        pbar = tqdm(loader, desc=f"Eval({split})", leave=False)

        ctx = self.autocast_ctor() if self.use_amp else torch.no_grad()
        with ctx:
            for X,E,M,y1,y2,y3 in pbar:
                X = X.to(self.device, non_blocking=True)
                E = E.to(self.device, non_blocking=True)
                M = M.to(self.device, non_blocking=True)
                y1 = y1.to(self.device, non_blocking=True)
                y2 = y2.to(self.device, non_blocking=True)
                y3 = y3.to(self.device, non_blocking=True)

                o1, o2, m_logits = self.model(X,E,M)
                
                l1 = self.bce(o1, y1)
                p1 = torch.sigmoid(o1)
                pred_onset = (p1 > 0.5).long().view(-1)
                y1_long = y1.long().view(-1)
                corr_onset += int((pred_onset == y1_long).sum().item())
                sum_l1 += float(l1.item()) * X.size(0)

                l2 = self.ce(o2, y2)
                pred_link = o2.argmax(dim=-1).view(-1)
                corr_link += int((pred_link == y2.view(-1)).sum().item())
                sum_l2 += float(l2.item()) * X.size(0)

                mask = (y2 > 0)
                motor_pred_per_b = [-1] * X.size(0)
                if mask.any():
                    logits_sel = []
                    y3_sel = []
                    for b in range(X.size(0)):
                        if mask[b]:
                            li = int(y2[b].item()) - 1
                            logits_b = m_logits[li][b:b+1]
                            logits_sel.append(logits_b)
                            y3_sel.append(y3[b:b+1])
                            motor_pred_per_b[b] = int(torch.argmax(m_logits[li][b]).item())
                    logits_sel = torch.cat(logits_sel, dim=0)
                    y3_sel = torch.cat(y3_sel, dim=0)
                    l3 = self.ce(logits_sel, y3_sel)
                    pred_motor = logits_sel.argmax(dim=-1)
                    corr_motor += int((pred_motor == y3_sel).sum().item())
                    counted_motor += y3_sel.size(0)
                else:
                    l3 = torch.tensor(0.0, device=self.device)
                sum_l3 += float(l3.item()) * X.size(0)

                loss = (self.cfg.lambda_l1 * l1
                        + self.cfg.lambda_l2 * l2
                        + self.cfg.lambda_l3 * l3)
                sum_loss += float(loss.item()) * X.size(0)
                tot += X.size(0)

                y2_cpu = y2.view(-1).tolist(); y3_cpu = y3.view(-1).tolist()
                pred_link_cpu = pred_link.tolist()
                pred_onset_cpu = pred_onset.tolist(); y1_long_cpu = y1_long.tolist()
                for b in range(X.size(0)):
                    ok1 = (pred_onset_cpu[b] == y1_long_cpu[b])
                    ok2 = (pred_link_cpu[b] == y2_cpu[b])
                    if y2_cpu[b] == 0: ok3 = True
                    else: ok3 = (motor_pred_per_b[b] == y3_cpu[b])
                    overall_correct += int(ok1 and ok2 and ok3)
                all_onset_probs.append(p1.detach().cpu().numpy())

        avg_loss = sum_loss / max(tot,1)
        l1_loss  = sum_l1 / max(tot,1)
        l2_loss  = sum_l2 / max(tot,1)
        l3_loss  = sum_l3 / max(tot,1)
        l1_acc   = corr_onset / max(tot,1)
        l2_acc   = corr_link  / max(tot,1)
        l3_acc   = (corr_motor / max(counted_motor,1)) if counted_motor>0 else 0.0
        overall_acc = overall_correct / max(tot,1)
        print(f"[{split}] loss={avg_loss:.4f}  Acc(L1/L2/L3/Overall)={l1_acc:.3f}/{l2_acc:.3f}/{l3_acc:.3f}/{overall_acc:.3f}  "
              f"Loss(L1/L2/L3)={l1_loss:.4f}/{l2_loss:.4f}/{l3_loss:.4f}  (L3 count={counted_motor})")
        if self.cfg.apply_post and len(all_onset_probs)>0:
            probs = np.concatenate(all_onset_probs, axis=0).reshape(-1)
            post = kofn_hysteresis(probs, self.cfg.post_k, self.cfg.post_n,
                                   self.cfg.post_up, self.cfg.post_down)
            thr = (probs>0.5).astype(int)
        return avg_loss, l1_loss, l2_loss, l3_loss, l1_acc, l2_acc, l3_acc, overall_acc, counted_motor

    def train(self):
        print(f"[INFO] Start training for {self.cfg.epochs} epochs...")
        for epoch in range(1, self.cfg.epochs+1):
            t0 = time.time()
            self.model.train()
            tot = 0
            sum_loss = sum_l1 = sum_l2 = sum_l3 = 0.0
            corr_onset = corr_link = corr_motor = 0
            counted_motor = 0
            overall_correct = 0

            if epoch <= self.cfg.motor_freeze_epochs:
                lambda_l3 = 0.0
            else:
                lambda_l3 = self.cfg.lambda_l3

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg.epochs}", leave=True)

            for X,E,M,y1,y2,y3 in pbar:
                X = X.to(self.device, non_blocking=True)
                E = E.to(self.device, non_blocking=True)
                M = M.to(self.device, non_blocking=True)
                y1 = y1.to(self.device, non_blocking=True)
                y2 = y2.to(self.device, non_blocking=True)
                y3 = y3.to(self.device, non_blocking=True)

                with self.autocast_ctor():
                    o1, o2, m_logits = self.model(X,E,M)
                    l1 = self.bce(o1, y1)
                    
                    l2 = self.ce(o2, y2)
                    
                    mask = (y2 > 0)
                    motor_pred_per_b = [-1] * X.size(0)
                    if mask.any() and lambda_l3 > 0.0:
                        logits_sel = []
                        y3_sel = []
                        for b in range(X.size(0)):
                            if mask[b]:
                                li = int(y2[b].item()) - 1
                                logits_b = m_logits[li][b:b+1]
                                logits_sel.append(logits_b)
                                y3_sel.append(y3[b:b+1])
                                motor_pred_per_b[b] = int(torch.argmax(m_logits[li][b]).item())
                        logits_sel = torch.cat(logits_sel, dim=0)
                        y3_sel = torch.cat(y3_sel, dim=0)
                        l3 = self.ce(logits_sel, y3_sel)
                        pred_motor = logits_sel.argmax(dim=-1)
                        corr_motor += int((pred_motor == y3_sel).sum().item())
                        counted_motor += y3_sel.size(0)
                    else:
                        if mask.any():
                             for b in range(X.size(0)):
                                if mask[b]:
                                    li = int(y2[b].item()) - 1
                                    motor_pred_per_b[b] = int(torch.argmax(m_logits[li][b]).item())
                        l3 = torch.tensor(0.0, device=self.device)

                    task_losses = torch.stack([
                        self.cfg.lambda_l1 * l1,
                        self.cfg.lambda_l2 * l2,
                        lambda_l3 * l3
                    ])
                    precision = torch.exp(-self.log_vars)
                    loss = (precision * task_losses).sum() + self.log_vars.sum()

                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()

                sum_l1 += float(l1.item()) * X.size(0)
                sum_l2 += float(l2.item()) * X.size(0)
                sum_l3 += float(l3.item()) * X.size(0)
                sum_loss += float(loss.item()) * X.size(0)
                tot += X.size(0)

                p1 = torch.sigmoid(o1)
                pred_onset = (p1 > 0.5).long().view(-1)
                y1_long = y1.long().view(-1)
                corr_onset += int((pred_onset == y1_long).sum().item())
                
                pred_link = o2.argmax(dim=-1).view(-1)
                corr_link  += int((pred_link == y2.view(-1)).sum().item())
                
                y2_cpu = y2.view(-1).tolist(); y3_cpu = y3.view(-1).tolist()
                pred_link_cpu = pred_link.tolist()
                pred_onset_cpu = pred_onset.tolist(); y1_long_cpu = y1_long.tolist()
                for b in range(X.size(0)):
                    ok1 = (pred_onset_cpu[b] == y1_long_cpu[b])
                    ok2 = (pred_link_cpu[b] == y2_cpu[b])
                    if y2_cpu[b] == 0: ok3 = True
                    else: ok3 = (motor_pred_per_b[b] == y3_cpu[b])
                    overall_correct += int(ok1 and ok2 and ok3)
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            train_loss = sum_loss / max(tot,1)
            l1_loss = sum_l1 / max(tot,1)
            l2_loss = sum_l2 / max(tot,1)
            l3_loss = sum_l3 / max(tot,1)
            l1_acc  = corr_onset / max(tot,1)
            l2_acc  = corr_link  / max(tot,1)
            l3_acc  = (corr_motor / max(counted_motor,1)) if counted_motor>0 else 0.0
            overall_acc = overall_correct / max(tot,1)
            cur_lr = self.opt.param_groups[0]['lr']
            elapsed = time.time() - t0
            print(f"Epoch {epoch} done | {elapsed:.1f}s | Loss={train_loss:.4f} | OverAcc={overall_acc:.3f}")
            self._write_csv(epoch, 'train', train_loss, l1_loss, l2_loss, l3_loss,
                            l1_acc, l2_acc, l3_acc, overall_acc, counted_motor, cur_lr)
            self.sched.step()

            val_metrics = self.eval_one_epoch(self.val_loader, split='val')
            self._write_csv(epoch, 'val', *val_metrics, self.opt.param_groups[0]['lr'])
            if self.cfg.eval_test_each_epoch:
                test_metrics = self.eval_one_epoch(self.test_loader, split='test')
                self._write_csv(epoch, 'test', *test_metrics, self.opt.param_groups[0]['lr'])

            val_loss = val_metrics[0]
            if val_loss < self.best_val:
                self.best_val = val_loss
                self._save_checkpoint('best_graph_fdi.pth', folder=self.save_dir_all)

            self._save_checkpoint('last_graph_fdi.pth', folder=self.save_dir_all)

            if self.cfg.save_every and (epoch % self.cfg.save_every == 0):
                self._save_checkpoint(f'ckpt_epoch_new_{epoch:03d}.pth',
                                      folder=self.save_dir_all)

            if epoch == self.cfg.epochs:
                final_name = f"GraphFDI_Transformer_L{self.links}.pth"
                self._save_checkpoint(final_name, folder=self.save_dir_last)

        print("Training done. Testing best model…")
        self._load_checkpoint('best_graph_fdi.pth', folder=self.save_dir_all)
        test_metrics = self.eval_one_epoch(self.test_loader, split='test(final)')
        self._write_csv(self.cfg.epochs, 'test_final', *test_metrics, self.opt.param_groups[0]['lr'])
        self.maybe_export_onset_series()

    def maybe_export_onset_series(self):
        if not self.cfg.export_onset_series:
            return
        self.model.eval()
        for X,E,M,y1,y2,y3 in self.val_loader:
            X = X.to(self.device); E = E.to(self.device); M = M.to(self.device)
            with self.autocast_ctor():
                o1, _, _ = self.model(X,E,M)
            probs = torch.sigmoid(o1).detach().cpu().numpy().reshape(-1)
            post = kofn_hysteresis(probs, self.cfg.post_k, self.cfg.post_n,
                                   self.cfg.post_up, self.cfg.post_down)
            with open('onset_series.csv','w',newline='') as f:
                w = csv.writer(f); w.writerow(['t','prob','post'])
                for t,(p,pp) in enumerate(zip(probs, post)):
                    w.writerow([t, f"{p:.6f}", int(pp)])
            print("Exported onset_series.csv (prob vs K-of-N post) from one val batch")
            break

    def _save_checkpoint(self, filename: str, folder: Optional[str] = None):
        """
        filename: 저장할 파일 이름 (예: 'best_graph_fdi.pth')
        folder  : 저장할 폴더 (None이면 Link{L}_All 사용)
        """
        if folder is None:
            folder = self.save_dir_all

        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)

        ckpt = {
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'cfg': self.cfg.__dict__,
            'log_vars': self.log_vars.detach().cpu()
        }
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")


    def _load_checkpoint(self, filename: str, folder: Optional[str] = None):
        """
        filename: 불러올 파일 이름 (예: 'best_graph_fdi.pth')
        folder  : 폴더 (None이면 Link{L}_All 사용)
        """
        if folder is None:
            folder = self.save_dir_all

        if os.path.isabs(filename):
            path = filename
        else:
            path = os.path.join(folder, filename)

        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        if 'log_vars' in ckpt:
            self.log_vars.data = ckpt['log_vars'].to(self.device)
        print(f"Loaded checkpoint: {path}")

# ------------------------- CLI -------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description='LASDRA Graph-Transformer FDI — training (frame-cache, transformer)')
    p.add_argument('--data_dir', type=str, required=True,
                   help='Path to link_L/ directory')
    p.add_argument('--shard_first', type=int, default=1)
    p.add_argument('--shard_last', type=int, default=12)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--T_win', type=int, default=128)
    p.add_argument('--stride', type=int, default=64)
    p.add_argument('--d_model', type=int, default=384)
    p.add_argument('--heads', type=int, default=6)
    p.add_argument('--depth', type=int, default=5)
    p.add_argument('--dropout', type=float, default=0.05)
    p.add_argument('--temporal', type=str, default='transformer', choices=['tcn','transformer'])
    p.add_argument('--motor_geom', type=str, default=None,
                   help='npz with rotvec_m2l, r_mL, f_dir, arm')
    p.add_argument('--apply_post', action='store_true')
    p.add_argument('--post_k', type=int, default=3)
    p.add_argument('--post_n', type=int, default=5)
    p.add_argument('--post_up', type=float, default=0.6)
    p.add_argument('--post_down', type=float, default=0.4)
    p.add_argument('--export_onset_series', type=int, default=0)
    p.add_argument('--log_csv', type=str, default='train_log.csv')
    p.add_argument('--eval_test_each_epoch', type=int, default=1)
    p.add_argument('--save_every', type=int, default=1)
    p.add_argument('--cache_windows', type=int, default=1,
                   help='1=enable frame-cache build/use')
    p.add_argument('--cache_dir', type=str, default='./cache_frames')
    p.add_argument('--cache_dtype', type=str, default='fp32', choices=['fp32','fp16'])
    p.add_argument('--amp', type=int, default=1)
    p.add_argument('--compile', type=int, default=0)
    # loss weights & warm-up
    p.add_argument('--lambda_l1', type=float, default=1.0)
    p.add_argument('--lambda_l2', type=float, default=1.0)
    p.add_argument('--lambda_l3', type=float, default=1.0)
    p.add_argument('--motor_freeze_epochs', type=int, default=6)
    # label smoothing
    p.add_argument('--label_smoothing', type=float, default=0.05)
    args = p.parse_args()
    return TrainConfig(**vars(args))

if __name__ == '__main__':
    cfg = parse_args()
    print('Config:', cfg)
    trainer = Trainer(cfg)
    trainer.train()

"""
python3 graph_fdi_train.py \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_1 \
  --shard_first 1 --shard_last 3 \
  --epochs 20 \
  --batch_size 16 \
  --device cuda \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom_link1.npz \
  --temporal transformer \
  --d_model 384 --heads 6 --depth 5 --dropout 0.1 \
  --T_win 128 --stride 64 \
  --lr 5e-5 --weight_decay 1e-4 \
  --apply_post --post_k 3 --post_n 5 --post_up 0.6 --post_down 0.4 \
  --eval_test_each_epoch 1 \
  --num_workers 0 \
  --cache_windows 1 --cache_dir ./cache_L1 --cache_dtype fp32 \
  --amp 1 \
  --lambda_l1 1.0 --lambda_l2 1.0 --lambda_l3 1.0 \
  --motor_freeze_epochs 1 \
  --label_smoothing 0.05 \
  --log_csv link1_train_log.csv

python3 graph_fdi_train.py \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_2 \
  --shard_first 1 --shard_last 4 \
  --epochs 45 \
  --batch_size 8 \
  --device cuda \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom_link2.npz \
  --temporal transformer \
  --d_model 384 --heads 6 --depth 5 --dropout 0.1 \
  --T_win 128 --stride 64 \
  --lr 5e-5 --weight_decay 1e-4 \
  --apply_post --post_k 3 --post_n 5 --post_up 0.6 --post_down 0.4 \
  --eval_test_each_epoch 1 \
  --num_workers 8 \
  --cache_windows 1 --cache_dir ./cache_L2 --cache_dtype fp32 \
  --amp 1 \
  --lambda_l1 1.0 --lambda_l2 1.0 --lambda_l3 1.0 \
  --motor_freeze_epochs 5 \
  --label_smoothing 0.05 \
  --log_csv link2_train_log.csv

python3 graph_fdi_train.py \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_3 \
  --shard_first 1 --shard_last 5 \
  --epochs 80 \
  --batch_size 16 \
  --device cuda \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom_link3.npz \
  --temporal transformer \
  --d_model 384 --heads 6 --depth 5 --dropout 0.1 \
  --T_win 128 --stride 64 \
  --lr 5e-5 --weight_decay 1e-4 \
  --apply_post --post_k 3 --post_n 5 --post_up 0.6 --post_down 0.4 \
  --eval_test_each_epoch 1 \
  --num_workers 0 \
  --cache_windows 1 --cache_dir ./cache_L3 --cache_dtype fp32 \
  --amp 1 \
  --lambda_l1 1.0 --lambda_l2 1.0 --lambda_l3 1.0 \
  --motor_freeze_epochs 8 \
  --label_smoothing 0.05

python3 graph_fdi_train.py \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_4 \
  --shard_first 1 --shard_last 5 \
  --epochs 80 \
  --batch_size 8 \
  --device cuda \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom_link4.npz \
  --temporal transformer \
  --d_model 384 --heads 6 --depth 5 --dropout 0.1 \
  --T_win 128 --stride 64 \
  --lr 5e-5 --weight_decay 1e-4 \
  --apply_post --post_k 3 --post_n 5 --post_up 0.6 --post_down 0.4 \
  --eval_test_each_epoch 1 \
  --num_workers 8 \
  --cache_windows 1 --cache_dir ./cache_L4 --cache_dtype fp32 \
  --amp 1 \
  --lambda_l1 1.0 --lambda_l2 1.0 --lambda_l3 1.0 \
  --motor_freeze_epochs 8 \
  --label_smoothing 0.05 \
  --log_csv link4_train_log.csv

python3 graph_fdi_train.py \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_5 \
  --shard_first 1 --shard_last 5 \
  --epochs 80 \
  --batch_size 8 \
  --device cuda \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom_link5.npz \
  --temporal transformer \
  --d_model 384 --heads 6 --depth 5 --dropout 0.1 \
  --T_win 128 --stride 64 \
  --lr 5e-5 --weight_decay 1e-4 \
  --apply_post --post_k 3 --post_n 5 --post_up 0.6 --post_down 0.4 \
  --eval_test_each_epoch 1 \
  --num_workers 8 \
  --cache_windows 1 --cache_dir ./cache_L5 --cache_dtype fp32 \
  --amp 1 \
  --lambda_l1 1.0 --lambda_l2 1.0 --lambda_l3 1.0 \
  --motor_freeze_epochs 8 \
  --label_smoothing 0.05 \
  --log_csv link5_train_log.csv


python3 graph_fdi_train.py \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_10 \
  --shard_first 1 --shard_last 6 \
  --epochs 80 \
  --batch_size 2 \
  --device cuda \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom_link10.npz \
  --temporal transformer \
  --d_model 384 --heads 6 --depth 5 --dropout 0.1 \
  --T_win 128 --stride 64 \
  --lr 5e-5 --weight_decay 1e-4 \
  --apply_post --post_k 3 --post_n 5 --post_up 0.6 --post_down 0.4 \
  --eval_test_each_epoch 1 \
  --num_workers 4 \
  --cache_windows 1 --cache_dir ./cache_L10 --cache_dtype fp32 \
  --amp 1 \
  --lambda_l1 1.0 --lambda_l2 1.0 --lambda_l3 1.0 \
  --motor_freeze_epochs 8 \
  --label_smoothing 0.05 \
  --log_csv link10_train_log.csv
"""
