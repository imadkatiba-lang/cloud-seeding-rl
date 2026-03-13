# ======================================================================================
# USV FULL PIPELINE
# End-to-end manuscript-aligned workflow in a single script
#
# Stages:
#   1. Data processing
#   2. Predictive models
#   3. Constrained decision-making
#   4. Deployment-oriented audit
#   5. Spatial linkage and continuous prioritization
#   6. Spatiotemporal clustering and operational routing
#
# Notes:
#   - Update the Windows paths below before execution.
#   - This script is organized by stage but kept in a single file.
#   - Each stage can be enabled or disabled through the RUN_STAGE_* flags.
# ======================================================================================

import os
import re
import json
import time
import math
import glob
import warnings
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import xarray as xr

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except Exception:
    yaml = None

try:
    from sklearn.cluster import DBSCAN
except Exception:
    DBSCAN = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ======================================================================================
# GLOBAL CONFIG
# ======================================================================================

# --------------------------------------
# Execution flags
# --------------------------------------
RUN_STAGE_1 = True
RUN_STAGE_2 = True
RUN_STAGE_3 = True
RUN_STAGE_4 = True
RUN_STAGE_5 = True
RUN_STAGE_6 = True

# --------------------------------------
# Paths
# --------------------------------------
BASE_ROOT = r"C:\Users\tuf-p\Desktop\ARTICLES\USV"

RAW_NC = r"C:\Users\tuf-p\Desktop\ARTICLES\USV\USV_PROJECT\01_clean\ERA5_JAN_2021_2025_USV_INSTANT_ONLY_clean_v2.nc"
SEQ_INFO = r"C:\Users\tuf-p\Desktop\ARTICLES\USV\USV_PROJECT\04_ready_seq_v5_fe\seq_build_info.json"

STAGE_1_DIR = os.path.join(BASE_ROOT, "RESULTS_OF_PROJECT", "STAGE_1_DATA_PROCESSING")
STAGE_2_DIR = os.path.join(BASE_ROOT, "RESULTS_OF_PROJECT", "STAGE_2_PREDICTIVE_MODELS")
STAGE_3_DIR = os.path.join(BASE_ROOT, "RESULTS_OF_PROJECT", "STAGE_3_CONSTRAINED_DECISION_MAKING")
STAGE_4_DIR = os.path.join(BASE_ROOT, "RESULTS_OF_PROJECT", "STAGE_4_DEPLOYMENT_ORIENTED_AUDIT")
STAGE_5_DIR = os.path.join(BASE_ROOT, "RESULTS_OF_PROJECT", "STAGE_5_SPATIAL_LINKAGE_AND_CONTINUOUS_PRIORITIZATION")
STAGE_6_DIR = os.path.join(BASE_ROOT, "RESULTS_OF_PROJECT", "STAGE_6_SPATIOTEMPORAL_CLUSTERING_AND_OPERATIONAL_ROUTING")

for _d in [STAGE_1_DIR, STAGE_2_DIR, STAGE_3_DIR, STAGE_4_DIR, STAGE_5_DIR, STAGE_6_DIR]:
    os.makedirs(_d, exist_ok=True)

# --------------------------------------
# Device and seed
# --------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------
# Shared variables
# --------------------------------------
STATE_VARS = ["blh", "msl", "tcc", "tcwv", "u10", "v10", "wspd10", "lcc"]
IDX_LCC = STATE_VARS.index("lcc")
IDX_TCWV = STATE_VARS.index("tcwv")
IDX_WSPD = STATE_VARS.index("wspd10")

KEEP_FRACTION_FALLBACK = 0.25
MAX_POINTS = 360
SPLIT = (0.70, 0.15, 0.15)
ROBUST_SAMPLES = 150_000

# --------------------------------------
# Predictor config
# --------------------------------------
PRED_EPOCHS = 16
PRED_BATCH = 4096
PRED_PATIENCE = 5
WM_LATENT = 48
WM_HID = 256
WM_KL_W = 0.08

# --------------------------------------
# Decision config
# --------------------------------------
DELTA_LCC_MAX = 0.06
ACTION_COST = 0.03
NO_GO_GATE = 0.15
ACT_DIM = 1

RL_STEPS = 18000
RL_BATCH = 512
ROLLOUT_H = 8
GAMMA = 0.97
TAU = 0.01
ALPHA = 0.10

N_EVAL_EPISODES = 80
EP_LEN = 24

CEM_POP = 160
CEM_ELITE = 32
CEM_ITERS = 5
CEM_H = 8

REWARD_SCALE = 0.10
Q_TARGET_CLAMP = 20.0
CRITIC_LR = 1e-4
ACTOR_LR = 3e-4
CRITIC_GRAD_CLIP = 3.0
ACTOR_GRAD_CLIP = 10.0

DIAG_ACTION_GRID = [0.0, 0.25, 0.5, 1.0]
N_DIAG_REWARD_SAMPLES = 60000

# --------------------------------------
# Audit config
# --------------------------------------
W_LCC_R2 = 0.55
W_LCC_RMSE = 0.20
W_LCC_MAE = 0.10
W_STATE_R2 = 0.10
W_STATE_RMSE = 0.03
W_STATE_MAE = 0.02

OVERFIT_GAP_THRESHOLD_R2 = 0.03
OVERFIT_PENALTY = 0.05

W_RET_MEAN = 0.55
W_RET_STD = 0.10
W_TEST_LCC_R2 = 0.15
W_TEST_LCC_RMSE = 0.07
W_TEST_STATE_R2 = 0.05
W_TEST_STATE_RMSE = 0.03
W_VIOL = 0.05

# --------------------------------------
# Stage 5 config
# --------------------------------------
MAX_SPRAY_L_PER_POINT = 50.0
ZONE_TOPK = 10
EPS = 1e-12

# --------------------------------------
# Stage 6 config
# --------------------------------------
LOG_PATH_STAGE_6 = os.path.join(STAGE_6_DIR, "operational_routing.log")
CONFIG_DIR_STAGE_6 = os.path.join(STAGE_6_DIR, "config")
OUT_DIR_STAGE_6 = os.path.join(STAGE_6_DIR, "outputs")
os.makedirs(CONFIG_DIR_STAGE_6, exist_ok=True)
os.makedirs(OUT_DIR_STAGE_6, exist_ok=True)

PATH_USV_PLATFORMS = os.path.join(CONFIG_DIR_STAGE_6, "usv_platforms.yaml")
PATH_MISSION_POLICY = os.path.join(CONFIG_DIR_STAGE_6, "mission_policy.yaml")
PATH_PORTS_BASES = os.path.join(CONFIG_DIR_STAGE_6, "ports_bases.csv")

DEFAULT_USV_PLATFORMS = {
    "platforms": [
        {
            "usv_id": "USV_01",
            "platform_name": "USV_Generic",
            "cruise_speed_kmh": 26.0,
            "max_endurance_min": 600.0,
            "reserve_min": 40.0,
            "spray_rate_lpm": 2.0,
            "agent_capacity_l": 300.0,
            "setup_spray_min": 2.0,
            "shutdown_spray_min": 2.0,
            "max_nav_wind_ms": 14.0,
            "max_spray_wind_ms": 10.0,
            "turnaround_time_min": 20.0,
            "base_id": None
        }
    ]
}

DEFAULT_MISSION_POLICY = {
    "policy": {
        "priority_score_min": 0.5,
        "default_time_step_min": 5.0,
        "time_buffer_min": 2.0,
        "dbscan_eps_km": 5.0,
        "dbscan_min_samples": 2,
        "max_bases": 220,
        "targets_dbscan_eps_km": 5.0,
        "targets_dbscan_min_samples": 2,
        "n_usv_per_base_default": 4,
        "priority_weights": {"score": 0.7, "distance": 0.2, "window": 0.1}
    }
}

DEFAULT_PORTS_BASES = pd.DataFrame([
    {"base_id": "BASE_01", "base_name": "Base 1", "lat": 33.5731, "lon": -7.5898, "n_usv_available": 1, "is_active": 1},
])


# ======================================================================================
# GENERAL HELPERS
# ======================================================================================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def nanfix(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def robust_fit(X):
    med = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    iqr = np.maximum(iqr, 1e-6)
    return med.astype(np.float32), iqr.astype(np.float32)

def robust_apply(X, med, iqr):
    return ((X - med) / iqr).astype(np.float32)

def idx_to_ij(p, Nx):
    return (p // Nx).astype(np.int32), (p % Nx).astype(np.int32)

def time_split(T):
    n_train = int(T * SPLIT[0])
    n_val = int(T * SPLIT[1])
    train_rng = (0, n_train - 1)
    val_rng = (n_train, n_train + n_val - 1)
    test_rng = (n_train + n_val, T - 1)
    return train_rng, val_rng, test_rng

def r2_rmse_mae_np(y_true, y_pred):
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
    r2 = float(1.0 - ss_res / ss_tot)
    return r2, rmse, mae

def r2_rmse_mae_torch(y_true, y_pred):
    return r2_rmse_mae_np(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

def bootstrap_ci(x, n_boot=400, alpha=0.05):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 5:
        return float(np.nan), float(np.nan)
    rng = np.random.default_rng(SEED + 7)
    means = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=x.size, replace=True)
        means.append(np.mean(samp))
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))

def minmax_series(s: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        out = pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    else:
        out = (s - mn) / (mx - mn)
    return 1.0 - out if invert else out

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
            return None
    return None

def save_df_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("Saved:", path)

def normalize_colname(c: str) -> str:
    c = str(c).strip().replace("\ufeff", "")
    c = c.replace(" ", "_").replace("/", "_").replace("-", "_")
    c = re.sub(r"__+", "_", c)
    return c

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_colname(c) for c in out.columns]
    return out

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def clip01(x):
    return np.clip(x, 0.0, 1.0)

def robust01(s: pd.Series, q_low=0.02, q_high=0.98) -> pd.Series:
    s = to_numeric_safe(s)
    v = s.values.astype(float)
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return pd.Series(np.nan, index=s.index)
    lo = np.nanquantile(v[finite], q_low)
    hi = np.nanquantile(v[finite], q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < EPS:
        return pd.Series(np.zeros(len(s)), index=s.index)
    out = (v - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return pd.Series(out, index=s.index)

def pick_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}
    for c in candidates:
        c_norm = normalize_colname(c)
        if c_norm in df.columns:
            return c_norm
    for c in candidates:
        c_norm = normalize_colname(c).lower()
        if c_norm in cols_lower:
            return cols_lower[c_norm]
    return None

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def latlon_to_km_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat0 = np.nanmean(lat)
    x = (lon - np.nanmean(lon)) * 111.32 * np.cos(np.deg2rad(lat0))
    y = (lat - np.nanmean(lat)) * 110.574
    return np.stack([x, y], axis=1)


# ======================================================================================
# STAGE 1 — DATA PROCESSING
# ======================================================================================

def load_keep_idx(seq_info_path, Ny, Nx):
    if os.path.exists(seq_info_path):
        try:
            with open(seq_info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            if "keep_idx" in info:
                k = np.array(info["keep_idx"], dtype=np.int32)
                k.sort()
                return k
        except Exception:
            pass
    P = Ny * Nx
    keep = np.random.choice(P, size=int(P * KEEP_FRACTION_FALLBACK), replace=False).astype(np.int32)
    keep.sort()
    return keep

def stage_1_load_data():
    ds = xr.open_dataset(RAW_NC, engine="netcdf4")
    T = ds.sizes["time"]
    Ny = ds.sizes["lat"]
    Nx = ds.sizes["lon"]
    keep = load_keep_idx(SEQ_INFO, Ny, Nx)
    P_full = Ny * Nx

    if MAX_POINTS is not None and keep.shape[0] > MAX_POINTS:
        sel = np.random.choice(keep.shape[0], size=MAX_POINTS, replace=False)
        sel.sort()
        keep = keep[sel]

    Pk = keep.shape[0]
    feats = []
    for v in STATE_VARS:
        a = ds[v].values.astype(np.float32).reshape(T, P_full)[:, keep]
        feats.append(a[..., None])

    S = np.concatenate(feats, axis=2)
    S = nanfix(S)

    iy, ix = idx_to_ij(keep, Nx)
    ij = np.stack([iy, ix], axis=1).astype(np.int32)
    ds.close()
    return S, ij, (T, Ny, Nx, Pk, len(STATE_VARS))

def run_stage_1():
    print("\n=== Stage 1 — Data processing ===")
    S, ij, meta = stage_1_load_data()
    T, Ny, Nx, Pk, sdim = meta
    train_rng, val_rng, test_rng = time_split(T)

    a, b = train_rng
    ts = np.random.randint(a, b + 1, size=ROBUST_SAMPLES)
    ps = np.random.randint(0, Pk, size=ROBUST_SAMPLES)
    med, iqr = robust_fit(S[ts, ps])
    S_norm = nanfix(robust_apply(S, med, iqr))

    np.save(os.path.join(STAGE_1_DIR, "environment_tensor.npy"), S)
    np.save(os.path.join(STAGE_1_DIR, "environment_tensor_normalized.npy"), S_norm)
    np.save(os.path.join(STAGE_1_DIR, "grid_indices.npy"), ij)

    pd.DataFrame({"iy": ij[:, 0], "ix": ij[:, 1]}).to_csv(
        os.path.join(STAGE_1_DIR, "grid_indices.csv"), index=False
    )

    info = {
        "STATE_VARS": STATE_VARS,
        "shape_raw": list(S.shape),
        "shape_norm": list(S_norm.shape),
        "Ny": int(Ny),
        "Nx": int(Nx),
        "Pk": int(Pk),
        "train_range": train_rng,
        "val_range": val_rng,
        "test_range": test_rng,
    }
    with open(os.path.join(STAGE_1_DIR, "data_processing_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("Stage 1 completed:", STAGE_1_DIR)


# ======================================================================================
# STAGE 2 — PREDICTIVE MODELS
# ======================================================================================

class TwinMLP(nn.Module):
    def __init__(self, sdim=8, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sdim, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, sdim),
        )

    def forward(self, s):
        return self.net(s)

class GRUDynamics(nn.Module):
    def __init__(self, sdim=8, hid=128):
        super().__init__()
        self.gru = nn.GRU(input_size=sdim, hidden_size=hid, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, 256), nn.SiLU(), nn.Linear(256, sdim))

    def forward(self, s_tm1, s_t):
        x = torch.stack([s_tm1, s_t], dim=1)
        h, _ = self.gru(x)
        return self.head(h[:, -1])

def gaussian_kl(mu_q, logstd_q, mu_p, logstd_p):
    var_q = torch.exp(2.0 * logstd_q)
    var_p = torch.exp(2.0 * logstd_p)
    return (logstd_p - logstd_q) + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p) - 0.5

def reparam(mu, logstd):
    eps = torch.randn_like(mu)
    return mu + eps * torch.exp(logstd)

class StochWM(nn.Module):
    def __init__(self, sdim=8, adim=1, latent=48, hid=256):
        super().__init__()
        self.post = nn.Sequential(nn.Linear(sdim, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU())
        self.post_mu = nn.Linear(hid, latent)
        self.post_ls = nn.Linear(hid, latent)

        self.prior = nn.Sequential(nn.Linear(latent + adim, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU())
        self.prior_mu = nn.Linear(hid, latent)
        self.prior_ls = nn.Linear(hid, latent)

        self.dec = nn.Sequential(nn.Linear(latent, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, sdim))

    def posterior_params(self, s):
        h = self.post(s)
        mu = self.post_mu(h)
        ls = torch.clamp(self.post_ls(h), -6.0, 2.0)
        return mu, ls

    def prior_params(self, z, a):
        h = self.prior(torch.cat([z, a], dim=1))
        mu = self.prior_mu(h)
        ls = torch.clamp(self.prior_ls(h), -6.0, 2.0)
        return mu, ls

    def decode(self, z):
        return self.dec(z)

def load_stage1_data():
    S_norm = np.load(os.path.join(STAGE_1_DIR, "environment_tensor_normalized.npy"))
    with open(os.path.join(STAGE_1_DIR, "data_processing_info.json"), "r", encoding="utf-8") as f:
        info = json.load(f)
    train_rng = tuple(info["train_range"])
    val_rng = tuple(info["val_range"])
    test_rng = tuple(info["test_range"])
    return S_norm, train_rng, val_rng, test_rng

@torch.no_grad()
def eval_pred_1step(model, kind, S_norm, t_rng, n=120_000, wm_mode="prior"):
    model.eval()
    a, b = t_rng
    T, Pk, _ = S_norm.shape
    ts = np.random.randint(max(a, 1), min(b, T - 2) + 1, size=n)
    ps = np.random.randint(0, Pk, size=n)

    s_t = torch.from_numpy(S_norm[ts, ps]).to(DEVICE)
    sp_true = torch.from_numpy(S_norm[ts + 1, ps]).to(DEVICE)

    if kind == "TWIN":
        sp_hat = model(s_t)
    elif kind == "GRU":
        s_tm1 = torch.from_numpy(S_norm[ts - 1, ps]).to(DEVICE)
        sp_hat = model(s_tm1, s_t)
    elif kind == "WM":
        if wm_mode == "recon":
            mu_q, ls_q = model.posterior_params(sp_true)
            z_q = reparam(mu_q, ls_q)
            sp_hat = model.decode(z_q)
        else:
            a0 = torch.zeros((s_t.shape[0], ACT_DIM), device=DEVICE)
            mu_z, ls_z = model.posterior_params(s_t)
            z_t = reparam(mu_z, ls_z)
            mu_p, ls_p = model.prior_params(z_t, a0)
            z_p = reparam(mu_p, ls_p)
            sp_hat = model.decode(z_p)
    else:
        raise ValueError(kind)

    lcc_true = sp_true[:, IDX_LCC]
    lcc_hat = sp_hat[:, IDX_LCC]
    lcc_r2, lcc_rmse, lcc_mae = r2_rmse_mae_torch(lcc_true, lcc_hat)
    s_r2, s_rmse, s_mae = r2_rmse_mae_torch(sp_true.reshape(-1), sp_hat.reshape(-1))
    return (lcc_r2, lcc_rmse, lcc_mae), (s_r2, s_rmse, s_mae)

@torch.no_grad()
def eval_pred_per_variable_test(model, kind, S_norm, t_rng, n=180_000, wm_mode="prior"):
    model.eval()
    a, b = t_rng
    T, Pk, _ = S_norm.shape
    ts = np.random.randint(max(a, 1), min(b, T - 2) + 1, size=n)
    ps = np.random.randint(0, Pk, size=n)

    s_t = torch.from_numpy(S_norm[ts, ps]).to(DEVICE)
    sp_true = torch.from_numpy(S_norm[ts + 1, ps]).to(DEVICE)

    if kind == "TWIN":
        sp_hat = model(s_t)
        suffix = ""
    elif kind == "GRU":
        s_tm1 = torch.from_numpy(S_norm[ts - 1, ps]).to(DEVICE)
        sp_hat = model(s_tm1, s_t)
        suffix = ""
    elif kind == "WM":
        if wm_mode == "recon":
            mu_q, ls_q = model.posterior_params(sp_true)
            z_q = reparam(mu_q, ls_q)
            sp_hat = model.decode(z_q)
            suffix = "_WM_RECON"
        else:
            a0 = torch.zeros((s_t.shape[0], ACT_DIM), device=DEVICE)
            mu_z, ls_z = model.posterior_params(s_t)
            z_t = reparam(mu_z, ls_z)
            mu_p, ls_p = model.prior_params(z_t, a0)
            z_p = reparam(mu_p, ls_p)
            sp_hat = model.decode(z_p)
            suffix = "_WM_PRIOR"
    else:
        raise ValueError(kind)

    rows = []
    for j, v in enumerate(STATE_VARS):
        r2, rmse, mae = r2_rmse_mae_torch(sp_true[:, j], sp_hat[:, j])
        rows.append({"predictor": kind + suffix, "variable": v, "R2": r2, "RMSE": rmse, "MAE": mae})
    return pd.DataFrame(rows)

def train_predictor(kind, S_norm, train_rng, val_rng):
    T, Pk, sdim = S_norm.shape
    best_path = os.path.join(STAGE_2_DIR, f"{kind}_best.pt")

    if kind == "TWIN":
        model = TwinMLP(sdim, hid=256).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    elif kind == "GRU":
        model = GRUDynamics(sdim, hid=128).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    elif kind == "WM":
        model = StochWM(sdim, ACT_DIM, latent=WM_LATENT, hid=WM_HID).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    else:
        raise ValueError(kind)

    best = -1e18
    bad = 0

    for ep in range(1, PRED_EPOCHS + 1):
        model.train()
        a, b = train_rng
        n_steps = max(1, 220_000 // PRED_BATCH)
        losses = []

        for _ in range(n_steps):
            ts = np.random.randint(max(a, 1), min(b, S_norm.shape[0] - 2) + 1, size=PRED_BATCH)
            ps = np.random.randint(0, Pk, size=PRED_BATCH)

            s_t = torch.from_numpy(S_norm[ts, ps]).to(DEVICE)
            sp = torch.from_numpy(S_norm[ts + 1, ps]).to(DEVICE)

            if kind == "TWIN":
                sp_hat = model(s_t)
                loss = F.mse_loss(sp_hat, sp)
            elif kind == "GRU":
                s_tm1 = torch.from_numpy(S_norm[ts - 1, ps]).to(DEVICE)
                sp_hat = model(s_tm1, s_t)
                loss = F.mse_loss(sp_hat, sp)
            elif kind == "WM":
                a01 = torch.rand((PRED_BATCH, ACT_DIM), device=DEVICE)
                mu_z, ls_z = model.posterior_params(s_t)
                z = reparam(mu_z, ls_z)

                mu_q, ls_q = model.posterior_params(sp)
                z_q = reparam(mu_q, ls_q)

                mu_p, ls_p = model.prior_params(z, a01)
                sp_hat = model.decode(z_q)

                loss_recon = F.mse_loss(sp_hat, sp)
                kl = gaussian_kl(mu_q, ls_q, mu_p, ls_p).mean()
                loss = loss_recon + WM_KL_W * kl
            else:
                raise ValueError(kind)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))

        tr_loss = float(np.mean(losses))
        if kind == "WM":
            (lcc_r2_v, lcc_rmse_v, lcc_mae_v), _ = eval_pred_1step(model, kind, S_norm, val_rng, n=100_000, wm_mode="prior")
            print(f"[{kind}] Ep {ep:02d} | loss={tr_loss:.5f} | VAL lcc R2(PRIOR)={lcc_r2_v:.4f} RMSE={lcc_rmse_v:.4f} MAE={lcc_mae_v:.4f}")
        else:
            (lcc_r2_v, lcc_rmse_v, lcc_mae_v), _ = eval_pred_1step(model, kind, S_norm, val_rng, n=100_000)
            print(f"[{kind}] Ep {ep:02d} | loss={tr_loss:.5f} | VAL lcc R2={lcc_r2_v:.4f} RMSE={lcc_rmse_v:.4f} MAE={lcc_mae_v:.4f}")

        if lcc_r2_v > best + 1e-4:
            best = lcc_r2_v
            bad = 0
            torch.save({"model": model.state_dict()}, best_path)
        else:
            bad += 1
            if bad >= PRED_PATIENCE:
                print(f"[{kind}] Early stopping.")
                break

    ck = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    return model, best_path

def run_stage_2():
    print("\n=== Stage 2 — Predictive models ===")
    S_norm, train_rng, val_rng, test_rng = load_stage1_data()

    pred_metrics = []
    per_var_rows = []

    for kind in ["TWIN", "GRU", "WM"]:
        print(f"\nTraining predictor: {kind}")
        model, best_path = train_predictor(kind, S_norm, train_rng, val_rng)

        if kind == "WM":
            (lcc_r2_v, lcc_rmse_v, lcc_mae_v), (s_r2_v, s_rmse_v, s_mae_v) = eval_pred_1step(model, kind, S_norm, val_rng, n=120_000, wm_mode="prior")
            (lcc_r2_t, lcc_rmse_t, lcc_mae_t), (s_r2_t, s_rmse_t, s_mae_t) = eval_pred_1step(model, kind, S_norm, test_rng, n=160_000, wm_mode="prior")
            (lcc_r2_v_re, lcc_rmse_v_re, lcc_mae_v_re), (s_r2_v_re, s_rmse_v_re, s_mae_v_re) = eval_pred_1step(model, kind, S_norm, val_rng, n=120_000, wm_mode="recon")
            (lcc_r2_t_re, lcc_rmse_t_re, lcc_mae_t_re), (s_r2_t_re, s_rmse_t_re, s_mae_t_re) = eval_pred_1step(model, kind, S_norm, test_rng, n=160_000, wm_mode="recon")

            pred_metrics.append({
                "predictor": "WM",
                "WM_eval_mode_comparable": "PRIOR",
                "VAL_lcc_R2": lcc_r2_v, "VAL_lcc_RMSE": lcc_rmse_v, "VAL_lcc_MAE": lcc_mae_v,
                "TEST_lcc_R2": lcc_r2_t, "TEST_lcc_RMSE": lcc_rmse_t, "TEST_lcc_MAE": lcc_mae_t,
                "VAL_state_R2": s_r2_v, "VAL_state_RMSE": s_rmse_v, "VAL_state_MAE": s_mae_v,
                "TEST_state_R2": s_r2_t, "TEST_state_RMSE": s_rmse_t, "TEST_state_MAE": s_mae_t,
                "VAL_lcc_R2_WM_RECON": lcc_r2_v_re, "VAL_lcc_RMSE_WM_RECON": lcc_rmse_v_re, "VAL_lcc_MAE_WM_RECON": lcc_mae_v_re,
                "TEST_lcc_R2_WM_RECON": lcc_r2_t_re, "TEST_lcc_RMSE_WM_RECON": lcc_rmse_t_re, "TEST_lcc_MAE_WM_RECON": lcc_mae_t_re,
                "VAL_state_R2_WM_RECON": s_r2_v_re, "VAL_state_RMSE_WM_RECON": s_rmse_v_re, "VAL_state_MAE_WM_RECON": s_mae_v_re,
                "TEST_state_R2_WM_RECON": s_r2_t_re, "TEST_state_RMSE_WM_RECON": s_rmse_t_re, "TEST_state_MAE_WM_RECON": s_mae_t_re,
                "best_ckpt": os.path.basename(best_path),
            })
            per_var_rows.append(eval_pred_per_variable_test(model, kind, S_norm, test_rng, n=160_000, wm_mode="prior"))
            per_var_rows.append(eval_pred_per_variable_test(model, kind, S_norm, test_rng, n=160_000, wm_mode="recon"))
        else:
            (lcc_r2_v, lcc_rmse_v, lcc_mae_v), (s_r2_v, s_rmse_v, s_mae_v) = eval_pred_1step(model, kind, S_norm, val_rng, n=120_000)
            (lcc_r2_t, lcc_rmse_t, lcc_mae_t), (s_r2_t, s_rmse_t, s_mae_t) = eval_pred_1step(model, kind, S_norm, test_rng, n=160_000)

            pred_metrics.append({
                "predictor": kind,
                "WM_eval_mode_comparable": "",
                "VAL_lcc_R2": lcc_r2_v, "VAL_lcc_RMSE": lcc_rmse_v, "VAL_lcc_MAE": lcc_mae_v,
                "TEST_lcc_R2": lcc_r2_t, "TEST_lcc_RMSE": lcc_rmse_t, "TEST_lcc_MAE": lcc_mae_t,
                "VAL_state_R2": s_r2_v, "VAL_state_RMSE": s_rmse_v, "VAL_state_MAE": s_mae_v,
                "TEST_state_R2": s_r2_t, "TEST_state_RMSE": s_rmse_t, "TEST_state_MAE": s_mae_t,
                "best_ckpt": os.path.basename(best_path),
            })
            per_var_rows.append(eval_pred_per_variable_test(model, kind, S_norm, test_rng, n=160_000))

    pd.DataFrame(pred_metrics).to_csv(os.path.join(STAGE_2_DIR, "predictors_metrics.csv"), index=False)
    pd.concat(per_var_rows, axis=0, ignore_index=True).to_csv(
        os.path.join(STAGE_2_DIR, "predictors_per_variable_test_metrics.csv"), index=False
    )

    print("Stage 2 completed:", STAGE_2_DIR)


# ======================================================================================
# STAGE 3 — CONSTRAINED DECISION-MAKING
# ======================================================================================

GLOBAL_POLICIES = {}

def load_stage_data_for_decision():
    S_norm = np.load(os.path.join(STAGE_1_DIR, "environment_tensor_normalized.npy"))
    ij = np.load(os.path.join(STAGE_1_DIR, "grid_indices.npy"))
    with open(os.path.join(STAGE_1_DIR, "data_processing_info.json"), "r", encoding="utf-8") as f:
        info = json.load(f)
    train_rng = tuple(info["train_range"])
    val_rng = tuple(info["val_range"])
    test_rng = tuple(info["test_range"])
    Ny, Nx = info["Ny"], info["Nx"]
    return S_norm, ij, Ny, Nx, train_rng, val_rng, test_rng

def load_predictor(kind):
    sdim = len(STATE_VARS)
    if kind == "TWIN":
        model = TwinMLP(sdim, hid=256).to(DEVICE)
    elif kind == "GRU":
        model = GRUDynamics(sdim, hid=128).to(DEVICE)
    elif kind == "WM":
        model = StochWM(sdim, ACT_DIM, latent=48, hid=256).to(DEVICE)
    else:
        raise ValueError(kind)
    ck = torch.load(os.path.join(STAGE_2_DIR, f"{kind}_best.pt"), map_location=DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model

def effect_delta_lcc(s_norm, a01):
    tcwv = s_norm[:, IDX_TCWV:IDX_TCWV + 1]
    wspd = s_norm[:, IDX_WSPD:IDX_WSPD + 1]
    gate_moist = torch.sigmoid(3.0 * (tcwv - 0.0))
    gate_wind = torch.sigmoid(2.0 * (1.0 - torch.abs(wspd)))
    gate = gate_moist * gate_wind
    delta = (a01 * gate) * DELTA_LCC_MAX
    delta = torch.clamp(delta, 0.0, DELTA_LCC_MAX)
    return delta, gate

@torch.no_grad()
def sim_step_predictor(model, kind, s_t, s_tm1=None, a01=None):
    if a01 is None:
        a01 = torch.zeros((s_t.shape[0], 1), device=DEVICE)

    if kind == "TWIN":
        sp_base = model(s_t)
    elif kind == "GRU":
        sp_base = model(s_tm1, s_t) if s_tm1 is not None else model(s_t, s_t)
    elif kind == "WM":
        mu_z, ls_z = model.posterior_params(s_t)
        z = reparam(mu_z, ls_z)
        mu_p, ls_p = model.prior_params(z, a01)
        z_p = reparam(mu_p, ls_p)
        sp_base = model.decode(z_p)
    else:
        raise ValueError(kind)

    delta, gate = effect_delta_lcc(s_t, a01)
    sp = sp_base.clone()
    sp[:, IDX_LCC:IDX_LCC + 1] = torch.clamp(sp[:, IDX_LCC:IDX_LCC + 1] + delta, -6.0, 6.0)

    lcc_next = sp[:, IDX_LCC:IDX_LCC + 1]
    r = lcc_next - ACTION_COST * (a01 * a01)
    r = r - (a01 * (gate < NO_GO_GATE).float()) * 0.05
    return sp, r.squeeze(1), gate.squeeze(1)

def mpc_cem_action(model, kind, s0, s_prev=None):
    mu, sig = 0.5, 0.35
    for _ in range(CEM_ITERS):
        A = np.clip(np.random.normal(mu, sig, size=(CEM_POP, CEM_H, 1)), 0.0, 1.0).astype(np.float32)
        with torch.no_grad():
            s = s0.repeat(CEM_POP, 1)
            spv = s_prev.repeat(CEM_POP, 1) if s_prev is not None else None
            ret = torch.zeros((CEM_POP,), device=DEVICE)
            for h in range(CEM_H):
                a = torch.from_numpy(A[:, h]).to(DEVICE)
                s, r, _ = sim_step_predictor(model, kind, s, spv, a)
                spv = s
                ret += r
        ret_np = ret.detach().cpu().numpy()
        elite = A[np.argsort(ret_np)[-CEM_ELITE:]]
        mu = float(np.mean(elite[:, 0, 0]))
        sig = float(np.std(elite[:, 0, 0]) + 1e-6)
    return float(np.clip(mu, 0.0, 1.0))

class Actor(nn.Module):
    def __init__(self, sdim=8, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sdim, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU()
        )
        self.mu = nn.Linear(hid, ACT_DIM)
        self.logstd = nn.Linear(hid, ACT_DIM)

    def forward(self, s):
        h = self.net(s)
        mu = self.mu(h)
        ls = torch.clamp(self.logstd(h), -5, 2)
        return mu, ls

    def sample(self, s):
        mu, ls = self(s)
        std = torch.exp(ls)
        z = mu + std * torch.randn_like(mu)
        a = torch.sigmoid(z)
        logp = -0.5 * (((z - mu) / (std + 1e-8)) ** 2 + 2 * ls + math.log(2 * math.pi))
        logp = logp.sum(dim=1, keepdim=True)
        corr = torch.log(a * (1 - a) + 1e-8).sum(dim=1, keepdim=True)
        logp = logp + corr
        return a, logp

class Critic(nn.Module):
    def __init__(self, sdim=8, hid=256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(sdim + ACT_DIM, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, 1)
        )

    def forward(self, s, a):
        return self.q(torch.cat([s, a], dim=1))

class Replay:
    def __init__(self, cap=400000, sdim=8):
        self.cap = int(cap)
        self.s = np.zeros((cap, sdim), np.float32)
        self.a = np.zeros((cap, ACT_DIM), np.float32)
        self.r = np.zeros((cap,), np.float32)
        self.sp = np.zeros((cap, sdim), np.float32)
        self.d = np.zeros((cap,), np.float32)
        self.n = 0
        self.i = 0

    def add(self, s, a, r, sp, d):
        self.s[self.i] = s
        self.a[self.i] = a
        self.r[self.i] = r
        self.sp[self.i] = sp
        self.d[self.i] = d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs):
        idx = np.random.randint(0, self.n, size=bs)
        return (
            torch.from_numpy(self.s[idx]).to(DEVICE),
            torch.from_numpy(self.a[idx]).to(DEVICE),
            torch.from_numpy(self.r[idx]).to(DEVICE),
            torch.from_numpy(self.sp[idx]).to(DEVICE),
            torch.from_numpy(self.d[idx]).to(DEVICE),
        )

@torch.no_grad()
def soft_update(tgt, src, tau):
    for pt, p in zip(tgt.parameters(), src.parameters()):
        pt.data.mul_(1.0 - tau).add_(tau * p.data)

def train_sac_like(model, kind, S_norm, train_rng, algo_name="REDQ"):
    sdim = S_norm.shape[-1]
    actor = Actor(sdim).to(DEVICE)
    q1, q2 = Critic(sdim).to(DEVICE), Critic(sdim).to(DEVICE)
    q1t, q2t = Critic(sdim).to(DEVICE), Critic(sdim).to(DEVICE)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())

    optA = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    optQ = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=CRITIC_LR)

    rb = Replay(sdim=sdim)
    a_rng, b_rng = train_rng
    T, Pk, _ = S_norm.shape

    warmup_n = 3500
    for _ in range(warmup_n):
        t = np.random.randint(max(a_rng, 1), min(b_rng, T - 2) + 1)
        p = np.random.randint(0, Pk)
        s0 = S_norm[t, p]
        s1 = S_norm[t - 1, p]
        s_t = torch.from_numpy(s0[None]).to(DEVICE)
        s_tm1 = torch.from_numpy(s1[None]).to(DEVICE)
        a0 = torch.rand((1, ACT_DIM), device=DEVICE)
        sp, r, _ = sim_step_predictor(model, kind, s_t, s_tm1, a0)
        rb.add(s0, a0.squeeze(0).cpu().numpy(), float(r.item()), sp.squeeze(0).cpu().numpy(), 0.0)

    t_start = time.time()
    ema_avg_r = 0.0
    extra_q_updates = 2 if algo_name == "REDQ" else 1

    for step in range(1, RL_STEPS + 1):
        with torch.no_grad():
            t = np.random.randint(max(a_rng, 1), min(b_rng, T - 2) + 1)
            p = np.random.randint(0, Pk)
            s = torch.from_numpy(S_norm[t, p][None]).to(DEVICE)
            s_tm1 = torch.from_numpy(S_norm[t - 1, p][None]).to(DEVICE)

            for _ in range(ROLLOUT_H):
                a_samp, _ = actor.sample(s)
                sp, r, _ = sim_step_predictor(model, kind, s, s_tm1, a_samp)
                rb.add(
                    s.squeeze(0).cpu().numpy(),
                    a_samp.squeeze(0).cpu().numpy(),
                    float(r.item()),
                    sp.squeeze(0).cpu().numpy(),
                    0.0
                )
                s_tm1 = s
                s = sp

        for _ in range(extra_q_updates):
            s, a_t, r, sp, d = rb.sample(RL_BATCH)
            r = r.unsqueeze(1)
            d = d.unsqueeze(1)
            r_sc = r * REWARD_SCALE

            with torch.no_grad():
                a2, logp2 = actor.sample(sp)
                qn = torch.min(q1t(sp, a2), q2t(sp, a2)) - ALPHA * logp2
                qn = torch.clamp(qn, -Q_TARGET_CLAMP, Q_TARGET_CLAMP)
                y = r_sc + (1 - d) * GAMMA * qn
                y = torch.clamp(y, -Q_TARGET_CLAMP, Q_TARGET_CLAMP)

            q1v = q1(s, a_t)
            q2v = q2(s, a_t)
            lossQ = F.smooth_l1_loss(q1v, y) + F.smooth_l1_loss(q2v, y)

            optQ.zero_grad(set_to_none=True)
            lossQ.backward()
            torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), CRITIC_GRAD_CLIP)
            optQ.step()

        s, a_t, r, sp, d = rb.sample(RL_BATCH)
        a_new, logp = actor.sample(s)
        qa = torch.min(q1(s, a_new), q2(s, a_new))
        if algo_name == "TQC":
            qa = torch.clamp(qa, -Q_TARGET_CLAMP, Q_TARGET_CLAMP)
        lossA = (ALPHA * logp - qa).mean()

        optA.zero_grad(set_to_none=True)
        lossA.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), ACTOR_GRAD_CLIP)
        optA.step()

        soft_update(q1t, q1, TAU)
        soft_update(q2t, q2, TAU)

        ema_avg_r = 0.99 * ema_avg_r + 0.01 * float(r.mean().item())

        if step % 500 == 0:
            dt = time.time() - t_start
            print(f"[{algo_name}] step {step:06d} | lossQ={lossQ.item():.4f} lossA={lossA.item():.4f} | ema_avg_r={ema_avg_r:.4f} | {dt:.1f}s")

    return actor

@torch.no_grad()
def run_policy_episode(model, kind, policy_name, policy_obj, s0, s0m1):
    s = s0.clone()
    sm1 = s0m1.clone()
    ret, ben, energy = 0.0, 0.0, 0.0
    viol = 0
    actions, gates, deltas = [], [], []

    for _ in range(EP_LEN):
        if policy_name == "MPC_CEM":
            a = mpc_cem_action(model, kind, s, sm1)
            a_t = torch.tensor([[a]], device=DEVICE, dtype=torch.float32)
        else:
            a_t, _ = policy_obj.sample(s)

        delta_probe, gate_probe = effect_delta_lcc(s, a_t)
        sp, r, gate = sim_step_predictor(model, kind, s, sm1, a_t)
        sp0, r0, _ = sim_step_predictor(model, kind, s, sm1, torch.zeros_like(a_t))

        ret += float(r.item())
        ben += float((r - r0).item())
        energy += float((a_t * a_t).item())
        if float(a_t.item()) > 1e-6 and float(gate.item()) < NO_GO_GATE:
            viol += 1

        actions.append(float(a_t.item()))
        gates.append(float(gate_probe.item()))
        deltas.append(float(delta_probe.item()))

        sm1 = s
        s = sp

    return ret, ben, energy, viol, actions, gates, deltas

@torch.no_grad()
def reward_action_probe(model, kind, S_norm, t_rng, n_samples=N_DIAG_REWARD_SAMPLES):
    a, b = t_rng
    T, Pk, _ = S_norm.shape
    ts = np.random.randint(max(a, 1), min(b, T - 2) + 1, size=n_samples)
    ps = np.random.randint(0, Pk, size=n_samples)

    s_t = torch.from_numpy(S_norm[ts, ps]).to(DEVICE)
    s_tm1 = torch.from_numpy(S_norm[ts - 1, ps]).to(DEVICE)

    rows = []
    for aval in DIAG_ACTION_GRID:
        a01 = torch.full((n_samples, 1), float(aval), device=DEVICE)
        _, r, _ = sim_step_predictor(model, kind, s_t, s_tm1, a01)
        delta, gate = effect_delta_lcc(s_t, a01)
        rows.append({
            "predictor": kind,
            "action_probe": float(aval),
            "reward_mean": float(r.mean().item()),
            "reward_median": float(torch.median(r).item()),
            "reward_p10": float(torch.quantile(r, 0.10).item()),
            "reward_p90": float(torch.quantile(r, 0.90).item()),
            "gate_mean": float(gate.mean().item()),
            "gate_p10": float(torch.quantile(gate, 0.10).item()),
            "gate_p90": float(torch.quantile(gate, 0.90).item()),
            "delta_mean": float(delta.mean().item()),
            "delta_p90": float(torch.quantile(delta, 0.90).item()),
        })
    return pd.DataFrame(rows)

@torch.no_grad()
def eval_deciders_for_predictor(model, kind, S_norm, test_rng):
    a, b = test_rng
    T, Pk, _ = S_norm.shape

    results = []
    usage_rows = []

    policies = {
        "REDQ": GLOBAL_POLICIES[(kind, "REDQ")],
        "TQC": GLOBAL_POLICIES[(kind, "TQC")]
    }

    for pol_name in ["MPC_CEM", "REDQ", "TQC"]:
        rets, bens, enes, vios = [], [], [], []
        mean_actions, mean_gates, mean_deltas = [], [], []

        for _ in range(N_EVAL_EPISODES):
            t = np.random.randint(max(a, 1), min(b, T - EP_LEN - 2) + 1)
            p = np.random.randint(0, Pk)
            s0 = torch.from_numpy(S_norm[t, p][None]).to(DEVICE)
            s0m1 = torch.from_numpy(S_norm[t - 1, p][None]).to(DEVICE)

            pobj = None if pol_name == "MPC_CEM" else policies[pol_name]
            ret, ben, ene, vio, actions, gates, deltas = run_policy_episode(model, kind, pol_name, pobj, s0, s0m1)

            rets.append(ret)
            bens.append(ben)
            enes.append(ene)
            vios.append(vio / EP_LEN)
            mean_actions.append(float(np.mean(actions)))
            mean_gates.append(float(np.mean(gates)))
            mean_deltas.append(float(np.mean(deltas)))

        rets = np.asarray(rets)
        bens = np.asarray(bens)
        enes = np.asarray(enes)
        vios = np.asarray(vios)
        mean_actions = np.asarray(mean_actions)
        mean_gates = np.asarray(mean_gates)
        mean_deltas = np.asarray(mean_deltas)

        ben_ci_lo, ben_ci_hi = bootstrap_ci(bens / EP_LEN)
        vio_ci_lo, vio_ci_hi = bootstrap_ci(vios)
        bpk = (bens / EP_LEN) / (enes / EP_LEN + 1e-8)
        bpk_ci_lo, bpk_ci_hi = bootstrap_ci(bpk)

        results.append({
            "predictor": kind,
            "decider": pol_name,
            "return_total_mean": float(np.mean(rets)),
            "return_total_std": float(np.std(rets)),
            "benefit_mean_mean": float(np.mean(bens / EP_LEN)),
            "benefit_mean_std": float(np.std(bens / EP_LEN)),
            "benefit_mean_median": float(np.median(bens / EP_LEN)),
            "benefit_mean_p10": float(np.quantile(bens / EP_LEN, 0.10)),
            "benefit_mean_p90": float(np.quantile(bens / EP_LEN, 0.90)),
            "benefit_mean_mean_ci_lo": ben_ci_lo,
            "benefit_mean_mean_ci_hi": ben_ci_hi,
            "energy_mean": float(np.mean(enes / EP_LEN)),
            "benefit_per_kwh_mean": float(np.mean(bpk)),
            "benefit_per_kwh_p90": float(np.quantile(bpk, 0.90)),
            "benefit_per_kwh_mean_ci_lo": bpk_ci_lo,
            "benefit_per_kwh_mean_ci_hi": bpk_ci_hi,
            "violation_rate_mean": float(np.mean(vios)),
            "violation_rate_std": float(np.std(vios)),
            "violation_rate_median": float(np.median(vios)),
            "violation_rate_p10": float(np.quantile(vios, 0.10)),
            "violation_rate_p90": float(np.quantile(vios, 0.90)),
            "violation_rate_mean_ci_lo": vio_ci_lo,
            "violation_rate_mean_ci_hi": vio_ci_hi,
            "action_mean": float(np.mean(mean_actions)),
            "action_median": float(np.median(mean_actions)),
            "action_p90": float(np.quantile(mean_actions, 0.90)),
            "gate_mean": float(np.mean(mean_gates)),
            "gate_p10": float(np.quantile(mean_gates, 0.10)),
            "delta_lcc_mean": float(np.mean(mean_deltas)),
            "delta_lcc_p90": float(np.quantile(mean_deltas, 0.90)),
            "collapse_no_action_frac": float(np.mean(mean_actions < 1e-3)),
        })

        usage_rows.append(pd.DataFrame({
            "predictor": [kind] * len(mean_actions),
            "decider": [pol_name] * len(mean_actions),
            "episode_mean_action": mean_actions,
            "episode_mean_gate": mean_gates,
            "episode_mean_delta_lcc": mean_deltas,
            "episode_violation_rate": vios,
            "episode_benefit_mean": bens / EP_LEN,
            "episode_energy_mean": enes / EP_LEN,
        }))

    return pd.DataFrame(results), pd.concat(usage_rows, axis=0, ignore_index=True)

def run_stage_3():
    print("\n=== Stage 3 — Constrained decision-making ===")
    S_norm, _, _, _, train_rng, _, test_rng = load_stage_data_for_decision()

    pred_probe_rows = []
    all_decision_rows = []
    all_usage_rows = []

    predictors = []
    for kind in ["TWIN", "GRU", "WM"]:
        predictors.append((kind, load_predictor(kind)))

    for kind, model in predictors:
        print(f"\nTraining decision strategies for predictor {kind}")
        actor_redq = train_sac_like(model, kind, S_norm, train_rng, algo_name="REDQ")
        actor_tqc = train_sac_like(model, kind, S_norm, train_rng, algo_name="TQC")

        GLOBAL_POLICIES[(kind, "REDQ")] = actor_redq
        GLOBAL_POLICIES[(kind, "TQC")] = actor_tqc

        df_probe = reward_action_probe(model, kind, S_norm, test_rng)
        pred_probe_rows.append(df_probe)

        df_dec, df_usage = eval_deciders_for_predictor(model, kind, S_norm, test_rng)
        all_decision_rows.append(df_dec)
        all_usage_rows.append(df_usage)

    pd.concat(pred_probe_rows, axis=0, ignore_index=True).to_csv(
        os.path.join(STAGE_3_DIR, "reward_action_probe.csv"), index=False
    )
    pd.concat(all_decision_rows, axis=0, ignore_index=True).to_csv(
        os.path.join(STAGE_3_DIR, "decision_metrics.csv"), index=False
    )
    pd.concat(all_usage_rows, axis=0, ignore_index=True).to_csv(
        os.path.join(STAGE_3_DIR, "rl_stability_and_action_usage.csv"), index=False
    )

    print("Stage 3 completed:", STAGE_3_DIR)


# ======================================================================================
# STAGE 4 — DEPLOYMENT-ORIENTED AUDIT
# ======================================================================================

def run_stage_4():
    print("\n=== Stage 4 — Deployment-oriented audit ===")
    fig_dir = os.path.join(STAGE_4_DIR, "figures")
    csv_dir = os.path.join(STAGE_4_DIR, "csv")
    ensure_dir(fig_dir)
    ensure_dir(csv_dir)

    df_pred = safe_read_csv(os.path.join(STAGE_2_DIR, "predictors_metrics.csv"))
    df_pervar = safe_read_csv(os.path.join(STAGE_2_DIR, "predictors_per_variable_test_metrics.csv"))
    df_dec = safe_read_csv(os.path.join(STAGE_3_DIR, "decision_metrics.csv"))
    df_usage = safe_read_csv(os.path.join(STAGE_3_DIR, "rl_stability_and_action_usage.csv"))
    df_probe = safe_read_csv(os.path.join(STAGE_3_DIR, "reward_action_probe.csv"))

    if df_pred is not None:
        for c in [
            "predictor", "VAL_lcc_R2", "VAL_lcc_RMSE", "VAL_lcc_MAE",
            "TEST_lcc_R2", "TEST_lcc_RMSE", "TEST_lcc_MAE",
            "VAL_state_R2", "VAL_state_RMSE", "VAL_state_MAE",
            "TEST_state_R2", "TEST_state_RMSE", "TEST_state_MAE", "best_ckpt"
        ]:
            if c not in df_pred.columns:
                df_pred[c] = np.nan

        df_pred["gap_lcc_R2_VAL_minus_TEST"] = df_pred["VAL_lcc_R2"] - df_pred["TEST_lcc_R2"]
        df_pred["gap_state_R2_VAL_minus_TEST"] = df_pred["VAL_state_R2"] - df_pred["TEST_state_R2"]
        df_pred["flag_overfit_lcc"] = df_pred["gap_lcc_R2_VAL_minus_TEST"] > OVERFIT_GAP_THRESHOLD_R2
        df_pred["flag_overfit_state"] = df_pred["gap_state_R2_VAL_minus_TEST"] > OVERFIT_GAP_THRESHOLD_R2

        s_lcc_r2 = minmax_series(df_pred["TEST_lcc_R2"], invert=False)
        s_lcc_rmse = minmax_series(df_pred["TEST_lcc_RMSE"], invert=True)
        s_lcc_mae = minmax_series(df_pred["TEST_lcc_MAE"], invert=True)
        s_state_r2 = minmax_series(df_pred["TEST_state_R2"], invert=False)
        s_state_rmse = minmax_series(df_pred["TEST_state_RMSE"], invert=True)
        s_state_mae = minmax_series(df_pred["TEST_state_MAE"], invert=True)

        df_pred["score_test_predictor_raw"] = (
            W_LCC_R2 * s_lcc_r2 +
            W_LCC_RMSE * s_lcc_rmse +
            W_LCC_MAE * s_lcc_mae +
            W_STATE_R2 * s_state_r2 +
            W_STATE_RMSE * s_state_rmse +
            W_STATE_MAE * s_state_mae
        )
        df_pred["score_penalty_overfit"] = 0.0
        df_pred.loc[df_pred["flag_overfit_lcc"], "score_penalty_overfit"] += OVERFIT_PENALTY
        df_pred.loc[df_pred["flag_overfit_state"], "score_penalty_overfit"] += OVERFIT_PENALTY * 0.5
        df_pred["score_test_predictor_final"] = df_pred["score_test_predictor_raw"] - df_pred["score_penalty_overfit"]
        df_pred["rank_predictor_test"] = df_pred["score_test_predictor_final"].rank(ascending=False, method="dense").astype(int)

        df_pred_audit = df_pred.sort_values(["rank_predictor_test", "TEST_lcc_RMSE"], ascending=[True, True]).reset_index(drop=True)
        save_df_csv(df_pred_audit, os.path.join(csv_dir, "audit_predictors_test_focused.csv"))

    if df_dec is not None:
        for c in [
            "predictor", "decider", "return_total_mean", "return_total_std",
            "TEST_lcc_R2", "TEST_lcc_RMSE", "TEST_lcc_MAE",
            "TEST_state_R2", "TEST_state_RMSE", "TEST_state_MAE",
            "VAL_lcc_R2", "VAL_lcc_RMSE", "VAL_lcc_MAE",
            "VAL_state_R2", "VAL_state_RMSE", "VAL_state_MAE"
        ]:
            if c not in df_dec.columns:
                df_dec[c] = np.nan

        if df_usage is not None:
            usage_cols = [c for c in [
                "predictor", "decider", "episodes", "mean_action", "median_action",
                "mean_gate", "mean_delta_lcc", "mean_benefit", "mean_violation_rate"
            ] if c in df_usage.columns]
            if set(["predictor", "decider"]).issubset(usage_cols):
                df_dec = df_dec.merge(df_usage[usage_cols], on=["predictor", "decider"], how="left")

        s_ret_mean = minmax_series(df_dec["return_total_mean"], invert=False)
        s_ret_std = minmax_series(df_dec["return_total_std"], invert=True)
        s_lcc_r2 = minmax_series(df_dec["TEST_lcc_R2"], invert=False)
        s_lcc_rmse = minmax_series(df_dec["TEST_lcc_RMSE"], invert=True)
        s_state_r2 = minmax_series(df_dec["TEST_state_R2"], invert=False)
        s_state_rmse = minmax_series(df_dec["TEST_state_RMSE"], invert=True)
        s_viol = minmax_series(df_dec["mean_violation_rate"], invert=True) if "mean_violation_rate" in df_dec.columns else pd.Series(np.zeros(len(df_dec)), index=df_dec.index)

        df_dec["score_test_decider"] = (
            W_RET_MEAN * s_ret_mean +
            W_RET_STD * s_ret_std +
            W_TEST_LCC_R2 * s_lcc_r2 +
            W_TEST_LCC_RMSE * s_lcc_rmse +
            W_TEST_STATE_R2 * s_state_r2 +
            W_TEST_STATE_RMSE * s_state_rmse +
            W_VIOL * s_viol
        )
        df_dec["rank_decider_global_test"] = df_dec["score_test_decider"].rank(ascending=False, method="dense").astype(int)
        df_dec["rank_decider_within_predictor_test"] = (
            df_dec.groupby("predictor")["score_test_decider"].rank(ascending=False, method="dense").astype(int)
        )

        df_dec_audit = df_dec.sort_values(
            ["rank_decider_global_test", "predictor", "rank_decider_within_predictor_test"],
            ascending=[True, True, True]
        ).reset_index(drop=True)
        save_df_csv(df_dec_audit, os.path.join(csv_dir, "audit_deciders_test_focused_global.csv"))

        df_best = (
            df_dec_audit.sort_values(["predictor", "rank_decider_within_predictor_test", "rank_decider_global_test"])
            .groupby("predictor", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        save_df_csv(df_best, os.path.join(csv_dir, "audit_best_decider_per_predictor_test.csv"))

    if df_probe is not None:
        save_df_csv(df_probe, os.path.join(csv_dir, "reward_action_probe_summary.csv"))
    if df_pervar is not None:
        save_df_csv(df_pervar, os.path.join(csv_dir, "predictors_per_variable_test_metrics_copy.csv"))

    print("Stage 4 completed:", STAGE_4_DIR)


# ======================================================================================
# STAGE 5 — SPATIAL LINKAGE AND CONTINUOUS PRIORITIZATION
# ======================================================================================

CANDIDATES = {
    "intensity_mean": ["intensity_mean", "mean_intensity", "action_mean", "mean_action", "avg_action"],
    "violation_rate": ["violation_rate", "mean_violation_rate", "viol_rate", "risk_rate"],
}

def build_continuous_spray_01(df: pd.DataFrame, intensity_col: Optional[str], violation_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()

    if intensity_col is not None and intensity_col in out.columns:
        x = to_numeric_safe(out[intensity_col])
        finite = np.isfinite(x.values)
        if finite.sum() > 0:
            xmin = np.nanmin(x.values[finite])
            xmax = np.nanmax(x.values[finite])
            if xmin >= -0.05 and xmax <= 1.05:
                spray01 = clip01(x.fillna(0.0).values)
                source = f"direct_clip_from_{intensity_col}"
            elif xmin >= -1.05 and xmax <= 1.05:
                spray01 = clip01((x.fillna(0.0).values + 1.0) / 2.0)
                source = f"rescaled_minus1_1_from_{intensity_col}"
            else:
                spray01 = robust01(x).fillna(0.0).values
                source = f"robust01_from_{intensity_col}"
        else:
            spray01 = np.zeros(len(out))
            source = "zeros_no_finite_intensity"
    else:
        spray01 = np.zeros(len(out))
        source = "zeros_no_intensity"

    out["spray_01_continuous"] = clip01(spray01).astype(float)
    out["spray_01_binary_threshold_05"] = (out["spray_01_continuous"] >= 0.5).astype(int)

    if violation_col is not None and violation_col in out.columns:
        v = to_numeric_safe(out[violation_col]).fillna(0.0)
        v = clip01(v.values)
        out["violation_rate"] = v
        out["spray_01_safe"] = clip01(out["spray_01_continuous"].values * (1.0 - v))
    else:
        out["violation_rate"] = 0.0
        out["spray_01_safe"] = out["spray_01_continuous"]

    out["spray_effective_proxy_liters_continuous"] = out["spray_01_safe"] * MAX_SPRAY_L_PER_POINT
    out["spray_proxy_liters_raw_continuous"] = out["spray_01_continuous"] * MAX_SPRAY_L_PER_POINT
    out["spray_source_rule"] = source
    return out

def run_stage_5():
    print("\n=== Stage 5 — Spatial linkage and continuous prioritization ===")
    csv_out = os.path.join(STAGE_5_DIR, "csv")
    fig_out = os.path.join(STAGE_5_DIR, "figures")
    ensure_dir(csv_out)
    ensure_dir(fig_out)

    df_pred = safe_read_csv(os.path.join(STAGE_4_DIR, "csv", "audit_predictors_test_focused.csv"))
    df_best = safe_read_csv(os.path.join(STAGE_4_DIR, "csv", "audit_best_decider_per_predictor_test.csv"))
    df_decg = safe_read_csv(os.path.join(STAGE_4_DIR, "csv", "audit_deciders_test_focused_global.csv"))
    df_decision = safe_read_csv(os.path.join(STAGE_3_DIR, "decision_metrics.csv"))

    if df_decision is None:
        raise RuntimeError("decision_metrics.csv is missing.")

    intensity_col = pick_existing_col(df_decision, CANDIDATES["intensity_mean"])
    violation_col = pick_existing_col(df_decision, CANDIDATES["violation_rate"])

    df_points = build_continuous_spray_01(df_decision, intensity_col, violation_col)

    if df_pred is not None:
        keep_pred_cols = [c for c in ["predictor", "TEST_lcc_R2", "TEST_lcc_RMSE", "TEST_lcc_MAE", "score_test_predictor_final"] if c in df_pred.columns]
        df_points = df_points.merge(df_pred[keep_pred_cols], on="predictor", how="left")

    if df_decg is not None:
        keep_dec_cols = [c for c in ["predictor", "decider", "return_total_mean", "score_test_decider"] if c in df_decg.columns]
        df_points = df_points.merge(df_decg[keep_dec_cols], on=["predictor", "decider"], how="left")

    if "benefit_mean" not in df_points.columns and "benefit_mean_mean" in df_points.columns:
        df_points["benefit_mean"] = df_points["benefit_mean_mean"]

    benefit_series = df_points["benefit_mean"] if "benefit_mean" in df_points.columns else pd.Series(np.zeros(len(df_points)))
    violation_series = pd.to_numeric(df_points["violation_rate"], errors="coerce").fillna(0.0)

    df_points["point_operational_score_proxy_continuous"] = clip01(
        0.45 * robust01(benefit_series).fillna(0.0).values +
        0.40 * pd.to_numeric(df_points["spray_01_safe"], errors="coerce").fillna(0.0).values +
        0.15 * (1.0 - violation_series.values)
    )

    df_points["score_linkage_spray_proxy_point_context"] = (
        0.35 * pd.to_numeric(df_points.get("score_test_predictor_final", 0.0), errors="coerce").fillna(0.0) +
        0.25 * pd.to_numeric(df_points.get("score_test_decider", 0.0), errors="coerce").fillna(0.0) +
        0.25 * pd.to_numeric(df_points["point_operational_score_proxy_continuous"], errors="coerce").fillna(0.0) +
        0.15 * pd.to_numeric(df_points["spray_01_safe"], errors="coerce").fillna(0.0)
    )

    save_df_csv(df_points, os.path.join(csv_out, "spatial_linkage_all_combinations.csv"))

    df_combo = df_points.groupby(["predictor", "decider"], dropna=False).agg(
        n_points=("predictor", "size"),
        spray_01_mean=("spray_01_continuous", "mean"),
        spray_01_safe_mean=("spray_01_safe", "mean"),
        spray_effective_proxy_liters_sum=("spray_effective_proxy_liters_continuous", "sum"),
        benefit_mean_global=("benefit_mean", "mean"),
        violation_rate_mean_global=("violation_rate", "mean"),
        point_operational_score_proxy_mean=("point_operational_score_proxy_continuous", "mean"),
        TEST_lcc_R2=("TEST_lcc_R2", "mean"),
        TEST_lcc_RMSE=("TEST_lcc_RMSE", "mean"),
        return_total_mean=("return_total_mean", "mean"),
        score_test_predictor_final=("score_test_predictor_final", "mean"),
        score_test_decider=("score_test_decider", "mean"),
    ).reset_index()

    df_combo["score_linkage_spray_proxy_continuous"] = (
        0.30 * minmax_series(df_combo["TEST_lcc_R2"], invert=False).fillna(0.0) +
        0.20 * minmax_series(df_combo["return_total_mean"], invert=False).fillna(0.0) +
        0.20 * minmax_series(df_combo["spray_01_safe_mean"], invert=False).fillna(0.0) +
        0.15 * minmax_series(df_combo["point_operational_score_proxy_mean"], invert=False).fillna(0.0) +
        0.15 * minmax_series(df_combo["violation_rate_mean_global"], invert=True).fillna(0.0)
    )
    df_combo = df_combo.sort_values("score_linkage_spray_proxy_continuous", ascending=False).reset_index(drop=True)
    df_combo["rank_linkage_spray_proxy_continuous"] = np.arange(1, len(df_combo) + 1)

    save_df_csv(df_combo, os.path.join(csv_out, "spatial_linkage_summary_by_combination.csv"))

    if plt is not None:
        plt.figure(figsize=(12, 5))
        top = df_combo.head(12).copy()
        labels = [f"{r.predictor}-{r.decider}" for r in top.itertuples()]
        vals = top["score_linkage_spray_proxy_continuous"].values
        plt.bar(range(len(top)), vals)
        plt.xticks(range(len(top)), labels, rotation=45, ha="right")
        plt.ylabel("Composite spatial linkage and prioritization score")
        plt.title("Ranking of predictor-decider combinations")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_out, "combination_ranking.png"), dpi=300)
        plt.close()

    print("Stage 5 completed:", STAGE_5_DIR)


# ======================================================================================
# STAGE 6 — SPATIOTEMPORAL CLUSTERING AND OPERATIONAL ROUTING
# ======================================================================================

@dataclass
class USVPlatform:
    usv_id: str
    platform_name: str
    cruise_speed_kmh: float
    max_endurance_min: float
    reserve_min: float
    spray_rate_lpm: float
    agent_capacity_l: float
    setup_spray_min: float
    shutdown_spray_min: float
    max_nav_wind_ms: float
    max_spray_wind_ms: float
    turnaround_time_min: float
    base_id: Optional[str] = None

def setup_logging_stage_6():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(LOG_PATH_STAGE_6, mode="w", encoding="utf-8"), logging.StreamHandler()]
    )

def safe_read_yaml(path: str, default_obj: dict) -> dict:
    if os.path.exists(path) and (yaml is not None):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = yaml.safe_load(f)
            return obj if obj is not None else default_obj
        except Exception:
            return default_obj
    return default_obj

def dump_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    def _default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)): return o.tolist()
        if isinstance(o, (pd.Timestamp,)): return o.isoformat()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)

def write_default_configs_stage_6():
    if yaml is not None:
        with open(PATH_USV_PLATFORMS, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_USV_PLATFORMS, f, sort_keys=False, allow_unicode=True)
        with open(PATH_MISSION_POLICY, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_MISSION_POLICY, f, sort_keys=False, allow_unicode=True)
    DEFAULT_PORTS_BASES.to_csv(PATH_PORTS_BASES, index=False)

def load_platforms_stage_6() -> List[USVPlatform]:
    raw = safe_read_yaml(PATH_USV_PLATFORMS, DEFAULT_USV_PLATFORMS)
    plats = raw.get("platforms", [])
    base_def = DEFAULT_USV_PLATFORMS["platforms"][0].copy()
    out = []
    for p in plats:
        merged = {**base_def, **p}
        allowed = set(USVPlatform.__annotations__.keys())
        merged = {k: v for k, v in merged.items() if k in allowed}
        out.append(USVPlatform(**merged))
    if not out:
        out = [USVPlatform(**base_def)]
    return out

def load_policy_stage_6() -> dict:
    raw = safe_read_yaml(PATH_MISSION_POLICY, DEFAULT_MISSION_POLICY)
    return raw.get("policy", DEFAULT_MISSION_POLICY["policy"])

def build_selected_opportunities_stage_6(policy: dict) -> pd.DataFrame:
    df = safe_read_csv(os.path.join(STAGE_5_DIR, "csv", "spatial_linkage_all_combinations.csv"))
    if df is None or df.empty:
        raise RuntimeError("Missing spatial_linkage_all_combinations.csv")
    if "score_linkage_spray_proxy_point_context" not in df.columns:
        raise RuntimeError("Expected score_linkage_spray_proxy_point_context column.")
    if "lat" not in df.columns or "lon" not in df.columns:
        # fallback pseudo-grid coordinates if spatial lat/lon are not available
        if "iy" in df.columns and "ix" in df.columns:
            df["lat"] = pd.to_numeric(df["iy"], errors="coerce")
            df["lon"] = pd.to_numeric(df["ix"], errors="coerce")
        else:
            raise RuntimeError("Expected lat/lon or iy/ix columns.")

    score_min = float(policy.get("priority_score_min", 0.5))
    d = df[pd.to_numeric(df["score_linkage_spray_proxy_point_context"], errors="coerce").fillna(0.0) >= score_min].copy()
    d["priority_score"] = pd.to_numeric(d["score_linkage_spray_proxy_point_context"], errors="coerce").fillna(0.0)
    d["decision_bin"] = 1
    d["time_start"] = pd.Timestamp("2025-01-01 00:00:00")
    d["time_end"] = pd.Timestamp("2025-01-01 01:30:00")
    d["safety_ok"] = True
    save_df_csv(d, os.path.join(OUT_DIR_STAGE_6, "selected_operational_opportunities.csv"))
    return d

def adaptive_dbscan_clusters(coords_km: np.ndarray, eps0: float, min0: int) -> Tuple[np.ndarray, dict]:
    if DBSCAN is None:
        raise RuntimeError("scikit-learn not available.")
    tried = []
    min_list = [min0, 4, 3, 2]
    eps_list = [eps0, 6.0, 8.0, 10.0, 12.0]

    for m in min_list:
        if m < 2:
            continue
        model = DBSCAN(eps=float(eps0), min_samples=int(m))
        labels = model.fit_predict(coords_km)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        tried.append({"eps_km": float(eps0), "min_samples": int(m), "n_clusters": int(n_clusters), "n_noise": n_noise})
        if n_clusters > 0:
            return labels, {"selected": tried[-1], "tried": tried}

    for e in eps_list[1:]:
        model = DBSCAN(eps=float(e), min_samples=2)
        labels = model.fit_predict(coords_km)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        tried.append({"eps_km": float(e), "min_samples": 2, "n_clusters": int(n_clusters), "n_noise": n_noise})
        if n_clusters > 0:
            return labels, {"selected": tried[-1], "tried": tried}

    return np.full(len(coords_km), -1, dtype=int), {"selected": None, "tried": tried}

def build_bases_from_dbscan_stage_6(opportunities: pd.DataFrame, policy: dict) -> pd.DataFrame:
    eps_km = float(policy.get("dbscan_eps_km", 5.0))
    min_samples = int(policy.get("dbscan_min_samples", 2))
    max_bases = int(policy.get("max_bases", 220))
    n_usv_default = int(policy.get("n_usv_per_base_default", 4))

    lat = opportunities["lat"].to_numpy(dtype=float)
    lon = opportunities["lon"].to_numpy(dtype=float)
    coords_km = latlon_to_km_xy(lat, lon)

    labels, audit = adaptive_dbscan_clusters(coords_km, eps0=eps_km, min0=min_samples)
    opportunities = opportunities.copy()
    opportunities["cluster_id"] = labels

    cl = opportunities[opportunities["cluster_id"] >= 0].copy()
    if cl.empty:
        dump_json(os.path.join(OUT_DIR_STAGE_6, "dbscan_bases_audit.json"), audit)
        raise RuntimeError("DBSCAN produced only noise.")

    g = cl.groupby("cluster_id").agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        n_points=("cluster_id", "size"),
        score_mean=("priority_score", "mean"),
        score_max=("priority_score", "max")
    ).reset_index()

    g = g.sort_values(["n_points", "score_max"], ascending=[False, False]).head(max_bases).reset_index(drop=True)

    bases = []
    for _, r in g.iterrows():
        bases.append({
            "base_id": f"OFF_{int(r['cluster_id']):03d}",
            "base_name": f"OffshoreStation_{int(r['cluster_id']):03d}",
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "n_usv_available": int(n_usv_default),
            "is_active": 1
        })
    bases = pd.DataFrame(bases)

    dump_json(os.path.join(OUT_DIR_STAGE_6, "dbscan_bases_audit.json"), audit)
    save_df_csv(bases, os.path.join(OUT_DIR_STAGE_6, "stations_used.csv"))
    return bases

def build_targets_stage_6(opportunities: pd.DataFrame, policy: dict) -> pd.DataFrame:
    eps_km = float(policy.get("targets_dbscan_eps_km", 5.0))
    min_samples = int(policy.get("targets_dbscan_min_samples", 2))

    coords_km = latlon_to_km_xy(opportunities["lat"].to_numpy(float), opportunities["lon"].to_numpy(float))
    labels, audit = adaptive_dbscan_clusters(coords_km, eps0=eps_km, min0=min_samples)
    opportunities = opportunities.copy()
    opportunities["target_cluster"] = labels

    cl = opportunities[opportunities["target_cluster"] >= 0].copy()
    if cl.empty:
        dump_json(os.path.join(OUT_DIR_STAGE_6, "dbscan_targets_audit.json"), audit)
        cl = opportunities.copy()
        cl["target_cluster"] = np.arange(len(cl))

    g = cl.groupby("target_cluster").agg(
        lat_c=("lat", "mean"),
        lon_c=("lon", "mean"),
        score_mean=("priority_score", "mean"),
        score_max=("priority_score", "max"),
        n_cells=("priority_score", "size"),
        window_start=("time_start", "min"),
        window_end=("time_end", "max"),
        safety_ok_rate=("safety_ok", "mean")
    ).reset_index()

    g["window_start"] = pd.to_datetime(g["window_start"], errors="coerce")
    g["window_end"] = pd.to_datetime(g["window_end"], errors="coerce")
    g["window_duration_min"] = (g["window_end"] - g["window_start"]).dt.total_seconds() / 60.0

    g["target_priority"] = (
        0.75 * (g["score_max"] - g["score_max"].min()) / max(g["score_max"].max() - g["score_max"].min(), 1e-12) +
        0.15 * (g["window_duration_min"] - g["window_duration_min"].min()) / max(g["window_duration_min"].max() - g["window_duration_min"].min(), 1e-12) +
        0.10 * (g["safety_ok_rate"] - g["safety_ok_rate"].min()) / max(g["safety_ok_rate"].max() - g["safety_ok_rate"].min(), 1e-12)
    )
    g["target_id"] = [f"TGT_{i + 1:05d}" for i in range(len(g))]
    g = g.sort_values(["target_priority", "score_max"], ascending=[False, False]).reset_index(drop=True)

    dump_json(os.path.join(OUT_DIR_STAGE_6, "dbscan_targets_audit.json"), audit)
    save_df_csv(g, os.path.join(OUT_DIR_STAGE_6, "targets.csv"))
    return g

def expand_base_target_pairs_stage_6(targets: pd.DataFrame, bases: pd.DataFrame, platforms: List[USVPlatform]) -> pd.DataFrame:
    rows = []
    for _, tgt in targets.iterrows():
        for _, b in bases.iterrows():
            for plat in platforms:
                rows.append({
                    "target_id": tgt["target_id"],
                    "lat_c": tgt["lat_c"],
                    "lon_c": tgt["lon_c"],
                    "target_priority": tgt["target_priority"],
                    "score_max": tgt["score_max"],
                    "window_start": tgt["window_start"],
                    "window_end": tgt["window_end"],
                    "window_duration_min": tgt["window_duration_min"],
                    "base_id": b["base_id"],
                    "base_name": b["base_name"],
                    "base_lat": b["lat"],
                    "base_lon": b["lon"],
                    "n_usv_available": int(b["n_usv_available"]),
                    "usv_id": plat.usv_id,
                    "platform_name": plat.platform_name,
                    "cruise_speed_kmh": plat.cruise_speed_kmh,
                    "max_endurance_min": plat.max_endurance_min,
                    "reserve_min": plat.reserve_min,
                    "spray_rate_lpm": plat.spray_rate_lpm,
                    "agent_capacity_l": plat.agent_capacity_l,
                    "setup_spray_min": plat.setup_spray_min,
                    "shutdown_spray_min": plat.shutdown_spray_min
                })
    return pd.DataFrame(rows)

def compute_target_feasibility_stage_6(targets: pd.DataFrame, bases: pd.DataFrame, platforms: List[USVPlatform], policy: dict) -> pd.DataFrame:
    pairs = expand_base_target_pairs_stage_6(targets, bases, platforms)
    pairs["dist_oneway_km"] = pairs.apply(lambda r: haversine_km(r["base_lat"], r["base_lon"], r["lat_c"], r["lon_c"]), axis=1)
    pairs["dist_roundtrip_km"] = 2.0 * pairs["dist_oneway_km"]
    pairs["transit_oneway_min"] = 60.0 * pairs["dist_oneway_km"] / pairs["cruise_speed_kmh"].replace(0, np.nan)
    pairs["transit_roundtrip_min"] = 2.0 * pairs["transit_oneway_min"]

    buffer_min = float(policy.get("time_buffer_min", 2.0))
    pairs["spray_plannable_min"] = (
        pairs["window_duration_min"] -
        pairs["transit_roundtrip_min"] -
        pairs["setup_spray_min"] -
        pairs["shutdown_spray_min"] -
        buffer_min
    ).clip(lower=0.0)

    pairs["total_mission_min"] = (
        pairs["transit_roundtrip_min"] +
        pairs["setup_spray_min"] +
        pairs["shutdown_spray_min"] +
        pairs["spray_plannable_min"]
    )
    pairs["reserve_remaining_min"] = pairs["max_endurance_min"] - pairs["reserve_min"] - pairs["total_mission_min"]

    pairs["feasible_window"] = pairs["spray_plannable_min"] > 0
    pairs["feasible_reserve"] = pairs["reserve_remaining_min"] >= 0
    pairs["feasible_precheck"] = pairs[["feasible_window", "feasible_reserve"]].all(axis=1)

    def reason_row(r):
        reasons = []
        if not bool(r["feasible_window"]):
            reasons.append("insufficient_operational_window")
        if not bool(r["feasible_reserve"]):
            reasons.append("reserve_violation")
        return "ok" if not reasons else "|".join(reasons)

    pairs["feasibility_reason"] = pairs.apply(reason_row, axis=1)
    save_df_csv(pairs, os.path.join(OUT_DIR_STAGE_6, "target_feasibility.csv"))
    return pairs

def assign_missions_multi_usv_stage_6(feas_df: pd.DataFrame) -> pd.DataFrame:
    cap = feas_df.groupby("base_id")["n_usv_available"].max().to_dict()
    used = {k: 0 for k in cap.keys()}
    missions = []

    for target_id, g in feas_df.groupby("target_id", sort=False):
        g = g.sort_values(["feasible_precheck", "target_priority"], ascending=[False, False])
        chosen = None
        for _, r in g.iterrows():
            b = r["base_id"]
            if used.get(b, 0) < cap.get(b, 0):
                chosen = r
                used[b] = used.get(b, 0) + 1
                break
        if chosen is None:
            chosen = g.iloc[0].copy()
            chosen["feasible_precheck"] = False
            chosen["feasibility_reason"] = str(chosen.get("feasibility_reason", "")) + "|base_capacity_full"
        missions.append(chosen)

    mdf = pd.DataFrame(missions).reset_index(drop=True)
    mdf["mission_id"] = [f"MISSION_{i + 1:05d}" for i in range(len(mdf))]
    mdf["agent_use_planned_l"] = mdf["spray_plannable_min"] * mdf["spray_rate_lpm"]
    mdf["agent_capacity_ok"] = mdf["agent_use_planned_l"] <= mdf["agent_capacity_l"]

    def classify(r):
        if not bool(r["feasible_precheck"]):
            return "NO-GO"
        if not bool(r["agent_capacity_ok"]):
            return "SURVEILLER"
        rr = float(r.get("reserve_remaining_min", -1))
        sp = float(r.get("spray_plannable_min", 0))
        if rr < 15 or sp < 10:
            return "SURVEILLER"
        return "GO"

    mdf["mission_go"] = mdf.apply(classify, axis=1)
    save_df_csv(mdf, os.path.join(OUT_DIR_STAGE_6, "mission_plan.csv"))
    return mdf

def compute_kpis_stage_6(missions: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    summary = {"missions_total": int(len(missions))}
    vc = missions["mission_go"].value_counts().to_dict()
    summary["missions_go"] = int(vc.get("GO", 0))
    summary["missions_surveiller"] = int(vc.get("SURVEILLER", 0))
    summary["missions_nogo"] = int(vc.get("NO-GO", 0))
    summary["go_rate"] = float(summary["missions_go"] / max(1, summary["missions_total"]))

    kpis = missions[[
        "mission_id", "mission_go", "target_id", "base_id", "usv_id",
        "dist_roundtrip_km", "transit_roundtrip_min", "spray_plannable_min",
        "total_mission_min", "reserve_remaining_min", "agent_use_planned_l", "feasibility_reason"
    ]].copy()
    return kpis, summary

def run_stage_6():
    print("\n=== Stage 6 — Spatiotemporal clustering and operational routing ===")
    setup_logging_stage_6()
    write_default_configs_stage_6()

    platforms = load_platforms_stage_6()
    policy = load_policy_stage_6()

    opportunities = build_selected_opportunities_stage_6(policy)
    bases = build_bases_from_dbscan_stage_6(opportunities, policy)
    targets = build_targets_stage_6(opportunities, policy)
    feasibility = compute_target_feasibility_stage_6(targets, bases, platforms, policy)
    missions = assign_missions_multi_usv_stage_6(feasibility)

    kpis, summary = compute_kpis_stage_6(missions)
    save_df_csv(kpis, os.path.join(OUT_DIR_STAGE_6, "mission_kpis.csv"))
    dump_json(os.path.join(OUT_DIR_STAGE_6, "operational_routing_report.json"), {
        "summary": summary,
        "n_platforms": len(platforms),
        "platforms": [asdict(p) for p in platforms],
        "n_bases": int(len(bases))
    })

    print("Stage 6 completed:", STAGE_6_DIR)


# ======================================================================================
# MASTER MAIN
# ======================================================================================

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("DEVICE:", DEVICE)

    if RUN_STAGE_1:
        run_stage_1()
    if RUN_STAGE_2:
        run_stage_2()
    if RUN_STAGE_3:
        run_stage_3()
    if RUN_STAGE_4:
        run_stage_4()
    if RUN_STAGE_5:
        run_stage_5()
    if RUN_STAGE_6:
        run_stage_6()

    print("\nAll requested stages completed.")

if __name__ == "__main__":
    main()
