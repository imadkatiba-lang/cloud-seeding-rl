#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, json, math, warnings, random, time
from pathlib import Path
from typing import List, Dict
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt

# -------------------------------------------------
# 0) CONFIG (générale + spécifiques algos)
# -------------------------------------------------
CFG = {
    # Entrées (adapter si besoin)
    "PRED_PATH": r"C:\Users\tuf-p\Desktop\ARTICLES\Seeding2_logs_SeedingElig01\test_predictions_seeding_elig01.csv",
    "MERRA_PATH": r"C:\Users\tuf-p\Desktop\ARTICLES\Seeding1\_MERGED_NEWBOX\MERRA2_PREC_2011-01_to_2024-02_newbox.csv",

    # Sorties
    "OUT_DIR": r"C:\Users\tuf-p\Desktop\ARTICLES\Seeding_Drone_TEST",

    # Stations & missions
    "STATIONS": [
        {"name": "Casablanca", "lat": 33.57, "lon": -7.59},
        {"name": "Rabat",      "lat": 34.02, "lon": -6.84},
        {"name": "Tangier",    "lat": 35.76, "lon": -5.83},
        {"name": "Tetouan",    "lat": 35.57, "lon": -5.37},
        {"name": "Kenitra",    "lat": 34.26, "lon": -6.57},
    ],
    "MISSIONS_PER_STATION": 10,
    "MAX_HOTSPOTS": 8,

    # NMS & gate
    "TAU": 0.45,              # seuil global (fallback)
    "RADIUS_KM": 45,          # <- plus serré pour garder plus de cibles proches
    "SEARCH_KM": 40,          # rayon autour de la base pour filtrer les hotspots (missions)

    # Sécurité (hard)
    "LIMITS": {
        "U10M_MAX": 14.0,
        "UCLD_MAX": 22.0,
        "RAIN_MAX": 2.0,      # mm/h
        "CAPE_MAX": 1200.0,
        "SHEAR_MAX": 16.0,    # m/s
        "GUST_FACTOR_MAX": 1.5,
        "T850_ICE": (-15.0, 0.0),
        "RH_ICE": 0.90,
    },

    # Drone
    "DRONE": {
        "SPEED_KMH": 70.0,
        "LOITER_MIN": 2.0,
        "SCAN_MIN": 1.0,
        "ENDURANCE_MIN": 110.0,
        "RESERVE_MIN": 10.0,
    },

    # RL global
    "SEED": 42,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EPOCHS": 50,
    "BATCH": 128,
    "GAMMA": 0.99,
    "TAU_SOFT": 0.005,
    "LR": 3e-4,

    # Récompense (échelle robuste multi-stations)
    "REF_RAIN": 1.0,
    "BETA": 1.2,          # pente (1 - exp(-BETA*a))
    "LAMBDA": 0.01,       # coût d'action
    "REWARD_SCALE": 12.0, # échelle globale

    # REDQ
    "REDQ_ENSEMBLE": 5,
    "REDQ_SUBSAMPLE": 2,
    "REDQ_TARGET_ENTROPY": -0.5,
    "REDQ_ALPHA_INIT": 0.4,

    # CEM
    "CEM_POP": 64,
    "CEM_ELITES": 8,
    "CEM_INIT_STD": 0.8,
    "CEM_ALPHA": 0.6,
    "CEM_MAX_EP_STEPS": 512,

    # PPO
    "PPO_CLIP": 0.2,
    "PPO_EPOCHS": 4,
    "PPO_LAMBDA_GAE": 0.95,
}

# Prépare sorties
OUT = Path(CFG["OUT_DIR"])
(OUT / "mission_set").mkdir(parents=True, exist_ok=True)

# Seeds + option deterministic
random.seed(CFG["SEED"])
np.random.seed(CFG["SEED"])
torch.manual_seed(CFG["SEED"])
if CFG["DEVICE"] == "cuda":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# -------------------------------------------------
# Utils généraux
# -------------------------------------------------
EARTH_R_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return EARTH_R_KM * c

def speed(u, v):
    return np.sqrt(u*u + v*v)

# -------------------------------------------------
# 1) Load & merge data
# -------------------------------------------------
REQ_COLS_PRED = ["lat", "lon", "eligibility_pred"]
REQ_COLS_MET  = ["U10M", "V10M", "U850", "V850", "U700", "V700", "T850", "RH850", "tp", "CAPE", "GUST10M"]

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] lecture CSV échouée: {path} -> {e}")
        return pd.DataFrame()

def load_and_merge(pred_path, met_path):
    dfp = safe_read_csv(pred_path)
    dfm = safe_read_csv(met_path)
    dfp.columns = [c.strip() for c in dfp.columns]
    dfm.columns = [c.strip() for c in dfm.columns]

    if dfm.empty:
        base_len = len(dfp) if len(dfp) > 0 else 0
        dfm = pd.DataFrame({
            'lat': dfp['lat'] if 'lat' in dfp.columns else (32 + np.random.rand(base_len) * 5),
            'lon': dfp['lon'] if 'lon' in dfp.columns else (-8 + np.random.rand(base_len) * 3),
        })

    for c in REQ_COLS_PRED:
        if c not in dfp.columns:
            if c == "eligibility_pred":
                dfp[c] = np.clip(np.random.rand(len(dfp))*0.4 + 0.4, 0, 1)
            elif c == "lat":
                dfp[c] = 32 + np.random.rand(len(dfp)) * 5
            elif c == "lon":
                dfp[c] = -8 + np.random.rand(len(dfp)) * 3

    for c in REQ_COLS_MET:
        if c not in dfm.columns:
            if len(dfm) > 0:
                base = {
                    "U10M": 8, "V10M": 2, "U850": 10, "V850": 3, "U700": 12, "V700": 4,
                    "T850": 3.0, "RH850": 0.6, "tp": 0.5, "CAPE": 400, "GUST10M": 10,
                }[c]
                dfm[c] = base + 0.5*np.random.randn(len(dfm))
            else:
                dfm[c] = []

    if 'lat' not in dfm.columns:
        dfm['lat'] = 32 + np.random.rand(len(dfm)) * 5
    if 'lon' not in dfm.columns:
        dfm['lon'] = -8 + np.random.rand(len(dfm)) * 3

    dfp['lat_q'] = dfp['lat'].round(2)
    dfp['lon_q'] = dfp['lon'].round(2)
    dfm['lat_q'] = dfm['lat'].round(2)
    dfm['lon_q'] = dfm['lon'].round(2)
    df = pd.merge(dfp, dfm, on=['lat_q','lon_q'], how='left', suffixes=("","_m"))

    defaults = {"U10M":8, "V10M":2, "U850":10, "V850":3, "U700":12, "V700":4,
                "T850":3.0, "RH850":0.6, "tp":0.5, "CAPE":400, "GUST10M":10}
    for c in REQ_COLS_MET:
        if c not in df.columns or df[c].isna().all():
            df[c] = defaults[c]
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(defaults[c])

    df['eligibility_pred'] = pd.to_numeric(df['eligibility_pred'], errors='coerce').fillna(0.5).clip(0,1)
    return df

# -------------------------------------------------
# 2) Sécurité (hard)
# -------------------------------------------------
def getf(row, key, default):
    val = row.get(key, default)
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or not np.isfinite(val))):
            return default
        return float(val)
    except Exception:
        return default

def meteorological_efficiency(row):
    ivt = speed(getf(row, 'U850', 0.0), getf(row, 'V850', 0.0)) + speed(getf(row, 'U700', 0.0), getf(row, 'V700', 0.0))
    rh = getf(row, 'RH850', 0.6)
    cape = getf(row, 'CAPE', 400.0)
    eff = (ivt/30.0) * (0.5 + 0.5*rh) * (1.0/(1.0 + max(0.0, cape-800.0)/800.0))
    return float(np.clip(eff if np.isfinite(eff) else 0.0, 0.0, 2.0))

def safety_flags(row, lim=CFG["LIMITS"]):
    u10 = getf(row, 'U10M', 0.0); v10 = getf(row, 'V10M', 0.0)
    u850 = getf(row, 'U850', 0.0); v850 = getf(row, 'V850', 0.0)
    u700 = getf(row, 'U700', 0.0); v700 = getf(row, 'V700', 0.0)
    tp = getf(row, 'tp', 0.0); cape = getf(row, 'CAPE', 0.0)
    gust = getf(row, 'GUST10M', 0.0); rh = getf(row, 'RH850', 0.6); T850 = getf(row, 'T850', 3.0)
    flags = {
        "wind_sfc": speed(u10, v10) <= lim["U10M_MAX"],
        "wind_cloud": max(speed(u850, v850), speed(u700, v700)) <= lim["UCLD_MAX"],
        "rain": tp <= lim["RAIN_MAX"],
        "cape": cape <= lim["CAPE_MAX"],
        "shear": abs(speed(u850, v850) - speed(u700, v700)) <= lim["SHEAR_MAX"],
        "gust": (gust / max(1e-6, speed(u10, v10))) <= lim["GUST_FACTOR_MAX"],
        "icing": not (lim["T850_ICE"][0] <= T850 <= lim["T850_ICE"][1] and rh >= lim["RH_ICE"])
    }
    ok = all(bool(v) for v in flags.values())
    return ok, flags

# -------------------------------------------------
# 3) NMS locale + calibration par station
# -------------------------------------------------
def nms_hotspots_local(df, base_lat, base_lon, tau=CFG["TAU"], r_km=CFG["RADIUS_KM"], max_k=CFG["MAX_HOTSPOTS"], search_km=CFG["SEARCH_KM"]):
    # Filtrer par distance à la base
    df2 = df.copy()
    df2["d_base_km"] = haversine_km(base_lat, base_lon, df2["lat"].values, df2["lon"].values)
    df2 = df2[df2["d_base_km"] <= search_km].copy()
    if df2.empty:
        return pd.DataFrame()

    # Backoff du TAU plus profond si peu de candidats
    tau_try = tau
    selected = pd.DataFrame()
    for _ in range(10):
        cand = df2[df2["eligibility_pred"] >= tau_try].copy()
        if len(cand) >= 3:
            selected = cand
            break
        tau_try = max(0.25, tau_try - 0.04)  # min 0.25, pas 0.04
    if selected.empty:
        selected = df2.nlargest(16, "eligibility_pred").copy()  # fallback plus large

    # NMS par distance
    selected = selected.sort_values('eligibility_pred', ascending=False).reset_index(drop=True)
    keep = []
    for _, row in selected.iterrows():
        lat_i, lon_i = float(row['lat']), float(row['lon'])
        if all(haversine_km(lat_i, lon_i, float(k['lat']), float(k['lon'])) > r_km for k in keep):
            keep.append(row)
        if len(keep) >= max_k:
            break
    return pd.DataFrame(keep)

def compute_station_calibration(df_all, base_lat, base_lon, search_km=CFG["SEARCH_KM"]):
    df2 = df_all.copy()
    df2["d_base_km"] = haversine_km(base_lat, base_lon, df2["lat"].values, df2["lon"].values)
    # Fenêtre élargie pour estimer des quantiles robustes
    loc = df2[df2["d_base_km"] <= max(search_km, 80)].copy()
    if loc.empty:
        return {"tau_local": CFG["TAU"], "y_p50": 0.5, "y_p90": 0.8, "reward_scale_loc": 1.0}

    y = pd.to_numeric(loc["eligibility_pred"], errors="coerce").fillna(0.0).clip(0,1)
    if len(y) < 50:
        p50, p60, p75, p90 = 0.45, 0.55, 0.60, 0.75  # garde-fous si peu de points
    else:
        p50, p60, p75, p90 = np.percentile(y, [50,60,75,90])

    tau_local = float(np.clip(p60, 0.30, 0.75))
    # Échelle locale un peu plus généreuse (bornée)
    if p75 > 1e-6:
        scale_fac = float(np.clip(0.75 / p75, 0.7, 2.0))
    else:
        scale_fac = 1.3

    return {
        "tau_local": tau_local,
        "y_p50": float(p50),
        "y_p90": float(max(p90, p50 + 0.08)),  # span min 0.08
        "reward_scale_loc": scale_fac
    }

# -------------------------------------------------
# 4) Missions (avec calibration locale)
# -------------------------------------------------
def build_missions(df_all: pd.DataFrame, station: Dict, n_missions: int) -> List[Dict]:
    missions = []
    base_lat, base_lon = station['lat'], station['lon']
    calib = compute_station_calibration(df_all, base_lat, base_lon, search_km=CFG["SEARCH_KM"])
    for mid in range(1, n_missions+1):
        hs = nms_hotspots_local(df_all, base_lat, base_lon, tau=calib["tau_local"])
        if hs.empty:
            continue
        visited = []
        cur_lat, cur_lon = base_lat, base_lon
        remaining = hs.copy()
        while not remaining.empty:
            remaining['dist'] = remaining.apply(lambda r: haversine_km(cur_lat, cur_lon, r['lat'], r['lon']), axis=1)
            j = remaining['dist'].idxmin()
            nxt = remaining.loc[j]
            visited.append(dict(nxt))
            cur_lat, cur_lon = float(nxt['lat']), float(nxt['lon'])
            remaining = remaining.drop(index=j)
        missions.append({
            "station": station['name'],
            "mission_id": mid,
            "base_lat": base_lat,
            "base_lon": base_lon,
            "hotspots": visited,
            "calib": calib,
        })
    return missions

# -------------------------------------------------
# 5) Environnement RL (reward v3 calibrée)
# -------------------------------------------------
class MissionEnv:
    def __init__(self, mission: Dict, cfg: Dict):
        self.mission = mission
        self.cfg = cfg
        self.idx = 0
        self.distance_total = 0.0
        self.loiter_total = 0.0
        self.seeded_total = 0
        self.violations = 0
        self.state_dim = 10
        self.action_low = 0.0
        self.action_high = 1.0
        self.calib = mission.get("calib", {"tau_local": CFG["TAU"], "y_p50": 0.5, "y_p90": 0.8, "reward_scale_loc": 1.0})
        self.reset()

    def _build_obs(self, hs):
        def gf(key, default):
            val = hs.get(key, default)
            try:
                if val is None or (isinstance(val, float) and (np.isnan(val) or not np.isfinite(val))):
                    return default
                return float(val)
            except Exception:
                return default
        y = gf('eligibility_pred', 0.5)
        eff = gf('eff', 1.0)
        u10 = gf('U10M', 8.0)
        ucl = max(speed(gf('U850',0.0), gf('V850',0.0)), speed(gf('U700',0.0), gf('V700',0.0)))
        tp = gf('tp', 0.5)
        cape = gf('CAPE', 400.0)
        rh = gf('RH850', 0.6)
        gust = gf('GUST10M', 10.0)
        dleft = (len(self.hotspots) - self.idx) / max(1, len(self.hotspots))
        obs = np.array([y, eff, u10/20, ucl/30, tp/5, cape/1500, rh, gust/25, dleft, 1.0], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def reset(self):
        self.hotspots = list(self.mission['hotspots'])
        for h in self.hotspots:
            h['eff'] = meteorological_efficiency(h)
            for key, default in [("U10M",8),("V10M",2),("U850",10),("V850",3),("U700",12),("V700",4),
                                 ("T850",3.0),("RH850",0.6),("tp",0.5),("CAPE",400),("GUST10M",10),
                                 ("eligibility_pred",0.5)]:
                try:
                    v = h.get(key, default)
                    if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))):
                        h[key] = default
                except Exception:
                    h[key] = default
        self.idx = 0
        self.distance_total = 0.0
        self.loiter_total = 0.0
        self.seeded_total = 0
        self.violations = 0
        self.cur_lat = float(self.mission.get('base_lat', CFG['STATIONS'][0]['lat']))
        self.cur_lon = float(self.mission.get('base_lon', CFG['STATIONS'][0]['lon']))
        if self.hotspots:
            return self._build_obs(self.hotspots[self.idx])
        return np.zeros(self.state_dim, dtype=np.float32)

    def step(self, a: float):
        if self.idx >= len(self.hotspots):
            return np.zeros(self.state_dim, dtype=np.float32), 0.0, True, {}
        hs = self.hotspots[self.idx]
        d_km = haversine_km(self.cur_lat, self.cur_lon, float(hs['lat']), float(hs['lon']))
        self.distance_total += d_km
        self.cur_lat, self.cur_lon = float(hs['lat']), float(hs['lon'])

        ok, flags = safety_flags(hs)
        a = float(np.clip(a, 0.0, 1.0))

        # -------- Reward v3 (calibrée par station) --------
        y_raw = float(hs.get('eligibility_pred', 0.5))
        p50 = float(self.calib.get("y_p50", 0.5))
        p90 = float(self.calib.get("y_p90", 0.8))
        span = max(0.08, p90 - p50)  # span minimal robuste
        y_loc = float(np.clip((y_raw - p50) / span * 0.8 + 0.5, 0.0, 1.0))

        eff = float(hs.get('eff', 1.0))
        rain = float(hs.get('tp', 0.5))

        def sigmoid(x): 
            return 1.0 / (1.0 + math.exp(-x))

        tau_l = float(self.calib.get("tau_local", CFG["TAU"]))
        g_soft = sigmoid((y_loc - tau_l) / 0.12)
        a_eff = a * (1.0 if ok else 0.0)

        y_mix = 0.90 * y_loc + 0.10 * min(rain / 2.0, 1.0)  # plus de poids au modèle
        base_gain = 0.35 + 0.65 * g_soft
        action_gain = (1.0 - math.exp(-CFG['BETA'] * a_eff))

        reward_scale_loc = float(self.calib.get("reward_scale_loc", 1.0))
        r_pos = (CFG['REWARD_SCALE'] * reward_scale_loc) * base_gain * y_mix * eff * action_gain
        r_cost = CFG['LAMBDA'] * (a ** 2)
        r = r_pos - r_cost
        # ---------------------------------------------------

        loiter_min = a_eff * CFG['DRONE']['LOITER_MIN'] * (1.0 + eff)
        self.loiter_total += loiter_min

        if not ok:
            self.violations += 1
        if a_eff > 0.01:
            self.seeded_total += 1

        self.idx += 1
        done = (self.idx >= len(self.hotspots))

        # --- Garde-fou endurance (durée & distance) ---
        t_flight_min = (self.distance_total / max(1e-6, self.cfg['DRONE']['SPEED_KMH'])) * 60.0
        t_total_min = t_flight_min + self.loiter_total
        max_min = self.cfg['DRONE']['ENDURANCE_MIN'] - self.cfg['DRONE']['RESERVE_MIN']
        endurance_cut = False
        if t_total_min > max_min:
            done = True
            endurance_cut = True

        obs = self._build_obs(self.hotspots[self.idx]) if not done else np.zeros(self.state_dim, dtype=np.float32)
        info = {
            "gate": 1.0 if y_loc >= tau_l else 0.0,
            "ok": ok,
            "flags": flags,
            "dist_km": d_km,
            "loiter_min": loiter_min,
            # logs reward
            "r_pos": r_pos, "r_cost": r_cost, "g_soft": g_soft, "y_mix": y_mix, "eff_comp": eff,
            "y_raw": y_raw, "y_loc": y_loc, "tau_local": tau_l, "reward_scale_loc": reward_scale_loc,
            # endurance
            "endurance_cut": int(endurance_cut),
            "t_total_min": t_total_min,
            "hotspots_count": len(self.hotspots)
        }
        return obs, float(r), done, info

# -------------------------------------------------
# 6) Réseaux & Buffers & Aides
# -------------------------------------------------
class MLP(nn.Module):
    def __init__(self, inp, out, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out)
        )
    def forward(self, x): return self.net(x)

class ActorGaussian(nn.Module):
    def __init__(self, inp, hid=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(inp, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        self.mu = nn.Linear(hid, 1)
        self.logstd = nn.Linear(hid, 1)
    def forward(self, x):
        z = self.backbone(x)
        mu = torch.tanh(self.mu(z))      # [-1,1]
        logstd = torch.clamp(self.logstd(z), -5, 2)
        return mu, logstd
    def sample(self, x):
        mu, logstd = self(x)
        std = logstd.exp()
        dist = Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)                # [-1,1]
        a01 = (a + 1) / 2                # [0,1]
        logp = dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        return a01, logp.sum(dim=-1, keepdim=True)

class Replay:
    def __init__(self, cap=20000, sdim=10):
        self.s = torch.zeros((cap, sdim), dtype=torch.float32)
        self.a = torch.zeros((cap, 1), dtype=torch.float32)
        self.r = torch.zeros((cap, 1), dtype=torch.float32)
        self.ns= torch.zeros((cap, sdim), dtype=torch.float32)
        self.d = torch.zeros((cap, 1), dtype=torch.float32)
        self.idx = 0; self.full=False; self.cap=cap
    def add(self, s,a,r,ns,d):
        i = self.idx
        self.s[i]=s; self.a[i]=a; self.r[i]=r; self.ns[i]=ns; self.d[i]=d
        self.idx=(i+1)%self.cap; self.full=self.full or self.idx==0
    def sample(self, bs):
        n = self.cap if self.full else self.idx
        idx = np.random.randint(0, n, size=bs)
        return (self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.d[idx])

def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def set_param_vector(module: nn.Module, vec: torch.Tensor) -> None:
    idx = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(vec[idx:idx+n].view_as(p))
        idx += n

# -------------------------------------------------
# 7) Agents : REDQ / CEM / PPO
# -------------------------------------------------
class REDQAgent:
    def __init__(self, sdim):
        self.actor = ActorGaussian(sdim).to(CFG['DEVICE'])
        self.qs = nn.ModuleList([MLP(sdim+1,1).to(CFG['DEVICE']) for _ in range(CFG['REDQ_ENSEMBLE'])])
        self.qs_t = nn.ModuleList([MLP(sdim+1,1).to(CFG['DEVICE']) for _ in range(CFG['REDQ_ENSEMBLE'])])
        for i in range(CFG['REDQ_ENSEMBLE']):
            self.qs_t[i].load_state_dict(self.qs[i].state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=CFG['LR'])
        self.opt_q = torch.optim.Adam([p for m in self.qs for p in m.parameters()], lr=CFG['LR'])
        self.log_alpha = torch.tensor(math.log(CFG['REDQ_ALPHA_INIT']), requires_grad=True, device=CFG['DEVICE'])
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=CFG['LR'])
        self.target_entropy = CFG['REDQ_TARGET_ENTROPY']

    def act(self, s):
        s = torch.tensor(s, dtype=torch.float32, device=CFG['DEVICE']).unsqueeze(0)
        with torch.no_grad():
            a, _ = self.actor.sample(s)
        return float(a.clamp(0,1).cpu().numpy()[0,0])

    def update(self, buf: Replay, bs=CFG['BATCH']):
        if (buf.idx if not buf.full else buf.cap) < bs:
            return {"q":0.0, "a":0.0}
        s,a,r,ns,d = buf.sample(bs)
        s=s.to(CFG['DEVICE']); a=a.to(CFG['DEVICE']); r=r.to(CFG['DEVICE']); ns=ns.to(CFG['DEVICE']); d=d.to(CFG['DEVICE'])
        with torch.no_grad():
            na, nlogp = self.actor.sample(ns)
            q_targets = [qt(torch.cat([ns, na], dim=-1)) for qt in self.qs_t]
            idx = np.random.choice(len(q_targets), size=CFG['REDQ_SUBSAMPLE'], replace=False)
            mins = torch.min(torch.stack([q_targets[i] for i in idx], dim=0), dim=0).values
            y = r + CFG['GAMMA'] * (1 - d) * (mins - math.exp(self.log_alpha.item()) * nlogp)
        q_losses = []
        for q in self.qs:
            qv = q(torch.cat([s,a], dim=-1))
            q_losses.append(F.mse_loss(qv, y))
        q_loss = torch.stack(q_losses).mean()
        self.opt_q.zero_grad(); q_loss.backward(); self.opt_q.step()
        a_s, logp = self.actor.sample(s)
        q_vals = torch.stack([q(torch.cat([s,a_s], dim=-1)) for q in self.qs], dim=0)
        qmin = torch.min(q_vals, dim=0).values
        a_loss = (math.exp(self.log_alpha.item())*logp - qmin).mean()
        self.opt_a.zero_grad(); a_loss.backward(); self.opt_a.step()
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()
        for qt, q in zip(self.qs_t, self.qs):
            for tp, p in zip(qt.parameters(), q.parameters()):
                tp.data.copy_(tp.data*(1-CFG['TAU_SOFT']) + p.data*CFG['TAU_SOFT'])
        return {"q": float(q_loss.item()), "a": float(a_loss.item())}

class CEMPolicy(nn.Module):
    """ Politique déterministe a = sigmoid(MLP(s)) ∈ [0,1] """
    def __init__(self, sdim, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sdim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )
    def forward(self, x):
        y = self.net(x)
        a01 = torch.sigmoid(y)
        return a01
    @torch.no_grad()
    def act_np(self, s_np):
        x = torch.tensor(s_np, dtype=torch.float32).unsqueeze(0).to(CFG['DEVICE'])
        a01 = self.forward(x)
        return float(a01.clamp(0,1).cpu().numpy()[0,0])

class CEMAgent:
    """ Cross-Entropy Method sur les paramètres d'une politique MLP compacte. """
    def __init__(self, sdim):
        self.policy = CEMPolicy(sdim).to(CFG['DEVICE'])
        self.dim = count_params(self.policy)
        self.mean = torch.zeros(self.dim, device=CFG['DEVICE'])
        self.std  = torch.full((self.dim,), CFG['CEM_INIT_STD'], device=CFG['DEVICE'])
        set_param_vector(self.policy, self.mean.clone())

    @torch.no_grad()
    def act(self, s):
        return self.policy.act_np(s)

    @torch.no_grad()
    def sample_params(self, n):
        eps = torch.randn((n, self.dim), device=CFG['DEVICE'])
        return self.mean + eps * self.std

    @torch.no_grad()
    def set_params(self, vec):
        set_param_vector(self.policy, vec)

    def update_distribution(self, elites):
        new_mean = elites.mean(dim=0)
        new_std  = elites.std(dim=0).clamp_min(1e-6)
        alpha = CFG['CEM_ALPHA']
        self.mean = alpha * new_mean + (1 - alpha) * self.mean
        self.std  = alpha * new_std  + (1 - alpha) * self.std
        set_param_vector(self.policy, self.mean.clone())

class PPOBuffer:
    def __init__(self, sdim, cap=4096):
        self.s = torch.zeros((cap, sdim), dtype=torch.float32)
        self.a = torch.zeros((cap, 1), dtype=torch.float32)
        self.r = torch.zeros((cap, 1), dtype=torch.float32)
        self.v = torch.zeros((cap, 1), dtype=torch.float32)
        self.logp = torch.zeros((cap,1), dtype=torch.float32)
        self.d = torch.zeros((cap, 1), dtype=torch.float32)
        self.ptr=0; self.cap=cap
    def add(self, s,a,r,v,logp,d):
        i=self.ptr; self.s[i]=s; self.a[i]=a; self.r[i]=r; self.v[i]=v; self.logp[i]=logp; self.d[i]=d
        self.ptr += 1
    def get(self):
        n=self.ptr
        return self.s[:n], self.a[:n], self.r[:n], self.v[:n], self.logp[:n], self.d[:n]
    def reset(self): self.ptr=0

class ValueNet(nn.Module):
    def __init__(self, sdim, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sdim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )
    def forward(self, x): return self.net(x)

class PPOAgent:
    def __init__(self, sdim):
        self.actor = ActorGaussian(sdim).to(CFG['DEVICE'])
        self.critic = ValueNet(sdim).to(CFG['DEVICE'])
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=CFG['LR'])
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=CFG['LR'])

    def act(self, s):
        s = torch.tensor(s, dtype=torch.float32, device=CFG['DEVICE']).unsqueeze(0)
        with torch.no_grad():
            mu, logstd = self.actor(s)
            std = logstd.exp()
            dist = Normal(mu, std)
            u = dist.sample(); a = torch.tanh(u)
            a01 = (a+1)/2
        return float(a01.clamp(0,1).cpu().numpy()[0,0])

    def collect(self, env: MissionEnv, steps=1024):
        buf = PPOBuffer(env.state_dim, cap=steps)
        s = env.reset()
        for _ in range(steps):
            st = torch.tensor(s, dtype=torch.float32, device=CFG['DEVICE']).unsqueeze(0)
            mu, logstd = self.actor(st)
            std = logstd.exp(); dist = Normal(mu, std)
            u = dist.rsample(); a = torch.tanh(u)
            a01 = (a+1)/2
            logp = dist.log_prob(u) - torch.log(1-a.pow(2)+1e-6)
            v = self.critic(st)
            a01_det = a01.clamp(0,1).detach()
            ns, r, done, info = env.step(float(a01_det.cpu().numpy()[0,0]))
            buf.add(torch.tensor(s), a01_det.cpu(), torch.tensor([[r]]), v.detach().cpu(), logp.sum(dim=-1, keepdim=True).detach().cpu(), torch.tensor([[float(done)]]))
            s = ns
            if done:
                break
        return buf

    def update(self, buf: PPOBuffer):
        s,a,r,v,logp_old,d = [t.to(CFG['DEVICE']) for t in buf.get()]
        if len(r) == 0:
            return {"a":0.0,"v":0.0}
        adv = torch.zeros_like(r, device=CFG['DEVICE'])
        ret = torch.zeros_like(r, device=CFG['DEVICE'])
        g = torch.zeros(1,1, device=CFG['DEVICE'])
        for t in reversed(range(len(r))):
            next_v = v[t+1] if t < len(r)-1 else torch.zeros_like(v[t])
            delta = r[t] + CFG['GAMMA'] * (1-d[t]) * next_v - v[t]
            g = delta + CFG['GAMMA']*CFG['PPO_LAMBDA_GAE']*(1-d[t])*g
            adv[t] = g
            ret[t] = adv[t] + v[t]
        adv = (adv - adv.mean()) / (adv.std()+1e-6)
        for _ in range(CFG['PPO_EPOCHS']):
            mu, logstd = self.actor(s)
            std = logstd.exp(); dist = Normal(mu, std)
            a_tanh = a*2-1
            atanh_clip = torch.atanh(torch.clamp(a_tanh, -0.999, 0.999))
            logp = (dist.log_prob(atanh_clip) - torch.log(1-a_tanh.pow(2)+1e-6)).sum(dim=-1, keepdim=True)
            ratio = torch.exp(logp - logp_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-CFG['PPO_CLIP'], 1+CFG['PPO_CLIP']) * adv
            a_loss = -torch.min(surr1, surr2).mean()
            v_pred = self.critic(s)
            c_loss = F.mse_loss(v_pred, ret.detach())
            self.opt_a.zero_grad(); a_loss.backward(); self.opt_a.step()
            self.opt_c.zero_grad(); c_loss.backward(); self.opt_c.step()
        return {"a": float(a_loss.item()), "v": float(c_loss.item())}

# -------------------------------------------------
# 8) Entraînement par algo & Exports mission
# -------------------------------------------------
def ensure_dirs(station_name, mission_id):
    base = OUT / "mission_set" / f"{station_name}_m{mission_id}"
    base.mkdir(parents=True, exist_ok=True)
    return base

def run_training_for_env(env: MissionEnv, algo_name: str):
    if algo_name == "REDQ":
        agent = REDQAgent(env.state_dim)
        buf = Replay(cap=15000, sdim=env.state_dim)
        train_curve = []
        for ep in range(1, CFG['EPOCHS']+1):
            s = env.reset(); ep_ret = 0.0; done=False
            while not done:
                a = agent.act(s)
                ns, r, done, info = env.step(a)
                buf.add(torch.tensor(s), torch.tensor([[a]], dtype=torch.float32), torch.tensor([[r]], dtype=torch.float32), torch.tensor(ns), torch.tensor([[float(done)]], dtype=torch.float32))
                _ = agent.update(buf)
                ep_ret += r; s = ns
            train_curve.append({"epoch": ep, "reward_episode": ep_ret})
            print(f"[EPOCH] REDQ station={env.mission['station']} mission={env.mission['mission_id']} epoch={ep} reward_episode={ep_ret:.4f}")
        return agent, pd.DataFrame(train_curve)

    if algo_name == "CEM":
        agent = CEMAgent(env.state_dim)
        POP = CFG['CEM_POP']; K = CFG['CEM_ELITES']
        train_curve = []
        for ep in range(1, CFG['EPOCHS']+1):
            thetas = agent.sample_params(POP)
            returns = torch.zeros(POP, device=CFG['DEVICE'])
            for i in range(POP):
                agent.set_params(thetas[i])
                s = env.reset(); ep_ret=0.0; steps=0; done=False
                while not done and steps < CFG['CEM_MAX_EP_STEPS']:
                    a = agent.act(s)
                    s, r, done, info = env.step(a)
                    ep_ret += r; steps += 1
                returns[i] = ep_ret
            k = min(K, POP)
            elite_idx = torch.topk(returns, k=k, largest=True).indices
            elites = thetas[elite_idx]
            agent.update_distribution(elites)
            best = float(returns[elite_idx[0]].item())
            train_curve.append({"epoch": ep, "reward_episode": best})
            print(f"[EPOCH] CEM station={env.mission['station']} mission={env.mission['mission_id']} epoch={ep} best_reward={best:.4f}")
        return agent, pd.DataFrame(train_curve)

    if algo_name == "PPO":
        agent = PPOAgent(env.state_dim)
        train_curve = []
        for ep in range(1, CFG['EPOCHS']+1):
            buf = agent.collect(env, steps=1024)
            _ = agent.update(buf)
            s = env.reset(); ep_ret=0.0; done=False
            while not done:
                a = agent.act(s)
                s, r, done, info = env.step(a)
                ep_ret += r
            train_curve.append({"epoch": ep, "reward_episode": ep_ret})
            print(f"[EPOCH] PPO station={env.mission['station']} mission={env.mission['mission_id']} epoch={ep} reward_episode={ep_ret:.4f}")
        return agent, pd.DataFrame(train_curve)

    raise ValueError(f"algo inconnu: {algo_name}")

def export_decisions(env: MissionEnv, agent, algo_name: str, base_dir: Path, station: str, mission_id: int):
    rows = []
    s = env.reset(); i=0
    while True:
        i += 1
        a = agent.act(s)
        ns, r, done, info = env.step(a)
        rows.append({
            "station": station,
            "mission": mission_id,
            "algo": algo_name,
            "step": i,
            "action": "GO" if a>0.01 and info.get('ok',False) and info.get('gate',0)>0 else "SKIP",
            "a_cont": a,
            "reward": r,
            "dist_km": info.get('dist_km',0.0),
            "loiter_min": info.get('loiter_min',0.0),
            "safety_ok": int(info.get('ok',False)),
            # logs de reward
            "r_pos": float(info.get("r_pos", 0.0)),
            "r_cost": float(info.get("r_cost", 0.0)),
            "g_soft": float(info.get("g_soft", 0.0)),
            "y_mix": float(info.get("y_mix", 0.0)),
            "eff_comp": float(info.get("eff_comp", 0.0)),
            "y_raw": float(info.get("y_raw", 0.0)),
            "y_loc": float(info.get("y_loc", 0.0)),
            "tau_local": float(info.get("tau_local", CFG["TAU"])),
            "reward_scale_loc": float(info.get("reward_scale_loc", 1.0)),
            # endurance
            "endurance_cut": int(info.get("endurance_cut", 0)),
            "t_total_min": float(info.get("t_total_min", 0.0)),
            "hotspots_count": int(info.get("hotspots_count", 0)),
        })
        s = ns
        if done:
            break

    df = pd.DataFrame(rows)
    df.to_csv(base_dir / f"decisions_{algo_name.lower()}_torch.csv", index=False)

    if df.empty:
        rep = {
            "station": station, "mission": mission_id, "algo": algo_name,
            "reward_sum": 0.0, "reward_mean": 0.0,
            "go_count": 0, "skip_count": 0,
            "loiter_sum_min": 0.0, "dist_sum_km": 0.0,
            "safety_ok_ratio": 0.0
        }
    else:
        rep = {
            "station": station,
            "mission": mission_id,
            "algo": algo_name,
            "reward_sum": float(df['reward'].sum()),
            "reward_mean": float(df['reward'].mean()),
            "go_count": int((df['action'] == "GO").sum()),
            "skip_count": int((df['action'] == "SKIP").sum()),
            "loiter_sum_min": float(df['loiter_min'].sum()),
            "dist_sum_km": float(df['dist_km'].sum()),
            "safety_ok_ratio": float(df['safety_ok'].mean())
        }

    pd.DataFrame([rep]).to_csv(base_dir / f"report_{algo_name.lower()}.csv", index=False)

    print("[STATS] {} m{} {}: reward_sum={:.3f} reward_mean={:.4f} GO={} SKIP={}".format(
        station, mission_id, algo_name,
        rep["reward_sum"], rep["reward_mean"],
        rep["go_count"], rep["skip_count"]
    ))

    return df, rep

def export_training_curve(curve_df: pd.DataFrame, base_dir: Path, algo_name: str):
    path = base_dir / f"{algo_name.lower()}_training_curve.csv"
    curve_df.to_csv(path, index=False)
    return path

# -------------------------------------------------
# 9) Consolidation avancée (grand tableau + figures)
# -------------------------------------------------
def _now_iso(): return pd.Timestamp.utcnow().isoformat()

def _write_manifest(out_dir: Path, cfg: dict, run_id: str, kpis: dict):
    manifest = {
        "run_id": run_id,
        "timestamp_utc": _now_iso(),
        "cfg": cfg,
        "kpis": kpis,
        "artifacts": {
            "per_mission_dir": str(out_dir / "mission_set"),
            "summary_all_stations_csv": str(out_dir / "summary_all_stations.csv"),
            "all_reports_csv": str(out_dir / "all_reports.csv"),
            "all_decisions_csv": str(out_dir / "all_decisions.csv"),
            "all_training_curves_csv": str(out_dir / "all_training_curves.csv"),
            "grand_table_csv": str(out_dir / "grand_table_all_stats.csv"),
            "summary_station_algo_csv": str(out_dir / "summary_station_algo.csv"),
            "comparison_pngs": [str(p) for p in out_dir.glob("cmp_*_*.png")],
        }
    }
    with open(out_dir/"project_summary.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def build_grand_table(all_reports_df, all_decisions_df, all_curves_df):
    rep = all_reports_df.copy()

    # Best epoch & reward depuis courbes (par station×mission×algo)
    best_rows = []
    if not all_curves_df.empty and all(c in all_curves_df.columns for c in ["station","mission","algo","epoch","reward_episode"]):
        for (st, mi, al), g in all_curves_df.groupby(["station","mission","algo"]):
            g2 = g.sort_values("epoch")
            idx = g2["reward_episode"].idxmax()
            best_rows.append({
                "station": st, "mission": int(mi), "algo": al,
                "best_epoch_curve": int(g2.loc[idx, "epoch"]),
                "best_reward_episode_curve": float(g2.loc[idx, "reward_episode"]),
                "epochs": int(g2["epoch"].max())
            })
    best_df = pd.DataFrame(best_rows)

    # Décisions agrégées (station×mission×algo)
    if not all_decisions_df.empty:
        dec = (all_decisions_df
               .groupby(["station","mission","algo"], as_index=False)
               .agg(go_count_dec=("action", lambda s: (s=="GO").sum()),
                    steps=("action","size"),
                    safety_ok_ratio_dec=("safety_ok","mean"),
                    mean_action=("a_cont","mean"),
                    mean_reward_step=("reward","mean"),
                    dist_sum_km_dec=("dist_km","sum"),
                    loiter_sum_min_dec=("loiter_min","sum")))
    else:
        dec = pd.DataFrame(columns=["station","mission","algo","go_count_dec","steps","safety_ok_ratio_dec","mean_action","mean_reward_step","dist_sum_km_dec","loiter_sum_min_dec"])

    # Merge (reports ⟂ best ⟂ decisions)
    out = rep.copy()
    if not best_df.empty:
        out = out.merge(best_df, on=["station","mission","algo"], how="left", validate="m:1")
    if not dec.empty:
        out = out.merge(dec, on=["station","mission","algo"], how="left", validate="m:1")

    # Nettoyage NaN
    for c in out.columns:
        if out[c].dtype.kind in "fc":
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out

def save_grand_table_and_summary(out_dir: Path, all_reports_df, all_decisions_df, all_curves_df):
    grand = build_grand_table(all_reports_df, all_decisions_df, all_curves_df)
    grand_path = out_dir / "grand_table_all_stats.csv"
    grand.to_csv(grand_path, index=False)
    # Résumé par station×algo
    summary = (grand.groupby(["station","algo"], as_index=False)
               .agg(reward_sum=("reward_sum","mean"),
                    reward_mean=("reward_mean","mean"),
                    go_count=("go_count","mean"),
                    go_count_dec=("go_count_dec","mean"),
                    safety_ok_ratio=("safety_ok_ratio","mean"),
                    safety_ok_ratio_dec=("safety_ok_ratio_dec","mean"),
                    dist_sum_km=("dist_sum_km","mean"),
                    dist_sum_km_dec=("dist_sum_km_dec","mean"),
                    loiter_sum_min=("loiter_sum_min","mean"),
                    loiter_sum_min_dec=("loiter_sum_min_dec","mean"),
                    best_epoch=("best_epoch","mean"),
                    best_reward_episode=("best_reward_episode","mean"),
                    best_reward_episode_curve=("best_reward_episode_curve","mean")))
    summary_path = out_dir / "summary_station_algo.csv"
    summary.to_csv(summary_path, index=False)
    return grand_path, summary_path, grand

def save_comparison_plots(out_dir: Path, grand_df: pd.DataFrame, run_id: str):
    if grand_df.empty:
        return []
    paths = []
    def _simple_bar(df, col, title, ylabel, fname):
        if col not in df.columns:
            return None
        d = (df.groupby("algo", as_index=False)[col].mean())
        if d.empty: 
            return None
        fig = plt.figure()
        plt.bar(d["algo"], d[col])
        plt.title(title)
        plt.ylabel(ylabel)
        p = out_dir / f"{fname}_{run_id}.png"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return p
    P1 = _simple_bar(grand_df, "reward_sum", "Reward_sum moyen par algo", "reward_sum (moy.)", "cmp_reward_sum")
    P2 = _simple_bar(grand_df, "go_count_dec", "Nombre de GO moyen par algo (decisions)", "GO (moy.)", "cmp_go_count")
    P3 = _simple_bar(grand_df, "safety_ok_ratio_dec", "Sécurité moyenne par algo", "safety_ok_ratio (moy.)", "cmp_safety")
    P4 = _simple_bar(grand_df, "dist_sum_km_dec", "Distance (km) moyenne par algo", "km (moy.)", "cmp_distance")
    P5 = _simple_bar(grand_df, "loiter_sum_min_dec", "Loiter (min) moyen par algo", "min (moy.)", "cmp_loiter")
    P6 = _simple_bar(grand_df, "best_reward_episode_curve", "Meilleur reward_episode moyen", "best_reward_episode (moy.)", "cmp_best_reward_ep")
    for P in [P1,P2,P3,P4,P5,P6]:
        if P is not None:
            paths.append(str(P))
    return paths

# -------------------------------------------------
# 10) Orchestration & Agrégats globaux
# -------------------------------------------------
def main():
    run_id = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    print("[INFO] Chargement données...")
    t0 = time.time()
    df_all = load_and_merge(CFG['PRED_PATH'], CFG['MERRA_PATH'])
    if df_all.empty:
        raise SystemExit("Aucune donnée chargée. Vérifiez PRED_PATH et MERRA_PATH.")

    all_reports = []
    all_decisions = []
    all_curves = []

    for station in CFG["STATIONS"]:
        n_m = station.get("missions", CFG["MISSIONS_PER_STATION"])
        missions = build_missions(df_all, station, n_m)
        for m in missions:
            base_dir = ensure_dirs(station["name"], m["mission_id"])
            print(f"\n=== {station['name']} — mission {m['mission_id']} ===")
            for algo in ["REDQ", "CEM", "PPO"]:
                print(f"[TRAIN] Algo={algo}")
                env = MissionEnv(m, CFG)
                agent, curve = run_training_for_env(env, algo)
                # enrichir + exporter courbe
                curve.insert(0, "algo", algo)
                curve.insert(0, "station", station["name"])
                curve.insert(1, "mission", m["mission_id"])
                curve.insert(0, "run_id", run_id)
                curve["total_steps"] = curve["epoch"] * 256
                all_curves.append(curve)
                export_training_curve(curve, base_dir, algo)
                # décisions & rapport mission
                env2 = MissionEnv(m, CFG)
                dec_df, rep = export_decisions(env2, agent, algo, base_dir, station["name"], m["mission_id"])
                dec_df.insert(0, "run_id", run_id)
                all_decisions.append(dec_df)
                # best epoch & reward (simple)
                if not curve.empty and "reward_episode" in curve.columns:
                    idx_best = int(curve["reward_episode"].idxmax())
                    best_epoch = int(curve.loc[idx_best, "epoch"])
                    best_reward = float(curve.loc[idx_best, "reward_episode"])
                else:
                    best_epoch, best_reward = 0, 0.0
                # enrich rapport
                rep.update({
                    "run_id": run_id,
                    "timestamp_utc": _now_iso(),
                    "station": station["name"],
                    "mission": m["mission_id"],
                    "algo": algo,
                    "hotspots": len(m["hotspots"]),
                    "tau_gate": CFG["TAU"],
                    "radius_km": CFG["RADIUS_KM"],
                    "search_km": CFG["SEARCH_KM"],
                    "epochs": CFG["EPOCHS"],
                    "drone_speed_kmh": CFG["DRONE"]["SPEED_KMH"],
                    "loiter_min_base": CFG["DRONE"]["LOITER_MIN"],
                    "endurance_min": CFG["DRONE"]["ENDURANCE_MIN"],
                    "reserve_min": CFG["DRONE"]["RESERVE_MIN"],
                    "best_epoch": best_epoch,
                    "best_reward_episode": best_reward
                })
                all_reports.append(rep)

    # Agrégats globaux
    all_reports_df = pd.DataFrame(all_reports)
    all_decisions_df = pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()
    all_curves_df = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()

    # Sauvegardes de base
    all_reports_df.to_csv(OUT/"all_reports.csv", index=False)
    all_decisions_df.to_csv(OUT/"all_decisions.csv", index=False)
    all_curves_df.to_csv(OUT/"all_training_curves.csv", index=False)

    # Résumé station × algo
    if not all_reports_df.empty:
        summary = (all_reports_df
                   .groupby(["station","algo"], as_index=False)
                   .agg({
                       "reward_sum":"mean",
                       "reward_mean":"mean",
                       "go_count":"mean",
                       "loiter_sum_min":"mean",
                       "dist_sum_km":"mean",
                       "safety_ok_ratio":"mean",
                       "hotspots":"mean",
                       "best_epoch":"mean",
                       "best_reward_episode":"mean"
                   }))
        summary.to_csv(OUT/"summary_all_stations.csv", index=False)

    # Grand tableau + résumé station×algo + figures
    grand_path, summary_sa_path, grand_df = save_grand_table_and_summary(OUT, all_reports_df, all_decisions_df, all_curves_df)
    plot_paths = save_comparison_plots(OUT, grand_df, run_id)

    kpis = {
        "stations": len(CFG["STATIONS"]),
        "missions_total": int(sum([st.get("missions", CFG["MISSIONS_PER_STATION"]) for st in CFG["STATIONS"]])),
        "algos": ["REDQ","CEM","PPO"],
        "duration_sec": round(time.time()-t0, 3)
    }
    _write_manifest(OUT, CFG, run_id, kpis)

    print(f"\n[OK] Exports complets écrits dans: {OUT}")
    print("- all_reports.csv, all_decisions.csv, all_training_curves.csv, summary_all_stations.csv")
    print("- grand_table_all_stats.csv, summary_station_algo.csv, project_summary.json")
    for p in plot_paths:
        print(f"- figure: {p}")

if __name__ == "__main__":
    main()


# In[ ]:




