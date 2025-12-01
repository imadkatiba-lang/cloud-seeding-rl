#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, gc, time, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from torch.amp import autocast, GradScaler
from tqdm import tqdm

# ---------------------------
# Paramètres globaux
# ---------------------------
CSV_PATH    = r"C:\Users\tuf-p\Desktop\ARTICLES\Seeding_Drone\ERA5_NEAtl_2000-2025\ERA5_NEAtl_monthly_2000-2025_merged.csv"
OUTPUT_DIR  = r"C:\Users\tuf-p\Desktop\ARTICLES\Seeding2_logs_SeedingElig01"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Données & modèle
SEQ_LEN     = 36
BATCH_SIZE  = 256
LR          = 2e-4
MAX_EPOCHS  = 120
ACCUM_STEPS = 1
USE_SPATIAL = True
USE_AMP     = True
HISTO_CSV   = "training_history_seeding_elig01.csv"

# Split & sampling
IID_MODE               = False   # True → split i.i.d. (moins prospectif mais R2 test ↑)
MAX_STEPS_PER_EPOCH    = None    # None = full pass

# ClimaX-lite
D_MODEL     = 192
N_HEAD      = 6
N_LAYERS    = 5
D_FF        = 384
DROPOUT     = 0.15

# Régularisation additionnelle
WEIGHT_DECAY    = 1e-4
INPUT_NOISE_STD = 0.01
TIME_MASK_P     = 0.10
TIME_MASK_FRAC  = 0.15

# EMA
USE_EMA      = True
EMA_DECAY    = 0.999

# Anomalies / climatologie
USE_ANOM     = True

# Early stopping
PATIENCE     = 10   # <— demandé
MIN_DELTA    = 1e-12

# DataLoader
NUM_WORKERS         = 0
PERSISTENT_WORKERS  = False
PREFETCH_FACTOR     = 2

# ---------------------------
# Device & perfs
# ---------------------------
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device:", device)
if use_cuda:
    print("GPU:", torch.cuda.get_device_name(0))
    torch.set_float32_matmul_precision("medium")

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.set_num_threads(1)

scaler_amp = GradScaler("cuda" if use_cuda else "cpu", enabled=(USE_AMP and use_cuda))

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------- Metrics ----------
def mape_masked(y_true, y_pred, eps=1e-6):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nz = np.abs(y_true) > eps
    if nz.sum() == 0: return 0.0
    return float(np.mean(np.abs((y_true[nz]-y_pred[nz]) / y_true[nz])) * 100)

# ============================================================
# Chargement & parsing robuste de 'time'
# ============================================================
print("▶️ Lecture CSV…")
df = pd.read_csv(CSV_PATH)
print("Brut:", df.shape)

# Harmonise time/lat/lon
if "time" not in df.columns:
    for alt in ["valid_time", "date", "timestamp", "time_utc", "TIME"]:
        if alt in df.columns:
            df = df.rename(columns={alt: "time"})
            break
assert "time" in df.columns, "Aucune colonne 'time' trouvée."

if "latitude" not in df.columns and "lat" in df.columns:
    df = df.rename(columns={"lat": "latitude"})
if "longitude" not in df.columns and "lon" in df.columns:
    df = df.rename(columns={"lon": "longitude"})

# Parse 'time'
df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True, infer_datetime_format=True)
if df["time"].isna().all():
    df["time"] = pd.to_datetime(df["time"].astype(str), format="%Y-%m", errors="coerce", utc=True)
    if df["time"].isna().all():
        df["time"] = pd.to_datetime(df["time"].astype(str), format="%Y/%m", errors="coerce", utc=True)
if hasattr(df["time"].dt, "tz_localize"):
    try:
        df["time"] = df["time"].dt.tz_localize(None)
    except Exception:
        pass

# Numérise toutes les autres colonnes
for c in [col for col in df.columns if col != "time"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Mois + sin/cos (sans .dt)
t = pd.to_datetime(df["time"], errors="coerce", infer_datetime_format=True)
df = df.loc[t.notna()].copy()
df["time"] = t = t.loc[t.notna()]
t64m = df["time"].values.astype("datetime64[M]")
df["month"] = ((t64m.astype("int64") % 12) + 1).astype(np.int16)
df["mon_sin"] = np.sin(2*np.pi*(df["month"]-1)/12.0).astype(np.float32)
df["mon_cos"] = np.cos(2*np.pi*(df["month"]-1)/12.0).astype(np.float32)

# ============================================================
# Cible: seeding_eligibility ∈ [0,1]
# ============================================================
target_col = None
for cand in ["seeding_eligibility", "seeding_prob", "eligibility", "seed_eligibility"]:
    if cand in df.columns:
        target_col = cand
        break

drivers_pref = ["cape", "tcwv", "tp", "cin", "viwve", "viwvn"]
drivers = [c for c in drivers_pref if c in df.columns]
features_all = [c for c in df.columns if c not in ["time","latitude","longitude"]]

# Imputation (mois -> globale -> 0) avant la cible
for c in features_all:
    if c == "month": 
        continue
    med_global = df[c].median(skipna=True)
    med_month = df.groupby("month")[c].transform("median")
    df[c] = df[c].fillna(med_month).fillna(med_global).fillna(0.0)

# Cible douce si absente
if target_col is None:
    if len(drivers) == 0:
        raise ValueError("Aucun driver disponible pour construire une cible douce. Ajoute cape/tcwv/tp/cin/viwve/viwvn ou une colonne seeding_eligibility.")
    def robust01(s):
        q5, q95 = np.nanpercentile(s, [5,95])
        if not np.isfinite(q5) or not np.isfinite(q95) or q95 <= q5:
            q5, q95 = np.nanmin(s), np.nanmax(s)
        out = (s - q5) / max(q95 - q5, 1e-9)
        return np.clip(out, 0, 1)

    parts, weights = [], []
    if "cape" in drivers:  parts.append(robust01(df["cape"].values));  weights.append(0.40)
    if "tcwv" in drivers:  parts.append(robust01(df["tcwv"].values));  weights.append(0.30)
    if "tp"   in drivers:  parts.append(robust01(df["tp"].values));    weights.append(0.20)
    if "cin"  in drivers:
        cin_mag = robust01(np.abs(df["cin"].values))
        parts.append(1.0 - cin_mag); weights.append(0.10)
    if "viwve" in drivers: parts.append(robust01(np.abs(df["viwve"].values))); weights.append(0.05)
    if "viwvn" in drivers: parts.append(robust01(np.abs(df["viwvn"].values))); weights.append(0.05)

    w = np.array(weights, dtype=np.float32); w = w / w.sum()
    M = np.vstack(parts)
    y_soft = np.clip((w[:,None] * M).sum(axis=0), 0, 1).astype(np.float32)
    y_soft = np.clip(y_soft ** 0.90, 0, 1)  # léger sharpen
    df["seeding_eligibility"] = y_soft
    target_col = "seeding_eligibility"
    print(f"ℹ️ Cible douce '{target_col}' construite à partir de: {drivers}")
else:
    df[target_col] = np.clip(df[target_col].astype(np.float32), 0.0, 1.0)

# Core mask
core_mask = df["time"].notna() & df["latitude"].notna() & df["longitude"].notna() & df[target_col].notna()
df = df.loc[core_mask].copy()
print("Après garde time/lat/lon/target:", df.shape)

# ============================================================
# Anomalies/climatologie (option)
# ============================================================
extra_feats = []
if USE_ANOM:
    df["gid"] = pd.factorize(list(zip(df["latitude"].values, df["longitude"].values)))[0].astype(np.int32)
    anom_candidates = ["tcwv","cape","tp","cin","viwve","viwvn"]
    for c in anom_candidates:
        if c in df.columns:
            clim_c = df.groupby(["gid","month"])[c].transform("mean")
            col = f"{c}_anom"
            df[col] = (df[c] - clim_c).astype(np.float32)
            extra_feats.append(col)

# ============================================================
# Features finales
# ============================================================
base_features = [c for c in df.columns if c not in ["time","latitude","longitude",target_col,"gid"]]
# Lags par station
df = df.sort_values(["latitude","longitude","time"]).reset_index(drop=True)
grp = df.groupby(["latitude","longitude"], sort=False, group_keys=False)
lag_candidates = [c for c in ["cape","tcwv","tp","cin","viwve","viwvn"] if c in df.columns]
lags = [1,3,6,12]
for c in lag_candidates:
    for L in lags:
        col = f"{c}_lag{L}"
        df[col] = grp[c].shift(L).fillna(0.0).astype(np.float32)
        base_features.append(col)

# Saison
for c in ["mon_sin","mon_cos"]:
    if c not in base_features: base_features.append(c)
base_features = list(dict.fromkeys(base_features))

# Nettoyage final
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df[target_col] = df[target_col].fillna(0.0)
for c in base_features:
    df[c] = df[c].fillna(0.0)

print("Nettoyé:", df.shape)
nan_total = int(df[[target_col]+base_features].isna().sum().sum())
print("NaN restants (doit être 0):", nan_total)
assert nan_total == 0
print(f"Features finales ({len(base_features)}): {base_features[:12]}{' ...' if len(base_features)>12 else ''}")

# ============================================================
# Split + scalers (X uniquement; y déjà 0–1)
# ============================================================
gid_all = pd.factorize(list(zip(df["latitude"].values, df["longitude"].values)))[0].astype(np.int32)

if IID_MODE:
    rng = np.random.RandomState(42)
    r = rng.rand(len(df))
    mask_tr = r < 0.70
    mask_va = (r >= 0.70) & (r < 0.85)
    mask_te = r >= 0.85
else:
    times_sorted = df["time"].sort_values().to_numpy()
    t_train_end = times_sorted[int(0.70 * len(times_sorted))]
    t_val_end   = times_sorted[int(0.85 * len(times_sorted))]
    print("Cutoffs:", t_train_end, t_val_end)
    mask_tr = df["time"] <= t_train_end
    mask_va = (df["time"] > t_train_end) & (df["time"] <= t_val_end)
    mask_te = df["time"] > t_val_end

df_tr, df_val, df_te = df.loc[mask_tr], df.loc[mask_va], df.loc[mask_te]
print("Split lignes:", df_tr.shape, df_val.shape, df_te.shape)

scalerX = StandardScaler().fit(df_tr[base_features])
X_all = scalerX.transform(df[base_features]).astype(np.float32, copy=False)
y_all = df[target_col].astype(np.float32).to_numpy()
lat_all = df["latitude"].to_numpy(np.float32)
lon_all = df["longitude"].to_numpy(np.float32)
time_all = df["time"].to_numpy()

# ============================================================
# Fenêtres panel (ordre station→temps) + fallback de SEQ_LEN
# ============================================================
order = np.lexsort((time_all, gid_all))
X_all = X_all[order]; y_all = y_all[order]
lat_all = lat_all[order]; lon_all = lon_all[order]
time_all = time_all[order]; gid_all  = gid_all[order]
mask_tr_ord = mask_tr.to_numpy()[order]
mask_va_ord = mask_va.to_numpy()[order]
mask_te_ord = mask_te.to_numpy()[order]

def build_index_map(gids, set_mask, seq_len):
    idx_map = []
    uniq, first_idx, counts = np.unique(gids, return_index=True, return_counts=True)
    for start_g, cnt in zip(first_idx, counts):
        g_slice = slice(start_g, start_g + cnt)
        ends = np.arange(g_slice.start + seq_len - 1, g_slice.stop)
        if ends.size == 0: continue
        ends = ends[ set_mask[ends] ]
        if ends.size == 0: continue
        starts = ends - (seq_len - 1)
        idx_map.extend(zip(starts.tolist(), ends.tolist()))
    return idx_map

def build_all_indices(seq_len):
    return (
        build_index_map(gid_all, mask_tr_ord, seq_len),
        build_index_map(gid_all, mask_va_ord, seq_len),
        build_index_map(gid_all, mask_te_ord, seq_len),
    )

train_idx, val_idx, test_idx = build_all_indices(SEQ_LEN)
fallbacks = [24, 18, 12, 9, 6, 3, 1]
i_fb = 0
while (len(train_idx)==0 or len(val_idx)==0 or len(test_idx)==0) and i_fb < len(fallbacks):
    newL = fallbacks[i_fb]; i_fb += 1
    print(f"[WARN] Fenêtres vides → SEQ_LEN={newL}")
    SEQ_LEN = newL
    train_idx, val_idx, test_idx = build_all_indices(SEQ_LEN)

print(f"Fenêtres: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")

# ============================================================
# Dataset & Loaders
# ============================================================
class PanelSeqDataset(Dataset):
    def __init__(self, X, y, lat, lon, idx_pairs, use_spatial=True):
        self.X, self.y, self.lat, self.lon = X, y, lat, lon
        self.idx = idx_pairs
        self.use_spatial = use_spatial
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        s, e = self.idx[i]
        x = torch.from_numpy(self.X[s:e+1])  # [T,F]
        y = torch.tensor([self.y[e]], dtype=torch.float32)
        if self.use_spatial:
            spat = torch.tensor([self.lat[e], self.lon[e]], dtype=torch.float32)
        else:
            spat = torch.zeros(2, dtype=torch.float32)
        return x, y, spat, torch.tensor(0, dtype=torch.long)

def _sampler_if_needed(n_items, max_steps):
    if (max_steps is None) or (max_steps <= 0): return None
    need = min(max_steps * BATCH_SIZE, n_items)
    idx = np.random.choice(n_items, size=need, replace=False)
    return SubsetRandomSampler(idx)

def make_loader(dataset, shuffle, max_steps=None):
    sampler = _sampler_if_needed(len(dataset), max_steps)
    kwargs = dict(batch_size=BATCH_SIZE,
                  shuffle=(shuffle and sampler is None),
                  sampler=sampler,
                  num_workers=NUM_WORKERS,
                  pin_memory=use_cuda,
                  drop_last=False)
    if NUM_WORKERS > 0:
        kwargs.update(persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR)
    return DataLoader(dataset, **kwargs)

ds_tr = PanelSeqDataset(X_all, y_all, lat_all, lon_all, train_idx, USE_SPATIAL)
ds_va = PanelSeqDataset(X_all, y_all, lat_all, lon_all, val_idx,   USE_SPATIAL)
ds_te = PanelSeqDataset(X_all, y_all, lat_all, lon_all, test_idx,  USE_SPATIAL)

tr_loader = make_loader(ds_tr, True,  MAX_STEPS_PER_EPOCH)
va_loader = make_loader(ds_va, False, min((MAX_STEPS_PER_EPOCH or 0)//4, len(ds_va)//BATCH_SIZE) or None)
te_loader = make_loader(ds_te, False, min((MAX_STEPS_PER_EPOCH or 0)//4, len(ds_te)//BATCH_SIZE) or None)

print(f"Loaders: train_steps≈{len(tr_loader)} | val_steps≈{len(va_loader)} | test_steps≈{len(te_loader)}")

# ============================================================
# ClimaX-lite (Transformer causal) + Sigmoid en tête
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe, persistent=False)
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

def causal_mask(T, device):
    m = torch.full((T, T), float("-inf"), device=device)
    m = torch.triu(m, diagonal=1)
    return m

class ClimaXLite(nn.Module):
    def __init__(self, in_features, d_model=192, nhead=6, num_layers=5, dim_feedforward=384,
                 dropout=0.15, use_spatial=True):
        super().__init__()
        self.use_spatial = use_spatial
        self.inp = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.Dropout(dropout/2)
        )
        self.pos = SinusoidalPositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        head_in = d_model + (2 if use_spatial else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, max(32, head_in//2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, head_in//2), 1),
            nn.Sigmoid()  # sortie ∈ [0,1]
        )
    def forward(self, x, spat=None):
        h = self.inp(x)
        h = self.pos(h)
        T = h.size(1)
        mask = causal_mask(T, h.device)
        h = self.encoder(h, mask=mask)   # [B,T,D]
        h_last = h[:, -1, :]
        if self.use_spatial and spat is not None:
            h_last = torch.cat([h_last, spat], dim=-1)
        return self.head(h_last)         # [B,1] in [0,1]

model = ClimaXLite(
    in_features=X_all.shape[1],
    d_model=D_MODEL, nhead=N_HEAD, num_layers=N_LAYERS, dim_feedforward=D_FF,
    dropout=DROPOUT, use_spatial=USE_SPATIAL
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.99))

# === Scheduler: OneCycleLR (step par batch) ===
from torch.optim.lr_scheduler import OneCycleLR
steps_per_epoch = max(1, len(tr_loader))
scheduler = OneCycleLR(
    optimizer, max_lr=8e-4,
    epochs=MAX_EPOCHS, steps_per_epoch=steps_per_epoch,
    pct_start=0.10, div_factor=10.0, final_div_factor=10.0,
    anneal_strategy="cos"
)

# Pertes (y∈[0,1])
loss_huber = nn.SmoothL1Loss(beta=1.0)
loss_mse   = nn.MSELoss()

# ---------------- EMA ----------------
class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point:
                self.shadow[name] = param.detach().clone()
    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow and param.dtype.is_floating_point:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
    def apply(self, model: nn.Module):
        self.backup = {}
        for name, param in model.state_dict().items():
            if name in self.shadow and param.dtype.is_floating_point:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].data)
    def restore(self, model: nn.Module):
        for name, param in model.state_dict().items():
            if name in self.backup and param.dtype.is_floating_point:
                param.data.copy_(self.backup[name].data)
        self.backup = {}

ema = EMA(model, decay=EMA_DECAY) if USE_EMA else None

# ---------- métriques (y en [0,1]) ----------
def metrics01(preds, trues):
    preds = np.clip(np.asarray(preds).reshape(-1), 0, 1)
    trues = np.clip(np.asarray(trues).reshape(-1), 0, 1)
    rmse = float(np.sqrt(np.mean((preds-trues)**2)))
    mae  = float(mean_absolute_error(trues, preds))
    try:
        r2 = float(r2_score(trues, preds))
    except Exception:
        r2 = np.nan
    return {
        "RMSE": rmse,
        "MAE":  mae,
        "R2":   r2,
        "mMAPE": mape_masked(trues, preds),
        "Brier": rmse**2  # MSE sur [0,1]
    }

# ---------- train / eval ----------
def train_epoch(model, loader, epoch, total, ema=None, scheduler=None):
    model.train()
    total_loss, n = 0.0, 0
    preds_all, trues_all = [], []
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total} [train]", leave=False)
    for step, (xs, ys, spat, _) in enumerate(pbar, 1):
        xs, ys, spat = xs.to(device), ys.to(device), spat.to(device)
        # régularisation entrée
        if INPUT_NOISE_STD > 0: xs = xs + INPUT_NOISE_STD * torch.randn_like(xs)
        if TIME_MASK_P > 0 and random.random() < TIME_MASK_P:
            B,T,F = xs.shape
            L = max(1, int(T * TIME_MASK_FRAC))
            st = random.randint(0, T - L)
            xs[:, st:st+L, :] = 0.0

        with autocast(device_type="cuda", enabled=(USE_AMP and use_cuda)):
            yhat = model(xs, spat)               # [B,1] in [0,1]
            if yhat.ndim==1: yhat = yhat.view(-1,1)
            if ys.ndim==1:   ys   = ys.view(-1,1)
            loss = (0.7 * loss_huber(yhat, ys) + 0.3 * loss_mse(yhat, ys)) / ACCUM_STEPS

        if USE_AMP and use_cuda:
            scaler_amp.scale(loss).backward()
            if step % ACCUM_STEPS == 0:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer); scaler_amp.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None: scheduler.step()
                if ema is not None: ema.update(model)
        else:
            loss.backward()
            if step % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                if scheduler is not None: scheduler.step()
                if ema is not None: ema.update(model)

        bs = xs.size(0)
        total_loss += loss.item() * bs * ACCUM_STEPS
        preds_all.append(yhat.detach().cpu().numpy().reshape(-1))
        trues_all.append(ys.detach().cpu().numpy().reshape(-1))
        n += bs
        pbar.set_postfix(loss=f"{(total_loss/max(1,n)):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    cleanup()

    preds = np.concatenate(preds_all) if len(preds_all)>0 else np.array([])
    trues = np.concatenate(trues_all) if len(trues_all)>0 else np.array([])
    met = {"loss": total_loss / max(1,n)}
    if preds.size>0:
        met.update(metrics01(preds, trues))
    else:
        met.update({"RMSE":np.nan,"MAE":np.nan,"R2":np.nan,"mMAPE":np.nan,"Brier":np.nan})
    return met, time.time()-t0

@torch.inference_mode()
def eval_epoch(model, loader, epoch, total, phase="val", ema=None):
    if len(loader)==0:
        return {"loss":np.nan,"RMSE":np.nan,"MAE":np.nan,"R2":np.nan,"mMAPE":np.nan,"Brier":np.nan}, np.array([]), np.array([]), 0.0
    model.eval()
    total_loss, n = 0.0, 0
    preds_all, trues_all = [], []
    t0 = time.time()

    if ema is not None: ema.apply(model)

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total} [{phase}]", leave=False)
    for xs,ys,spat,_ in pbar:
        xs, ys, spat = xs.to(device), ys.to(device), spat.to(device)
        yhat = model(xs, spat)                # [B,1] in [0,1]
        if yhat.ndim==1: yhat = yhat.view(-1,1)
        if ys.ndim==1:   ys   = ys.view(-1,1)
        loss = 0.7 * loss_huber(yhat, ys) + 0.3 * loss_mse(yhat, ys)
        total_loss += loss.item()*xs.size(0)
        preds_all.append(yhat.cpu().numpy().reshape(-1))
        trues_all.append(ys.cpu().numpy().reshape(-1))
        n += xs.size(0)
        pbar.set_postfix(loss=f"{(total_loss/max(1,n)):.4f}")

    if ema is not None: ema.restore(model)

    preds = np.concatenate(preds_all) if len(preds_all)>0 else np.array([])
    trues = np.concatenate(trues_all) if len(trues_all)>0 else np.array([])
    if preds.size==0:
        met = {"loss":np.nan,"RMSE":np.nan,"MAE":np.nan,"R2":np.nan,"mMAPE":np.nan,"Brier":np.nan}
    else:
        met = {"loss": total_loss/max(1,n), **metrics01(preds, trues)}
    return met, preds, trues, time.time()-t0

# ============================================================
# Entraînement (sélection sur val_RMSE) + EMA + EarlyStopping
# ============================================================
history_rows = []
best_sel = float("inf")
best_state, best_epoch = None, 0
epochs_no_improve = 0
CURRENT_EPOCH = 0

print("▶️ Démarrage entraînement (Seeding eligibility 0–1)…")
for epoch in range(1, MAX_EPOCHS+1):
    CURRENT_EPOCH = epoch
    tr_metrics, tr_time = train_epoch(model, tr_loader, epoch, MAX_EPOCHS, ema=ema, scheduler=scheduler)
    va_metrics, _, _, va_time = eval_epoch(model, va_loader, epoch, MAX_EPOCHS, phase="val", ema=ema)

    # Sélection + early stopping sur RMSE (plus petit = mieux)
    metric_for_sel = va_metrics["RMSE"]
    if metric_for_sel < best_sel - MIN_DELTA:
        best_sel = metric_for_sel
        best_epoch = epoch
        epochs_no_improve = 0
        if ema is not None:
            ema.apply(model)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ema.restore(model)
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1

    print(f"[Epoch {epoch:03d}/{MAX_EPOCHS}] "
          f"Train: loss={tr_metrics['loss']:.4f} RMSE={tr_metrics['RMSE']:.4f} MAE={tr_metrics['MAE']:.4f} R2={tr_metrics['R2']:.3f} Brier={tr_metrics['Brier']:.5f} "
          f"| Val(EMA): loss={va_metrics['loss']:.4f} RMSE={va_metrics['RMSE']:.4f} MAE={va_metrics['MAE']:.4f} R2={va_metrics['R2']:.3f} Brier={va_metrics['Brier']:.5f} mMAPE={va_metrics['mMAPE']:.2f}% "
          f"| t={tr_time:.1f}s/{va_time:.1f}s | best@{best_epoch:03d} RMSE={best_sel:.4f} | no_improve={epochs_no_improve}")

    history_rows.append({
        "epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]),
        **{f"train_{k}": v for k,v in tr_metrics.items()},
        **{f"val_{k}":   v for k,v in va_metrics.items()},
        "train_time_s": tr_time, "val_time_s": va_time
    })
    pd.DataFrame(history_rows).to_csv(os.path.join(OUTPUT_DIR, HISTO_CSV), index=False)

    # Early stopping
    if epochs_no_improve >= PATIENCE:
        print(f"⏹️ Early stopping (patience={PATIENCE}).")
        break

print(f"Historique écrit: {os.path.join(OUTPUT_DIR, HISTO_CSV)}")

# Recharge le meilleur modèle
if best_state is not None:
    model.load_state_dict(best_state)
    print(f"✅ Meilleur modèle (val_RMSE={best_sel:.4f} @epoch={best_epoch}) rechargé pour le test.")

# ============================================================
# Test final
# ============================================================
print("▶️ Évaluation test…")
te_metrics, te_pred, te_true, te_time = eval_epoch(model, te_loader, CURRENT_EPOCH, MAX_EPOCHS, phase="test", ema=None)
print(f"[TEST] loss={te_metrics['loss']:.4f} RMSE={te_metrics['RMSE']:.4f} MAE={te_metrics['MAE']:.4f} "
      f"R2={te_metrics['R2']:.3f} Brier={te_metrics['Brier']:.5f} mMAPE={te_metrics['mMAPE']:.2f}% | time={te_time:.1f}s")

# Sauvegarde des prédictions test (utile pour RL/vols)
out_csv = os.path.join(OUTPUT_DIR, "test_predictions_seeding_elig01.csv")
te_n = len(te_pred)
df_te_sorted = df.loc[mask_te].sort_values(["latitude","longitude","time"]).tail(te_n)
pd.DataFrame({
    "time": df_te_sorted["time"].values,
    "latitude": df_te_sorted["latitude"].values,
    "longitude": df_te_sorted["longitude"].values,
    "eligibility_true": te_true,
    "eligibility_pred": te_pred
}).to_csv(out_csv, index=False)
print(f"Preds test écrites : {out_csv}")

