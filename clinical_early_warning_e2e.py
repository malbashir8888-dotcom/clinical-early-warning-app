%%writefile /content/clinical_early_warning_e2e.py
# clinical_early_warning_e2e.py (single-file)
# End-to-end: Zenodo -> Inspect -> Prepare -> Train -> UI (Streamlit)
# UI upgrades: Heatmap + Top features + Top timestep + Similar patients (no retraining)
# Exports: CSV + Excel (multi-sheet) + PDF report (with plots)

import sys, os, json, argparse, gzip, time, math, tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

try:
    import ijson
    HAS_IJSON = True
except Exception:
    HAS_IJSON = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

ZENODO_API = "https://zenodo.org/api/records/{record_id}"


# -------------------- argv guard for Jupyter/Colab --------------------
def _sanitize_argv_for_jupyter():
    try:
        if "ipykernel" in sys.modules:
            new_argv = [sys.argv[0]]
            skip_next = False
            for i, a in enumerate(sys.argv[1:], start=1):
                if skip_next:
                    skip_next = False
                    continue
                if a == "-f" and i + 1 < len(sys.argv):
                    skip_next = True
                    continue
                if isinstance(a, str) and a.endswith(".json") and "kernel-" in a:
                    continue
                new_argv.append(a)
            sys.argv = new_argv
    except Exception:
        pass


# -------------------- HTTP helpers (robust) --------------------
def _session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ColabZenodoDownloader/1.0",
        "Accept": "*/*",
    })
    return s


def _http_get_with_retry(url: str, timeout: int = 180, max_tries: int = 10, backoff: float = 2.0, stream: bool = False):
    last_err = None
    s = _session()
    for i in range(max_tries):
        try:
            r = s.get(url, timeout=timeout, stream=stream, allow_redirects=True)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(60, backoff ** i)
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        wait = max(wait, int(ra))
                    except Exception:
                        pass
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(min(60, backoff ** i))
    raise RuntimeError(f"Failed GET after retries: {url} ; last_err={last_err}")


def _download_with_resume(url: str, out_path: Path, expected_size: Optional[int] = None, chunk_mb: int = 4):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s = _session()

    mode = "wb"
    headers = {}
    existing = out_path.stat().st_size if out_path.exists() else 0

    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with s.get(url, stream=True, timeout=240, headers=headers, allow_redirects=True) as r:
        if existing > 0 and r.status_code == 200:
            existing = 0
            mode = "wb"
        r.raise_for_status()

        total = None
        cl = r.headers.get("Content-Length")
        if cl is not None:
            try:
                total = int(cl) + (existing if mode == "ab" else 0)
            except Exception:
                total = None
        if expected_size and expected_size > 0:
            total = expected_size

        pbar = tqdm(total=total, initial=existing, unit="B", unit_scale=True, desc=f"Downloading {out_path.name}")
        with out_path.open(mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        pbar.close()

    if expected_size and expected_size > 0:
        got = out_path.stat().st_size
        if got != expected_size:
            raise RuntimeError(f"Downloaded size mismatch: got={got} expected={expected_size}. Try --force_redownload.")


def download_zenodo(record_id: int, out_dir: Path, out_name: str = "json", force_redownload: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    if force_redownload and out_path.exists():
        out_path.unlink()

    try:
        r = _http_get_with_retry(ZENODO_API.format(record_id=record_id), timeout=60, max_tries=6, backoff=2.0, stream=False)
        meta = r.json()
        files = meta.get("files", [])
        if files:
            target = sorted(files, key=lambda x: int(x.get("size", 0) or 0), reverse=True)[0]
            size = int(target.get("size", 0) or 0)
            dl = target.get("links", {}).get("self") or target.get("links", {}).get("download")

            if dl:
                api_out = out_path
                if force_redownload and api_out.exists():
                    api_out.unlink()
                if api_out.exists() and size > 0 and api_out.stat().st_size == size:
                    print(f"[download] exists (API): {api_out}")
                    return api_out
                print(f"[download] API: {dl} -> {api_out}")
                _download_with_resume(dl, api_out, expected_size=size if size > 0 else None)
                return api_out
    except Exception as e:
        print(f"[download] API failed, fallback to direct. reason={e}")

    direct = f"https://zenodo.org/records/{record_id}/files/{out_name}?download=1"
    if out_path.exists() and out_path.stat().st_size > 0 and not force_redownload:
        print(f"[download] exists (direct): {out_path}")
        return out_path

    print(f"[download] DIRECT: {direct} -> {out_path}")
    _download_with_resume(direct, out_path, expected_size=None)
    return out_path


# -------------------- file format helpers --------------------
def _is_gzip(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            sig = f.read(2)
        return sig == b"\x1f\x8b"
    except Exception:
        return False


def open_maybe_gzip(path: Path, mode: str = "rt", encoding: str = "utf-8"):
    if _is_gzip(path):
        return gzip.open(path, mode=mode, encoding=encoding if "t" in mode else None)
    return path.open(mode=mode, encoding=encoding if "t" in mode else None)


def _as_float(x) -> np.ndarray:
    return np.array(x, dtype=np.float32)


# -------------------- JSON loaders --------------------
def load_json_records(path: Path, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        with open_maybe_gzip(path, "rt") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj[:max_records] if max_records else obj
        if isinstance(obj, dict):
            for k in ["data", "records", "patients", "examples", "train", "test", "items"]:
                if k in obj and isinstance(obj[k], list):
                    recs = obj[k]
                    return recs[:max_records] if max_records else recs
            return [obj]
    except Exception:
        pass

    recs: List[Dict[str, Any]] = []
    try:
        with open_maybe_gzip(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                recs.append(json.loads(line))
                if max_records and len(recs) >= max_records:
                    break
        if recs:
            return recs
    except Exception:
        recs = []

    if not HAS_IJSON:
        raise RuntimeError("Failed to parse JSON. Please install ijson: pip install ijson")

    try:
        recs = []
        with open_maybe_gzip(path, "rb") as f:
            for rec in ijson.items(f, "item"):
                if isinstance(rec, dict):
                    recs.append(rec)
                else:
                    recs.append({"_value": rec})
                if max_records and len(recs) >= max_records:
                    break
        if recs:
            return recs
    except Exception:
        pass

    for key in ["data", "records", "patients", "examples", "train", "test", "items"]:
        try:
            recs = []
            with open_maybe_gzip(path, "rb") as f:
                for rec in ijson.items(f, f"{key}.item"):
                    if isinstance(rec, dict):
                        recs.append(rec)
                    else:
                        recs.append({"_value": rec})
                    if max_records and len(recs) >= max_records:
                        break
            if recs:
                return recs
        except Exception:
            continue

    raise RuntimeError("Failed to parse JSON with ijson. Format unexpected or file corrupted.")


def inspect_dataset(json_path: Path, n: int = 1):
    print(f"[inspect] file: {json_path}")
    print(f"[inspect] size_mb: {json_path.stat().st_size / (1024*1024):.2f}")
    print(f"[inspect] gzip: {_is_gzip(json_path)}  ijson: {HAS_IJSON}")
    recs = load_json_records(json_path, max_records=max(n, 1))
    print(f"[inspect] loaded_records: {len(recs)}")
    for i, r in enumerate(recs[:n]):
        if not isinstance(r, dict):
            print(f"[inspect] rec[{i}] type={type(r)} value_preview={str(r)[:200]}")
            continue
        keys = list(r.keys())
        print(f"[inspect] rec[{i}] keys({len(keys)}): {keys[:40]}")
        for k in ["data", "forward", "backward", "values", "masks", "X", "M", "label", "labels", "y", "target"]:
            if k in r:
                v = r[k]
                if isinstance(v, dict):
                    print(f"  - {k}: dict keys={list(v.keys())[:30]}")
                elif isinstance(v, list):
                    print(f"  - {k}: list len={len(v)}")
                else:
                    print(f"  - {k}: {type(v)}")


# -------------------- Feature map (English + Arabic) --------------------
def default_feature_map_35() -> pd.DataFrame:
    rows = [
        (0,  "DiasABP",   "ضغط الدم الانبساطي الشرياني"),
        (1,  "HR",        "معدل ضربات القلب"),
        (2,  "Na",        "الصوديوم"),
        (3,  "Lactate",   "اللاكتات"),
        (4,  "NIDiasABP", "ضغط الدم الانبساطي (غير غازي)"),
        (5,  "PaO2",      "ضغط الأكسجين الشرياني"),
        (6,  "WBC",       "كريات الدم البيضاء"),
        (7,  "pH",        "الأس الهيدروجيني"),
        (8,  "Albumin",   "الألبومين"),
        (9,  "ALT",       "ALT إنزيم الكبد"),
        (10, "Glucose",   "سكر الدم"),
        (11, "SaO2",      "إشباع الأكسجين الشرياني"),
        (12, "Temp",      "درجة الحرارة"),
        (13, "AST",       "AST إنزيم الكبد"),
        (14, "Bilirubin", "البيليروبين"),
        (15, "HCO3",      "البيكربونات"),
        (16, "BUN",       "يوريا الدم (BUN)"),
        (17, "RespRate",  "معدل التنفس"),
        (18, "Mg",        "المغنيسيوم"),
        (19, "HCT",       "الهيماتوكريت"),
        (20, "SysABP",    "ضغط الدم الانقباضي الشرياني"),
        (21, "FiO2",      "نسبة الأكسجين المستنشق (FiO2)"),
        (22, "K",         "البوتاسيوم"),
        (23, "GCS",       "مقياس غلاسكو للغيبوبة"),
        (24, "Cholesterol","الكوليسترول"),
        (25, "NISysABP",  "ضغط الدم الانقباضي (غير غازي)"),
        (26, "TroponinT", "تروبونين T"),
        (27, "MAP",       "متوسط الضغط الشرياني (MAP)"),
        (28, "TroponinI", "تروبونين I"),
        (29, "PaCO2",     "ضغط ثاني أكسيد الكربون الشرياني"),
        (30, "Platelets", "الصفائح الدموية"),
        (31, "Creatinine","الكرياتينين"),
        (32, "ALP",       "الفوسفاتاز القلوي (ALP)"),
        (33, "NIMAP",     "متوسط الضغط الشرياني (غير غازي)"),
        (34, "AnionGap",  "فجوة الأيونات (Anion Gap)"),
    ]
    df = pd.DataFrame(rows, columns=["feat_idx", "feature_en", "feature_ar"])
    df["feat_name"] = df["feat_idx"].apply(lambda i: f"feat_{i}")
    return df[["feat_idx", "feat_name", "feature_en", "feature_ar"]]


def save_feature_map_xlsx(out_path: Path, feat_dim: int):
    df = default_feature_map_35()
    if feat_dim != 35:
        df = pd.DataFrame({
            "feat_idx": list(range(feat_dim)),
            "feat_name": [f"feat_{i}" for i in range(feat_dim)],
            "feature_en": [f"feat_{i}" for i in range(feat_dim)],
            "feature_ar": [f"ميزة {i}" for i in range(feat_dim)],
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    return df


# -------------------- Robust record parser (BRITS/PhysioNet) --------------------
def extract_record(rec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[str]], int]:
    def _find_label(obj: Dict[str, Any]) -> Optional[int]:
        for k in ["label", "y", "target", "mortality", "outcome", "Y"]:
            if k in obj:
                try:
                    return int(obj[k])
                except Exception:
                    pass
        if "labels" in obj:
            try:
                if isinstance(obj["labels"], (list, tuple)) and len(obj["labels"]) > 0:
                    return int(obj["labels"][0])
                return int(obj["labels"])
            except Exception:
                pass
        if "meta" in obj and isinstance(obj["meta"], dict):
            return _find_label(obj["meta"])
        return None

    label = _find_label(rec)
    if label is None and "data" in rec and isinstance(rec["data"], dict):
        label = _find_label(rec["data"])
    if label is None:
        raise ValueError("No label found in record.")

    feature_names = None
    if "feature_names" in rec and isinstance(rec["feature_names"], list):
        feature_names = [str(x) for x in rec["feature_names"]]
    elif "features" in rec and isinstance(rec["features"], list) and all(isinstance(x, str) for x in rec["features"]):
        feature_names = rec["features"]
    elif "data" in rec and isinstance(rec["data"], dict):
        d0 = rec["data"]
        if "feature_names" in d0 and isinstance(d0["feature_names"], list):
            feature_names = [str(x) for x in d0["feature_names"]]

    if "values" in rec and ("masks" in rec or "mask" in rec):
        v = _as_float(rec["values"])
        m = _as_float(rec.get("masks", rec.get("mask")))
        t = _as_float(rec.get("times", rec.get("t", np.arange(v.shape[0])))).reshape(-1)
        return v, m, t, feature_names, label

    if "data" in rec and isinstance(rec["data"], dict):
        d = rec["data"]
        if "values" in d and ("masks" in d or "mask" in d):
            v = _as_float(d["values"])
            m = _as_float(d.get("masks", d.get("mask")))
            t = _as_float(d.get("times", d.get("t", np.arange(v.shape[0])))).reshape(-1)
            return v, m, t, feature_names, label

    def _try_brits(obj: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if "forward" in obj and isinstance(obj["forward"], dict):
            fwd = obj["forward"]
            if "values" in fwd and ("masks" in fwd or "mask" in fwd):
                v = _as_float(fwd["values"])
                m = _as_float(fwd.get("masks", fwd.get("mask")))
                t = np.arange(v.shape[0], dtype=np.float32)
                return v, m, t
        return None

    got = _try_brits(rec)
    if got is None and "data" in rec and isinstance(rec["data"], dict):
        got = _try_brits(rec["data"])
    if got is not None:
        v, m, t = got
        return v, m, t, feature_names, label

    def _try_brits_alt(obj: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if "forward" in obj and isinstance(obj["forward"], dict):
            fwd = obj["forward"]
            if "X" in fwd and ("M" in fwd or "mask" in fwd):
                v = _as_float(fwd["X"])
                m = _as_float(fwd.get("M", fwd.get("mask")))
                t = np.arange(v.shape[0], dtype=np.float32)
                return v, m, t
        return None

    got = _try_brits_alt(rec)
    if got is None and "data" in rec and isinstance(rec["data"], dict):
        got = _try_brits_alt(rec["data"])
    if got is not None:
        v, m, t = got
        return v, m, t, feature_names, label

    if "X" in rec and ("M" in rec or "mask" in rec):
        v = _as_float(rec["X"])
        m = _as_float(rec.get("M", rec.get("mask")))
        t = _as_float(rec.get("t", np.arange(v.shape[0]))).reshape(-1)
        return v, m, t, feature_names, label

    raise ValueError("Unsupported record format (BRITS/PhysioNet parser did not match).")


# -------------------- dataset preparation --------------------
def pad_batch(values_list, mask_list, times_list):
    lengths = [v.shape[0] for v in values_list]
    max_len = max(lengths)
    feat_dim = values_list[0].shape[1]

    V = np.zeros((len(values_list), max_len, feat_dim), dtype=np.float32)
    M = np.zeros((len(values_list), max_len, feat_dim), dtype=np.float32)
    Tm = np.zeros((len(values_list), max_len), dtype=np.float32)
    L = np.array(lengths, dtype=np.int64)

    for i, (v, m, t) in enumerate(zip(values_list, mask_list, times_list)):
        l = v.shape[0]
        V[i, :l] = v
        if m.shape != v.shape:
            m = np.ones_like(v, dtype=np.float32)
        M[i, :l] = m
        if t.shape[0] != l:
            t = np.arange(l, dtype=np.float32)
        Tm[i, :l] = t
    return V, M, Tm, L


def prepare_dataset(json_path: Path, out_dir: Path, max_records: Optional[int] = None, min_ok: int = 200) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_json_records(json_path, max_records=max_records)

    values_list, mask_list, times_list, y_list = [], [], [], []
    feature_names: Optional[List[str]] = None
    bad = 0

    for rec in tqdm(records, desc="Parsing"):
        try:
            v, m, t, fn, y = extract_record(rec)
            if feature_names is None and fn is not None:
                feature_names = fn
            v = v.reshape(v.shape[0], -1).astype(np.float32)
            m = m.reshape(m.shape[0], -1).astype(np.float32)
            if m.shape != v.shape:
                m = np.ones_like(v, dtype=np.float32)
            t = np.array(t, dtype=np.float32).reshape(-1)

            values_list.append(v); mask_list.append(m); times_list.append(t); y_list.append(int(y))
        except Exception:
            bad += 1

    if len(values_list) < min_ok:
        raise RuntimeError(f"Too few parsed records ({len(values_list)}), bad={bad}. Run inspect and verify parser.")

    V, M, Tm, L = pad_batch(values_list, mask_list, times_list)
    y = np.array(y_list, dtype=np.int64)

    obs = M.reshape(-1, M.shape[-1])
    vals = V.reshape(-1, V.shape[-1])
    eps = 1e-6
    mean = np.sum(vals * obs, axis=0) / (np.sum(obs, axis=0) + eps)
    var = np.sum(((vals - mean) ** 2) * obs, axis=0) / (np.sum(obs, axis=0) + eps)
    std = np.sqrt(var + eps)

    Vn = ((V - mean) / std) * M

    out_path = out_dir / "prepared_dataset.npz"
    np.savez_compressed(
        out_path,
        V=Vn.astype(np.float32),
        M=M.astype(np.float32),
        Tm=Tm.astype(np.float32),
        L=L.astype(np.int64),
        y=y.astype(np.int64),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        feature_names=np.array(feature_names if feature_names else [], dtype=object),
        n_records=int(V.shape[0]),
        feat_dim=int(V.shape[-1]),
        bad_records=int(bad),
    )

    fmap_path = out_dir / "feature_map.xlsx"
    save_feature_map_xlsx(fmap_path, int(V.shape[-1]))
    print(f"[prepare] feature_map saved: {fmap_path}")
    print(f"[prepare] saved: {out_path} n={int(V.shape[0])} feat_dim={int(V.shape[-1])} bad={bad}")
    return out_path


# -------------------- Torch dataset + model --------------------
class ICUTimeSeriesDataset(Dataset):
    def __init__(self, V, M, L, y):
        self.V = torch.from_numpy(V)
        self.M = torch.from_numpy(M)
        self.L = torch.from_numpy(L)
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.V.shape[0]

    def __getitem__(self, idx):
        return self.V[idx], self.M[idx], self.L[idx], self.y[idx]


class GRUAttnClassifier(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(input_size=feat_dim * 2, hidden_size=hidden, batch_first=True)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, V, M, L):
        x = torch.cat([V, M], dim=-1)  # [B,T,2F]
        packed = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        h_packed, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h_packed, batch_first=True)  # [B,T,H]

        att = self.attn(h).squeeze(-1)  # [B,T]
        T = h.size(1)
        rng = torch.arange(T, device=L.device).unsqueeze(0)
        mask_t = (rng < L.unsqueeze(1)).float()
        att = att + (mask_t - 1) * 1e9
        alpha = torch.softmax(att, dim=1)

        ctx = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # [B,H]
        logit = self.head(ctx).squeeze(-1)
        return logit, alpha

    def encode_context(self, V, M, L):
        """Return context vector (attention-weighted) for similarity search. No training needed."""
        x = torch.cat([V, M], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        h_packed, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h_packed, batch_first=True)

        att = self.attn(h).squeeze(-1)
        T = h.size(1)
        rng = torch.arange(T, device=L.device).unsqueeze(0)
        mask_t = (rng < L.unsqueeze(1)).float()
        att = att + (mask_t - 1) * 1e9
        alpha = torch.softmax(att, dim=1)

        ctx = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # [B,H]
        return ctx


def train_model(prepared_npz: Path, out_dir: Path, epochs: int, batch_size: int, lr: float, seed: int = 7) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed); torch.manual_seed(seed)

    d = np.load(prepared_npz, allow_pickle=True)
    V, M, L, y = d["V"], d["M"], d["L"], d["y"]
    feat_dim = int(V.shape[-1])

    idx = np.arange(V.shape[0])
    tr, te = train_test_split(idx, test_size=0.15, random_state=seed, stratify=y)
    tr, va = train_test_split(tr, test_size=0.15, random_state=seed, stratify=y[tr])

    train_ds = ICUTimeSeriesDataset(V[tr], M[tr], L[tr], y[tr])
    val_ds   = ICUTimeSeriesDataset(V[va], M[va], L[va], y[va])
    test_ds  = ICUTimeSeriesDataset(V[te], M[te], L[te], y[te])

    def collate(batch):
        Vb, Mb, Lb, yb = zip(*batch)
        return torch.stack(Vb), torch.stack(Mb), torch.stack(Lb), torch.stack(yb)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUAttnClassifier(feat_dim=feat_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_path = out_dir / "best_model.pt"

    def eval_dl(dl):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for Vb, Mb, Lb, yb in dl:
                Vb, Mb, Lb = Vb.to(device), Mb.to(device), Lb.to(device)
                logit, _ = model(Vb, Mb, Lb)
                prob = torch.sigmoid(logit).cpu().numpy()
                ys.append(yb.numpy()); ps.append(prob)
        yt = np.concatenate(ys); yp = np.concatenate(ps)
        return float(roc_auc_score(yt, yp)), float(average_precision_score(yt, yp))

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for Vb, Mb, Lb, yb in tqdm(train_dl, desc=f"Epoch {ep}/{epochs}"):
            Vb, Mb, Lb, yb = Vb.to(device), Mb.to(device), Lb.to(device), yb.to(device)
            logit, _ = model(Vb, Mb, Lb)
            loss = loss_fn(logit, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * Vb.size(0)

        val_auc, val_ap = eval_dl(val_dl)
        print(f"[Epoch {ep}] loss={total/len(train_ds):.4f} val_auc={val_auc:.4f} val_ap={val_ap:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model_state": model.state_dict(), "feat_dim": feat_dim}, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_auc, test_ap = eval_dl(test_dl)

    meta = {
        "prepared": str(prepared_npz),
        "model": str(best_path),
        "feat_dim": feat_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "val_best_auc": best_auc,
        "test_auc": test_auc,
        "test_ap": test_ap,
        "device": str(device),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[TEST] auc={test_auc:.4f} ap={test_ap:.4f}")
    return best_path


# -------------------- Captum explanations (no retraining) --------------------
def _expand_to_batch(M: torch.Tensor, L: torch.Tensor, B: int):
    if M.size(0) == 1 and B > 1:
        M = M.repeat(B, 1, 1)
    if L.size(0) == 1 and B > 1:
        L = L.repeat(B)
    return M, L


class CaptumWrapper(nn.Module):
    def __init__(self, model: nn.Module, M_fixed: torch.Tensor, L_fixed: torch.Tensor):
        super().__init__()
        self.model = model
        self.M_fixed = M_fixed
        self.L_fixed = L_fixed

    def forward(self, V_in: torch.Tensor):
        B = V_in.size(0)
        M, L = _expand_to_batch(self.M_fixed, self.L_fixed, B)
        logit, _ = self.model(V_in, M, L)
        return logit


def explain_ig(model, V_np, M_np, L_np, idx: int, n_steps: int = 64, topk: int = 15):
    from captum.attr import IntegratedGradients
    device = next(model.parameters()).device
    model.eval()

    v = torch.from_numpy(V_np[idx:idx+1]).to(device).requires_grad_(True)
    m = torch.from_numpy(M_np[idx:idx+1]).to(device)
    l = torch.from_numpy(L_np[idx:idx+1]).to(device)
    wrapper = CaptumWrapper(model, m, l).to(device)

    base = torch.zeros_like(v)
    ig = IntegratedGradients(wrapper)
    attr = ig.attribute(v, baselines=base, n_steps=n_steps)
    attr = attr.detach().cpu().numpy()[0]

    feat_imp = np.sum(np.abs(attr), axis=0)
    time_imp = np.sum(np.abs(attr), axis=1)
    top_feat = np.argsort(-feat_imp)[:topk]
    t_star = int(np.argmax(time_imp))

    return {
        "method": "IntegratedGradients",
        "attr_full": attr,            # [T,F]
        "feat_importance": feat_imp,  # [F]
        "time_importance": time_imp,  # [T]
        "top_feat_idx": top_feat,
        "top_time": t_star,
    }


def explain_gradshap(model, V_np, M_np, L_np, idx: int, n_samples: int = 50, stdevs: float = 0.01, topk: int = 15):
    from captum.attr import GradientShap
    device = next(model.parameters()).device
    model.eval()

    v = torch.from_numpy(V_np[idx:idx+1]).to(device).requires_grad_(True)
    m = torch.from_numpy(M_np[idx:idx+1]).to(device)
    l = torch.from_numpy(L_np[idx:idx+1]).to(device)
    wrapper = CaptumWrapper(model, m, l).to(device)

    base0 = torch.zeros_like(v)
    base1 = torch.zeros_like(v) + torch.randn_like(v) * stdevs
    baselines = torch.cat([base0, base1], dim=0)

    gs = GradientShap(wrapper)
    attr = gs.attribute(v, baselines=baselines, n_samples=n_samples)
    attr = attr.detach().cpu().numpy()[0]

    feat_imp = np.sum(np.abs(attr), axis=0)
    time_imp = np.sum(np.abs(attr), axis=1)
    top_feat = np.argsort(-feat_imp)[:topk]
    t_star = int(np.argmax(time_imp))

    return {
        "method": "GradientShap",
        "attr_full": attr,
        "feat_importance": feat_imp,
        "time_importance": time_imp,
        "top_feat_idx": top_feat,
        "top_time": t_star,
    }


# -------------------- Case storage + exports --------------------
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def save_case(cases_dir: Path, payload: Dict[str, Any]) -> Path:
    cases_dir.mkdir(parents=True, exist_ok=True)
    cid = payload.get("case_id") or f"case_{time.strftime('%Y%m%d_%H%M%S')}"
    p = cases_dir / f"{cid}.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p

def list_cases(cases_dir: Path) -> List[Path]:
    if not cases_dir.exists():
        return []
    return sorted(cases_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

def load_case(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def export_case_csv(out_path: Path, case: Dict[str, Any]) -> Path:
    rows = []
    core = {
        "case_id": case.get("case_id"),
        "patient_index": case.get("patient_index"),
        "risk": case.get("risk"),
        "threshold": case.get("threshold"),
        "decision": case.get("decision"),
        "label": case.get("label"),
        "top_time": case.get("top_time"),
        "explain_method": case.get("explain_method"),
    }
    for k, v in core.items():
        rows.append((k, v))
    df = pd.DataFrame(rows, columns=["key", "value"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path

def export_case_excel(out_path: Path, case: Dict[str, Any]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        summary = pd.DataFrame([{
            "case_id": case.get("case_id"),
            "patient_index": case.get("patient_index"),
            "risk": case.get("risk"),
            "threshold": case.get("threshold"),
            "decision": case.get("decision"),
            "label": case.get("label"),
            "top_time": case.get("top_time"),
            "explain_method": case.get("explain_method"),
            "time_len": case.get("time_len"),
        }])
        summary.to_excel(w, index=False, sheet_name="summary")

        tf = pd.DataFrame(case.get("top_features_agg", []))
        if len(tf) == 0:
            tf = pd.DataFrame(case.get("top_features", []))
        tf.to_excel(w, index=False, sheet_name="top_features_agg")

        tft = pd.DataFrame(case.get("top_features_at_time", []))
        tft.to_excel(w, index=False, sheet_name="top_features_at_top_time")

        sim = pd.DataFrame(case.get("similar_patients", []))
        sim.to_excel(w, index=False, sheet_name="similar_patients")

    return out_path

def export_case_pdf(out_path: Path, case: Dict[str, Any], images: Optional[Dict[str, str]] = None) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    w, h = A4

    y = h - 2*cm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, y, "Early Warning System - Case Report")
    y -= 1.0*cm

    c.setFont("Helvetica", 11)
    items = [
        ("Case ID", case.get("case_id")),
        ("Patient index", case.get("patient_index")),
        ("Risk", case.get("risk")),
        ("Threshold", case.get("threshold")),
        ("Decision", case.get("decision")),
        ("Label", case.get("label")),
        ("Explain method", case.get("explain_method")),
        ("Top time step", case.get("top_time")),
    ]
    for k, v in items:
        c.drawString(2*cm, y, f"{k}: {v}")
        y -= 0.65*cm

    y -= 0.4*cm

    # Insert images if provided (heatmap, temporal plot)
    if images:
        for title, img_path in images.items():
            if y < 8*cm:
                c.showPage()
                y = h - 2*cm
            c.setFont("Helvetica-Bold", 12)
            c.drawString(2*cm, y, title)
            y -= 0.6*cm
            try:
                c.drawImage(img_path, 2*cm, y-9*cm, width=16*cm, height=9*cm, preserveAspectRatio=True, anchor='sw')
                y -= 9.5*cm
            except Exception:
                c.setFont("Helvetica", 10)
                c.drawString(2*cm, y, f"(Failed to render image: {img_path})")
                y -= 0.6*cm

    def _draw_table(title: str, rows: List[Dict[str, Any]], max_rows: int = 12):
        nonlocal y
        if not rows:
            return
        if y < 6*cm:
            c.showPage()
            y = h - 2*cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y, title)
        y -= 0.7*cm
        c.setFont("Helvetica", 9)
        for r in rows[:max_rows]:
            line = " | ".join([str(r.get(k)) for k in ["feat", "feature_en", "feature_ar"] if k in r])
            if "score(abs)" in r:
                line += f" | score={r.get('score(abs)')}"
            if "value(norm)" in r:
                line += f" | value={r.get('value(norm)')}"
            c.drawString(2*cm, y, line[:120])
            y -= 0.45*cm
            if y < 2*cm:
                c.showPage()
                y = h - 2*cm
                c.setFont("Helvetica", 9)

    _draw_table("Top Features (Aggregated over time)", case.get("top_features_agg", []), max_rows=12)
    _draw_table("Top Features at Top Time", case.get("top_features_at_time", []), max_rows=12)

    sim = case.get("similar_patients", [])
    if sim:
        if y < 6*cm:
            c.showPage()
            y = h - 2*cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y, "Similar patients (no training)")
        y -= 0.7*cm
        c.setFont("Helvetica", 10)
        for r in sim[:10]:
            line = f"- idx={r.get('patient_index')}  sim={r.get('cosine_sim')}  risk={r.get('risk')}  label={r.get('label')}"
            c.drawString(2*cm, y, line[:110])
            y -= 0.5*cm
            if y < 2*cm:
                c.showPage()
                y = h - 2*cm
                c.setFont("Helvetica", 10)

    c.save()
    return out_path


# -------------------- UI helpers --------------------
def _load_feature_map(processed_dir: Path, feat_dim: int) -> pd.DataFrame:
    xlsx = processed_dir / "feature_map.xlsx"
    if xlsx.exists():
        try:
            df = pd.read_excel(xlsx)
            need = {"feat_idx", "feat_name", "feature_en", "feature_ar"}
            if need.issubset(set(df.columns)):
                return df
        except Exception:
            pass
    return save_feature_map_xlsx(xlsx, feat_dim)

def _feat_lookup(df_map: pd.DataFrame, i: int) -> Tuple[str, str, str]:
    row = df_map[df_map["feat_idx"] == i]
    if len(row) == 0:
        return f"feat_{i}", f"feat_{i}", f"ميزة {i}"
    r = row.iloc[0]
    return str(r["feat_name"]), str(r["feature_en"]), str(r["feature_ar"])

def _cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # A: [N,H], b: [H]
    eps = 1e-9
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    bn = b / (np.linalg.norm(b) + eps)
    return An @ bn

def _plot_heatmap(mat: np.ndarray, title: str, xlab: str, ylab: str, xticklabels: Optional[List[str]] = None):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.colorbar()
    if xticklabels is not None:
        plt.xticks(ticks=np.arange(len(xticklabels)), labels=xticklabels, rotation=45, ha="right")
    plt.tight_layout()
    return fig

def _plot_line(y: np.ndarray, title: str, xlab: str, ylab: str):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(y)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    return fig


# -------------------- Streamlit UI --------------------
def run_ui(prepared_npz: Path, model_path: Path, threshold: float = 0.6, cases_dir: Path = Path("cases")):
    import streamlit as st
    import matplotlib.pyplot as plt

    st.set_page_config(page_title="Early Warning System", layout="wide")
    st.title("نظام الإنذار المبكر  |  Early Warning System")
    # Removed (as requested): explanatory caption about XAI methods

    d = np.load(prepared_npz, allow_pickle=True)
    V, M, L, y = d["V"], d["M"], d["L"], d["y"]
    feat_dim = int(V.shape[-1])
    processed_dir = prepared_npz.parent
    fmap = _load_feature_map(processed_dir, feat_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    model = GRUAttnClassifier(feat_dim=int(ckpt["feat_dim"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Cache embeddings for similarity search (computed once per session)
    @st.cache_data(show_spinner=True)
    def compute_all_embeddings(_prepared_path: str, _model_path: str) -> np.ndarray:
        # Use model.encode_context to compute [N,H]
        VV = torch.from_numpy(V).to(device)
        MM = torch.from_numpy(M).to(device)
        LL = torch.from_numpy(L).to(device)

        bs = 128
        embs = []
        with torch.no_grad():
            for i0 in range(0, VV.size(0), bs):
                v_b = VV[i0:i0+bs]
                m_b = MM[i0:i0+bs]
                l_b = LL[i0:i0+bs]
                ctx = model.encode_context(v_b, m_b, l_b)  # [B,H]
                embs.append(ctx.detach().cpu().numpy())
        return np.concatenate(embs, axis=0)

    all_emb = compute_all_embeddings(str(prepared_npz), str(model_path))  # [N,H]

    # Sidebar
    st.sidebar.header("Case / الحالة")
    mode = st.sidebar.radio("Mode", ["Dataset patient", "Saved case"], index=0)

    if mode == "Dataset patient":
        idx = st.sidebar.number_input("Patient index", 0, int(V.shape[0]-1), 0, 1)
        thr = st.sidebar.slider("Decision Threshold", 0.1, 0.9, float(threshold), 0.05)
        explain_method = st.sidebar.selectbox("Explain method", ["IntegratedGradients", "GradientShap"], index=0)
        topk = st.sidebar.slider("TopK features", 5, 30, 15, 1)
        n_sim = st.sidebar.slider("Similar patients (N)", 3, 20, 8, 1)

        colb1, colb2 = st.sidebar.columns(2)
        run_explain = colb1.button(f"Explain ({'IG' if explain_method=='IntegratedGradients' else 'GradientShap'})")
        save_this = colb2.button("Save case")
    else:
        thr = st.sidebar.slider("Decision Threshold", 0.1, 0.9, float(threshold), 0.05)
        files = list_cases(cases_dir)
        if not files:
            st.sidebar.info("No saved cases yet.")
            return
        choices = {p.name: p for p in files}
        picked = st.sidebar.selectbox("Select saved case", list(choices.keys()))
        case = load_case(choices[picked])
        idx = int(case.get("patient_index", 0))
        explain_method = case.get("explain_method", "IntegratedGradients")
        topk = 15
        n_sim = 8
        run_explain = False
        save_this = False

    idx = int(idx)

    # Inference
    Vt = torch.from_numpy(V[idx:idx+1]).to(device)
    Mt = torch.from_numpy(M[idx:idx+1]).to(device)
    Lt = torch.from_numpy(L[idx:idx+1]).to(device)

    with torch.no_grad():
        logit, alpha = model(Vt, Mt, Lt)
        risk = float(torch.sigmoid(logit).cpu().numpy()[0])
        alpha = alpha.cpu().numpy()[0, :int(L[idx])]

    decision = "HIGH" if risk >= thr else "LOW"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk", f"{risk:.3f}")
    c2.metric("Decision", decision)
    c3.metric("Label", int(y[idx]))
    c4.metric("Time length", int(L[idx]))

    # Attention
    st.subheader("Attention over time")  # Removed "Weak-XAI"
    fig_att = _plot_line(alpha, "Attention over time", "Time step", "Attention")
    st.pyplot(fig_att)

    # Top time step by attention + top observed features at that time
    st.subheader("Top time step + top observed features")  # Removed "Weak-XAI" phrasing
    t_star_att = int(np.argmax(alpha))
    obs_att = M[idx, t_star_att] > 0.5
    vals_att = V[idx, t_star_att]
    scores_att = np.abs(vals_att) * obs_att
    top_att = np.argsort(-scores_att)[:topk]

    rows_att = []
    for fi in top_att:
        feat_name, en, ar = _feat_lookup(fmap, int(fi))
        rows_att.append({
            "feat": feat_name,
            "feature_en": en,
            "feature_ar": ar,
            "abs_value(norm)": float(scores_att[int(fi)]),
            "value(norm)": float(vals_att[int(fi)]),
            "observed": int(obs_att[int(fi)]),
            "time_step(attention)": t_star_att
        })
    st.dataframe(pd.DataFrame(rows_att), use_container_width=True)

    # Attribution explanations
    st.subheader("Explanation")  # Simplified title
    # Removed (as requested): caption explaining what will appear after Explain

    exp = None
    if mode == "Saved case":
        if "attr_full" in case:
            exp = case
    else:
        if run_explain:
            with st.spinner("Computing attributions..."):
                if explain_method == "IntegratedGradients":
                    exp = explain_ig(model, V, M, L, idx=idx, n_steps=64, topk=topk)
                else:
                    exp = explain_gradshap(model, V, M, L, idx=idx, n_samples=50, stdevs=0.01, topk=topk)

    if exp is None or "attr_full" not in exp:
        st.info("اضغط Explain لعرض النتائج.")
    else:
        Tlen = int(L[idx])
        attr_full = exp["attr_full"][:Tlen]  # [T,F]
        feat_imp = np.sum(np.abs(attr_full), axis=0)  # [F]
        time_imp = np.sum(np.abs(attr_full), axis=1)  # [T]

        top_feat = np.argsort(-feat_imp)[:topk]
        t_star = int(np.argmax(time_imp))

        # Top features aggregated
        rows_agg = []
        for fi in top_feat:
            feat_name, en, ar = _feat_lookup(fmap, int(fi))
            rows_agg.append({
                "feat": feat_name,
                "feature_en": en,
                "feature_ar": ar,
                "score(abs)_agg": float(feat_imp[int(fi)]),
            })

        st.markdown(f"**Method:** {exp.get('method', explain_method)}")
        st.markdown(f"**Top time step:** {t_star}")

        colx1, colx2 = st.columns(2)
        with colx1:
            st.markdown("**Top features (aggregated over time)**")
            st.dataframe(pd.DataFrame(rows_agg), use_container_width=True)

        # Top features at top time step (values at that time, and attribution at that time)
        obs_t = (M[idx, t_star] > 0.5)
        vals_t = V[idx, t_star]
        attr_t = attr_full[t_star]  # [F]
        rows_time = []
        for fi in top_feat:
            feat_name, en, ar = _feat_lookup(fmap, int(fi))
            rows_time.append({
                "feat": feat_name,
                "feature_en": en,
                "feature_ar": ar,
                "value(norm)": float(vals_t[int(fi)]),
                "observed": int(obs_t[int(fi)]),
                "attr(time)": float(attr_t[int(fi)]),
                "abs_attr(time)": float(abs(attr_t[int(fi)])),
                "time_step": t_star,
            })

        with colx2:
            st.markdown("**Top features at Top time step**")
            st.dataframe(pd.DataFrame(rows_time), use_container_width=True)

        # Heatmap (Time x Features) for topk features
        st.markdown("### Heatmap (Time × Features)")
        topk_feat = top_feat
        mat = attr_full[:, topk_feat]  # [T,K]
        fig_hm = _plot_heatmap(mat, "Attribution Heatmap (Time x Top Features)", "Top features", "Time step", xticklabels=None)
        st.pyplot(fig_hm)

        # Temporal importance plot
        st.markdown("### Temporal importance")
        fig_time = _plot_line(time_imp, "Temporal importance (sum abs attr)", "Time step", "Importance")
        st.pyplot(fig_time)

        # Similar patients (no training)
        st.markdown("### Similar patients")
        with st.spinner("Finding similar patients..."):
            b = all_emb[idx]  # [H]
            sims = _cosine_sim_matrix(all_emb, b)  # [N]
            sims[idx] = -1.0  # exclude itself
            top_sim_idx = np.argsort(-sims)[:int(n_sim)]

            # compute their risk quickly (batch)
            Vs = torch.from_numpy(V[top_sim_idx]).to(device)
            Ms = torch.from_numpy(M[top_sim_idx]).to(device)
            Ls = torch.from_numpy(L[top_sim_idx]).to(device)
            with torch.no_grad():
                logit_s, _ = model(Vs, Ms, Ls)
                risk_s = torch.sigmoid(logit_s).detach().cpu().numpy()

            rows_sim = []
            for j, pid in enumerate(top_sim_idx):
                rows_sim.append({
                    "patient_index": int(pid),
                    "cosine_sim": float(sims[int(pid)]),
                    "risk": float(risk_s[j]),
                    "label": int(y[int(pid)]),
                    "time_len": int(L[int(pid)]),
                })
        st.dataframe(pd.DataFrame(rows_sim), use_container_width=True)

        # Build payload for saving + exports
        exp_payload = {
            "case_id": f"patient_{idx}_{time.strftime('%Y%m%d_%H%M%S')}",
            "patient_index": int(idx),
            "risk": _safe_float(risk),
            "threshold": _safe_float(thr),
            "decision": decision,
            "label": int(y[idx]),
            "time_len": int(Tlen),
            "explain_method": exp.get("method", explain_method),
            "top_time": int(t_star),
            "top_features_agg": rows_agg,
            "top_features_at_time": rows_time,
            "similar_patients": rows_sim,
            "attr_topk_heatmap": mat.tolist(),
            "top_feat_idx": [int(x) for x in topk_feat.tolist()],
        }

        # Exports (CSV/Excel/PDF)
        st.markdown("### Exports (CSV / Excel / PDF) for this case")
        cexp1, cexp2, cexp3, cexp4 = st.columns(4)

        if cexp1.button("Save case (JSON)"):
            pth = save_case(cases_dir, exp_payload)
            st.success(f"Saved: {pth}")

        if cexp2.button("Export CSV (summary)"):
            out_csv = cases_dir / "exports" / f"{exp_payload['case_id']}.csv"
            export_case_csv(out_csv, exp_payload)
            st.success(f"Saved: {out_csv}")

        if cexp3.button("Export Excel (multi-sheet)"):
            out_xlsx = cases_dir / "exports" / f"{exp_payload['case_id']}.xlsx"
            export_case_excel(out_xlsx, exp_payload)
            st.success(f"Saved: {out_xlsx}")

        if cexp4.button("Export PDF report"):
            tmpdir = Path(tempfile.mkdtemp())
            hm_png = tmpdir / "heatmap.png"
            time_png = tmpdir / "temporal.png"
            att_png = tmpdir / "attention.png"

            try:
                fig_hm.savefig(hm_png, dpi=160, bbox_inches="tight")
                fig_time.savefig(time_png, dpi=160, bbox_inches="tight")
                fig_att.savefig(att_png, dpi=160, bbox_inches="tight")
            except Exception:
                pass

            out_pdf = cases_dir / "exports" / f"{exp_payload['case_id']}.pdf"
            export_case_pdf(
                out_pdf,
                exp_payload,
                images={
                    "Attention over time": str(att_png),
                    "Attribution heatmap (Time x Features)": str(hm_png),
                    "Temporal importance": str(time_png),
                }
            )
            st.success(f"Saved: {out_pdf}")

    # Save basic case even without explanation
    if mode == "Dataset patient" and save_this:
        payload = {
            "case_id": f"patient_{idx}_{time.strftime('%Y%m%d_%H%M%S')}",
            "patient_index": int(idx),
            "risk": _safe_float(risk),
            "threshold": _safe_float(thr),
            "decision": decision,
            "label": int(y[idx]),
            "top_time_attention": int(t_star_att),
            "explain_method": explain_method,
            "notes": "",
        }
        pth = save_case(cases_dir, payload)
        st.success(f"Saved: {pth}")

    st.divider()
    st.subheader("Offline usage / التشغيل بدون تدريب أو تحميل")
    st.markdown(
        "- احفظ الملفات: **prepared_dataset.npz** و **best_model.pt**\n"
        "- شغّل فقط الواجهة:\n"
        "  - `streamlit run clinical_early_warning_e2e.py -- ui --prepared <path>/prepared_dataset.npz --model <path>/best_model.pt`\n"
        "- التطبيق يحتاج عملية Streamlit تعمل (محليًا أو سيرفر)."
    )


# -------------------- CLI --------------------
def build_parser():
    p = argparse.ArgumentParser(prog="clinical_early_warning_e2e.py")
    sub = p.add_subparsers(dest="cmd", required=False)

    dl = sub.add_parser("download")
    dl.add_argument("--record_id", type=int, default=5330730)
    dl.add_argument("--out_dir", type=str, default="data/raw")
    dl.add_argument("--out_name", type=str, default="json")
    dl.add_argument("--force_redownload", action="store_true")

    ins = sub.add_parser("inspect")
    ins.add_argument("--input", type=str, required=True)
    ins.add_argument("--n", type=int, default=1)

    pr = sub.add_parser("prepare")
    pr.add_argument("--input", type=str, required=True)
    pr.add_argument("--out_dir", type=str, default="data/processed")
    pr.add_argument("--max_records", type=int, default=None)

    tr = sub.add_parser("train")
    tr.add_argument("--prepared", type=str, default="data/processed/prepared_dataset.npz")
    tr.add_argument("--out_dir", type=str, default="artifacts")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-3)

    ui = sub.add_parser("ui")
    ui.add_argument("--prepared", type=str, default="data/processed/prepared_dataset.npz")
    ui.add_argument("--model", type=str, default="artifacts/best_model.pt")
    ui.add_argument("--threshold", type=float, default=0.6)
    ui.add_argument("--cases_dir", type=str, default="cases")

    pipe = sub.add_parser("pipeline")
    pipe.add_argument("--record_id", type=int, default=5330730)
    pipe.add_argument("--raw_dir", type=str, default="data/raw")
    pipe.add_argument("--processed_dir", type=str, default="data/processed")
    pipe.add_argument("--artifacts_dir", type=str, default="artifacts")
    pipe.add_argument("--epochs", type=int, default=10)
    pipe.add_argument("--batch_size", type=int, default=64)
    pipe.add_argument("--lr", type=float, default=1e-3)
    pipe.add_argument("--force_redownload", action="store_true")

    return p


def main():
    _sanitize_argv_for_jupyter()
    parser = build_parser()
    args, _ = parser.parse_known_args()

    if not getattr(args, "cmd", None):
        print("Examples:")
        print("  !pip -q install ijson tqdm scikit-learn streamlit captum reportlab openpyxl matplotlib")
        print("  !python clinical_early_warning_e2e.py pipeline --record_id 5330730 --raw_dir data/raw --processed_dir data/processed --artifacts_dir artifacts --epochs 10")
        print("  !streamlit run clinical_early_warning_e2e.py -- ui --prepared data/processed/prepared_dataset.npz --model artifacts/best_model.pt")
        try:
            parser.print_help()
        except Exception:
            pass
        return 0

    if args.cmd == "download":
        pth = download_zenodo(args.record_id, Path(args.out_dir), out_name=args.out_name, force_redownload=args.force_redownload)
        print(str(pth))

    elif args.cmd == "inspect":
        inspect_dataset(Path(args.input), n=args.n)

    elif args.cmd == "prepare":
        npz = prepare_dataset(Path(args.input), Path(args.out_dir), max_records=args.max_records)
        print(str(npz))

    elif args.cmd == "train":
        best = train_model(Path(args.prepared), Path(args.out_dir), args.epochs, args.batch_size, args.lr)
        print(str(best))

    elif args.cmd == "ui":
        run_ui(Path(args.prepared), Path(args.model), threshold=args.threshold, cases_dir=Path(args.cases_dir))

    elif args.cmd == "pipeline":
        raw_dir = Path(args.raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir = Path(args.processed_dir); processed_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = Path(args.artifacts_dir); artifacts_dir.mkdir(parents=True, exist_ok=True)

        fpath = download_zenodo(args.record_id, raw_dir, out_name="json", force_redownload=args.force_redownload)
        inspect_dataset(fpath, n=1)
        npz = prepare_dataset(fpath, processed_dir, max_records=None)
        best = train_model(npz, artifacts_dir, args.epochs, args.batch_size, args.lr)

        print("[pipeline] done")
        print("prepared:", npz)
        print("model:", best)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
