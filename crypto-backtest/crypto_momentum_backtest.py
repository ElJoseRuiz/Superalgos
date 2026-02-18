#!/usr/bin/env python3
"""
Crypto Futures Momentum Backtest
================================
Backtest de momentum sobre futuros de Binance (velas 5 min) para el año 2025.

Estrategias: TOP 1, TOP 3, TOP 5 por subida % en la última hora.
Filtros: volumen relativo, anti-duplicados, magnitud mínima.

Uso:
    python crypto_momentum_backtest.py [--data-dir RUTA] [--output-dir RUTA]
"""

import os
import sys
import json
import argparse
import warnings
import time as _time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN POR DEFECTO
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = r"C:\Users\portatil\Dropbox\CryptoData\Binance\F_5m"
DEFAULT_OUTPUT_DIR = "results"

YEAR = 2025
CANDLE_MINUTES = 5
SIGNAL_INTERVAL = 2          # cada 2 velas = 10 min
LOOKBACK_CANDLES = 12         # 12 velas = 1 hora
FORWARD_CANDLES = 288         # 288 velas = 24 horas
TOP_N_LIST = [1, 3, 5]
MAGNITUDE_THRESHOLDS = [0.0, 1.0, 2.0, 3.0]  # %
VOLUME_TOP_N = 10             # filtro momentum+volumen
CHECKPOINT_EVERY = 1000       # guardar checkpoint cada N señales procesadas
CHECKPOINT_FILE = "checkpoint_signals.pkl"


# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------------------
def _read_sample(filepath: str) -> pd.DataFrame:
    """Lee las primeras filas de un archivo (Parquet o CSV) para detectar estructura."""
    ext = Path(filepath).suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(filepath)
        return df.head(5)
    else:
        return pd.read_csv(filepath, nrows=5)


def detect_columns(filepath: str) -> dict:
    """Lee las primeras líneas de un archivo y detecta la estructura de columnas."""
    sample = _read_sample(filepath)
    cols_lower = [c.lower().strip() for c in sample.columns]
    print(f"\n  Columnas detectadas ({Path(filepath).name}): {list(sample.columns)}")
    print(f"  Shape muestra: {sample.shape}")
    print(sample.head(2).to_string(index=False))

    mapping = {}
    # Mapeo flexible
    for i, c in enumerate(cols_lower):
        if "time" in c and "open" in c:
            mapping["timestamp"] = sample.columns[i]
        elif c in ("open_time", "opentime", "timestamp", "time", "date", "datetime"):
            mapping["timestamp"] = sample.columns[i]
        if c in ("open",):
            mapping["open"] = sample.columns[i]
        if c in ("high",):
            mapping["high"] = sample.columns[i]
        if c in ("low",):
            mapping["low"] = sample.columns[i]
        if c in ("close",):
            mapping["close"] = sample.columns[i]
        if c in ("volume", "vol"):
            mapping["volume"] = sample.columns[i]

    # Fallback posicional si no se detectaron nombres
    if "close" not in mapping and len(sample.columns) >= 6:
        print("  Usando mapeo posicional: [timestamp, open, high, low, close, volume, ...]")
        mapping["timestamp"] = sample.columns[0]
        mapping["open"] = sample.columns[1]
        mapping["high"] = sample.columns[2]
        mapping["low"] = sample.columns[3]
        mapping["close"] = sample.columns[4]
        mapping["volume"] = sample.columns[5]

    print(f"  Mapeo: {mapping}")
    return mapping


def load_pair(filepath: str, col_map: dict) -> pd.DataFrame:
    """Carga un archivo Parquet/CSV de un par, filtra al año objetivo, y devuelve df limpio."""
    usecols = list(col_map.values())
    rename = {v: k for k, v in col_map.items()}

    ext = Path(filepath).suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(filepath, columns=usecols)
    else:
        df = pd.read_csv(filepath, usecols=usecols)
    df.rename(columns=rename, inplace=True)

    # Parsear timestamp
    ts = df["timestamp"]
    if ts.dtype in (np.int64, np.float64) and ts.iloc[0] > 1e12:
        df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True)
    elif ts.dtype in (np.int64, np.float64) and ts.iloc[0] > 1e9:
        df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True)

    # Filtrar al año
    df = df[df["timestamp"].dt.year == YEAR].copy()
    if df.empty:
        return df

    # Tipos eficientes
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = df[c].astype(np.float32)

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_all_pairs(data_dir: str) -> dict:
    """Carga todos los pares del directorio (Parquet o CSV). Devuelve {pair_name: DataFrame}."""
    data_dir = Path(data_dir)

    # Buscar Parquet primero, luego CSV como fallback
    data_files = sorted(data_dir.glob("*.parquet"))
    file_type = "Parquet"
    if not data_files:
        data_files = sorted(data_dir.glob("*.csv"))
        file_type = "CSV"

    if not data_files:
        print(f"ERROR: No se encontraron archivos Parquet ni CSV en {data_dir}")
        sys.exit(1)

    print(f"\nEncontrados {len(data_files)} archivos {file_type} en {data_dir}")

    # Detectar columnas con el primer archivo
    col_map = detect_columns(str(data_files[0]))

    pairs = {}
    for f in tqdm(data_files, desc="Cargando pares"):
        name = f.stem  # nombre del archivo sin extensión
        df = load_pair(str(f), col_map)
        if len(df) > LOOKBACK_CANDLES:
            pairs[name] = df

    print(f"\nPares cargados con datos de {YEAR}: {len(pairs)}")
    total_rows = sum(len(v) for v in pairs.values())
    print(f"Total de filas: {total_rows:,}")
    mem_mb = sum(v.memory_usage(deep=True).sum() for v in pairs.values()) / 1e6
    print(f"Memoria aprox.: {mem_mb:.1f} MB")
    return pairs


# ---------------------------------------------------------------------------
# 2. ALINEAR TIMESTAMPS
# ---------------------------------------------------------------------------
def build_aligned_matrices(pairs: dict):
    """
    Construye matrices NumPy alineadas por timestamp para close y volume.
    Devuelve: timestamps (array), pair_names (list), close_matrix, volume_matrix
    """
    # Unión de todos los timestamps
    all_ts = set()
    for df in pairs.values():
        all_ts.update(df["timestamp"].values)
    all_ts = np.sort(np.array(list(all_ts)))

    pair_names = sorted(pairs.keys())
    n_times = len(all_ts)
    n_pairs = len(pair_names)

    close_matrix = np.full((n_times, n_pairs), np.nan, dtype=np.float32)
    volume_matrix = np.full((n_times, n_pairs), np.nan, dtype=np.float32)

    ts_to_idx = {ts: i for i, ts in enumerate(all_ts)}

    for j, name in enumerate(tqdm(pair_names, desc="Alineando datos")):
        df = pairs[name]
        indices = np.array([ts_to_idx[ts] for ts in df["timestamp"].values])
        close_matrix[indices, j] = df["close"].values
        volume_matrix[indices, j] = df["volume"].values

    print(f"\nMatriz alineada: {n_times} timestamps x {n_pairs} pares")
    return all_ts, pair_names, close_matrix, volume_matrix


# ---------------------------------------------------------------------------
# 3. GENERAR SEÑALES
# ---------------------------------------------------------------------------
def load_checkpoint(output_dir: str):
    """Carga checkpoint previo si existe. Retorna (records, last_sig_idx) o ([], -1)."""
    ckpt_path = os.path.join(output_dir, CHECKPOINT_FILE)
    if os.path.exists(ckpt_path):
        try:
            ckpt = pd.read_pickle(ckpt_path)
            records = ckpt["records"]
            last_sig_idx = ckpt["last_sig_idx"]
            print(f"  Checkpoint encontrado: {len(records):,} señales, último idx={last_sig_idx}")
            return records, last_sig_idx
        except Exception as e:
            print(f"  Checkpoint corrupto, reiniciando: {e}")
    return [], -1


def save_checkpoint(records: list, last_sig_idx: int, output_dir: str):
    """Guarda checkpoint parcial."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, CHECKPOINT_FILE)
    pd.to_pickle({"records": records, "last_sig_idx": last_sig_idx}, ckpt_path)


def compute_signals(all_ts, pair_names, close_matrix, volume_matrix, output_dir="results"):
    """
    Recorre cada punto de señal (cada SIGNAL_INTERVAL velas),
    calcula momentum y volumen relativo, y genera registros de señales.
    Guarda checkpoints cada CHECKPOINT_EVERY señales procesadas.
    """
    n_times, n_pairs = close_matrix.shape
    vol_lookback_24h = 288  # 24h en velas de 5 min

    # Puntos de señal: cada SIGNAL_INTERVAL velas, empezando tras suficiente lookback
    start_idx = max(LOOKBACK_CANDLES, vol_lookback_24h)
    signal_indices = list(range(start_idx, n_times - FORWARD_CANDLES, SIGNAL_INTERVAL))

    # Intentar cargar checkpoint
    records, last_sig_idx = load_checkpoint(output_dir)
    if last_sig_idx >= 0:
        # Filtrar signal_indices ya procesados
        original_len = len(signal_indices)
        signal_indices = [si for si in signal_indices if si > last_sig_idx]
        print(f"  Retomando: {original_len - len(signal_indices)} ya procesados, "
              f"quedan {len(signal_indices)}")
    else:
        records = []

    print(f"\nPuntos de señal a procesar: {len(signal_indices):,}")
    if signal_indices:
        print(f"Rango: {pd.Timestamp(all_ts[signal_indices[0]])} -> {pd.Timestamp(all_ts[signal_indices[-1]])}")

    signals_since_checkpoint = 0

    for sig_idx in tqdm(signal_indices, desc="Generando señales"):
        ts = all_ts[sig_idx]

        # --- Momentum: variación % del close en las últimas 12 velas ---
        close_now = close_matrix[sig_idx, :]
        close_past = close_matrix[sig_idx - LOOKBACK_CANDLES, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            momentum_pct = ((close_now - close_past) / close_past) * 100.0

        # --- Volumen relativo: vol última hora / media vol 24h ---
        vol_1h = np.nanmean(volume_matrix[sig_idx - LOOKBACK_CANDLES:sig_idx, :], axis=0)
        vol_24h = np.nanmean(volume_matrix[sig_idx - vol_lookback_24h:sig_idx, :], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_ratio = vol_1h / vol_24h

        # Máscara de pares válidos (con datos)
        valid = np.isfinite(momentum_pct) & np.isfinite(close_now) & (close_now > 0)

        if valid.sum() < 5:
            continue

        # Ranking por momentum (descendente)
        mom_vals = momentum_pct.copy()
        mom_vals[~valid] = -np.inf
        ranked_indices = np.argsort(mom_vals)[::-1]  # mayor primero

        # Ranking por volumen relativo
        vol_vals = vol_ratio.copy()
        vol_vals[~valid] = -np.inf
        vol_ranked = np.argsort(vol_vals)[::-1]

        # Top 10 momentum y top 10 volumen (para filtro momentum+volumen)
        top10_mom = set(ranked_indices[:10])
        top10_vol = set(vol_ranked[:10])
        mom_vol_set = top10_mom & top10_vol  # intersección

        # Forward returns: 288 velas futuras
        future_closes = close_matrix[sig_idx:sig_idx + FORWARD_CANDLES + 1, :]  # +1 para incluir t=0
        entry_prices = close_now

        # Checkpoints de retorno
        checkpoints = {
            "1h": min(12, FORWARD_CANDLES),
            "4h": min(48, FORWARD_CANDLES),
            "12h": min(144, FORWARD_CANDLES),
            "24h": min(288, FORWARD_CANDLES),
        }

        # Para cada top N, registrar señales
        for top_n in TOP_N_LIST:
            selected = ranked_indices[:top_n]
            # Solo pares válidos
            selected = [s for s in selected if valid[s]]
            if not selected:
                continue

            for rank, pair_idx in enumerate(selected):
                entry_price = entry_prices[pair_idx]
                if entry_price <= 0 or np.isnan(entry_price):
                    continue

                pair_name = pair_names[pair_idx]
                mom_value = float(momentum_pct[pair_idx])
                vr_value = float(vol_ratio[pair_idx]) if np.isfinite(vol_ratio[pair_idx]) else 0.0

                # Retornos en checkpoints
                returns_at = {}
                for label, offset in checkpoints.items():
                    future_price = close_matrix[sig_idx + offset, pair_idx]
                    if np.isfinite(future_price) and future_price > 0:
                        returns_at[label] = float((future_price - entry_price) / entry_price * 100)
                    else:
                        returns_at[label] = np.nan

                # Max drawdown en las 24h post-señal
                fwd = close_matrix[sig_idx:sig_idx + FORWARD_CANDLES + 1, pair_idx]
                with np.errstate(divide="ignore", invalid="ignore"):
                    fwd_returns = (fwd - entry_price) / entry_price * 100
                valid_fwd = np.isfinite(fwd_returns)
                if valid_fwd.any():
                    max_dd = float(np.nanmin(fwd_returns))
                else:
                    max_dd = np.nan

                # Curva completa de retornos (289 puntos: t=0 a t=288)
                fwd_curve = fwd_returns.tolist() if len(fwd_returns) == FORWARD_CANDLES + 1 else []

                # Es parte del filtro momentum+volumen?
                passes_vol_filter = pair_idx in mom_vol_set

                ts_dt = pd.Timestamp(ts)
                record = {
                    "signal_ts": ts_dt,
                    "hour": ts_dt.hour,
                    "dow": ts_dt.dayofweek,
                    "pair": pair_name,
                    "top_n": top_n,
                    "rank": rank + 1,
                    "momentum_pct": mom_value,
                    "vol_ratio": vr_value,
                    "entry_price": float(entry_price),
                    "ret_1h": returns_at.get("1h", np.nan),
                    "ret_4h": returns_at.get("4h", np.nan),
                    "ret_12h": returns_at.get("12h", np.nan),
                    "ret_24h": returns_at.get("24h", np.nan),
                    "max_dd_24h": max_dd,
                    "passes_vol_filter": passes_vol_filter,
                    "fwd_curve": fwd_curve,
                }
                records.append(record)
                signals_since_checkpoint += 1

        # Guardar checkpoint periódicamente
        if signals_since_checkpoint >= CHECKPOINT_EVERY:
            save_checkpoint(records, sig_idx, output_dir)
            signals_since_checkpoint = 0
            tqdm.write(f"  [Checkpoint] {len(records):,} señales guardadas (idx={sig_idx})")

    # Checkpoint final
    if records:
        save_checkpoint(records, signal_indices[-1] if signal_indices else last_sig_idx, output_dir)
        print(f"  [Checkpoint final] {len(records):,} señales guardadas")

    print(f"\nSeñales generadas: {len(records):,}")
    return records


# ---------------------------------------------------------------------------
# 4. POST-PROCESAR SEÑALES
# ---------------------------------------------------------------------------
def build_signals_dataframe(records: list) -> pd.DataFrame:
    """Convierte records en DataFrame y añade columnas derivadas."""
    # Separar curvas forward del record principal para el CSV
    curves = [r.pop("fwd_curve") for r in records]
    df = pd.DataFrame(records)

    # Guardar curvas como array numpy aparte (para análisis)
    max_len = FORWARD_CANDLES + 1
    curve_matrix = np.full((len(curves), max_len), np.nan, dtype=np.float32)
    for i, c in enumerate(curves):
        if c and len(c) == max_len:
            curve_matrix[i, :] = c

    # Marcar solapamiento (anti-duplicados)
    df.sort_values(["top_n", "pair", "signal_ts"], inplace=True)
    df["is_overlapping"] = False

    for top_n in TOP_N_LIST:
        mask_tn = df["top_n"] == top_n
        for pair in df.loc[mask_tn, "pair"].unique():
            mask = mask_tn & (df["pair"] == pair)
            idx_list = df.loc[mask].index.tolist()
            times = df.loc[mask, "signal_ts"].values

            last_signal_time = None
            for k, idx in enumerate(idx_list):
                t = times[k]
                if last_signal_time is not None:
                    delta = (t - last_signal_time) / np.timedelta64(1, "h")
                    if delta < 24:
                        df.loc[idx, "is_overlapping"] = True
                # Actualizar siempre (la primera señal marca el inicio de ventana)
                if not df.loc[idx, "is_overlapping"]:
                    last_signal_time = t

    df.sort_values(["signal_ts", "top_n", "rank"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    n_overlap = df["is_overlapping"].sum()
    print(f"Señales overlapping: {n_overlap:,} ({n_overlap / len(df) * 100:.1f}%)")

    return df, curve_matrix


# ---------------------------------------------------------------------------
# 5. ANÁLISIS Y GRÁFICOS
# ---------------------------------------------------------------------------
def run_analysis(df: pd.DataFrame, curve_matrix: np.ndarray, output_dir: str):
    """Genera todos los análisis, gráficos y CSVs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "figure.dpi": 150,
    })

    x_minutes = np.arange(0, FORWARD_CANDLES + 1) * CANDLE_MINUTES  # 0 a 1440

    # ===================================================================
    # A. Funciones auxiliares
    # ===================================================================
    def get_top_n_avg_curves(df_, cm_, top_n, extra_filter=None):
        """Retorna curva promedio equiponderada de una estrategia TOP N."""
        mask = df_["top_n"] == top_n
        if extra_filter is not None:
            mask = mask & extra_filter
        # Agrupar por señal (signal_ts): promediar los N pares seleccionados
        sub = df_[mask].copy()
        sub_curves = cm_[mask.values]

        if len(sub) == 0:
            return np.full(FORWARD_CANDLES + 1, np.nan), np.array([]), np.array([])

        # Agrupar por signal_ts y promediar curvas
        signal_groups = sub.groupby("signal_ts").groups
        avg_per_signal = []
        for ts, idxs in signal_groups.items():
            pos = [sub.index.get_loc(i) for i in idxs]
            group_curves = sub_curves[pos]
            avg_curve = np.nanmean(group_curves, axis=0)
            avg_per_signal.append(avg_curve)

        stacked = np.array(avg_per_signal)
        mean_curve = np.nanmean(stacked, axis=0)
        p25 = np.nanpercentile(stacked, 25, axis=0)
        p75 = np.nanpercentile(stacked, 75, axis=0)
        return mean_curve, p25, p75

    def get_checkpoint_returns(df_, top_n, extra_filter=None):
        """Retorna DataFrame con retornos promedios por señal para un TOP N."""
        mask = df_["top_n"] == top_n
        if extra_filter is not None:
            mask = mask & extra_filter
        sub = df_[mask].copy()
        if sub.empty:
            return pd.DataFrame()
        # Promediar retornos de los pares por señal
        grouped = sub.groupby("signal_ts").agg({
            "ret_1h": "mean",
            "ret_4h": "mean",
            "ret_12h": "mean",
            "ret_24h": "mean",
            "max_dd_24h": "mean",
            "momentum_pct": "mean",
            "hour": "first",
            "dow": "first",
            "is_overlapping": "first",
            "passes_vol_filter": "first",
        }).reset_index()
        return grouped

    colors = {1: "#e74c3c", 3: "#2ecc71", 5: "#3498db"}
    labels = {1: "TOP 1", 3: "TOP 3", 5: "TOP 5"}

    # ===================================================================
    # B. GRÁFICO 1: Curva promedio post-señal (TOP 1 vs 3 vs 5)
    # ===================================================================
    print("\n[1/13] Curva promedio post-señal...")
    fig, ax = plt.subplots(figsize=(14, 7))
    for tn in TOP_N_LIST:
        mean_c, p25, p75 = get_top_n_avg_curves(df, curve_matrix, tn)
        ax.plot(x_minutes, mean_c, label=labels[tn], color=colors[tn], lw=2)
        if len(p25) > 0:
            ax.fill_between(x_minutes, p25, p75, alpha=0.15, color=colors[tn])
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Minutos post-señal")
    ax.set_ylabel("Retorno medio (%)")
    ax.set_title("Curva promedio de evolución post-señal (con bandas P25-P75)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_avg_curve_post_signal.png"))
    plt.close(fig)

    # ===================================================================
    # C. GRÁFICO 2: Distribución de retornos a 1h, 4h, 12h, 24h
    # ===================================================================
    print("[2/13] Distribución de retornos...")
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    periods = ["ret_1h", "ret_4h", "ret_12h", "ret_24h"]
    period_labels = ["1h", "4h", "12h", "24h"]

    for row, tn in enumerate(TOP_N_LIST):
        cp = get_checkpoint_returns(df, tn)
        for col, (period, plabel) in enumerate(zip(periods, period_labels)):
            ax = axes[row, col]
            data = cp[period].dropna()
            if len(data) > 0:
                ax.hist(data, bins=80, color=colors[tn], alpha=0.7, edgecolor="white", lw=0.3)
                ax.axvline(data.mean(), color="black", ls="--", lw=1.2, label=f"Media: {data.mean():.2f}%")
                ax.axvline(data.median(), color="orange", ls="--", lw=1.2, label=f"Mediana: {data.median():.2f}%")
                ax.legend(fontsize=8)
            ax.set_title(f"{labels[tn]} @ {plabel}")
            ax.set_xlabel("Retorno (%)")
            if col == 0:
                ax.set_ylabel("Frecuencia")
    fig.suptitle("Distribución de retornos post-señal", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_return_distributions.png"))
    plt.close(fig)

    # ===================================================================
    # D. GRÁFICO 3: Win rate
    # ===================================================================
    print("[3/13] Win rate...")
    wr_data = []
    for tn in TOP_N_LIST:
        cp = get_checkpoint_returns(df, tn)
        for period, plabel in zip(periods, period_labels):
            data = cp[period].dropna()
            wr = (data > 0).mean() * 100 if len(data) > 0 else 0
            wr_data.append({"Estrategia": labels[tn], "Período": plabel, "Win Rate (%)": wr, "N": len(data)})
    wr_df = pd.DataFrame(wr_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = wr_df.pivot(index="Período", columns="Estrategia", values="Win Rate (%)")
    pivot = pivot.reindex(period_labels)
    pivot.plot(kind="bar", ax=ax, color=[colors[1], colors[3], colors[5]], edgecolor="white")
    ax.axhline(50, color="gray", ls="--", lw=1)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate por período post-señal")
    ax.set_xticklabels(period_labels, rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_win_rate.png"))
    plt.close(fig)

    # ===================================================================
    # E. GRÁFICO 4: Estadísticas resumen
    # ===================================================================
    print("[4/13] Estadísticas resumen...")
    summary_rows = []
    for tn in TOP_N_LIST:
        cp = get_checkpoint_returns(df, tn)
        for period, plabel in zip(periods, period_labels):
            data = cp[period].dropna()
            if len(data) == 0:
                continue
            row = {
                "Estrategia": labels[tn],
                "Período": plabel,
                "N señales": len(data),
                "Media (%)": data.mean(),
                "Mediana (%)": data.median(),
                "Std (%)": data.std(),
                "Sharpe": data.mean() / data.std() if data.std() > 0 else 0,
                "Win Rate (%)": (data > 0).mean() * 100,
                "Max DD medio (%)": cp["max_dd_24h"].mean(),
                "Mejor (%)": data.max(),
                "Peor (%)": data.min(),
            }
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "summary_statistics.csv"), index=False)
    print(f"  Guardado: summary_statistics.csv")

    # Tabla visual
    fig, ax = plt.subplots(figsize=(16, max(4, len(summary_df) * 0.35 + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=summary_df.round(3).values,
        colLabels=summary_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax.set_title("Estadísticas Resumen Comparativas", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_summary_table.png"))
    plt.close(fig)

    # ===================================================================
    # F. GRÁFICO 5: Heatmap por hora del día
    # ===================================================================
    print("[5/13] Heatmap por hora del día...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, tn in enumerate(TOP_N_LIST):
        cp = get_checkpoint_returns(df, tn)
        if cp.empty:
            continue
        hm = cp.groupby("hour")["ret_1h"].mean()
        hm_full = hm.reindex(range(24), fill_value=0)
        sns.heatmap(
            hm_full.values.reshape(1, -1),
            ax=axes[i],
            xticklabels=range(24),
            yticklabels=[labels[tn]],
            cmap="RdYlGn",
            center=0,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Ret 1h (%)"},
        )
        axes[i].set_title(f"{labels[tn]} - Retorno 1h por hora UTC")
        axes[i].set_xlabel("Hora (UTC)")
    fig.suptitle("Heatmap: Retorno medio 1h post-señal por hora del día", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_heatmap_hour.png"))
    plt.close(fig)

    # ===================================================================
    # G. GRÁFICO 6: Heatmap por día de la semana + hora
    # ===================================================================
    print("[6/13] Heatmap día de la semana x hora...")
    dow_labels = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for i, tn in enumerate(TOP_N_LIST):
        cp = get_checkpoint_returns(df, tn)
        if cp.empty:
            continue
        pivot_hm = cp.pivot_table(values="ret_1h", index="dow", columns="hour", aggfunc="mean")
        pivot_hm = pivot_hm.reindex(index=range(7), columns=range(24), fill_value=0)
        sns.heatmap(
            pivot_hm,
            ax=axes[i],
            cmap="RdYlGn",
            center=0,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 6},
            yticklabels=dow_labels,
            cbar_kws={"label": "Ret 1h (%)"},
        )
        axes[i].set_title(f"{labels[tn]}")
        axes[i].set_xlabel("Hora (UTC)")
        axes[i].set_ylabel("Día de la semana")
    fig.suptitle("Heatmap: Retorno 1h por día de la semana y hora", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_heatmap_dow_hour.png"))
    plt.close(fig)

    # ===================================================================
    # H. GRÁFICO 7: Top pares más seleccionados
    # ===================================================================
    print("[7/13] Top pares más frecuentes...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for i, tn in enumerate(TOP_N_LIST):
        sub = df[df["top_n"] == tn]
        top_pairs = sub["pair"].value_counts().head(20)
        top_pairs.plot(kind="barh", ax=axes[i], color=colors[tn], edgecolor="white")
        axes[i].set_title(f"{labels[tn]} - Top 20 pares más frecuentes")
        axes[i].set_xlabel("Frecuencia")
        axes[i].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "07_top_pairs.png"))
    plt.close(fig)

    # ===================================================================
    # I. GRÁFICO 8: Análisis por quintil de magnitud
    # ===================================================================
    print("[8/13] Análisis por quintil de magnitud...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    quintile_colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]

    for i, tn in enumerate(TOP_N_LIST):
        mask = df["top_n"] == tn
        sub = df[mask].copy()
        sub_curves = curve_matrix[mask.values]

        if sub.empty:
            continue

        # Calcular quintil por señal (usar momentum_pct del primer par de cada señal)
        signal_mom = sub.groupby("signal_ts")["momentum_pct"].mean()
        quintile_edges = np.nanpercentile(signal_mom, [0, 20, 40, 60, 80, 100])
        q_labels = ["Q1 (débil)", "Q2", "Q3", "Q4", "Q5 (fuerte)"]

        for q in range(5):
            lo, hi = quintile_edges[q], quintile_edges[q + 1]
            if q == 4:
                q_signals = signal_mom[(signal_mom >= lo) & (signal_mom <= hi)].index
            else:
                q_signals = signal_mom[(signal_mom >= lo) & (signal_mom < hi)].index

            q_mask = mask & df["signal_ts"].isin(q_signals)
            if q_mask.sum() == 0:
                continue

            q_sub = df[q_mask]
            q_curves = curve_matrix[q_mask.values]

            # Promediar por señal
            signal_groups = q_sub.groupby("signal_ts").groups
            avg_per_signal = []
            for ts, idxs in signal_groups.items():
                pos = [q_sub.index.get_loc(idx) for idx in idxs]
                group_curves = q_curves[pos]
                avg_per_signal.append(np.nanmean(group_curves, axis=0))

            if avg_per_signal:
                stacked = np.array(avg_per_signal)
                mean_c = np.nanmean(stacked, axis=0)
                axes[i].plot(x_minutes, mean_c, label=f"{q_labels[q]} (n={len(avg_per_signal)})",
                            color=quintile_colors[q], lw=1.5)

        axes[i].axhline(0, color="gray", ls="--", lw=0.8)
        axes[i].set_title(f"{labels[tn]}")
        axes[i].set_xlabel("Minutos post-señal")
        axes[i].set_ylabel("Retorno medio (%)")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    fig.suptitle("Curva post-señal por quintil de magnitud de momentum", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "08_quintile_analysis.png"))
    plt.close(fig)

    # ===================================================================
    # J. GRÁFICO 9: Curva de equity simulada
    # ===================================================================
    print("[9/13] Curva de equity simulada...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for tn in TOP_N_LIST:
        for ax_idx, (filter_overlap, title_suffix) in enumerate([
            (False, "todas las señales"),
            (True, "sin solapamiento"),
        ]):
            extra = (~df["is_overlapping"]) if filter_overlap else None
            cp = get_checkpoint_returns(df, tn, extra_filter=extra)
            if cp.empty:
                continue
            cp = cp.sort_values("signal_ts")
            # Equity: cada señal aporta su retorno a 1h (asumiendo mismo nocional)
            equity = (1 + cp["ret_1h"].fillna(0) / 100).cumprod() * 100
            axes[ax_idx].plot(cp["signal_ts"].values, equity.values, label=labels[tn], color=colors[tn], lw=1)

    for ax_idx, title_suffix in enumerate(["Todas las señales", "Sin solapamiento"]):
        axes[ax_idx].set_title(f"Curva de equity simulada (ret 1h) - {title_suffix}")
        axes[ax_idx].set_ylabel("Equity (base 100)")
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
    axes[1].set_xlabel("Fecha")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "09_equity_curve.png"))
    plt.close(fig)

    # ===================================================================
    # K. GRÁFICO 10: Histograma de drawdowns post-señal
    # ===================================================================
    print("[10/13] Histograma de drawdowns...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, tn in enumerate(TOP_N_LIST):
        cp = get_checkpoint_returns(df, tn)
        if cp.empty:
            continue
        dd = cp["max_dd_24h"].dropna()
        axes[i].hist(dd, bins=80, color=colors[tn], alpha=0.7, edgecolor="white", lw=0.3)
        axes[i].axvline(dd.mean(), color="black", ls="--", lw=1.2, label=f"Media: {dd.mean():.2f}%")
        axes[i].axvline(dd.median(), color="orange", ls="--", lw=1.2, label=f"Mediana: {dd.median():.2f}%")
        axes[i].set_title(f"{labels[tn]} - Max Drawdown 24h")
        axes[i].set_xlabel("Drawdown (%)")
        axes[i].set_ylabel("Frecuencia")
        axes[i].legend()
    fig.suptitle("Distribución del máximo drawdown post-señal (24h)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "10_drawdown_hist.png"))
    plt.close(fig)

    # ===================================================================
    # L. GRÁFICO 11: Comparativa filtros de magnitud mínima
    # ===================================================================
    print("[11/13] Comparativa filtros magnitud mínima...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    thresh_colors = {0.0: "#95a5a6", 1.0: "#3498db", 2.0: "#f39c12", 3.0: "#e74c3c"}

    for i, tn in enumerate(TOP_N_LIST):
        for thresh in MAGNITUDE_THRESHOLDS:
            filt = df["momentum_pct"] >= thresh
            mean_c, _, _ = get_top_n_avg_curves(df, curve_matrix, tn, extra_filter=filt)
            n_sigs = len(df[(df["top_n"] == tn) & filt].groupby("signal_ts"))
            axes[i].plot(x_minutes, mean_c,
                        label=f">{thresh}% (n={n_sigs})",
                        color=thresh_colors[thresh], lw=1.5)
        axes[i].axhline(0, color="gray", ls="--", lw=0.8)
        axes[i].set_title(f"{labels[tn]}")
        axes[i].set_xlabel("Minutos post-señal")
        axes[i].set_ylabel("Retorno medio (%)")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    fig.suptitle("Curva post-señal por umbral de magnitud mínima", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "11_magnitude_threshold.png"))
    plt.close(fig)

    # ===================================================================
    # M. GRÁFICO 12: Filtro momentum + volumen
    # ===================================================================
    print("[12/13] Análisis filtro momentum + volumen...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for i, tn in enumerate(TOP_N_LIST):
        for passes, label_extra, ls in [(True, "Con filtro vol", "-"), (False, "Sin filtro vol", "--")]:
            filt = df["passes_vol_filter"] == passes
            mean_c, _, _ = get_top_n_avg_curves(df, curve_matrix, tn, extra_filter=filt)
            n_sigs = len(df[(df["top_n"] == tn) & filt].groupby("signal_ts"))
            axes[i].plot(x_minutes, mean_c, label=f"{label_extra} (n={n_sigs})", lw=1.5, ls=ls)
        axes[i].axhline(0, color="gray", ls="--", lw=0.8)
        axes[i].set_title(f"{labels[tn]}")
        axes[i].set_xlabel("Minutos post-señal")
        axes[i].set_ylabel("Retorno medio (%)")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

    # Tabla comparativa win rate con y sin filtro
    vol_wr_rows = []
    for tn in TOP_N_LIST:
        for passes, label_extra in [(True, "Con vol filter"), (False, "Sin vol filter")]:
            filt = df["passes_vol_filter"] == passes
            cp = get_checkpoint_returns(df, tn, extra_filter=filt)
            if cp.empty:
                continue
            for period, plabel in zip(periods, period_labels):
                data = cp[period].dropna()
                vol_wr_rows.append({
                    "Estrategia": labels[tn],
                    "Filtro": label_extra,
                    "Período": plabel,
                    "Win Rate (%)": (data > 0).mean() * 100 if len(data) > 0 else 0,
                    "Media (%)": data.mean() if len(data) > 0 else 0,
                    "N": len(data),
                })
    vol_wr_df = pd.DataFrame(vol_wr_rows)
    vol_wr_df.to_csv(os.path.join(output_dir, "volume_filter_comparison.csv"), index=False)

    fig.suptitle("Filtro Momentum + Volumen: con vs sin filtro de volumen", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "12_volume_filter.png"))
    plt.close(fig)

    # ===================================================================
    # N. GRÁFICO 13: Curva de equity con y sin solapamiento + 4h/12h/24h
    # ===================================================================
    print("[13/13] Equity curves adicionales (4h, 12h, 24h)...")
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    ret_cols = ["ret_4h", "ret_12h", "ret_24h"]
    ret_labels_eq = ["4h", "12h", "24h"]

    for col_idx, (rcol, rlabel) in enumerate(zip(ret_cols, ret_labels_eq)):
        for row_idx, (filter_overlap, title_suffix) in enumerate([
            (False, "Todas"),
            (True, "Sin solap."),
        ]):
            extra = (~df["is_overlapping"]) if filter_overlap else None
            for tn in TOP_N_LIST:
                cp = get_checkpoint_returns(df, tn, extra_filter=extra)
                if cp.empty:
                    continue
                cp = cp.sort_values("signal_ts")
                equity = (1 + cp[rcol].fillna(0) / 100).cumprod() * 100
                axes[row_idx, col_idx].plot(cp["signal_ts"].values, equity.values,
                                            label=labels[tn], color=colors[tn], lw=1)
            axes[row_idx, col_idx].set_title(f"Equity {rlabel} - {title_suffix}")
            axes[row_idx, col_idx].legend(fontsize=8)
            axes[row_idx, col_idx].grid(True, alpha=0.3)
            axes[row_idx, col_idx].set_ylabel("Equity (base 100)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "13_equity_curves_extended.png"))
    plt.close(fig)

    return summary_df, wr_df, vol_wr_df


# ---------------------------------------------------------------------------
# 6. GUARDAR CSV MAESTRO
# ---------------------------------------------------------------------------
def save_master_csv(df: pd.DataFrame, output_dir: str):
    """Guarda el CSV maestro con todas las señales."""
    cols_to_save = [
        "signal_ts", "hour", "dow", "pair", "top_n", "rank",
        "momentum_pct", "vol_ratio", "entry_price",
        "ret_1h", "ret_4h", "ret_12h", "ret_24h",
        "max_dd_24h", "passes_vol_filter", "is_overlapping",
    ]
    df[cols_to_save].to_csv(os.path.join(output_dir, "master_signals.csv"), index=False)
    print(f"  Guardado: master_signals.csv ({len(df):,} registros)")


# ---------------------------------------------------------------------------
# 7. RESUMEN EJECUTIVO
# ---------------------------------------------------------------------------
def print_executive_summary(summary_df, wr_df, vol_wr_df, df):
    """Imprime conclusiones principales."""
    print("\n" + "=" * 80)
    print("  RESUMEN EJECUTIVO")
    print("=" * 80)

    # Mejor estrategia por retorno medio a 1h
    s1h = summary_df[summary_df["Período"] == "1h"].copy()
    if not s1h.empty:
        best = s1h.loc[s1h["Media (%)"].idxmax()]
        worst = s1h.loc[s1h["Media (%)"].idxmin()]
        print(f"\n  RETORNO MEDIO A 1H:")
        for _, row in s1h.iterrows():
            print(f"    {row['Estrategia']}: {row['Media (%)']:.3f}% "
                  f"(mediana: {row['Mediana (%)']:.3f}%, WR: {row['Win Rate (%)']:.1f}%)")

    # Mejor a 4h
    s4h = summary_df[summary_df["Período"] == "4h"].copy()
    if not s4h.empty:
        print(f"\n  RETORNO MEDIO A 4H:")
        for _, row in s4h.iterrows():
            print(f"    {row['Estrategia']}: {row['Media (%)']:.3f}% "
                  f"(mediana: {row['Mediana (%)']:.3f}%, WR: {row['Win Rate (%)']:.1f}%)")

    # Win rate
    print(f"\n  WIN RATE:")
    if not wr_df.empty:
        for _, row in wr_df.iterrows():
            print(f"    {row['Estrategia']} @ {row['Período']}: {row['Win Rate (%)']:.1f}% (n={row['N']})")

    # Filtro de volumen
    print(f"\n  FILTRO MOMENTUM + VOLUMEN:")
    if not vol_wr_df.empty:
        for _, row in vol_wr_df[vol_wr_df["Período"] == "1h"].iterrows():
            print(f"    {row['Estrategia']} - {row['Filtro']}: WR={row['Win Rate (%)']:.1f}%, "
                  f"Media={row['Media (%)']:.3f}% (n={row['N']})")

    # Señales totales
    for tn in TOP_N_LIST:
        sub = df[df["top_n"] == tn]
        n_signals = sub.groupby("signal_ts").ngroups
        n_overlap = sub[sub["is_overlapping"]].groupby("signal_ts").ngroups
        print(f"\n  TOP {tn}: {n_signals} señales totales, {n_overlap} overlapping")

    # Conclusiones
    print(f"\n  {'─' * 60}")
    print("  CONCLUSIONES:")
    print("  ─────────────")

    if not s1h.empty:
        best_strat = s1h.loc[s1h["Media (%)"].idxmax(), "Estrategia"]
        best_mean = s1h.loc[s1h["Media (%)"].idxmax(), "Media (%)"]
        best_wr = s1h.loc[s1h["Media (%)"].idxmax(), "Win Rate (%)"]

        if best_mean > 0.1 and best_wr > 52:
            tradeable = "POTENCIALMENTE TRADEABLE"
        elif best_mean > 0 and best_wr > 50:
            tradeable = "MARGINAL - requiere optimización"
        else:
            tradeable = "NO TRADEABLE tal como está"

        print(f"  - Mejor estrategia (1h): {best_strat} con retorno medio {best_mean:.3f}%")
        print(f"  - Win rate de la mejor: {best_wr:.1f}%")
        print(f"  - Veredicto: {tradeable}")

    if not vol_wr_df.empty:
        vol_yes = vol_wr_df[(vol_wr_df["Filtro"] == "Con vol filter") & (vol_wr_df["Período"] == "1h")]
        vol_no = vol_wr_df[(vol_wr_df["Filtro"] == "Sin vol filter") & (vol_wr_df["Período"] == "1h")]
        if not vol_yes.empty and not vol_no.empty:
            avg_wr_yes = vol_yes["Win Rate (%)"].mean()
            avg_wr_no = vol_no["Win Rate (%)"].mean()
            if avg_wr_yes > avg_wr_no:
                print(f"  - Filtro volumen MEJORA el win rate: {avg_wr_yes:.1f}% vs {avg_wr_no:.1f}%")
            else:
                print(f"  - Filtro volumen NO mejora: {avg_wr_yes:.1f}% vs {avg_wr_no:.1f}%")

    print(f"\n  Revisa los gráficos y CSVs en la carpeta de resultados para un")
    print(f"  análisis detallado. En particular, revisa los heatmaps para")
    print(f"  identificar las mejores horas/días y los quintiles de magnitud")
    print(f"  para entender si señales más fuertes dan mejores resultados.")
    print("=" * 80)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Crypto Futures Momentum Backtest")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directorio con archivos Parquet/CSV de velas 5 min")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directorio de salida para resultados")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    print("=" * 80)
    print("  CRYPTO FUTURES MOMENTUM BACKTEST")
    print(f"  Datos: {data_dir}")
    print(f"  Año: {YEAR}")
    print(f"  Intervalo señal: cada {SIGNAL_INTERVAL * CANDLE_MINUTES} min")
    print(f"  Lookback: {LOOKBACK_CANDLES * CANDLE_MINUTES} min ({LOOKBACK_CANDLES} velas)")
    print(f"  Forward: {FORWARD_CANDLES * CANDLE_MINUTES} min ({FORWARD_CANDLES} velas)")
    print("=" * 80)

    t_start = _time.time()

    # 1. Cargar datos
    print("\n[PASO 1] Cargando datos...")
    pairs = load_all_pairs(data_dir)

    # 2. Alinear matrices
    print("\n[PASO 2] Alineando datos en matrices...")
    all_ts, pair_names, close_matrix, volume_matrix = build_aligned_matrices(pairs)

    # Liberar memoria de los DataFrames individuales
    del pairs
    import gc
    gc.collect()

    # 3. Generar señales (con checkpoints)
    print("\n[PASO 3] Generando señales...")
    os.makedirs(output_dir, exist_ok=True)
    records = compute_signals(all_ts, pair_names, close_matrix, volume_matrix, output_dir)

    if not records:
        print("ERROR: No se generaron señales. Verifica que los datos cubran el año 2025.")
        sys.exit(1)

    # 4. Post-procesar
    print("\n[PASO 4] Post-procesando señales...")
    df, curve_matrix_signals = build_signals_dataframe(records)

    # 5. Guardar CSV maestro
    print("\n[PASO 5] Guardando CSV maestro...")
    save_master_csv(df, output_dir)

    # 6. Análisis y gráficos
    print("\n[PASO 6] Generando análisis y gráficos...")
    summary_df, wr_df, vol_wr_df = run_analysis(df, curve_matrix_signals, output_dir)

    # 7. Resumen ejecutivo
    print_executive_summary(summary_df, wr_df, vol_wr_df, df)

    # 8. Limpiar checkpoint (ejecución completada con éxito)
    ckpt_path = os.path.join(output_dir, CHECKPOINT_FILE)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print("  Checkpoint eliminado (ejecución completa).")

    elapsed = _time.time() - t_start
    print(f"\nTiempo total: {elapsed / 60:.1f} minutos")
    print(f"Resultados guardados en: {os.path.abspath(output_dir)}/")


if __name__ == "__main__":
    main()
