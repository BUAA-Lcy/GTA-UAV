import os
import argparse
import math
import glob
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rx(a: float) -> np.ndarray:
    r = math.radians(a)
    return np.array([[1, 0, 0],
                     [0, math.cos(r), -math.sin(r)],
                     [0, math.sin(r), math.cos(r)]])


def ry(a: float) -> np.ndarray:
    r = math.radians(a)
    return np.array([[math.cos(r), 0, math.sin(r)],
                     [0, 1, 0],
                     [-math.sin(r), 0, math.cos(r)]])


def rz(a: float) -> np.ndarray:
    r = math.radians(a)
    return np.array([[math.cos(r), -math.sin(r), 0],
                     [math.sin(r), math.cos(r), 0],
                     [0, 0, 1]])


def r_omega_phi_kappa(omega: float, phi: float, kappa: float) -> np.ndarray:
    return rz(kappa) @ ry(phi) @ rx(omega)


def compute_tilt(omega: float, phi: float, kappa: float) -> float:
    R = r_omega_phi_kappa(omega, phi, kappa)
    f_cam = np.array([1.0, 0.0, 0.0], dtype=float)
    f_world = R @ f_cam
    down = np.array([0.0, 0.0, -1.0], dtype=float)
    num = float(np.dot(f_world, down))
    den = float(np.linalg.norm(f_world) * np.linalg.norm(down))
    if den == 0.0:
        return float("nan")
    c = max(-1.0, min(1.0, num / den))
    return math.degrees(math.acos(c))


def discover_csvs(root: str) -> List[str]:
    patterns = [
        os.path.join(root, "**", "*.csv"),
        os.path.join(root, "*.csv"),
    ]
    csvs = []
    for p in patterns:
        csvs.extend(glob.glob(p, recursive=True))
    csvs = [c for c in csvs if os.path.isfile(c)]
    return sorted(csvs)


def compute_stats(values: np.ndarray, thresholds: List[float]) -> Dict[str, Any]:
    v = values[np.isfinite(values)]
    total = int(v.size)
    out: Dict[str, Any] = {"count": total}
    for t in thresholds:
        out[f"ratio_ge_{int(t)}"] = float((v >= t).mean()) if total > 0 else 0.0
    q_levels = [5, 25, 50, 75, 90, 95, 99]
    if total > 0:
        quants = np.percentile(v, q_levels).tolist()
    else:
        quants = [float("nan")] * len(q_levels)
    out["quantiles"] = {str(q): float(val) for q, val in zip(q_levels, quants)}
    return out


def plot_hist(values: np.ndarray, bins: int, save_path: str) -> None:
    v = values[np.isfinite(values)]
    plt.figure(figsize=(6, 4), dpi=200)
    plt.hist(v, bins=bins, range=(0, 90), color="#4C84B6", edgecolor="black", alpha=0.9)
    plt.xlabel("Tilt angle (deg)")
    plt.ylabel("Count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def load_and_compute(csv_paths: List[str], bins: int, save_dir: str, thresholds: List[float]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    tilts_phi1 = []
    tilts_phi2 = []
    perfile_phi1: List[Dict[str, Any]] = []
    perfile_phi2: List[Dict[str, Any]] = []
    perfile_auto: List[Dict[str, Any]] = []
    tilts_auto = []
    for csv in csv_paths:
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        if not all(x in cols for x in ["omega", "kappa"]) or not ("phi1" in cols or "phi2" in cols):
            continue
        omega_col = cols["omega"]
        kappa_col = cols["kappa"]
        phi1_col = cols.get("phi1", None)
        phi2_col = cols.get("phi2", None)
        v1 = []
        v2 = []
        for _, row in df.iterrows():
            try:
                omega = float(row[omega_col])
                kappa = float(row[kappa_col])
            except Exception:
                continue
            best = None
            if phi1_col is not None:
                try:
                    phi1 = float(row[phi1_col])
                    t1 = compute_tilt(omega, phi1, kappa)
                    v1.append(t1)
                    best = t1 if best is None else min(best, t1)
                except Exception:
                    pass
            if phi2_col is not None:
                try:
                    phi2 = float(row[phi2_col])
                    t2 = compute_tilt(omega, phi2, kappa)
                    v2.append(t2)
                    best = t2 if best is None else min(best, t2)
                except Exception:
                    pass
            if best is not None:
                tilts_auto.append(best)
        if len(v1) > 0:
            arr = np.array(v1, dtype=float)
            tilts_phi1.extend(arr.tolist())
            st = compute_stats(arr, thresholds)
            st_row = {"file": os.path.basename(csv), "count": st["count"]}
            for t in thresholds:
                st_row[f"ratio_ge_{int(t)}"] = st[f"ratio_ge_{int(t)}"]
            st_row.update({f"q{q}": st["quantiles"][str(q)] for q in [5,25,50,75,90,95,99]})
            perfile_phi1.append(st_row)
        if len(v2) > 0:
            arr = np.array(v2, dtype=float)
            tilts_phi2.extend(arr.tolist())
            st = compute_stats(arr, thresholds)
            st_row = {"file": os.path.basename(csv), "count": st["count"]}
            for t in thresholds:
                st_row[f"ratio_ge_{int(t)}"] = st[f"ratio_ge_{int(t)}"]
            st_row.update({f"q{q}": st["quantiles"][str(q)] for q in [5,25,50,75,90,95,99]})
            perfile_phi2.append(st_row)
        if len(v1) > 0 or len(v2) > 0:
            arr = np.array([min(a, b) if (not math.isnan(a) and not math.isnan(b)) else (a if not math.isnan(a) else b) for a, b in zip(v1 if len(v1)==len(v2) else v1+[float('nan')]*(max(len(v1),len(v2))-len(v1)), v2 if len(v2)==len(v1) else v2+[float('nan')]*(max(len(v1),len(v2))-len(v2)) )], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                st = compute_stats(arr, thresholds)
                st_row = {"file": os.path.basename(csv), "count": st["count"]}
                for t in thresholds:
                    st_row[f"ratio_ge_{int(t)}"] = st[f"ratio_ge_{int(t)}"]
                st_row.update({f"q{q}": st["quantiles"][str(q)] for q in [5,25,50,75,90,95,99]})
                perfile_auto.append(st_row)
    if len(tilts_phi1) > 0:
        v = np.array(tilts_phi1, dtype=float)
        stats = compute_stats(v, thresholds)
        pd.DataFrame({"tilt_deg": v}).to_csv(os.path.join(save_dir, "tilts_phi1.csv"), index=False)
        plot_hist(v, bins, os.path.join(save_dir, "tilt_hist_phi1.png"))
        pd.DataFrame(perfile_phi1).to_csv(os.path.join(save_dir, "tilt_stats_phi1_perfile.csv"), index=False)
        with open(os.path.join(save_dir, "tilt_stats_phi1.txt"), "w", encoding="utf-8") as f:
            f.write(f"count: {stats['count']}\n")
            for t in thresholds:
                f.write(f"ratio_ge_{int(t)}: {stats[f'ratio_ge_{int(t)}']:.6f}\n")
            f.write("quantiles:\n")
            for k, val in stats["quantiles"].items():
                f.write(f"  {k}%: {val:.6f}\n")
    if len(tilts_phi2) > 0:
        v = np.array(tilts_phi2, dtype=float)
        stats = compute_stats(v, thresholds)
        pd.DataFrame({"tilt_deg": v}).to_csv(os.path.join(save_dir, "tilts_phi2.csv"), index=False)
        plot_hist(v, bins, os.path.join(save_dir, "tilt_hist_phi2.png"))
        pd.DataFrame(perfile_phi2).to_csv(os.path.join(save_dir, "tilt_stats_phi2_perfile.csv"), index=False)
        with open(os.path.join(save_dir, "tilt_stats_phi2.txt"), "w", encoding="utf-8") as f:
            f.write(f"count: {stats['count']}\n")
            for t in thresholds:
                f.write(f"ratio_ge_{int(t)}: {stats[f'ratio_ge_{int(t)}']:.6f}\n")
            f.write("quantiles:\n")
            for k, val in stats["quantiles"].items():
                f.write(f"  {k}%: {val:.6f}\n")
    if len(tilts_auto) > 0:
        v = np.array(tilts_auto, dtype=float)
        stats = compute_stats(v, thresholds)
        pd.DataFrame({"tilt_deg": v}).to_csv(os.path.join(save_dir, "tilts_auto.csv"), index=False)
        plot_hist(v, bins, os.path.join(save_dir, "tilt_hist_auto.png"))
        pd.DataFrame(perfile_auto).to_csv(os.path.join(save_dir, "tilt_stats_auto_perfile.csv"), index=False)
        with open(os.path.join(save_dir, "tilt_stats_auto.txt"), "w", encoding="utf-8") as f:
            f.write(f"count: {stats['count']}\n")
            for t in thresholds:
                f.write(f"ratio_ge_{int(t)}: {stats[f'ratio_ge_{int(t)}']:.6f}\n")
            f.write("quantiles:\n")
            for k, val in stats["quantiles"].items():
                f.write(f"  {k}%: {val:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Compute tilt stats for UAV_VisLoc_dataset CSVs.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--save_dir", type=str, default="output/tilt_stats_visloc")
    parser.add_argument("--thresholds", type=str, default="30,50")
    args = parser.parse_args()
    csvs = discover_csvs(args.data_root)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    compute_stats_dir = os.path.abspath(args.save_dir)
    load_and_compute(csvs, args.bins, compute_stats_dir, thresholds)
    print(f"CSV files: {len(csvs)}")
    print(f"Saved directory: {compute_stats_dir}")


if __name__ == "__main__":
    main()
