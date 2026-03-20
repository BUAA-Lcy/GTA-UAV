import os
import json
import argparse
import math
import glob
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def euler_to_rotation_matrix(pitch: float, roll: float, yaw: float) -> np.ndarray:
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    yaw = np.radians(yaw)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(roll), 0, np.sin(roll)],
                   [0, 1, 0],
                   [-np.sin(roll), 0, np.cos(roll)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R


def compute_tilt_deg_from_euler(pitch: float, roll: float, yaw: float) -> float:
    R = euler_to_rotation_matrix(pitch, roll, yaw)
    f_cam = np.array([-1.0, 0.0, 0.0], dtype=float)
    f_world = R @ f_cam
    down = np.array([0.0, 0.0, -1.0], dtype=float)
    num = float(np.dot(f_world, down))
    den = float(np.linalg.norm(f_world) * np.linalg.norm(down))
    if den == 0.0:
        return float("nan")
    cosang = max(-1.0, min(1.0, num / den))
    tilt = math.degrees(math.acos(cosang))
    return float(tilt)


def pick_euler(entry: Dict[str, Any], prefer_camera: bool = True) -> Optional[Tuple[float, float, float]]:
    meta = entry.get("drone_metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    keys_cam = ("cam_pitch", "cam_roll", "cam_yaw")
    keys_drone = ("drone_pitch", "drone_roll", "drone_yaw")
    if prefer_camera and all(k in meta for k in keys_cam):
        try:
            return float(meta["cam_pitch"]), float(meta["cam_roll"]), float(meta["cam_yaw"])
        except Exception:
            pass
    if all(k in meta for k in keys_drone):
        try:
            return float(meta["drone_pitch"]), float(meta["drone_roll"]), float(meta["drone_yaw"])
        except Exception:
            pass
    top_keys = ("drone_pitch", "drone_roll", "drone_yaw")
    if all(k in entry for k in top_keys):
        try:
            return float(entry["drone_pitch"]), float(entry["drone_roll"]), float(entry["drone_yaw"])
        except Exception:
            pass
    return None


def discover_meta_files(data_root: str) -> List[str]:
    patterns = [
        os.path.join(data_root, "*drone2sate*.json"),
        os.path.join(data_root, "*.json"),
    ]
    seen = set()
    files = []
    for p in patterns:
        for f in glob.glob(p):
            if not os.path.isfile(f):
                continue
            name = os.path.basename(f)
            if "drone2sate" in name or "pairs" in name or "train" in name or "test" in name:
                if f not in seen:
                    files.append(f)
                    seen.add(f)
    if not files:
        for f in glob.glob(os.path.join(data_root, "*.json")):
            if os.path.isfile(f) and f not in seen:
                files.append(f)
                seen.add(f)
    return sorted(files)


def load_all_entries(meta_files: List[str]) -> List[Dict[str, Any]]:
    entries = []
    for mf in meta_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    entries.extend(data)
        except Exception:
            continue
    return entries


def compute_stats(tilts: np.ndarray, thr_list: List[float]) -> Dict[str, Any]:
    tilts = tilts[np.isfinite(tilts)]
    total = int(tilts.size)
    stats = {}
    stats["count"] = total
    for thr in thr_list:
        key = f"ratio_ge_{int(thr)}"
        stats[key] = float((tilts >= thr).mean()) if total > 0 else 0.0
    q_levels = [5, 25, 50, 75, 90, 95, 99]
    if total > 0:
        quants = np.percentile(tilts, q_levels).tolist()
    else:
        quants = [float("nan")] * len(q_levels)
    stats["quantiles"] = {str(q): float(v) for q, v in zip(q_levels, quants)}
    return stats


def plot_hist(tilts: np.ndarray, bins: int, save_path: str) -> None:
    tilts = tilts[np.isfinite(tilts)]
    plt.figure(figsize=(6, 4), dpi=200)
    plt.hist(tilts, bins=bins, range=(0, 90), color="#5975A4", edgecolor="black", alpha=0.9)
    plt.xlabel("Tilt angle (deg)")
    plt.ylabel("Count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compute tilt distribution statistics for GTA-UAV.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to GTA-UAV dataset root.")
    parser.add_argument("--meta_files", type=str, nargs="*", default=None, help="Explicit meta JSON files. If omitted, auto-discover.")
    parser.add_argument("--prefer_camera", action="store_true", help="Prefer camera pose if available.")
    parser.add_argument("--bins", type=int, default=72, help="Histogram bins.")
    parser.add_argument("--save_dir", type=str, default="output/tilt_stats", help="Directory to save outputs.")
    parser.add_argument("--thresholds", type=str, default="30,50", help="Comma-separated thresholds in degrees.")
    args = parser.parse_args()

    if args.meta_files and len(args.meta_files) > 0:
        meta_files = [mf if os.path.isabs(mf) else os.path.join(args.data_root, mf) for mf in args.meta_files]
        meta_files = [mf for mf in meta_files if os.path.isfile(mf)]
    else:
        meta_files = discover_meta_files(args.data_root)
    if not meta_files:
        raise FileNotFoundError("No meta JSON files found. Please specify --meta_files.")

    entries = load_all_entries(meta_files)
    tilts = []
    for e in entries:
        eu = pick_euler(e, prefer_camera=args.prefer_camera)
        if eu is None:
            continue
        p, r, y = eu
        t = compute_tilt_deg_from_euler(p, r, y)
        if np.isfinite(t):
            tilts.append(t)
    tilts = np.array(tilts, dtype=float)

    thr_list = [float(x) for x in args.thresholds.split(",") if x.strip()]
    stats = compute_stats(tilts, thr_list)
    os.makedirs(args.save_dir, exist_ok=True)

    df = pd.DataFrame({"tilt_deg": tilts})
    csv_path = os.path.join(args.save_dir, "tilts.csv")
    df.to_csv(csv_path, index=False)

    hist_path = os.path.join(args.save_dir, "tilt_hist.png")
    plot_hist(tilts, args.bins, hist_path)

    txt_path = os.path.join(args.save_dir, "tilt_stats.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"count: {stats['count']}\n")
        for thr in thr_list:
            key = f"ratio_ge_{int(thr)}"
            f.write(f"{key}: {stats[key]:.6f}\n")
        f.write("quantiles:\n")
        for k, v in stats["quantiles"].items():
            f.write(f"  {k}%: {v:.6f}\n")

    json_path = os.path.join(args.save_dir, "tilt_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Meta files: {len(meta_files)}")
    print(f"Samples used: {stats['count']}")
    for thr in thr_list:
        key = f"ratio_ge_{int(thr)}"
        print(f"{key}: {stats[key]:.6f}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {hist_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
