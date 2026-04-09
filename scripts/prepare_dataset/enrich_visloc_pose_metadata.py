import argparse
import csv
import glob
import json
import os


def build_visloc_drone_metadata(row):
    row_ci = {key.lower(): value for key, value in row.items()}
    height = float(row_ci["height"])
    omega = float(row_ci["omega"])
    kappa = float(row_ci["kappa"])
    phi1 = float(row_ci["phi1"])
    phi2 = float(row_ci["phi2"])
    return {
        "height": height,
        "drone_roll": kappa,
        "drone_pitch": omega,
        "drone_yaw": phi1,
        "cam_roll": kappa,
        "cam_pitch": omega,
        "cam_yaw": phi1,
        "cam_yaw_phi2": phi2,
        "phi1": phi1,
        "phi2": phi2,
        "query_rotation_cw_deg": phi1,
    }


def load_pose_lookup(data_root):
    pose_lookup = {}
    for idx in range(1, 12):
        scene_id = f"{idx:02}"
        csv_path = os.path.join(data_root, scene_id, f"{scene_id}.csv")
        if not os.path.exists(csv_path):
            continue

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames_ci = {str(name).lower() for name in (reader.fieldnames or [])}
            required = {"filename", "height", "omega", "kappa", "phi1", "phi2"}
            if not required.issubset(fieldnames_ci):
                print(f"[Skip] {os.path.basename(csv_path)} has no full pose columns: {sorted(fieldnames_ci)}")
                continue
            for row in reader:
                row_ci = {key.lower(): value for key, value in row.items()}
                filename = row_ci["filename"]
                pose_lookup[filename] = build_visloc_drone_metadata(row)

    if not pose_lookup:
        raise FileNotFoundError(f"No per-scene CSV metadata found under {data_root}")

    return pose_lookup


def discover_json_files(data_root):
    patterns = [
        os.path.join(data_root, "*-drone2sate-*.json"),
    ]
    json_files = []
    for pattern in patterns:
        json_files.extend(glob.glob(pattern))
    json_files = sorted(set(json_files))
    return [path for path in json_files if not path.endswith("-pose.json")]


def enrich_json_file(input_json, output_json, pose_lookup):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing = []
    for entry in data:
        img_name = entry["drone_img_name"]
        if img_name not in pose_lookup:
            missing.append(img_name)
            continue
        entry["drone_metadata"] = pose_lookup[img_name]

    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"Missing pose metadata for {len(missing)} entries in {os.path.basename(input_json)}. "
            f"Examples: {preview}"
        )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return len(data)


def main():
    parser = argparse.ArgumentParser(description="Fill UAV-VisLoc JSON metadata with pose fields from per-scene CSV files.")
    parser.add_argument("--data_root", type=str, required=True, help="Root of UAV_VisLoc_dataset")
    parser.add_argument(
        "--input_json",
        type=str,
        nargs="*",
        default=None,
        help="One or more JSON files to enrich. Defaults to all *-drone2sate-*.json under data_root.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="-pose",
        help="Suffix appended before .json when writing enriched files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the input JSON files instead of writing *-pose.json copies.",
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    pose_lookup = load_pose_lookup(data_root)

    if args.input_json:
        input_jsons = [
            path if os.path.isabs(path) else os.path.join(data_root, path)
            for path in args.input_json
        ]
    else:
        input_jsons = discover_json_files(data_root)

    if not input_jsons:
        raise FileNotFoundError(f"No JSON files found under {data_root}")

    total_entries = 0
    for input_json in input_jsons:
        if not os.path.exists(input_json):
            raise FileNotFoundError(input_json)
        if args.overwrite:
            output_json = input_json
        else:
            stem, ext = os.path.splitext(input_json)
            output_json = f"{stem}{args.suffix}{ext}"
        entry_count = enrich_json_file(input_json, output_json, pose_lookup)
        total_entries += entry_count
        print(f"[OK] {os.path.basename(input_json)} -> {os.path.basename(output_json)} ({entry_count} entries)")

    print(f"Finished. Files: {len(input_jsons)}, entries: {total_entries}")


if __name__ == "__main__":
    main()
