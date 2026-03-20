from pathlib import Path
import re
import csv
from pypdf import PdfReader
from datetime import datetime


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
            texts.append(t)
        except Exception:
            continue
    return "\n".join(texts)


def detect_setting(line: str) -> str | None:
    l = line.lower()
    if "same-area" in l or "same area" in l:
        return "same-area"
    if "cross-area" in l or "cross area" in l:
        return "cross-area"
    return None


def detect_dataset(line: str) -> str | None:
    l = line.lower()
    if "gta-uav" in l or "gtauav" in l:
        return "GTA-UAV"
    if "uav-visloc" in l or "visloc" in l:
        return "UAV-VisLoc"
    return None


def parse_rows(text: str) -> list[dict]:
    rows = []
    current_dataset = None
    current_setting = None
    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        if not raw or raw.strip().startswith("Table"):
            ds = detect_dataset(raw)
            if ds:
                current_dataset = ds
            st = detect_setting(raw)
            if st:
                current_setting = st
            continue
        ds = detect_dataset(raw)
        if ds:
            current_dataset = ds
        st = detect_setting(raw)
        if st:
            current_setting = st
        num_matches = list(re.finditer(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%?", raw))
        if len(num_matches) < 4:
            continue
        name_end = num_matches[0].start()
        name = raw[:name_end].strip()
        if len(name) < 2:
            continue
        ign = name.lower()
        if any(k in ign for k in ["recall", "r@1", "r@5", "r@10", "map"]):
            continue
        vals = []
        for m in num_matches[:4]:
            try:
                vals.append(float(m.group(1)))
            except Exception:
                vals.append(None)
        if any(v is None for v in vals):
            continue
        if not all(0.0 <= v <= 100.0 for v in vals):
            continue
        v1, v5, v10, mp = vals
        if current_dataset is None:
            for back in range(1, 4):
                if idx - back >= 0:
                    ds2 = detect_dataset(lines[idx - back])
                    if ds2:
                        current_dataset = ds2
                        break
        if current_setting is None:
            for back in range(1, 4):
                if idx - back >= 0:
                    st2 = detect_setting(lines[idx - back])
                    if st2:
                        current_setting = st2
                        break
        dataset = current_dataset or "GTA-UAV"
        setting = current_setting or "same-area"
        rows.append({
            "model": name,
            "dataset": dataset,
            "setting": setting,
            "R1": v1,
            "R5": v5,
            "R10": v10,
            "mAP": mp,
        })
    return rows


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", "setting", "R1", "R5", "R10", "mAP"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    proj_root = Path(__file__).resolve().parents[1]
    pdf_path = proj_root / "Game4Loc.pdf"
    text = extract_text(pdf_path)
    rows = parse_rows(text)
    out_csv = proj_root / "Log" / "external_results.csv"
    write_csv(rows, out_csv)
    print(str(out_csv))
    print(str(len(rows)))


if __name__ == "__main__":
    main()
