from __future__ import annotations

import csv
import hashlib
import py_compile
from pathlib import Path, PureWindowsPath

from robustness_common import interpolate_stress, change_post_stress

ROOT = Path(__file__).resolve().parent

for name in (
    "robustness_common.py",
    "run_observation_sweep.py",
    "run_stress_ladder.py",
    "run_changepoint.py",
):
    py_compile.compile(str(ROOT / name), doraise=True)

manifest = ROOT / "code_sha256_manifest.csv"
if not manifest.is_file():
    raise RuntimeError("code_sha256_manifest.csv is missing")
with manifest.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise RuntimeError("Frozen-code hash manifest is empty")

# Accommodate either PowerShell Export-Csv Path/Hash columns or lowercase variants.
for row in rows:
    path_value = row.get("Path") or row.get("path") or row.get("File") or row.get("file")
    expected = row.get("Hash") or row.get("hash") or row.get("SHA256") or row.get("sha256")
    if not path_value or not expected:
        continue
    name = PureWindowsPath(path_value).name if "\\" in path_value or ":" in path_value else Path(path_value).name
    p = ROOT / name
    if not p.is_file():
        raise RuntimeError(f"Frozen file missing: {name}")
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual.lower() != expected.strip().lower():
        raise RuntimeError(f"Frozen file hash mismatch: {name}")

for a in (0.25, 0.50, 0.75):
    st = interpolate_stress(a)
    if not (4.0 < float(st.actual_shadowing_std_db) < 6.0):
        raise RuntimeError("Stress ladder interpolation failed")
    if not (1 <= int(st.stale_csi_slots) <= 5):
        raise RuntimeError("Stress ladder stale-CSI interpolation failed")

post = change_post_stress()
if post.actual_speed_mean_mps is not None or post.actual_speed_std_mps is not None:
    raise RuntimeError("Change-point post stress must not resample mobility")

print("ForeSafe-RL robustness-evaluation package validation passed.")
print("Frozen core hashes match the pre-final manifest.")
print("Observation sweep, stress ladder, and change-point definitions are valid.")
