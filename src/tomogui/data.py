"""Data inspection with bounded display sampling and strict record validation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import csv
import math
from pathlib import Path
import re

import numpy as np
from scipy.spatial import cKDTree

from .workspace import load_module


@dataclass
class Event:
    parts: list[str]
    picks: list[list[str]] = field(default_factory=list)


def read_src_rec(path: Path) -> list[Event]:
    events = []
    event = None
    with path.open(encoding="utf-8-sig") as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            parts = raw.split()
            try:
                try:
                    is_header = (len(parts) >= 12 and 1800 <= int(parts[1]) <= 2500
                                 and 1 <= int(parts[2]) <= 12 and 1 <= int(parts[3]) <= 31)
                except (ValueError, IndexError):
                    is_header = False
                if is_header:
                    if event is not None and int(event.parts[11]) != len(event.picks):
                        raise ValueError("Source receiver count does not match its records.")
                    if len(parts) < 12 or not 1800 <= int(parts[1]) <= 2500:
                        raise ValueError("Expected a TomoATT source header.")
                    lat, lon, dep = map(float, parts[7:10])
                    if not all(math.isfinite(v) for v in (lat, lon, dep)) or abs(lat) > 90 or abs(lon) > 180:
                        raise ValueError("Invalid source coordinates.")
                    int(parts[11])
                    if len(parts) >= 14 and (not math.isfinite(float(parts[13])) or float(parts[13]) < 0):
                        raise ValueError('Invalid event weight.')
                    event = Event(parts)
                    events.append(event)
                else:
                    if event is None or len(parts) < 8:
                        raise ValueError("Receiver record without a source or missing fields.")
                    if parts[0] != event.parts[0]:
                        raise ValueError("Receiver source index does not match its header.")
                    # The preparation workflow is for absolute arrivals. Differential
                    # records have another layout and must not be silently rewritten.
                    if len(parts) not in (8,9):
                        raise ValueError("Data preparation currently accepts absolute receiver rows with 8 or 9 fields; use the original scripts for differential or extended records.")
                    lat, lon, elev, tt = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[7])
                    if not all(math.isfinite(v) for v in (lat, lon, elev, tt)) or abs(lat) > 90 or abs(lon) > 180:
                        raise ValueError("Invalid receiver coordinates or travel time.")
                    if len(parts) >= 9 and (not math.isfinite(float(parts[8])) or float(parts[8]) < 0):
                        raise ValueError("Invalid receiver weight.")
                    event.picks.append(parts)
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path.name}, line {number}: {exc}") from exc
    if not events or not any(event.picks for event in events):
        raise ValueError("No source/receiver arrivals found.")
    if int(events[-1].parts[11]) != len(events[-1].picks):
        raise ValueError("Final source receiver count does not match its records.")
    return events


def write_src_rec(path: Path, events: list[Event]) -> None:
    if path.exists():
        raise ValueError(f"Refusing to overwrite an existing data file: {path}")
    with path.open("x", encoding="ascii", newline="\n") as handle:
        for event in events:
            header = list(event.parts)
            header[11] = str(len(event.picks))
            handle.write(" ".join(header) + "\n")
            for pick in event.picks:
                handle.write("    " + " ".join(pick) + "\n")


def canonicalize(path: Path) -> None:
    """Normalize only owned copies for whitespace-sensitive legacy helpers."""
    events = read_src_rec(path)
    temp = path.with_name(path.name + '.canonical.tmp')
    write_src_rec(temp, events)
    temp.replace(path)


def receiver_spacing(stations: list[list]) -> float | None:
    """Same spherical nearest-neighbor metric as the scanner, without NxN storage."""
    if len(stations) < 2:
        return None
    lat, lon = np.radians(np.array([[s[1], s[2]] for s in stations], dtype=float)).T
    xyz = np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))
    distances = cKDTree(xyz).query(xyz, k=2)[0][:, 1]
    return float(np.mean(2 * 6371.0 * np.arcsin(np.clip(distances / 2, 0, 1))))


def summarize(path: Path) -> dict:
    events = read_src_rec(path)
    stations = {}
    phases = Counter()
    for event in events:
        for pick in event.picks:
            stations.setdefault(pick[2], [pick[2], float(pick[3]), float(pick[4]), float(pick[5])])
            phases[pick[6].upper()] += 1
    points = [[float(e.parts[7]), float(e.parts[8]), float(e.parts[9])] for e in events]
    station_list = list(stations.values())
    lat = [p[0] for p in points] + [p[1] for p in station_list]
    lon = [p[1] for p in points] + [p[2] for p in station_list]
    depth = [p[2] for p in points]
    stride = max(1, math.ceil(len(points) / 2500))
    return {"path": str(path), "n_events": len(events), "n_stations": len(stations),
            "n_picks": sum(phases.values()), "phases": dict(phases),
            "events": points[::stride], "stations": station_list[:5000], "display_sampled": stride > 1,
            "avg_receiver_spacing_km": receiver_spacing(station_list),
            "region_bounds": {"lat_min": min(lat), "lat_max": max(lat), "lon_min": min(lon),
                              "lon_max": max(lon), "depth_min": min(0, math.floor(min(depth))),
                              "depth_max": math.ceil(max(depth))}}


def detect(path: Path, workspace: Path) -> dict:
    if not path.is_file():
        raise ValueError("Choose a data file.")
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        sample = handle.read(65536)
    lines = [s for s in sample.splitlines() if s.strip() and not s.lstrip().startswith("#")]
    if not lines:
        raise ValueError("The input file is empty.")
    first = lines[0]
    isc_header = next((i for i,line in enumerate(lines) if re.match(r'^\s*EVENTID\s*,', line, re.I)), None)
    if isc_header is not None:
        lines = lines[isc_header:]
        first = lines[0]
    tokens = first.split()
    result = {"path": str(path), "columns": [], "needs_stations": False, "ready": False,
              "preview": {"text": '\n'.join(sample.splitlines()[:40]), "bytes": path.stat().st_size,
                          "note": "Read-only preview: first 40 lines, up to 64 KiB. The original file is unchanged."}}
    try:
        is_src = len(tokens) >= 12 and 1800 <= int(tokens[1]) <= 2500 and abs(float(tokens[7])) <= 90 and abs(float(tokens[8])) <= 180
    except (ValueError, IndexError):
        is_src = False
    if is_src:
        return {**result, "kind": "src_rec", "label": "TomoATT src_rec", "ready": True,
                "summary": summarize(path), "adapter": "Validated source/receiver records"}
    if "," in first or "\t" in first:
        delimiter = "\t" if "\t" in first else ","
        columns = [column.strip() for column in next(csv.reader([first], delimiter=delimiter))]
        if {'evtid', 'quality', 'lat', 'lon'}.issubset({c.lower() for c in columns}):
            # This workspace's eqall catalog has a comma-separated header and
            # whitespace records. It has hypocentres/grades but no arrival picks.
            rows = []
            for line in lines[1:7]:
                parts = line.split()
                if len(parts) >= len(columns):
                    rows.append(dict(zip(columns, parts[:12] + [' '.join(parts[12:-2]), parts[-2], parts[-1]])))
            return {**result, 'kind': 'event_catalog', 'label': 'Event catalog (no arrival picks)',
                    'columns': columns, 'sample': rows, 'preparable': False,
                    'message': 'This file contains event locations and grades, but no station arrival times. Choose a phase-arrival catalog (such as simulp_evt.dat or hypo3d_w3.cwb) to prepare src_rec. Grades can be previewed here.'}
        if isc_header is not None:
            # Use the same duplicate-header normalization as the existing ISC
            # reader (origin DATE/TIME/LAT/LON must not overwrite arrival fields).
            counts = Counter()
            unique = []
            for column in columns:
                key = column.upper()
                occurrence = counts[key]; counts[key] += 1
                unique.append(column + '.' + str(occurrence) if occurrence else column)
            lines[0] = delimiter.join(unique)
        rows = list(csv.DictReader(lines[:7], delimiter=delimiter))
        return {**result, "kind": "csv", "label": "Arrival table", "columns": list(rows[0]) if rows else next(csv.reader([first], delimiter=delimiter)),
                "sample": rows[:6], "delimiter": delimiter, "adapter": "region_scanner_pipeline.py"}
    salute = load_module(workspace / "Scripts/convert_salute_to_src_rec.py")
    if any(salute.parse_event_header(line, 70) for line in lines[:10]):
        return {**result, "kind": "salute", "label": "SALUTE event catalog", "needs_stations": True,
                "adapter": "convert_salute_to_src_rec.py"}
    if re.match(r"^\d{10}", first.strip()) and len(first) >= 36:
        return {**result, "kind": "cwb", "label": "CWB catalog", "needs_stations": True,
                "adapter": "convert_cwb_to_src_rec.py"}
    return {**result, "kind": "unknown", "label": "Unrecognized data structure",
            "message": "No supported record signature found. Convert this source with a workspace adapter first."}
