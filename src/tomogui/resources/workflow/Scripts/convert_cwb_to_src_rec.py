#!/usr/bin/env python3
"""
Convert a CWB .cwb file (block format) into TomoATT src_rec.dat format.

Default inputs:
  - CWB catalog: CWBcatalog/eqsearch_PFILE_t20210101-20220831+20230101-20231231.salute.cwb
  - Station file: CWBcatalog/hypo3d.sta (repeatable with --sta or via --sta-dir)

Station files can be:
  - fixed-width .sta format
  - CSV with columns for station code, latitude, longitude, and depth/elevation

The .sta file must follow the fixed-width format:
  (a4,i2,f5.2,i3,f5.2,f7.1,3f5.2,2(1x,i8))
which provides station name, lat deg/min, lon deg/min, elevation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _safe_int(value: str, default: Optional[int] = None) -> Optional[int]:
    value = value.strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _safe_float(value: str, default: Optional[float] = None) -> Optional[float]:
    value = value.strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _normalize_header(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _get_column(columns: List[str], options: Iterable[str]) -> str | None:
    for option in options:
        if option in columns:
            return option
    return None


def parse_station_csv(path: Path) -> Dict[str, Tuple[float, float, float]]:
    station_map: Dict[str, Tuple[float, float, float]] = {}

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return station_map
        headers = [_normalize_header(name) for name in reader.fieldnames]
        reader.fieldnames = headers

        sta_col = _get_column(headers, ["station", "sta", "code", "station_code"])
        lat_col = _get_column(headers, ["latitude", "lat"])
        lon_col = _get_column(headers, ["longitude", "lon", "long"])
        dep_col = _get_column(headers, ["depth", "elev", "elevation", "stel"])

        if not sta_col or not lat_col or not lon_col:
            return station_map

        for row in reader:
            sta = str(row.get(sta_col, "")).strip()
            if not sta:
                continue
            try:
                lat = float(str(row.get(lat_col, "")).strip())
                lon = float(str(row.get(lon_col, "")).strip())
            except ValueError:
                continue
            dep_text = str(row.get(dep_col, "")).strip() if dep_col else ""
            try:
                depth = float(dep_text) if dep_text else 0.0
            except ValueError:
                depth = 0.0

            key = sta.strip().upper()[:6]
            station_map[key] = (lat, lon, depth)

    return station_map


def parse_station_file(path: Path) -> Dict[str, Tuple[float, float, float]]:
    if path.suffix.lower() == ".csv":
        return parse_station_csv(path)

    station_map: Dict[str, Tuple[float, float, float]] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            if idx == 0:
                # The first line in these files is a reference location, not a station record.
                continue
            if not line.strip():
                continue

            name = line[0:4].strip()
            lat_deg = _safe_int(line[4:6])
            lat_min = _safe_float(line[6:11])
            lon_deg = _safe_int(line[11:14])
            lon_min = _safe_float(line[14:19])
            elev = _safe_float(line[19:26])

            if not name or lat_deg is None or lat_min is None or lon_deg is None or lon_min is None:
                continue

            lat = lat_deg + lat_min / 60.0
            lon = lon_deg + lon_min / 60.0
            ele = elev if elev is not None else 0.0

            key = name.strip()
            station_map[key] = (lat, lon, ele)

            # Some files repeat the station code at the end; keep it as an alias if present.
            tail = line[-4:].strip()
            if tail and tail not in station_map:
                station_map[tail] = (lat, lon, ele)

    return station_map


def parse_station_files(paths: Iterable[Path], prefer: str) -> Dict[str, Tuple[float, float, float]]:
    merged: Dict[str, Tuple[float, float, float]] = {}
    for path in paths:
        station_map = parse_station_file(path)
        for key, value in station_map.items():
            if key in merged and prefer == "first":
                continue
            merged[key] = value
    return merged


def _year_from_two_digits(year2: int, cutoff: int) -> int:
    return 2000 + year2 if year2 < cutoff else 1900 + year2


def parse_cwb_header(line: str, year_cutoff: int) -> Dict[str, object]:
    padded = line.ljust(70)

    year2 = _safe_int(padded[0:2], 0) or 0
    month = _safe_int(padded[2:4], 1) or 1
    day = _safe_int(padded[4:6], 1) or 1
    hour = _safe_int(padded[6:8], 0) or 0
    minute = _safe_int(padded[8:10], 0) or 0
    second = _safe_float(padded[10:15], 0.0) or 0.0

    lat_deg = _safe_int(padded[15:17], 0) or 0
    lat_min = _safe_float(padded[17:22], 0.0) or 0.0
    lon_deg = _safe_int(padded[22:25], 0) or 0
    lon_min = _safe_float(padded[25:30], 0.0) or 0.0
    depth = _safe_float(padded[30:36], 0.0) or 0.0

    mag_raw = padded[36:40].strip()
    mag = _safe_float(mag_raw, 0.0) if mag_raw != "****" else 0.0

    tail = padded[62:].strip()
    tail_parts = tail.split()
    quality = tail_parts[0] if len(tail_parts) > 0 else ""
    nph = tail_parts[1] if len(tail_parts) > 1 else ""
    evtid = tail_parts[2] if len(tail_parts) > 2 else ""

    lat = lat_deg + lat_min / 60.0
    lon = lon_deg + lon_min / 60.0

    return {
        "year": _year_from_two_digits(year2, year_cutoff),
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
        "lat": lat,
        "lon": lon,
        "depth": depth,
        "mag": mag if mag is not None else 0.0,
        "quality": quality,
        "nph": nph,
        "evtid": evtid,
    }


def _looks_int_token(token: str, max_len: int) -> bool:
    return token.isdigit() and len(token) <= max_len


def _looks_float_token(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _find_time_index(tokens: List[str]) -> Optional[int]:
    # Look for YY MM DD HHMM SS.ss pattern after the station code.
    for i in range(1, len(tokens) - 4):
        if (
            _looks_int_token(tokens[i], 2)
            and _looks_int_token(tokens[i + 1], 2)
            and _looks_int_token(tokens[i + 2], 2)
            and _looks_int_token(tokens[i + 3], 4)
            and _looks_float_token(tokens[i + 4])
        ):
            return i
    return None


def parse_cwb_station_line_fixed(line: str) -> Optional[Dict[str, object]]:
    if len(line) < 36:
        return None

    sta = line[0:4].strip()
    if not sta:
        return None

    p_phase = line[5].strip()
    p_wei = line[7].strip()
    p_sec = _safe_float(line[19:25], None)
    s_sec = _safe_float(line[26:32], None)
    s_phase = line[33].strip()
    s_wei = line[35].strip()

    return {
        "sta": sta,
        "p_phase": p_phase,
        "p_wei": p_wei,
        "p_sec": p_sec,
        "s_phase": s_phase,
        "s_wei": s_wei,
        "s_sec": s_sec,
    }


def parse_cwb_station_line(line: str) -> Optional[Dict[str, object]]:
    fixed = parse_cwb_station_line_fixed(line)
    if fixed is not None:
        return fixed

    tokens = line.split()
    if len(tokens) < 6:
        return None

    sta = tokens[0]
    time_idx = _find_time_index(tokens)
    if time_idx is None:
        return None

    p_phase = ""
    p_wei = ""
    if time_idx >= 3:
        candidate_phase = tokens[time_idx - 2]
        if len(candidate_phase) == 1 and candidate_phase.isalpha():
            p_phase = candidate_phase
            p_wei = tokens[time_idx - 1]
        else:
            p_wei = tokens[time_idx - 1]
    elif time_idx == 2:
        p_phase = tokens[time_idx - 1]

    yy, mm, dd, hhmm, p_sec_token = tokens[time_idx : time_idx + 5]
    remaining = tokens[time_idx + 5 :]

    s_sec_token = None
    s_phase = ""
    s_wei = ""

    if remaining and remaining[0].isalpha():
        # No P seconds were present; the parsed p_sec is actually S seconds.
        s_sec_token = p_sec_token
        p_sec_token = None
        s_phase = remaining[0]
        if len(remaining) > 1:
            s_wei = remaining[1]
    elif len(remaining) >= 2 and remaining[1].isalpha():
        s_sec_token = remaining[0]
        s_phase = remaining[1]
        if len(remaining) > 2:
            s_wei = remaining[2]

    return {
        "sta": sta,
        "p_phase": p_phase,
        "p_wei": p_wei,
        "p_sec": _safe_float(p_sec_token, None) if p_sec_token else None,
        "s_phase": s_phase,
        "s_wei": s_wei,
        "s_sec": _safe_float(s_sec_token, None) if s_sec_token else None,
    }


def weight_from_cwb(code: str, mode: str) -> float:
    if mode == "unity":
        return 1.0

    code = code.strip()
    if not code.isdigit():
        return 1.0

    val = int(code)
    return max(0.0, 1.0 - 0.25 * val)


def _skip_weight(code: str, max_code: int | None) -> bool:
    if max_code is None:
        return False
    code = code.strip()
    if not code.isdigit():
        return False
    return int(code) > max_code


def resolve_station(
    sta: str,
    station_map: Dict[str, Tuple[float, float, float]],
    allow_prefix: bool,
) -> Optional[Tuple[str, Tuple[float, float, float]]]:
    key = sta.strip()
    if key in station_map:
        return key, station_map[key]

    if allow_prefix and len(key) == 3:
        matches = [name for name in station_map if name.startswith(key)]
        if len(matches) == 1:
            match = matches[0]
            return match, station_map[match]

    return None


def iter_cwb_blocks(path: Path) -> Iterable[Tuple[str, List[str]]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        _ = handle.readline()  # skip count line
        header = None
        stations: List[str] = []

        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.strip() == "**":
                if header:
                    yield header, stations
                header = None
                stations = []
                continue
            if line[0].isdigit():
                if header:
                    yield header, stations
                header = line
                stations = []
            else:
                if header:
                    stations.append(line)

        if header:
            yield header, stations


def write_src_rec(
    cwb_path: Path,
    station_map: Dict[str, Tuple[float, float, float]],
    out_path: Path,
    year_cutoff: int,
    weight_mode: str,
    event_weight: float,
    allow_prefix: bool,
    max_weight_code: int | None,
    missing_stations_out: Path | None,
    missing_stations_top: int,
) -> None:
    station_ids: Dict[str, int] = {}
    next_station_id = 0
    missing_stations: Dict[str, int] = {}

    ev_id = 0
    total_picks = 0
    total_events = 0

    with out_path.open("w", encoding="utf-8") as out:
        for header_line, station_lines in iter_cwb_blocks(cwb_path):
            header = parse_cwb_header(header_line, year_cutoff)
            picks = []

            origin_sec = float(header["second"])
            for station_line in station_lines:
                parsed = parse_cwb_station_line(station_line)
                if not parsed:
                    continue

                sta = parsed["sta"]
                resolved = resolve_station(sta, station_map, allow_prefix)
                if not resolved:
                    missing_stations[sta] = missing_stations.get(sta, 0) + 1
                    continue

                sta_key, (lat, lon, ele) = resolved
                if sta_key not in station_ids:
                    station_ids[sta_key] = next_station_id
                    next_station_id += 1
                stid = station_ids[sta_key]

                if parsed["p_phase"].upper() == "P" and parsed["p_sec"] is not None:
                    tt = float(parsed["p_sec"]) - origin_sec
                    if tt >= 0:
                        if _skip_weight(parsed["p_wei"], max_weight_code):
                            continue
                        picks.append(
                            ("P", tt, stid, sta_key, lat, lon, ele, parsed["p_wei"])
                        )

                if parsed["s_phase"].upper() == "S" and parsed["s_sec"] is not None:
                    tt = float(parsed["s_sec"]) - origin_sec
                    if tt >= 0:
                        if _skip_weight(parsed["s_wei"], max_weight_code):
                            continue
                        picks.append(
                            ("S", tt, stid, sta_key, lat, lon, ele, parsed["s_wei"])
                        )

            if not picks:
                continue

            ndata = len(picks)
            ev_name = header["evtid"] or f"EV{ev_id:06d}"

            out.write(
                "%7d %6d %2d %2d %2d %2d %5.2f %9.4f %9.4f %9.4f %5.2f %7d %s %7.3f\n"
                % (
                    ev_id,
                    header["year"],
                    header["month"],
                    header["day"],
                    header["hour"],
                    header["minute"],
                    header["second"],
                    header["lat"],
                    header["lon"],
                    header["depth"],
                    header["mag"],
                    ndata,
                    ev_name,
                    event_weight,
                )
            )

            for phase, tt, stid, sta_key, lat, lon, ele, wt in picks:
                weight = weight_from_cwb(wt, weight_mode)
                out.write(
                    "%7d %7d %6s %9.4f %9.4f %9.4f %s %8.4f %7.3f\n"
                    % (ev_id, stid, sta_key, lat, lon, ele, phase, tt, weight)
                )

            ev_id += 1
            total_events += 1
            total_picks += ndata

    print(f"Wrote {total_events} events and {total_picks} picks to {out_path}")
    if missing_stations:
        missing_count = sum(missing_stations.values())
        unique = len(missing_stations)
        print(f"Skipped {missing_count} picks from {unique} unmatched stations.")
        if missing_stations_top > 0:
            top = sorted(missing_stations.items(), key=lambda item: item[1], reverse=True)[:missing_stations_top]
            print("Top unmatched stations:")
            for name, count in top:
                print(f"  {name}: {count}")
        if missing_stations_out is not None:
            missing_stations_out.parent.mkdir(parents=True, exist_ok=True)
            with missing_stations_out.open("w", encoding="utf-8") as handle:
                handle.write("station,count\n")
                for name, count in sorted(
                    missing_stations.items(), key=lambda item: item[1], reverse=True
                ):
                    handle.write(f"{name},{count}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CWB .cwb file to TomoATT src_rec.dat format."
    )
    parser.add_argument(
        "--cwb",
        type=Path,
        default=Path(
            "CWBcatalog/eqsearch_PFILE_t20210101-20220831+20230101-20231231.salute.cwb"
        ),
        help="Input CWB catalog file.",
    )
    parser.add_argument(
        "--sta",
        action="append",
        type=Path,
        help="Station file (.sta fixed-width or station CSV). Repeatable.",
    )
    parser.add_argument(
        "--sta-dir",
        type=Path,
        help="Directory containing .sta or station CSV files to load (non-recursive).",
    )
    parser.add_argument(
        "--sta-priority",
        choices=["first", "last"],
        default="first",
        help="When station codes repeat across files, keep the first or last match.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src_rec.dat"),
        help="Output src_rec.dat file.",
    )
    parser.add_argument(
        "--year-cutoff",
        type=int,
        default=70,
        help="Two-digit years below this map to 2000+, otherwise 1900+.",
    )
    parser.add_argument(
        "--weight-mode",
        choices=["unity", "cwb"],
        default="cwb",
        help="Pick weight mapping mode.",
    )
    parser.add_argument(
        "--event-weight",
        type=float,
        default=1.0,
        help="Event-level weight written to src_rec.dat.",
    )
    parser.add_argument(
        "--max-weight-code",
        type=int,
        help="Skip picks with CWB weight code greater than this value.",
    )
    parser.add_argument(
        "--missing-stations-out",
        type=Path,
        help="Write unmatched station counts to a CSV file.",
    )
    parser.add_argument(
        "--missing-stations-top",
        type=int,
        default=0,
        help="Print the top N unmatched station codes to stdout.",
    )
    parser.add_argument(
        "--allow-prefix",
        action="store_true",
        help="Allow 3-letter station codes to match a unique 4-letter prefix in the .sta file.",
    )

    args = parser.parse_args()

    sta_paths = list(args.sta) if args.sta else []
    if args.sta_dir:
        sta_paths.extend(sorted(args.sta_dir.glob("*.sta")))
        sta_paths.extend(sorted(args.sta_dir.glob("stations*.csv")))
    if not sta_paths:
        sta_paths = [Path("CWBcatalog/hypo3d.sta")]
    station_map = parse_station_files(sta_paths, args.sta_priority)
    if not station_map:
        sta_list = ", ".join(str(p) for p in sta_paths)
        raise SystemExit(f"No stations parsed from {sta_list}")

    write_src_rec(
        args.cwb,
        station_map,
        args.out,
        args.year_cutoff,
        args.weight_mode,
        args.event_weight,
        args.allow_prefix,
        args.max_weight_code,
        args.missing_stations_out,
        args.missing_stations_top,
    )


if __name__ == "__main__":
    main()
