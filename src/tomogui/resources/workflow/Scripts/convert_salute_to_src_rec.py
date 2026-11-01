#!/usr/bin/env python3
"""
Convert SALUTEv5 event/station files to TomoATT src_rec.dat.

Primary inputs:
  - simulp_evt.dat  (event origin time + per-pick travel times)
  - simulp_sta.dat  (station metadata; default)

Optional station alternative:
  - stat_ft.dat (auto-detected format, converted to src_rec station elevation units)

This converter intentionally does NOT use rays.dat.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PICK_RE = re.compile(r"([A-Za-z0-9.]{3,8})\s*([PS])\s*([0-9])\s*([+-]?\d+(?:\.\d+)?)")


@dataclass
class Station:
    station_id: int
    name: str
    lat: float
    lon: float
    stel_m: float


@dataclass
class EventMeta:
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: float
    lat: float
    lon: float
    depth_km: float


@dataclass
class Pick:
    sta_raw: str
    phase: str
    weight_code: int
    tt: float


@dataclass
class HypoStationParse:
    sta_raw: str
    p_phase: str
    p_weight_code: Optional[int]
    p_sec: Optional[float]
    s_phase: str
    s_weight_code: Optional[int]
    s_sec: Optional[float]


def _year_from_two_digits(year2: int, cutoff: int) -> int:
    return 2000 + year2 if year2 < cutoff else 1900 + year2


def _parse_latlon_token(token: str) -> float:
    token = token.strip()
    if not token:
        raise ValueError("Empty coordinate token")
    hemi = token[-1].upper()
    if hemi in {"N", "S", "E", "W"}:
        value = float(token[:-1])
        if hemi in {"S", "W"}:
            value = -value
        return value
    return float(token)


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _is_decimal_float(text: str) -> bool:
    t = text.strip()
    if "." not in t:
        return False
    return _is_float(t)


def _is_latlon_token(text: str, kind: str) -> bool:
    t = text.strip().upper()
    if kind == "lat":
        if not (t.endswith("N") or t.endswith("S")):
            return False
    elif kind == "lon":
        if not (t.endswith("E") or t.endswith("W")):
            return False
    else:
        return False
    return _is_float(t[:-1])


def _norm_sta(name: str) -> str:
    return name.strip().upper()


def _base_sta(name: str) -> str:
    return _norm_sta(name)[:4]


def parse_station_file(path: Path) -> Dict[str, Station]:
    """
    Parse either:
      1) simulp_sta.dat: STA LON LAT ELEV_M
      2) stat_ft.dat   : LON LAT DEPTH_KM STA
    """
    station_map: Dict[str, Station] = {}
    station_id = 0

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue

            if _is_float(parts[0]) and _is_float(parts[1]):
                # stat_ft.dat style: lon lat depth_km sta
                lon = float(parts[0])
                lat = float(parts[1])
                depth_km = float(parts[2])
                name = parts[3]
                # stat_ft depth is opposite sign to src_rec station elevation convention.
                stel_m = -depth_km * 1000.0
            else:
                # simulp_sta.dat style: sta lon lat elev_m
                name = parts[0]
                lon = float(parts[1])
                lat = float(parts[2])
                stel_m = float(parts[3])

            key = _norm_sta(name)
            if key in station_map:
                continue
            station_map[key] = Station(
                station_id=station_id,
                name=key,
                lat=lat,
                lon=lon,
                stel_m=stel_m,
            )
            station_id += 1

    if not station_map:
        raise RuntimeError(f"No stations parsed from {path}")
    return station_map


def build_station_indices(stations: Dict[str, Station]) -> Tuple[Dict[str, Station], Dict[str, List[Station]]]:
    by_exact = dict(stations)
    by_base: Dict[str, List[Station]] = {}
    for sta in stations.values():
        by_base.setdefault(_base_sta(sta.name), []).append(sta)
    return by_exact, by_base


def resolve_station(sta_raw: str, by_exact: Dict[str, Station], by_base: Dict[str, List[Station]]) -> Optional[Station]:
    sta = _norm_sta(sta_raw)
    if sta in by_exact:
        return by_exact[sta]

    base = _base_sta(sta)
    candidates = by_base.get(base, [])
    if len(candidates) == 1:
        return candidates[0]

    # When short codes map to both .00 and .10 variants, prefer .00 then .10.
    if len(candidates) > 1:
        for suffix in (".00", ".10"):
            target = f"{base}{suffix}"
            for cand in candidates:
                if cand.name == target:
                    return cand

    return None


def _int_or_none(text: str) -> Optional[int]:
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return None


def _float_or_none(text: str) -> Optional[float]:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_hypo3d_line(line: str) -> str:
    # Expand compact tokens like TAWH.00P1 or P0/S2 into tokenized form.
    line = re.sub(r"([A-Za-z0-9.]{3,8})([PS])([0-9])", r"\1 \2 \3", line)
    line = re.sub(r"\b([PS])([0-9])\b", r"\1 \2", line)
    return line


def parse_hypo3d_station_line(line: str) -> Optional[HypoStationParse]:
    tokens = _normalize_hypo3d_line(line).split()
    if len(tokens) < 6:
        return None

    sta = tokens[0]
    p_idx: Optional[int] = None
    for i in range(1, len(tokens) - 1):
        if tokens[i] == "P" and tokens[i + 1].isdigit():
            p_idx = i
            break
    if p_idx is None:
        return None

    p_phase = "P"
    p_wei = _int_or_none(tokens[p_idx + 1])

    p_sec: Optional[float] = None
    p_sec_idx: Optional[int] = None
    for i in range(p_idx + 2, len(tokens)):
        if _is_decimal_float(tokens[i]):
            p_sec = float(tokens[i])
            p_sec_idx = i
            break

    s_phase = ""
    s_wei: Optional[int] = None
    s_sec: Optional[float] = None

    if p_sec_idx is not None:
        s_idx: Optional[int] = None
        for i in range(p_sec_idx + 1, len(tokens) - 1):
            if tokens[i] == "S" and tokens[i + 1].isdigit():
                s_idx = i
                s_wei = _int_or_none(tokens[i + 1])
                break
        if s_idx is not None:
            floats_before_s = [float(tokens[i]) for i in range(p_sec_idx + 1, s_idx) if _is_decimal_float(tokens[i])]
            if floats_before_s:
                s_sec = floats_before_s[-1]
                s_phase = "S"

    return HypoStationParse(
        sta_raw=sta,
        p_phase=p_phase,
        p_weight_code=p_wei,
        p_sec=p_sec,
        s_phase=s_phase,
        s_weight_code=s_wei,
        s_sec=s_sec,
    )


def _travel_time_from_seconds(
    phase_sec: Optional[float],
    origin_sec: float,
) -> Optional[float]:
    if phase_sec is None:
        return None
    tt = phase_sec - origin_sec
    # In hypo3d, phase seconds can exceed 60 for minute rollover; if still
    # negative, roll forward by one or more minutes.
    while tt < -1e-6:
        tt += 60.0
    return tt


def parse_event_header(line: str, year_cutoff: int) -> Optional[EventMeta]:
    parts = line.split()
    if len(parts) < 8:
        return None
    if not parts[0].isdigit() or not parts[1].isdigit() or not parts[2].isdigit():
        return None

    year2 = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])

    # Format A: YY M D HHMM SS.ss LATN LONE DEPTH
    # Format B: YY M D H M SS.ss LATN LONE DEPTH
    if (
        len(parts) >= 9
        and parts[3].isdigit()
        and parts[4].isdigit()
        and _is_float(parts[5])
        and _is_latlon_token(parts[6], "lat")
        and _is_latlon_token(parts[7], "lon")
    ):
        hour = int(parts[3])
        minute = int(parts[4])
        second = float(parts[5])
        lat = _parse_latlon_token(parts[6])
        lon = _parse_latlon_token(parts[7])
        depth_km = float(parts[8])
        return EventMeta(
            year=_year_from_two_digits(year2, year_cutoff),
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            lat=lat,
            lon=lon,
            depth_km=depth_km,
        )

    if (
        len(parts) >= 8
        and re.fullmatch(r"\d{3,4}", parts[3] or "")
        and _is_float(parts[4])
        and _is_latlon_token(parts[5], "lat")
        and _is_latlon_token(parts[6], "lon")
    ):
        hhmm = parts[3].zfill(4)
        hour = int(hhmm[:-2])
        minute = int(hhmm[-2:])
        second = float(parts[4])
        lat = _parse_latlon_token(parts[5])
        lon = _parse_latlon_token(parts[6])
        depth_km = float(parts[7])
        return EventMeta(
            year=_year_from_two_digits(year2, year_cutoff),
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            lat=lat,
            lon=lon,
            depth_km=depth_km,
        )

    return None


def parse_pick_line(line: str) -> List[Pick]:
    picks: List[Pick] = []
    for match in PICK_RE.finditer(line):
        sta = match.group(1)
        phase = match.group(2).upper()
        weight_code = int(match.group(3))
        tt = float(match.group(4))
        picks.append(Pick(sta_raw=sta, phase=phase, weight_code=weight_code, tt=tt))
    return picks


def iter_events(evt_path: Path, year_cutoff: int) -> Iterable[Tuple[EventMeta, List[Pick]]]:
    current_meta: Optional[EventMeta] = None
    current_picks: List[Pick] = []

    with evt_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip():
                if current_meta is not None:
                    yield current_meta, current_picks
                    current_meta = None
                    current_picks = []
                continue

            header = parse_event_header(line, year_cutoff)
            if header is not None:
                if current_meta is not None:
                    yield current_meta, current_picks
                current_meta = header
                current_picks = []
                continue

            if current_meta is None:
                continue
            current_picks.extend(parse_pick_line(line))

    if current_meta is not None:
        yield current_meta, current_picks


def pick_weight(weight_code: int, mode: str) -> float:
    if mode == "unity":
        return 1.0
    # CWB-like code mapping: 0->1.0, 1->0.8, ..., 4->0.2
    return max(0.2, 1.0 - 0.2 * float(weight_code))


def event_name(meta: EventMeta, idx: int) -> str:
    sec_int = int(meta.second)
    return f"SALUTE_{meta.year:04d}{meta.month:02d}{meta.day:02d}{meta.hour:02d}{meta.minute:02d}{sec_int:02d}_{idx:06d}"


def write_src_rec(
    evt_path: Path,
    stations: Dict[str, Station],
    out_path: Path,
    phases: set[str],
    year_cutoff: int,
    weight_mode: str,
    max_weight_code: Optional[int],
    event_weight: float,
) -> None:
    by_exact, by_base = build_station_indices(stations)

    total_events = 0
    written_events = 0
    total_picks = 0
    written_picks = 0
    written_p = 0
    written_s = 0
    skipped_negative_tt = 0
    skipped_weight = 0
    skipped_bad_phase = 0
    unmatched = Counter()

    with out_path.open("w", encoding="utf-8") as out:
        for meta, picks in iter_events(evt_path, year_cutoff):
            total_events += 1
            total_picks += len(picks)

            rows: List[Tuple[Station, Pick, float]] = []
            for pick in picks:
                if pick.phase not in phases:
                    skipped_bad_phase += 1
                    continue
                if pick.tt < 0:
                    skipped_negative_tt += 1
                    continue
                if max_weight_code is not None and pick.weight_code > max_weight_code:
                    skipped_weight += 1
                    continue

                station = resolve_station(pick.sta_raw, by_exact, by_base)
                if station is None:
                    unmatched[_norm_sta(pick.sta_raw)] += 1
                    continue

                wt = pick_weight(pick.weight_code, weight_mode)
                rows.append((station, pick, wt))

            if not rows:
                continue

            ev_idx = written_events
            out.write(
                "%7d %6d %2d %2d %2d %2d %5.2f %9.4f %9.4f %9.4f %5.2f %7d %s %7.3f\n"
                % (
                    ev_idx,
                    meta.year,
                    meta.month,
                    meta.day,
                    meta.hour,
                    meta.minute,
                    meta.second,
                    meta.lat,
                    meta.lon,
                    meta.depth_km,
                    0.0,
                    len(rows),
                    event_name(meta, ev_idx),
                    event_weight,
                )
            )

            for station, pick, wt in rows:
                out.write(
                    "%7d %7d %6s %9.4f %9.4f %9.4f %s %8.4f %7.3f\n"
                    % (
                        ev_idx,
                        station.station_id,
                        station.name,
                        station.lat,
                        station.lon,
                        station.stel_m,
                        pick.phase,
                        pick.tt,
                        wt,
                    )
                )
                written_picks += 1
                if pick.phase == "P":
                    written_p += 1
                elif pick.phase == "S":
                    written_s += 1

            written_events += 1

    print(f"Wrote {written_events} events and {written_picks} picks to {out_path}")
    print(f"P picks: {written_p}, S picks: {written_s}")
    print(f"Input events: {total_events}, input picks parsed: {total_picks}")
    if skipped_bad_phase:
        print(f"Skipped by phase filter: {skipped_bad_phase}")
    if skipped_weight:
        print(f"Skipped by max weight code: {skipped_weight}")
    if skipped_negative_tt:
        print(f"Skipped negative travel times: {skipped_negative_tt}")
    if unmatched:
        skipped_unmatched = sum(unmatched.values())
        print(f"Skipped unmatched stations: {skipped_unmatched} picks from {len(unmatched)} station codes")
        print("Top unmatched station codes:")
        for sta, cnt in unmatched.most_common(20):
            print(f"  {sta}: {cnt}")


def write_src_rec_from_hypo3d(
    hypo_path: Path,
    stations: Dict[str, Station],
    out_path: Path,
    phases: set[str],
    year_cutoff: int,
    weight_mode: str,
    max_weight_code: Optional[int],
    event_weight: float,
) -> None:
    # Reuse robust CWB block/header parser from existing converter.
    from convert_cwb_to_src_rec import iter_cwb_blocks, parse_cwb_header

    by_exact, by_base = build_station_indices(stations)

    total_events = 0
    written_events = 0
    total_picks = 0
    written_picks = 0
    written_p = 0
    written_s = 0
    skipped_negative_tt = 0
    skipped_weight = 0
    skipped_bad_phase = 0
    unmatched = Counter()

    with out_path.open("w", encoding="utf-8") as out:
        for header_line, station_lines in iter_cwb_blocks(hypo_path):
            total_events += 1
            header = parse_cwb_header(header_line, year_cutoff)
            origin_sec = float(header["second"])

            rows: List[Tuple[Station, str, float, float]] = []
            for station_line in station_lines:
                parsed = parse_hypo3d_station_line(station_line)
                if parsed is None:
                    continue

                station = resolve_station(parsed.sta_raw, by_exact, by_base)
                if station is None:
                    unmatched[_norm_sta(parsed.sta_raw)] += 1
                    continue

                # P pick
                if parsed.p_phase:
                    total_picks += 1
                    if parsed.p_phase not in phases:
                        skipped_bad_phase += 1
                    else:
                        if max_weight_code is not None and parsed.p_weight_code is not None and parsed.p_weight_code > max_weight_code:
                            skipped_weight += 1
                        else:
                            tt = _travel_time_from_seconds(parsed.p_sec, origin_sec)
                            if tt is None or tt < 0:
                                skipped_negative_tt += 1
                            else:
                                wt = pick_weight(parsed.p_weight_code or 0, weight_mode)
                                rows.append((station, "P", tt, wt))

                # S pick
                if parsed.s_phase:
                    total_picks += 1
                    if parsed.s_phase not in phases:
                        skipped_bad_phase += 1
                    else:
                        if max_weight_code is not None and parsed.s_weight_code is not None and parsed.s_weight_code > max_weight_code:
                            skipped_weight += 1
                        else:
                            tt = _travel_time_from_seconds(parsed.s_sec, origin_sec)
                            if tt is None or tt < 0:
                                skipped_negative_tt += 1
                            else:
                                wt = pick_weight(parsed.s_weight_code or 0, weight_mode)
                                rows.append((station, "S", tt, wt))

            if not rows:
                continue

            ev_idx = written_events
            ev_name = str(header.get("evtid") or event_name(
                EventMeta(
                    year=int(header["year"]),
                    month=int(header["month"]),
                    day=int(header["day"]),
                    hour=int(header["hour"]),
                    minute=int(header["minute"]),
                    second=float(header["second"]),
                    lat=float(header["lat"]),
                    lon=float(header["lon"]),
                    depth_km=float(header["depth"]),
                ),
                ev_idx,
            ))
            out.write(
                "%7d %6d %2d %2d %2d %2d %5.2f %9.4f %9.4f %9.4f %5.2f %7d %s %7.3f\n"
                % (
                    ev_idx,
                    int(header["year"]),
                    int(header["month"]),
                    int(header["day"]),
                    int(header["hour"]),
                    int(header["minute"]),
                    float(header["second"]),
                    float(header["lat"]),
                    float(header["lon"]),
                    float(header["depth"]),
                    float(header["mag"]),
                    len(rows),
                    ev_name,
                    event_weight,
                )
            )

            for station, phase, tt, wt in rows:
                out.write(
                    "%7d %7d %6s %9.4f %9.4f %9.4f %s %8.4f %7.3f\n"
                    % (
                        ev_idx,
                        station.station_id,
                        station.name,
                        station.lat,
                        station.lon,
                        station.stel_m,
                        phase,
                        tt,
                        wt,
                    )
                )
                written_picks += 1
                if phase == "P":
                    written_p += 1
                elif phase == "S":
                    written_s += 1

            written_events += 1

    print(f"Wrote {written_events} events and {written_picks} picks to {out_path}")
    print(f"P picks: {written_p}, S picks: {written_s}")
    print(f"Input events: {total_events}, input picks parsed: {total_picks}")
    if skipped_bad_phase:
        print(f"Skipped by phase filter: {skipped_bad_phase}")
    if skipped_weight:
        print(f"Skipped by max weight code: {skipped_weight}")
    if skipped_negative_tt:
        print(f"Skipped negative travel times: {skipped_negative_tt}")
    if unmatched:
        skipped_unmatched = sum(unmatched.values())
        print(f"Skipped unmatched stations: {skipped_unmatched} picks from {len(unmatched)} station codes")
        print("Top unmatched station codes:")
        for sta, cnt in unmatched.most_common(20):
            print(f"  {sta}: {cnt}")


def parse_phases(value: str) -> set[str]:
    phases = {v.strip().upper() for v in value.split(",") if v.strip()}
    if not phases:
        raise argparse.ArgumentTypeError("At least one phase must be provided.")
    bad = phases - {"P", "S"}
    if bad:
        raise argparse.ArgumentTypeError(f"Unsupported phases: {sorted(bad)}")
    return phases


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SALUTE files to src_rec.dat (simulp_evt or hypo3d modes).")
    parser.add_argument(
        "--input-format",
        choices=["simulp_evt", "hypo3d"],
        default="simulp_evt",
        help="Input parser mode: simulp_evt (default) or hypo3d.",
    )
    parser.add_argument("--events", type=Path, default=Path("SALUTEv5/simulp_evt.dat"), help="simulp_evt.dat path")
    parser.add_argument("--hypo3d", type=Path, default=Path("SALUTEv5/hypo3d_w3.cwb"), help="hypo3d_w3.cwb path")
    parser.add_argument(
        "--stations",
        type=Path,
        default=Path("SALUTEv5/simulp_sta.dat"),
        help="Station file path (simulp_sta.dat or stat_ft.dat)",
    )
    parser.add_argument("--out", type=Path, default=Path("src_rec_SALUTEv5.dat"), help="Output src_rec.dat path")
    parser.add_argument("--phases", type=parse_phases, default=parse_phases("P,S"), help="Comma-separated phases")
    parser.add_argument("--year-cutoff", type=int, default=70, help="YY < cutoff => 20YY, else 19YY")
    parser.add_argument("--weight-mode", choices=["cwb", "unity"], default="cwb", help="Weight mapping mode")
    parser.add_argument("--max-weight-code", type=int, default=None, help="Skip picks with code > this value")
    parser.add_argument("--event-weight", type=float, default=1.0, help="Event-level weight in src_rec")
    args = parser.parse_args()

    if args.input_format == "simulp_evt" and not args.events.exists():
        raise SystemExit(f"Events file not found: {args.events}")
    if args.input_format == "hypo3d" and not args.hypo3d.exists():
        raise SystemExit(f"Hypo3d file not found: {args.hypo3d}")
    if not args.stations.exists():
        raise SystemExit(f"Stations file not found: {args.stations}")

    stations = parse_station_file(args.stations)
    if args.input_format == "simulp_evt":
        write_src_rec(
            evt_path=args.events,
            stations=stations,
            out_path=args.out,
            phases=args.phases,
            year_cutoff=args.year_cutoff,
            weight_mode=args.weight_mode,
            max_weight_code=args.max_weight_code,
            event_weight=args.event_weight,
        )
    else:
        write_src_rec_from_hypo3d(
            hypo_path=args.hypo3d,
            stations=stations,
            out_path=args.out,
            phases=args.phases,
            year_cutoff=args.year_cutoff,
            weight_mode=args.weight_mode,
            max_weight_code=args.max_weight_code,
            event_weight=args.event_weight,
        )


if __name__ == "__main__":
    main()
