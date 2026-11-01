#!/usr/bin/env python3
"""Build custom TOMOATT HDF5 models from an ASCII spec file.

Spec format:
  - Global settings: key=value (one per line)
  - Shape operations: shape <type> key=value key=value ...
  - Empty lines and lines starting with '#' are ignored.

Supported shapes:
  - box
  - layer
  - sphere
  - cylinder (vertical)
  - polygon (lon/lat polygon + depth range)

Horizontal coordinates can be given either as absolute lon/lat values
(`lon_min`, `lat0`, `vertices`, etc.) or as fractions of the model domain
(`rel_lon_min`, `rel_lat0`, `rel_vertices`, etc.). Relative values must be in
the range [0, 1], where 0 is the west/south model edge and 1 is the east/north
model edge.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from pytomoatt.model import ATTModel


DEG_TO_KM = 111.32
EPS = 1e-12


def _load_attmodel():
    try:
        from pytomoatt.model import ATTModel
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "ERROR: pytomoatt is required. Install PyTomoATT into the active "
            "environment so `import pytomoatt` works."
        ) from exc
    return ATTModel


def _to_float(params: dict[str, str], key: str, default: float | None = None) -> float:
    if key not in params:
        if default is None:
            raise ValueError(f"Missing required key '{key}'.")
        return float(default)
    return float(params[key])


def _parse_keyvals(tokens: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"Invalid token '{tok}'; expected key=value.")
        k, v = tok.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if not k:
            raise ValueError(f"Invalid token '{tok}'; empty key.")
        out[k] = v
    return out


def _resolve_axis_value(
    params: dict[str, str],
    abs_key: str,
    rel_key: str,
    vmin: float,
    vmax: float,
) -> float:
    if abs_key in params and rel_key in params:
        raise ValueError(f"Specify only one of '{abs_key}' or '{rel_key}'.")
    if abs_key in params:
        return float(params[abs_key])
    if rel_key in params:
        rel = float(params[rel_key])
        if rel < 0.0 or rel > 1.0:
            raise ValueError(f"{rel_key} must be within [0, 1]; got {rel}.")
        return vmin + rel * (vmax - vmin)
    raise ValueError(f"Missing required key '{abs_key}' or '{rel_key}'.")


def _parse_vertices(vertices_raw: str) -> list[tuple[float, float]]:
    vertices: list[tuple[float, float]] = []
    for pair in vertices_raw.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_s, y_s = pair.split(":", 1)
        vertices.append((float(x_s), float(y_s)))
    if len(vertices) < 3:
        raise ValueError("polygon needs at least 3 vertices.")
    return vertices


def _parse_spec(path: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    globals_cfg: dict[str, str] = {}
    shapes: list[dict[str, object]] = []

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("shape "):
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{lineno}: shape line missing shape type.")
            shape_type = parts[1].lower()
            params = _parse_keyvals(parts[2:])
            shapes.append({"type": shape_type, "params": params, "lineno": lineno})
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{lineno}: expected key=value or shape line.")
        k, v = line.split("=", 1)
        globals_cfg[k.strip().lower()] = v.strip()

    return globals_cfg, shapes


def _apply_background(model: ATTModel, cfg: dict[str, str], spec_dir: Path) -> None:
    bg = cfg.get("background", "linear").strip().lower()

    if bg == "from_h5":
        src = Path(cfg["background_model"]).expanduser()
        if not src.is_absolute():
            src = (spec_dir / src).resolve()
        if not src.exists():
            raise FileNotFoundError(f"background_model not found: {src}")
        with h5py.File(src, "r") as f:
            for field in ("vel", "xi", "eta"):
                if field not in f:
                    raise ValueError(f"background_model missing dataset '{field}'.")
                data = np.asarray(f[field])
                if data.shape != model.vel.shape:
                    raise ValueError(
                        f"background_model '{field}' shape {data.shape} "
                        f"!= model grid shape {model.vel.shape}"
                    )
                getattr(model, field)[:, :, :] = data
        return

    if bg == "constant":
        vel = float(cfg.get("constant_vel", cfg.get("vel_min", "6.0")))
        model.vel[:, :, :] = vel
    elif bg == "linear":
        vel_min = float(cfg.get("vel_min", "5.0"))
        vel_max = float(cfg.get("vel_max", "8.0"))
        top_km = float(cfg.get("linear_top_km", "40.0"))
        d = model.depths
        model.vel[d < 0.0, :, :] = vel_min
        inside = np.where((d >= 0.0) & (d < top_km))[0]
        if inside.size > 0:
            vals = np.linspace(vel_min, vel_max, inside.size)[::-1]
            model.vel[inside, :, :] = vals[:, np.newaxis, np.newaxis]
        model.vel[d >= top_km, :, :] = vel_max
    else:
        raise ValueError("background must be one of: linear, constant, from_h5")

    model.xi[:, :, :] = 0.0
    model.eta[:, :, :] = 0.0

    if "default_eps" in cfg:
        eps = float(cfg["default_eps"])
        phi_deg = float(cfg.get("default_phi_deg", "0.0"))
        phi = math.radians(phi_deg)
        xi = eps * math.cos(2.0 * phi)
        eta = eps * math.sin(2.0 * phi)
        model.xi[:, :, :] = xi
        model.eta[:, :, :] = eta


def _polygon_mask(lon2d: np.ndarray, lat2d: np.ndarray, vertices: list[tuple[float, float]]) -> np.ndarray:
    x = lon2d.ravel()
    y = lat2d.ravel()
    inside = np.zeros(x.shape[0], dtype=bool)

    xv = np.array([v[0] for v in vertices], dtype=float)
    yv = np.array([v[1] for v in vertices], dtype=float)
    n = len(vertices)
    j = n - 1
    for i in range(n):
        yi = yv[i]
        yj = yv[j]
        xi = xv[i]
        xj = xv[j]
        cond = (yi > y) != (yj > y)
        x_cross = (xj - xi) * (y - yi) / ((yj - yi) + EPS) + xi
        inside ^= cond & (x < x_cross)
        j = i
    return inside.reshape(lon2d.shape)


def _shape_mask(
    shape_type: str,
    params: dict[str, str],
    dep3: np.ndarray,
    lat3: np.ndarray,
    lon3: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    s = shape_type.lower()
    lon_domain_min = float(np.min(lon2))
    lon_domain_max = float(np.max(lon2))
    lat_domain_min = float(np.min(lat2))
    lat_domain_max = float(np.max(lat2))

    if s == "box":
        lon_min = _resolve_axis_value(
            params, "lon_min", "rel_lon_min", lon_domain_min, lon_domain_max
        )
        lon_max = _resolve_axis_value(
            params, "lon_max", "rel_lon_max", lon_domain_min, lon_domain_max
        )
        lat_min = _resolve_axis_value(
            params, "lat_min", "rel_lat_min", lat_domain_min, lat_domain_max
        )
        lat_max = _resolve_axis_value(
            params, "lat_max", "rel_lat_max", lat_domain_min, lat_domain_max
        )
        dep_min = _to_float(params, "dep_min")
        dep_max = _to_float(params, "dep_max")
        return (
            (lon3 >= min(lon_min, lon_max))
            & (lon3 <= max(lon_min, lon_max))
            & (lat3 >= min(lat_min, lat_max))
            & (lat3 <= max(lat_min, lat_max))
            & (dep3 >= min(dep_min, dep_max))
            & (dep3 <= max(dep_min, dep_max))
        )

    if s == "layer":
        dep_min = _to_float(params, "dep_min")
        dep_max = _to_float(params, "dep_max")
        mask = (dep3 >= min(dep_min, dep_max)) & (dep3 <= max(dep_min, dep_max))
        if any(k in params for k in ("lat_min", "lat_max", "rel_lat_min", "rel_lat_max")):
            lat_min = _resolve_axis_value(
                params, "lat_min", "rel_lat_min", lat_domain_min, lat_domain_max
            )
            lat_max = _resolve_axis_value(
                params, "lat_max", "rel_lat_max", lat_domain_min, lat_domain_max
            )
            mask &= (lat3 >= min(lat_min, lat_max)) & (lat3 <= max(lat_min, lat_max))
        if any(k in params for k in ("lon_min", "lon_max", "rel_lon_min", "rel_lon_max")):
            lon_min = _resolve_axis_value(
                params, "lon_min", "rel_lon_min", lon_domain_min, lon_domain_max
            )
            lon_max = _resolve_axis_value(
                params, "lon_max", "rel_lon_max", lon_domain_min, lon_domain_max
            )
            mask &= (lon3 >= min(lon_min, lon_max)) & (lon3 <= max(lon_min, lon_max))
        return mask

    if s == "sphere":
        lon0 = _resolve_axis_value(params, "lon0", "rel_lon0", lon_domain_min, lon_domain_max)
        lat0 = _resolve_axis_value(params, "lat0", "rel_lat0", lat_domain_min, lat_domain_max)
        dep0 = _to_float(params, "dep0")
        radius = _to_float(params, "radius_km")
        dx = (lon3 - lon0) * DEG_TO_KM * max(math.cos(math.radians(lat0)), 1e-6)
        dy = (lat3 - lat0) * DEG_TO_KM
        dz = dep3 - dep0
        return (dx * dx + dy * dy + dz * dz) <= (radius * radius)

    if s == "cylinder":
        lon0 = _resolve_axis_value(params, "lon0", "rel_lon0", lon_domain_min, lon_domain_max)
        lat0 = _resolve_axis_value(params, "lat0", "rel_lat0", lat_domain_min, lat_domain_max)
        radius = _to_float(params, "radius_km")
        dep_min = _to_float(params, "dep_min")
        dep_max = _to_float(params, "dep_max")
        dx = (lon3 - lon0) * DEG_TO_KM * max(math.cos(math.radians(lat0)), 1e-6)
        dy = (lat3 - lat0) * DEG_TO_KM
        horiz = (dx * dx + dy * dy) <= (radius * radius)
        dep_ok = (dep3 >= min(dep_min, dep_max)) & (dep3 <= max(dep_min, dep_max))
        return horiz & dep_ok

    if s == "polygon":
        vertices_raw = params.get("vertices", "")
        rel_vertices_raw = params.get("rel_vertices", "")
        if vertices_raw and rel_vertices_raw:
            raise ValueError("polygon accepts only one of vertices=... or rel_vertices=...")
        if rel_vertices_raw:
            rel_vertices = _parse_vertices(rel_vertices_raw)
            vertices = [
                (
                    lon_domain_min + rel_lon * (lon_domain_max - lon_domain_min),
                    lat_domain_min + rel_lat * (lat_domain_max - lat_domain_min),
                )
                for rel_lon, rel_lat in rel_vertices
            ]
            for rel_lon, rel_lat in rel_vertices:
                if rel_lon < 0.0 or rel_lon > 1.0 or rel_lat < 0.0 or rel_lat > 1.0:
                    raise ValueError("rel_vertices values must be within [0, 1].")
        else:
            if not vertices_raw:
                raise ValueError(
                    "polygon requires vertices=lon1:lat1;... or rel_vertices=x1:y1;..."
                )
            vertices = _parse_vertices(vertices_raw)
        dep_min = _to_float(params, "dep_min")
        dep_max = _to_float(params, "dep_max")
        poly2 = _polygon_mask(lon2, lat2, vertices)
        return (dep3 >= min(dep_min, dep_max)) & (dep3 <= max(dep_min, dep_max)) & poly2[None, :, :]

    raise ValueError(f"Unsupported shape type '{shape_type}'.")


def _apply_properties(model: ATTModel, mask: np.ndarray, params: dict[str, str]) -> None:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != model.vel.shape:
        try:
            mask = np.broadcast_to(mask, model.vel.shape)
        except ValueError as exc:
            raise ValueError(
                f"Shape mask {mask.shape} is not compatible with model grid {model.vel.shape}."
            ) from exc
    if not np.any(mask):
        return

    if "vel_abs" in params:
        model.vel[mask] = float(params["vel_abs"])
    if "vel_scale" in params:
        model.vel[mask] *= float(params["vel_scale"])
    if "vel_pct" in params:
        model.vel[mask] *= 1.0 + float(params["vel_pct"])
    if "vel_delta" in params:
        model.vel[mask] += float(params["vel_delta"])

    if "eps" in params:
        eps = float(params["eps"])
        phi_deg = float(params.get("phi_deg", "0.0"))
        phi = math.radians(phi_deg)
        model.xi[mask] = eps * math.cos(2.0 * phi)
        model.eta[mask] = eps * math.sin(2.0 * phi)
    if "eps_delta" in params:
        eps_delta = float(params["eps_delta"])
        phi_deg = float(params.get("phi_delta_deg", params.get("phi_deg", "0.0")))
        phi = math.radians(phi_deg)
        model.xi[mask] += eps_delta * math.cos(2.0 * phi)
        model.eta[mask] += eps_delta * math.sin(2.0 * phi)


def build_model(spec_path: Path) -> Path:
    spec_path = spec_path.resolve()
    spec_dir = spec_path.parent
    cfg, shapes = _parse_spec(spec_path)
    att_model_cls = _load_attmodel()

    par_file = cfg.get("par_file")
    if not par_file:
        raise ValueError("Spec must define par_file=<path to input_params yaml>.")
    par_file_path = Path(par_file).expanduser()
    if not par_file_path.is_absolute():
        par_file_path = (spec_dir / par_file_path).resolve()

    output_model = Path(cfg.get("output_model", "custom_model.h5")).expanduser()
    if not output_model.is_absolute():
        output_model = (spec_dir / output_model).resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)

    model = att_model_cls(str(par_file_path))
    _apply_background(model, cfg, spec_dir)

    dep3 = model.depths[:, None, None]
    lat3 = model.latitudes[None, :, None]
    lon3 = model.longitudes[None, None, :]
    lon2, lat2 = np.meshgrid(model.longitudes, model.latitudes)

    for item in shapes:
        shape_type = str(item["type"])
        params = dict(item["params"])
        lineno = int(item["lineno"])
        mask = _shape_mask(shape_type, params, dep3, lat3, lon3, lat2, lon2)
        _apply_properties(model, mask, params)
        print(f"Applied shape line {lineno}: {shape_type} (cells={int(mask.sum())})")

    if "clip_vel_min" in cfg or "clip_vel_max" in cfg:
        vmin = float(cfg.get("clip_vel_min", np.nanmin(model.vel)))
        vmax = float(cfg.get("clip_vel_max", np.nanmax(model.vel)))
        model.vel[:, :, :] = np.clip(model.vel, vmin, vmax)

    model.write(str(output_model))
    print(f"Wrote model: {output_model}")
    print(
        "Velocity range: "
        f"{float(np.nanmin(model.vel)):.4f} .. {float(np.nanmax(model.vel)):.4f} km/s; "
        f"epsilon max(abs): {float(np.nanmax(np.sqrt(model.xi**2 + model.eta**2))):.4f}"
    )
    return output_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="ASCII model spec file.")
    args = parser.parse_args()

    build_model(Path(args.spec).expanduser())


if __name__ == "__main__":
    main()
