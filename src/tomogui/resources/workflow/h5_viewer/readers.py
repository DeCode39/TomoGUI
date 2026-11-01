from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from .data_model import ViewerError, VolumeSet
from .utils import is_numeric_dataset, safe_identifier, sibling_file, unique_name

try:
    from pytomoatt.model import ATTModel
except Exception:  # pragma: no cover - optional runtime dependency
    ATTModel = None

COORDINATE_CANDIDATES = (
    (("x", "y", "z"), {"x": "x", "y": "y", "z": "z"}),
    (("lon", "lat", "dep"), {"x": "lon", "y": "lat", "z": "dep"}),
    (("lon", "lat", "depth"), {"x": "lon", "y": "lat", "z": "depth"}),
    (("p", "t", "r"), {"x": "p", "y": "t", "z": "r"}),
)

INV_FIELD_RE = re.compile(r"^(?P<base>[A-Za-z0-9]+)_inv_(?P<step>\d+)$")


def read_volume(path: str | Path) -> VolumeSet:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ViewerError(f"File not found: {path}")
    try:
        with h5py.File(path, "r") as h5:
            if _looks_like_output_group(h5):
                return _read_output_group(path, h5)
            return _read_structured_hdf5(path, h5)
    except ViewerError:
        raise
    except OSError as exc:
        raise ViewerError(f"Could not open HDF5 file: {path}") from exc
    except Exception as exc:
        raise ViewerError(f"Could not read HDF5 file: {path}: {exc}") from exc


def _looks_like_output_group(h5: h5py.File) -> bool:
    if "model" not in h5:
        return False
    model_group = h5["model"]
    for key in model_group:
        obj = model_group[key]
        if is_numeric_dataset(obj) and obj.ndim == 1:
            return True
    return False


def _read_structured_hdf5(path: Path, h5: h5py.File) -> VolumeSet:
    datasets: list[tuple[str, tuple[int, int, int], h5py.Dataset]] = []
    candidate_coords: dict[str, np.ndarray] = {}

    def visitor(name: str, obj: object) -> None:
        if not is_numeric_dataset(obj):
            return
        if obj.ndim == 3:
            datasets.append((name, tuple(int(item) for item in obj.shape), obj))
        elif obj.ndim == 1:
            leaf = Path(name).name.lower()
            candidate_coords.setdefault(leaf, obj[...])

    h5.visititems(visitor)
    if not datasets:
        raise ViewerError(f"No 3D numeric datasets were found in {path.name}")

    grouped: dict[tuple[int, int, int], list[tuple[str, h5py.Dataset]]] = defaultdict(list)
    for dataset_name, dataset_shape, dataset in datasets:
        grouped[dataset_shape].append((dataset_name, dataset))

    shape, selected = max(grouped.items(), key=lambda item: (len(item[1]), np.prod(item[0])))
    ignored = sorted(name for other_shape, items in grouped.items() if other_shape != shape for name, _ in items)
    tomoatt_mapping = _try_tomoatt_mapping(path, shape)
    if tomoatt_mapping is None:
        tomoatt_mapping = _try_self_described_tomoatt_mapping(h5, shape, candidate_coords)
    if tomoatt_mapping is None:
        tomoatt_mapping = _try_inferred_tomoatt_dep_lat_lon_mapping(shape, candidate_coords)

    fields: dict[str, np.ndarray] = {}
    field_sources: dict[str, str] = {}
    used_names: set[str] = set()
    for dataset_name, dataset in sorted(selected):
        leaf = safe_identifier(Path(dataset_name).name)
        display_name = leaf if leaf not in used_names else safe_identifier(dataset_name.replace("/", "_"))
        display_name = unique_name(display_name, used_names)
        array = np.asarray(dataset[...])
        if tomoatt_mapping is not None:
            array = np.transpose(array, tomoatt_mapping["transpose_axes"])
        fields[display_name] = array
        field_sources[display_name] = dataset_name

    if tomoatt_mapping is not None:
        coords = tomoatt_mapping["coords"]
        coord_labels = tomoatt_mapping["coord_labels"]
        metadata = {
            "ignored_3d_datasets": ignored,
            "coordinate_space": "physical",
            "coordinate_source": tomoatt_mapping["parameter_file"],
            "tomoatt_dims": tomoatt_mapping["dims"],
        }
        file_type = "tomoatt_structured_hdf5"
    else:
        coords, coord_labels = _infer_coords_from_candidates(shape, candidate_coords)
        metadata = {
            "ignored_3d_datasets": ignored,
            "coordinate_space": "physical"
            if coord_labels != {"x": "longitude index", "y": "latitude index", "z": "depth index"}
            else "index",
        }
        file_type = "structured_hdf5"

    return VolumeSet(
        path=path,
        file_type=file_type,
        fields=fields,
        coords=coords,
        coord_labels=coord_labels,
        field_sources=field_sources,
        metadata=metadata,
    )


def _infer_coords_from_candidates(
    shape: tuple[int, int, int],
    candidates: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    for names, labels in COORDINATE_CANDIDATES:
        arrays = []
        ok = True
        for dim, name in zip(shape, names):
            values = candidates.get(name)
            if values is None or values.ndim != 1 or len(values) != dim:
                ok = False
                break
            arrays.append(np.asarray(values))
        if ok:
            return {"x": arrays[0], "y": arrays[1], "z": arrays[2]}, labels.copy()

    return (
        {
            "x": np.arange(shape[0], dtype=float),
            "y": np.arange(shape[1], dtype=float),
            "z": np.arange(shape[2], dtype=float),
        },
        {"x": "longitude index", "y": "latitude index", "z": "depth index"},
    )


def _try_tomoatt_mapping(path: Path, shape: tuple[int, int, int]) -> dict[str, object] | None:
    if ATTModel is None:
        return None

    for candidate in _candidate_parameter_files(path):
        try:
            model = ATTModel.read(str(path), str(candidate))
            xr = model.to_xarray()
            if "vel" not in xr:
                continue
            dims = xr["vel"].dims
            if tuple(int(xr.sizes[dim]) for dim in dims) != shape:
                continue
            transpose_axes = tuple(dims.index(dim) for dim in ("p", "t", "r"))
            coords = {
                "x": np.asarray(xr["lon"].values, dtype=float),
                "y": np.asarray(xr["lat"].values, dtype=float),
                "z": np.asarray(xr["dep"].values, dtype=float),
            }
            return {
                "coords": coords,
                "coord_labels": {"x": "longitude", "y": "latitude", "z": "depth"},
                "transpose_axes": transpose_axes,
                "parameter_file": str(candidate),
                "dims": dims,
            }
        except Exception:
            continue
    return None


def _try_self_described_tomoatt_mapping(
    h5: h5py.File,
    shape: tuple[int, int, int],
    candidate_coords: dict[str, np.ndarray],
) -> dict[str, object] | None:
    model_order = str(h5.attrs.get("axis_order_model", "")).strip().lower()
    coord_order = str(h5.attrs.get("axis_order_coords", "")).strip().lower()
    layout = str(h5.attrs.get("layout", "")).strip().lower()

    if not model_order and layout != "tomoatt_initial_model":
        return None

    model_dims = tuple(part.strip() for part in model_order.split(",") if part.strip())
    if model_dims != ("dep", "lat", "lon"):
        return None

    lon = candidate_coords.get("lon")
    lat = candidate_coords.get("lat")
    dep = candidate_coords.get("dep")
    if lon is None or lat is None or dep is None:
        return None
    if lon.ndim != 1 or lat.ndim != 1 or dep.ndim != 1:
        return None
    if shape != (len(dep), len(lat), len(lon)):
        return None

    metadata_source = "hdf5 attributes"
    if coord_order and coord_order != "lon,lat,dep":
        metadata_source = f"hdf5 attributes ({coord_order})"

    return {
        "coords": {
            "x": np.asarray(lon, dtype=float),
            "y": np.asarray(lat, dtype=float),
            "z": np.asarray(dep, dtype=float),
        },
        "coord_labels": {"x": "longitude", "y": "latitude", "z": "depth"},
        "transpose_axes": (2, 1, 0),
        "parameter_file": metadata_source,
        "dims": ("r", "t", "p"),
    }


def _try_inferred_tomoatt_dep_lat_lon_mapping(
    shape: tuple[int, int, int],
    candidate_coords: dict[str, np.ndarray],
) -> dict[str, object] | None:
    lon = candidate_coords.get("lon")
    lat = candidate_coords.get("lat")
    dep = candidate_coords.get("dep")
    if lon is None or lat is None or dep is None:
        return None
    if lon.ndim != 1 or lat.ndim != 1 or dep.ndim != 1:
        return None
    if shape != (len(dep), len(lat), len(lon)):
        return None

    return {
        "coords": {
            "x": np.asarray(lon, dtype=float),
            "y": np.asarray(lat, dtype=float),
            "z": np.asarray(dep, dtype=float),
        },
        "coord_labels": {"x": "longitude", "y": "latitude", "z": "depth"},
        "transpose_axes": (2, 1, 0),
        "parameter_file": "inferred from dep/lat/lon datasets",
        "dims": ("r", "t", "p"),
    }


def _candidate_parameter_files(path: Path) -> list[Path]:
    names = (
        "input_params_signal.yaml",
        "input_params_inv.yaml",
        "input_params_joint_step1.yaml",
        "input_params_joint_step2.yaml",
        "input_params_joint_step3.yaml",
        "input_params.yaml",
    )
    candidates: list[Path] = []
    seen: set[Path] = set()
    for parent in (path.parent,) + tuple(path.parents):
        search_dirs = [parent]
        input_dir = parent / "3_input_params"
        if input_dir.is_dir():
            search_dirs.insert(0, input_dir)
        for search_dir in search_dirs:
            for name in names:
                candidate = search_dir / name
                if candidate.is_file() and candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)
    return candidates


def _read_output_group(path: Path, h5: h5py.File) -> VolumeSet:
    grid_path = sibling_file(path, "out_data_grid.h5")
    if not grid_path.exists():
        raise ViewerError(
            f"{path.name} looks like an output group file, but sibling out_data_grid.h5 was not found."
        )

    with h5py.File(grid_path, "r") as grid_h5:
        coords, linear_index, shape = _read_output_grid(grid_h5)
        coord_labels = {"x": "longitude", "y": "latitude", "z": "depth"}

        raw_fields = _read_output_model_fields(h5)
        if not raw_fields:
            raise ViewerError(f"No usable output model fields were found in {path.name}")

        fields: dict[str, np.ndarray] = {}
        field_sources: dict[str, str] = {}
        used_names: set[str] = set()
        latest_aliases: dict[str, tuple[int, str, np.ndarray]] = {}

        for raw_name, values in sorted(raw_fields.items()):
            display_name = unique_name(safe_identifier(raw_name), used_names)
            reshaped = _reshape_output_field(values, linear_index, shape)
            fields[display_name] = reshaped
            field_sources[display_name] = f"model/{raw_name}"

            match = INV_FIELD_RE.match(raw_name)
            if match:
                base = match.group("base")
                step = int(match.group("step"))
                current = latest_aliases.get(base)
                if current is None or step >= current[0]:
                    latest_aliases[base] = (step, display_name, reshaped)

        for base, (step, source_name, reshaped) in sorted(latest_aliases.items()):
            alias = unique_name(base, used_names)
            fields[alias] = reshaped
            field_sources[alias] = field_sources[source_name]

        metadata = {
            "grid_path": str(grid_path),
            "coordinate_space": "physical",
            "latest_aliases": {base: name for base, (_, name, _) in latest_aliases.items()},
        }

        return VolumeSet(
            path=path,
            file_type="tomoatt_output_group",
            fields=fields,
            coords=coords,
            coord_labels=coord_labels,
            field_sources=field_sources,
            metadata=metadata,
        )


def _read_output_grid(grid_h5: h5py.File) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[int, int, int]]:
    coord_sets = (
        ("Mesh/node_coords_p", "Mesh/node_coords_t", "Mesh/node_coords_r", "ptr"),
        ("Mesh/node_coords_x", "Mesh/node_coords_y", "Mesh/node_coords_z", "xyz"),
    )

    for keys in coord_sets:
        mode = keys[3]
        coord_keys = keys[:3]
        if not all(key in grid_h5 for key in coord_keys):
            continue
        a = np.asarray(grid_h5[coord_keys[0]][:])
        b = np.asarray(grid_h5[coord_keys[1]][:])
        c = np.asarray(grid_h5[coord_keys[2]][:])
        ua, inv_a = np.unique(np.round(a, 12), return_inverse=True)
        ub, inv_b = np.unique(np.round(b, 12), return_inverse=True)
        uc, inv_c = np.unique(np.round(c, 12), return_inverse=True)
        shape = (len(ua), len(ub), len(uc))
        if np.prod(shape) != len(a):
            continue
        linear_index = inv_a * (shape[1] * shape[2]) + inv_b * shape[2] + inv_c
        if np.unique(linear_index).size != linear_index.size:
            raise ViewerError("out_data_grid.h5 contains duplicate node coordinates and cannot define a regular grid.")
        if mode == "ptr":
            coords = {
                "x": ua.astype(float),
                "y": ub.astype(float),
                "z": 6371.0 - uc.astype(float),
            }
        else:
            coords = {"x": ua.astype(float), "y": ub.astype(float), "z": uc.astype(float)}
        return (coords, linear_index, shape)

    raise ViewerError("Could not reconstruct a regular grid from out_data_grid.h5")


def _read_output_model_fields(h5: h5py.File) -> dict[str, np.ndarray]:
    model_group = h5.get("model")
    if model_group is None:
        return {}
    fields: dict[str, np.ndarray] = {}
    for key, obj in model_group.items():
        if is_numeric_dataset(obj) and obj.ndim == 1:
            fields[key] = np.asarray(obj[:])
    return fields


def _reshape_output_field(values: np.ndarray, linear_index: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    values = np.asarray(values)
    expected = int(np.prod(shape))
    if linear_index.size != expected:
        raise ViewerError(
            f"Output grid has {linear_index.size} nodes, but regular shape {shape} requires {expected} nodes."
        )
    if values.size != linear_index.size:
        raise ViewerError(
            f"Output field length {values.size} does not match output grid node count {linear_index.size}."
        )
    if linear_index.size and (int(np.min(linear_index)) < 0 or int(np.max(linear_index)) >= expected):
        raise ViewerError("Output grid index reconstruction produced out-of-range node indices.")
    if np.unique(linear_index).size != expected:
        raise ViewerError("Output grid index reconstruction does not cover every grid node exactly once.")
    reshaped = np.empty(expected, dtype=values.dtype)
    reshaped[linear_index] = values.reshape(-1)
    return reshaped.reshape(shape)
