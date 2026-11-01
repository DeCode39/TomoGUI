from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .utils import strict_allclose


class ViewerError(RuntimeError):
    """Application-specific error with user-facing meaning."""


@dataclass(frozen=True)
class VolumeMatchResult:
    ok: bool
    reason: str | None = None


@dataclass
class Slice2D:
    values: np.ndarray
    u_coords: np.ndarray
    v_coords: np.ndarray
    u_label: str
    v_label: str
    slice_axis: str
    slice_index: int
    slice_coord: float


@dataclass
class VolumeSet:
    path: Path
    file_type: str
    fields: dict[str, np.ndarray]
    coords: dict[str, np.ndarray]
    axis_order: tuple[str, str, str] = ("x", "y", "z")
    coord_labels: dict[str, str] = field(default_factory=lambda: {"x": "x", "y": "y", "z": "z"})
    field_sources: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fields:
            raise ViewerError(f"No usable fields were loaded from {self.path}")
        first_shape = next(iter(self.fields.values())).shape
        if len(first_shape) != 3:
            raise ViewerError(f"Fields in {self.path} must be 3D")
        for name, values in self.fields.items():
            if values.shape != first_shape:
                raise ViewerError(
                    f"Field '{name}' in {self.path} has shape {values.shape}, expected {first_shape}"
                )
        for axis, size in zip(self.axis_order, first_shape):
            if axis not in self.coords:
                raise ViewerError(f"Missing coordinate array for axis '{axis}' in {self.path}")
            if len(self.coords[axis]) != size:
                raise ViewerError(
                    f"Coordinate axis '{axis}' has length {len(self.coords[axis])}, expected {size}"
                )
            if axis not in self.coord_labels:
                self.coord_labels[axis] = axis

    @property
    def shape(self) -> tuple[int, int, int]:
        return next(iter(self.fields.values())).shape

    def list_fields(self) -> list[str]:
        return sorted(self.fields)

    def get_field(self, name: str) -> np.ndarray:
        try:
            return self.fields[name]
        except KeyError as exc:
            raise ViewerError(f"Field '{name}' is not available in {self.path.name}") from exc

    def get_coord(self, axis: str) -> np.ndarray:
        try:
            return self.coords[axis]
        except KeyError as exc:
            raise ViewerError(f"Axis '{axis}' is not available in {self.path.name}") from exc


def match_volumes(volume_a: VolumeSet, volume_b: VolumeSet) -> VolumeMatchResult:
    if volume_a.axis_order != volume_b.axis_order:
        return VolumeMatchResult(False, "Axis order does not match.")
    if volume_a.shape != volume_b.shape:
        return VolumeMatchResult(False, f"Shape mismatch: {volume_a.shape} vs {volume_b.shape}.")
    for axis in volume_a.axis_order:
        if not strict_allclose(volume_a.get_coord(axis), volume_b.get_coord(axis)):
            return VolumeMatchResult(False, f"Coordinate mismatch along axis '{axis}'.")
    return VolumeMatchResult(True, None)


def extract_slice(
    volume: np.ndarray,
    coords: dict[str, np.ndarray],
    axis: str,
    index: int,
    coord_labels: dict[str, str] | None = None,
) -> Slice2D:
    axis = axis.lower()
    labels = coord_labels or {"x": "x", "y": "y", "z": "z"}
    if axis not in {"x", "y", "z"}:
        raise ViewerError(f"Unsupported axis '{axis}'.")
    if volume.ndim != 3:
        raise ViewerError(f"Expected a 3D volume, got shape {volume.shape}.")

    if axis == "x":
        if not 0 <= index < volume.shape[0]:
            raise ViewerError(f"Slice index {index} is out of range for axis x.")
        values = volume[index, :, :]
        u_axis, v_axis = "y", "z"
    elif axis == "y":
        if not 0 <= index < volume.shape[1]:
            raise ViewerError(f"Slice index {index} is out of range for axis y.")
        values = volume[:, index, :]
        u_axis, v_axis = "x", "z"
    else:
        if not 0 <= index < volume.shape[2]:
            raise ViewerError(f"Slice index {index} is out of range for axis z.")
        values = volume[:, :, index]
        u_axis, v_axis = "x", "y"

    return Slice2D(
        values=np.asarray(values),
        u_coords=np.asarray(coords[u_axis]),
        v_coords=np.asarray(coords[v_axis]),
        u_label=labels.get(u_axis, u_axis),
        v_label=labels.get(v_axis, v_axis),
        slice_axis=axis,
        slice_index=index,
        slice_coord=float(np.asarray(coords[axis])[index]),
    )
