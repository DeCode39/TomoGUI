# %%
import math
import os

import numpy as np
import pygmt

from pytomoatt.model import ATTModel


pygmt.config(FONT="16p", IO_SEGMENT_MARKER="<<<")


def _region_from_points(points):
    x = points[:, 0]
    y = points[:, 1]
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return [0.0, 1.0, 0.0, 1.0]
    return [
        float(np.min(x[mask])),
        float(np.max(x[mask])),
        float(np.min(y[mask])),
        float(np.max(y[mask])),
    ]


def _section_region(points):
    x = points[:, 3]
    y = points[:, 1]
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return [0.0, 1.0, 0.0, 1.0]
    return [
        float(np.min(x[mask])),
        float(np.max(x[mask])),
        float(np.min(y[mask])),
        float(np.max(y[mask])),
    ]


def _align_region(region, x_inc, y_inc):
    x_min, x_max, y_min, y_max = region
    if x_inc > 0:
        x_min = math.floor(x_min / x_inc) * x_inc
        x_max = x_min + math.ceil((x_max - x_min) / x_inc) * x_inc
    if y_inc > 0:
        y_min = math.floor(y_min / y_inc) * y_inc
        y_max = y_min + math.ceil((y_max - y_min) / y_inc) * y_inc
    return [float(x_min), float(x_max), float(y_min), float(y_max)]


def _safe_surface(*, x=None, y=None, z=None, spacing=None, region=None, **kwargs):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]

    if isinstance(spacing, str):
        parts = spacing.split("/")
        x_inc = float(parts[0])
        y_inc = float(parts[1]) if len(parts) > 1 else float(parts[0])
    elif isinstance(spacing, (tuple, list)):
        x_inc = float(spacing[0])
        y_inc = float(spacing[1])
    else:
        x_inc = float(spacing)
        y_inc = float(spacing)

    x0 = float(region[0])
    y0 = float(region[2])
    if x_inc > 0:
        x = np.round((x - x0) / x_inc) * x_inc + x0
    if y_inc > 0:
        y = np.round((y - y0) / y_inc) * y_inc + y0

    xy = np.column_stack((x, y))
    unique_xy, inverse = np.unique(xy, axis=0, return_inverse=True)
    z_sum = np.zeros(unique_xy.shape[0], dtype=float)
    z_count = np.zeros(unique_xy.shape[0], dtype=float)
    np.add.at(z_sum, inverse, z)
    np.add.at(z_count, inverse, 1.0)
    z_mean = z_sum / z_count

    if unique_xy.shape[0] == 0:
        raise ValueError("No finite points remain for gridding.")

    xg = np.ascontiguousarray(unique_xy[:, 0].copy(), dtype=float)
    yg = np.ascontiguousarray(unique_xy[:, 1].copy(), dtype=float)
    zg = np.ascontiguousarray(z_mean.copy(), dtype=float)

    return pygmt.xyz2grd(
        x=xg,
        y=yg,
        z=zg,
        spacing=spacing,
        region=region,
    )


pygmt.surface = _safe_surface


def _depth_tag(depth):
    text = f"{depth:g}"
    return text.replace(".", "p")


def _build_depth_list(dep_min, dep_max):
    if dep_min < 0:
        dep_min = 0.0
    depths = []
    first = dep_min + 1.0
    if dep_min <= first <= dep_max:
        depths.append(first)
    d = 5
    while d <= dep_max + 1e-9:
        if d >= dep_min - 1e-9:
            depths.append(float(d))
        d += 5
    out = []
    for depth in depths:
        if not any(abs(depth - existing) < 1e-6 for existing in out):
            out.append(depth)
    return out


def _safe_ratio(num, den):
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den != 0)


# %%
# read models
par_file = "3_input_params/input_params_ref.yaml"

model_p = ATTModel.read("2_models/model_final_P.h5", par_file)
model_s = ATTModel.read("2_models/model_final_S.h5", par_file)

vp = model_p.to_xarray()
vs = model_s.to_xarray()

model_depths = np.asarray(getattr(model_p, "depths", []), dtype=float)
if model_depths.size == 0:
    raise ValueError("No model depths available.")
dep_min = float(np.nanmin(model_depths))
dep_max = float(np.nanmax(model_depths))
depth_list = _build_depth_list(dep_min, dep_max)
if not depth_list:
    raise ValueError("No depths resolved from model depth range.")

# plan view region and section definition
reference_depth = depth_list[0]
vp_ref = vp.interp_dep(reference_depth, field="vel")
region_plan = _region_from_points(vp_ref)
spacing_plan = 0.04
spacing_sec = (1.0, 0.04)
spacing_sec_str = f"{spacing_sec[0]}/{spacing_sec[1]}"
region_plan = _align_region(region_plan, spacing_plan, spacing_plan)
start = [region_plan[0], region_plan[2]]
end = [region_plan[1], region_plan[3]]

vp_sec = vp.interp_sec(start, end, field="vel", val=1)
vs_sec = vs.interp_sec(start, end, field="vel", val=1)
n_sec = min(vp_sec.shape[0], vs_sec.shape[0])
vp_sec = vp_sec[:n_sec]
vs_sec = vs_sec[:n_sec]
vpvs_sec = _safe_ratio(vp_sec[:, 4], vs_sec[:, 4])
region_sec = _section_region(vp_sec)
region_sec = _align_region(region_sec, spacing_sec[0], spacing_sec[1])

# compute Vp/Vs range from the requested depths
vpvs_min, vpvs_max = np.inf, -np.inf
for depth in depth_list:
    vp_dep = vp.interp_dep(depth, field="vel")
    vs_dep = vs.interp_dep(depth, field="vel")
    n_dep = min(vp_dep.shape[0], vs_dep.shape[0])
    if n_dep == 0:
        continue
    vpvs = _safe_ratio(vp_dep[:n_dep, 2], vs_dep[:n_dep, 2])
    vpvs = vpvs[np.isfinite(vpvs)]
    if vpvs.size == 0:
        continue
    vpvs_min = min(vpvs_min, float(np.min(vpvs)))
    vpvs_max = max(vpvs_max, float(np.max(vpvs)))

if not np.isfinite(vpvs_min) or not np.isfinite(vpvs_max):
    raise ValueError("No finite Vp/Vs values found.")

vpvs_pad = (vpvs_max - vpvs_min) * 0.05
vpvs_min -= vpvs_pad
vpvs_max += vpvs_pad

os.makedirs("img", exist_ok=True)

depth_text = ", ".join(f"{d:g}" for d in depth_list)
print(f"Horizontal slice depths: {depth_text} km")
print(
    f"Vertical slice line: start lon/lat {start[0]:.4f}, {start[1]:.4f} -> "
    f"end lon/lat {end[0]:.4f}, {end[1]:.4f}"
)

for depth in depth_list:
    depth_tag = _depth_tag(depth)
    vp_dep = vp.interp_dep(depth, field="vel")
    vs_dep = vs.interp_dep(depth, field="vel")
    n_dep = min(vp_dep.shape[0], vs_dep.shape[0])
    vp_dep = vp_dep[:n_dep]
    vs_dep = vs_dep[:n_dep]
    vpvs = _safe_ratio(vp_dep[:, 2], vs_dep[:, 2])

    fig = pygmt.Figure()
    fig.basemap(
        region=region_plan,
        frame=["xa1", "ya1", f"+tVp/Vs (depth {depth:g} km)"],
        projection="M10c",
    )
    pygmt.makecpt(cmap="../../utils/svel13_chen.cpt", series=[vpvs_min, vpvs_max], background=True, reverse=True)
    grid = pygmt.surface(
        x=vp_dep[:, 0],
        y=vp_dep[:, 1],
        z=vpvs,
        spacing=spacing_plan,
        region=region_plan,
    )
    fig.grdimage(grid=grid)
    fig.coast(
        region=region_plan,
        projection="M10c",
        borders="1/0.5p,black",
        shorelines="0.5p,black",
        frame=False,
    )

    fig.shift_origin(xshift=11)
    fig.basemap(
        region=region_sec,
        frame=["xa20+lDepth (km)", "ya1", "Nswe"],
        projection="X2c/10c",
    )
    grid = pygmt.surface(
        x=vp_sec[:, 3],
        y=vp_sec[:, 1],
        z=vpvs_sec,
        spacing=spacing_sec_str,
        region=region_sec,
    )
    fig.grdimage(grid=grid)

    fig.shift_origin(xshift=-11, yshift=-1.5)
    tick = (vpvs_max - vpvs_min) / 4.0
    fig.colorbar(
        frame=[f"a{tick:.2f}", "x+lVp/Vs"],
        position="+e+w4c/0.3c+h",
    )
    fig.shift_origin(xshift=11, yshift=1.5)

    fig.savefig(f"img/vpvs_depth_{depth_tag}.png", dpi=300)
