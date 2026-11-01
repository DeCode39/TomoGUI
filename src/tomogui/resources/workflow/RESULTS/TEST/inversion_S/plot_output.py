# %%
import pygmt
pygmt.config(FONT="16p", IO_SEGMENT_MARKER="<<<")

import math
import os
from pathlib import Path
import re

# %%
from pytomoatt.model import ATTModel
from pytomoatt.data import ATTData
import numpy as np

# %%
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


def _depth_bins(loc, dep_min=None, bin_size=10.0):
    ev_idx1, ev_idx2, ev_idx3 = [], [], []
    if loc.size == 0:
        return ev_idx1, ev_idx2, ev_idx3
    if dep_min is None:
        depths = loc[2, :]
        depths = depths[np.isfinite(depths)]
        if depths.size == 0:
            return ev_idx1, ev_idx2, ev_idx3
        dep_min = float(np.min(depths))
    dep_min = max(0.0, float(dep_min))
    d1 = dep_min + float(bin_size)
    d2 = dep_min + 2.0 * float(bin_size)
    for i in range(loc.shape[1]):
        dep = loc[2, i]
        if not np.isfinite(dep):
            continue
        if dep < d1:
            ev_idx1.append(i)
        elif dep < d2:
            ev_idx2.append(i)
        else:
            ev_idx3.append(i)
    return ev_idx1, ev_idx2, ev_idx3


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


def _plot_events(fig, loc, idx, xcol, ycol, **kwargs):
    if not idx:
        return
    fig.plot(x=loc[xcol, idx], y=loc[ycol, idx], **kwargs)


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


def _find_single_model(pattern):
    matches = sorted(Path("2_models").glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No model files found for pattern: {pattern}")
    return str(matches[0])


def _numeric_key(path):
    nums = [int(token) for token in re.findall(r"\d+", path.stem)]
    return tuple(nums) if nums else (-1,)


def _latest_output_file(*patterns):
    matches = []
    for pattern in patterns:
        matches.extend(Path(".").glob(pattern))
    matches = [path for path in matches if path.is_file() and not path.name.endswith("_obs.dat")]
    if not matches:
        return None
    return sorted(matches, key=_numeric_key)[-1]


def _plot_ani_safe(fig, data, **kwargs):
    if data is None:
        return
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0:
        return
    arr = np.ascontiguousarray(arr)
    try:
        fig.plot(arr, **kwargs)
    except Exception as exc:
        print(f"Warning: anisotropy plot skipped: {exc}")


def _build_ani_vectors(model, depth, samp_interval, length, width, ani_thd):
    try:
        ani_phi = model.interp_dep(depth, field='phi', samp_interval=samp_interval)
        ani_epsilon = model.interp_dep(depth, field='epsilon', samp_interval=samp_interval)
    except Exception as exc:
        print(f"Warning: anisotropy fields unavailable at depth {depth}: {exc}")
        return None
    ani_phi = np.asarray(ani_phi, dtype=float)
    ani_epsilon = np.asarray(ani_epsilon, dtype=float)
    if ani_phi.size == 0 or ani_epsilon.size == 0:
        return None
    ani = np.hstack(
        [
            ani_phi,
            ani_epsilon[:, 2].reshape(-1, 1) * length,
            np.ones((ani_epsilon.shape[0], 1)) * width,
        ]
    )
    idx = np.where(np.isfinite(ani_epsilon[:, 2]) & (ani_epsilon[:, 2] > ani_thd))[0]
    return ani[idx, :]

# %%
# read models

par_file = '3_input_params/input_params_signal.yaml'  
data_file = _find_single_model("model_init_N*.h5")
model = ATTModel.read(data_file, par_file)
initial_model = model.to_xarray()

plot_true_model = False

# initial model
model_depths = np.asarray(getattr(model, "depths", []), dtype=float)
if model_depths.size == 0:
    raise ValueError("No model depths available.")
dep_min = float(np.nanmin(model_depths))
dep_max = float(np.nanmax(model_depths))
depth_list = _build_depth_list(dep_min, dep_max)
if not depth_list:
    raise ValueError("No depths resolved for plotting.")
reference_depth = 10 if any(abs(d - 10.0) < 1e-6 for d in depth_list) else depth_list[0]
vel_init = initial_model.interp_dep(reference_depth, field='vel')
region = _region_from_points(vel_init)
spacing = 0.04
spacing_sec = (1.0, 0.04)
spacing_sec_str = f"{spacing_sec[0]}/{spacing_sec[1]}"
region = _align_region(region, spacing, spacing)
start = [region[0], region[2]]
end = [region[1], region[3]]
vel_init_sec = initial_model.interp_sec(start, end, field='vel', val = 1)
region_sec = _section_region(vel_init_sec)
region_sec = _align_region(region_sec, spacing_sec[0], spacing_sec[1])
depth = depth_list[0]
print("Horizontal slice depths: " + ", ".join(f"{d:g}" for d in depth_list) + " km")

# anisotropic arrow
samp_interval = 3
length = 7
width = 0.1
ani_thd = 0.02

try:
    os.mkdir('img')
except:
    pass

# %%
# read src_rec_file for data
from pytomoatt.src_rec import SrcRec

sr = SrcRec.read('1_src_rec_files/src_rec_config.dat')
station = sr.receivers[['stlo','stla','stel']].values.T
true_loc = sr.sources[['evlo','evla','evdp']].values.T
earthquake = true_loc

sr = SrcRec.read('1_src_rec_files/src_rec_config.dat')
init_loc = sr.sources[['evlo','evla','evdp']].values.T
event_depths = earthquake[2, :]
event_depths = event_depths[np.isfinite(event_depths)]
event_dep_min = max(0.0, float(np.min(event_depths))) if event_depths.size else 0.0
ev_idx1, ev_idx2, ev_idx3 = _depth_bins(earthquake, dep_min=event_dep_min)
ev_idx1_init, ev_idx2_init, ev_idx3_init = _depth_bins(init_loc, dep_min=event_dep_min)
ev_idx1_true, ev_idx2_true, ev_idx3_true = _depth_bins(true_loc, dep_min=event_dep_min)

# %%
# plot the model setting
fig = pygmt.Figure()
frame  = ["xa1","ya1"]
projection = "M10c"

vel_range = 20

# -------------- initial model and earthquake location --------------
fig.basemap(region=region, frame=["xa1","ya1","+tInitial model and locations"], projection=projection)
# velocity perturbation
pygmt.makecpt(cmap="../../utils/svel13_chen.cpt", series=[-vel_range, vel_range], background=True, reverse=False)
x = vel_init[:,0]; y = vel_init[:,1]; value = np.zeros_like(vel_init[:,2])
grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing,region=region)
fig.grdimage(grid = grid)
fig.coast(region=region, projection=projection, borders="1/0.5p,black", shorelines="0.5p,black", frame=False)
# earthquakes
fig.plot(x = init_loc[0,ev_idx1_init], y = init_loc[1,ev_idx1_init], style = "c0.1c", fill = "red")
fig.plot(x = init_loc[0,ev_idx2_init], y = init_loc[1,ev_idx2_init], style = "c0.1c", fill = "green")
fig.plot(x = init_loc[0,ev_idx3_init], y = init_loc[1,ev_idx3_init], style = "c0.1c", fill = "black")

# stations
fig.plot(x = station[0,:], y = station[1,:], style = "t0.4c", fill = "blue", pen = "black", label = "Station")

# # anisotropic arrow
# fig.plot(ani_ckb, style='j', fill='yellow1', pen='0.5p,black')

fig.shift_origin(xshift=11)

fig.basemap(region=region_sec, frame=["xa20+lDepth (km)","ya1","Nswe"], projection="X2c/10c")
x = vel_init_sec[:,3]; y = vel_init_sec[:,1]; value = np.zeros_like(vel_init_sec[:,4])
grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing_sec_str, region=region_sec)
fig.grdimage(grid = grid)

# earthquakes
fig.plot(x = init_loc[2,ev_idx1_init], y = init_loc[1,ev_idx1_init], style = "c0.1c", fill = "red")
fig.plot(x = init_loc[2,ev_idx2_init], y = init_loc[1,ev_idx2_init], style = "c0.1c", fill = "green")
fig.plot(x = init_loc[2,ev_idx3_init], y = init_loc[1,ev_idx3_init], style = "c0.1c", fill = "black")


if plot_true_model:
    data_file = _find_single_model("model_ckb_N*.h5")
    if os.path.exists(data_file):
        model = ATTModel.read(data_file, par_file)
        ckb_model = model.to_xarray()

        # checkerboard model
        vel_ckb = ckb_model.interp_dep(depth, field='vel') # lon = [:,0], lat = [:,1], vel = [:,2]
        vel_ckb_sec = ckb_model.interp_sec(start, end, field='vel', val = 1)

        ani_ckb = _build_ani_vectors(ckb_model, depth, samp_interval, length, width, ani_thd)

        fig.shift_origin(xshift=4)

        # -------------- true model and earthquake location --------------
        fig.basemap(region=region, frame=["xa1","ya1","+tTrue model and locations"], projection=projection)
        # velocity perturbation
        pygmt.makecpt(cmap="../../utils/svel13_chen.cpt", series=[-vel_range, vel_range], background=True, reverse=False)
        x = vel_ckb[:,0]; y = vel_ckb[:,1]; value = (vel_ckb[:,2] - vel_init[:,2])/vel_init[:,2] * 100
        grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing,region=region)
        fig.grdimage(grid = grid)
        fig.coast(region=region, projection=projection, borders="1/0.5p,black", shorelines="0.5p,black", frame=False)
        # earthquakes
        fig.plot(x = true_loc[0,ev_idx1_true], y = true_loc[1,ev_idx1_true], style = "c0.1c", fill = "red")
        fig.plot(x = true_loc[0,ev_idx2_true], y = true_loc[1,ev_idx2_true], style = "c0.1c", fill = "green")
        fig.plot(x = true_loc[0,ev_idx3_true], y = true_loc[1,ev_idx3_true], style = "c0.1c", fill = "black")

        # anisotropic arrow
        fig.plot(ani_ckb, style='j', fill='yellow1', pen='0.5p,black')

        fig.shift_origin(xshift=11)

        fig.basemap(region=region_sec, frame=["xa20+lDepth (km)","ya1","Nswe"], projection="X2c/10c")
        x = vel_ckb_sec[:,3]; y = vel_ckb_sec[:,1]; value = (vel_ckb_sec[:,4] - vel_init_sec[:,4])/vel_init_sec[:,4] * 100
        grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing_sec_str, region=region_sec)
        fig.grdimage(grid = grid)

        # earthquakes
        fig.plot(x = true_loc[2,ev_idx1_true], y = true_loc[1,ev_idx1_true], style = "c0.1c", fill = "red")
        fig.plot(x = true_loc[2,ev_idx2_true], y = true_loc[1,ev_idx2_true], style = "c0.1c", fill = "green")
        fig.plot(x = true_loc[2,ev_idx3_true], y = true_loc[1,ev_idx3_true], style = "c0.1c", fill = "black")


# ------------------- colorbar -------------------
fig.shift_origin(xshift=-11, yshift=-1.5)
fig.colorbar(frame = ["a%f"%(vel_range),"x+ldlnVs (%)"], position="+e+w4c/0.3c+h")

fig.shift_origin(xshift=6, yshift=-1)
fig.basemap(region=[0,1,0,1], frame=["wesn"], projection="X6c/1.5c")
ani = [
    [0.2, 0.6, 45, 0.02*length, width],    # lon, lat, phi, epsilon, size
    [0.5, 0.6, 45, 0.05*length, width],
    [0.8, 0.6, 45, 0.10*length, width],
    ]
_plot_ani_safe(fig, ani, style='j', fill='yellow1', pen='0.5p,black')
fig.text(text=["0.02", "0.05", "0.10"], x=[0.2,0.5,0.8], y=[0.2]*3, font="16p,Helvetica", justify="CM")
fig.shift_origin(xshift= 11, yshift=2.5)

fig.show()
fig.savefig('img/model_setting.png', dpi=300)

# %%
# plot the joint inversion results

projection = "M10c"
spacing = 0.04
vel_range = 20
joint_model = None
joint_model_file = Path("OUTPUT_FILES/OUTPUT_FILES_joint_step2/final_model.h5")
if joint_model_file.exists():
    joint_model = ATTModel.read(str(joint_model_file), par_file).to_xarray()

step_configs = []
step1_src = _latest_output_file("OUTPUT_FILES/OUTPUT_FILES_joint_step1/src_rec_file_reloc_*.dat")
if step1_src is not None:
    step_configs.append(("joint_step1", step1_src))
if joint_model is not None:
    step2_src = _latest_output_file("OUTPUT_FILES/OUTPUT_FILES_joint_step2/src_rec_file_inv_*_reloc_*.dat")
    step_configs.append(("joint_step2", step2_src))
    step3_src = _latest_output_file("OUTPUT_FILES/OUTPUT_FILES_joint_step3/src_rec_file_reloc_*.dat")
    if step3_src is not None:
        step_configs.append(("joint_step3", step3_src))

if not step_configs:
    raise FileNotFoundError("No joint-step outputs were found for plotting.")

print("Joint-step panels: " + ", ".join(tag for tag, _ in step_configs))

for depth in depth_list:
    vel_init = initial_model.interp_dep(depth, field='vel')
    vel_init_sec = initial_model.interp_sec(start, end, field='vel', val=1)
    fig = pygmt.Figure()

    for itag, (tag, src_file) in enumerate(step_configs):
        if tag == "joint_step1":
            x = vel_init[:, 0]; y = vel_init[:, 1]; value = np.zeros_like(vel_init[:, 2])
            x_sec = vel_init_sec[:, 3]; y_sec = vel_init_sec[:, 1]; value_sec = np.zeros_like(vel_init_sec[:, 4])
            sr = SrcRec.read(str(src_file))
            re_loc = sr.sources[['evlo', 'evla', 'evdp']].values.T
            re_idx1, re_idx2, re_idx3 = _depth_bins(re_loc, dep_min=event_dep_min)
            frame = ["xa1", "ya1", "+tStep %d, preliminary location" % (itag + 1)]
        else:
            inv_model = joint_model
            vel_inv = inv_model.interp_dep(depth, field='vel')
            x = vel_inv[:, 0]; y = vel_inv[:, 1]; value = (vel_inv[:, 2] - vel_init[:, 2]) / vel_init[:, 2] * 100
            vel_inv_sec = inv_model.interp_sec(start, end, field='vel', val=1)
            x_sec = vel_inv_sec[:, 3]; y_sec = vel_inv_sec[:, 1]; value_sec = (vel_inv_sec[:, 4] - vel_init_sec[:, 4]) / vel_init_sec[:, 4] * 100
            ani = _build_ani_vectors(inv_model, depth, samp_interval, length, width, ani_thd)
            if src_file is not None:
                sr = SrcRec.read(str(src_file))
                re_loc = sr.sources[['evlo', 'evla', 'evdp']].values.T
                re_idx1, re_idx2, re_idx3 = _depth_bins(re_loc, dep_min=event_dep_min)
            else:
                re_loc = init_loc
                re_idx1, re_idx2, re_idx3 = ev_idx1_init, ev_idx2_init, ev_idx3_init
            if tag == "joint_step2":
                frame = ["xa1", "ya1", "+tStep %d, joint inversion" % (itag + 1)]
            else:
                frame = ["xa1", "ya1", "+tStep %d, relocation" % (itag + 1)]

        fig.basemap(region=region, frame=frame, projection=projection)
        pygmt.makecpt(cmap="../../utils/svel13_chen.cpt", series=[-vel_range, vel_range], background=True, reverse=False)
        grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing, region=region)
        fig.grdimage(grid=grid)
        fig.coast(region=region, projection=projection, borders="1/0.5p,black", shorelines="0.5p,black", frame=False)
        if tag != "joint_step2":
            _plot_events(fig, re_loc, re_idx1, 0, 1, style="c0.1c", fill="red")
            _plot_events(fig, re_loc, re_idx2, 0, 1, style="c0.1c", fill="green")
            _plot_events(fig, re_loc, re_idx3, 0, 1, style="c0.1c", fill="black")
        if tag in {"joint_step2", "joint_step3"}:
            _plot_ani_safe(fig, ani, style='j', fill='yellow1', pen='0.5p,black')

        fig.shift_origin(xshift=11)
        fig.basemap(region=region_sec, frame=["xa20+lDepth (km)", "ya1", "Nswe"], projection="X2c/10c")
        grid = pygmt.surface(x=x_sec, y=y_sec, z=value_sec, spacing=spacing_sec_str, region=region_sec)
        fig.grdimage(grid=grid)
        if tag != "joint_step2":
            _plot_events(fig, re_loc, re_idx1, 2, 1, style="c0.1c", fill="red")
            _plot_events(fig, re_loc, re_idx2, 2, 1, style="c0.1c", fill="green")
            _plot_events(fig, re_loc, re_idx3, 2, 1, style="c0.1c", fill="black")
        fig.shift_origin(xshift=4)

    fig.shift_origin(xshift=-4)
    fig.shift_origin(xshift=-11, yshift=-1.5)
    fig.colorbar(frame=["a%f" % (vel_range), "x+ldlnVs (%)"], position="+e+w4c/0.3c+h")
    fig.shift_origin(xshift=6, yshift=-1)
    fig.basemap(region=[0, 1, 0, 1], frame=["wesn"], projection="X6c/1.5c")
    ani = [
        [0.2, 0.6, 45, 0.02 * length, width],
        [0.5, 0.6, 45, 0.05 * length, width],
        [0.8, 0.6, 45, 0.10 * length, width],
    ]
    _plot_ani_safe(fig, ani, style='j', fill='yellow1', pen='0.5p,black')
    fig.text(text=["0.02", "0.05", "0.10"], x=[0.2, 0.5, 0.8], y=[0.2] * 3, font="16p,Helvetica", justify="CM")
    fig.shift_origin(xshift=11, yshift=2.5)

    depth_suffix = _depth_tag(depth)
    fig.savefig(f'img/model_joint_depth_{depth_suffix}.png', dpi=300)
