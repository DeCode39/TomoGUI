# %%
import pygmt
pygmt.config(FONT="16p", IO_SEGMENT_MARKER="<<<")

import os
import math

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


def _build_ani_vectors(phi, epsilon, length, width, threshold):
    if phi.size == 0 or epsilon.size == 0:
        return np.empty((0, 5), dtype=float)
    n = min(phi.shape[0], epsilon.shape[0])
    phi = phi[:n]
    epsilon = epsilon[:n]
    ani = np.hstack(
        [
            phi,
            epsilon[:, 2].reshape(-1, 1) * length,
            np.ones((n, 1)) * width,
        ]
    )
    mask = np.isfinite(ani).all(axis=1)
    if threshold is not None:
        mask &= epsilon[:n, 2] > threshold
    ani = ani[mask]
    return np.ascontiguousarray(ani, dtype=float)


def _draw_borders(fig, region, projection):
    fig.coast(
        region=region,
        projection=projection,
        borders="1/0.5p,black",
        shorelines="0.5p,black",
        frame=False,
    )

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

REFERENCE_LABEL = "Reference (0%)"
REFERENCE_CMAP = "gray"
REFERENCE_SERIES = [-1, 1]
VELOCITY_LIMIT_PCT = 3.0
VELOCITY_TICK_PCT = 1.0


# %%
# read models

Ngrid = [61, 61, 61]
data_file = '2_models/model_init_N%d_%d_%d.h5'%(Ngrid[0],Ngrid[1],Ngrid[2])
par_file = '3_input_params/input_params_signal.yaml'  
model = ATTModel.read(data_file, par_file)
initial_model = model.to_xarray()
model_depths = np.asarray(getattr(model, "depths", []), dtype=float)
if model_depths.size == 0:
    raise ValueError("No model depths available.")
dep_min = float(np.nanmin(model_depths))
dep_max = float(np.nanmax(model_depths))
depth_list = _build_depth_list(dep_min, dep_max)
if not depth_list:
    raise ValueError("No depths resolved from model depth range.")

data_file = '2_models/model_ckb_N%d_%d_%d.h5'%(Ngrid[0],Ngrid[1],Ngrid[2])
model = ATTModel.read(data_file, par_file)
ckb_model = model.to_xarray()

# initial model
reference_depth = 10 if any(abs(d - 10.0) < 1e-6 for d in depth_list) else depth_list[0]
vel_init_ref = initial_model.interp_dep(reference_depth, field='vel')
region_plan = _region_from_points(vel_init_ref)
spacing_plan = 0.04
spacing_sec = (1.0, 0.04)
spacing_sec_str = f"{spacing_sec[0]}/{spacing_sec[1]}"
region_plan = _align_region(region_plan, spacing_plan, spacing_plan)
mid_lon = (region_plan[0] + region_plan[1]) * 0.5
mid_lat = (region_plan[2] + region_plan[3]) * 0.5
start = [region_plan[0], region_plan[2]]
end = [region_plan[1], region_plan[3]]
vel_init_sec = initial_model.interp_sec(start, end, field='vel', val = 1)
region_sec = _section_region(vel_init_sec)
region_sec = _align_region(region_sec, spacing_sec[0], spacing_sec[1])
mid_sec_x = (region_sec[0] + region_sec[1]) * 0.5
mid_sec_y = (region_sec[2] + region_sec[3]) * 0.5
depth_text = ", ".join(f"{d:g}" for d in depth_list)
print(f"Horizontal slice depths: {depth_text} km")
print(f"Vertical slice line: start lon/lat {start[0]:.4f}, {start[1]:.4f} -> end lon/lat {end[0]:.4f}, {end[1]:.4f}")
vel_ckb_sec = ckb_model.interp_sec(start, end, field='vel', val = 1)
setting_depth = reference_depth

# anisotropic arrow
samp_interval = 3
length = 7
width = 0.1
ani_thd = 0.02
ani_lat = region_plan[2] + 0.6
ani_lons = [region_plan[0] + 0.2, region_plan[0] + 0.5, region_plan[0] + 0.9]

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

# %%
# categorize earthquakes
ev_idx1 = []
ev_idx2 = []
ev_idx3 = []
depths = earthquake[2, :]
depths = depths[np.isfinite(depths)]
if depths.size:
    dep_min = float(np.min(depths))
    dep_max = float(np.max(depths))
else:
    dep_min = 0.0
    dep_max = 0.0
if dep_max < dep_min:
    dep_min, dep_max = dep_max, dep_min
dep_min = max(0.0, dep_min)
bin1_max = dep_min + 10.0
bin2_max = dep_min + 20.0

def _plot_quakes(fig, x, y, idx, style, fill):
    if not idx:
        return
    fig.plot(x=x[idx], y=y[idx], style=style, fill=fill)

for i in range(earthquake.shape[1]):
    dep = earthquake[2,i]
    if not np.isfinite(dep):
        continue
    if dep <= bin1_max:
        ev_idx1.append(i)
    elif dep <= bin2_max:
        ev_idx2.append(i)
    else:
        ev_idx3.append(i)

# %%
# plot the setting (initial + true models)
region = region_plan
frame  = ["xa1","ya1","nSWe"]
projection = "M10c"
spacing = spacing_plan
vel_range = VELOCITY_LIMIT_PCT

for depth in [setting_depth]:
    vel_init = initial_model.interp_dep(depth, field='vel')
    vel_ckb = ckb_model.interp_dep(depth, field='vel') # lon = [:,0], lat = [:,1], vel = [:,2]
    ani_ckb_phi = ckb_model.interp_dep(depth, field='phi', samp_interval=samp_interval)
    ani_ckb_epsilon = ckb_model.interp_dep(depth, field='epsilon', samp_interval=samp_interval)
    ani_ckb = _build_ani_vectors(ani_ckb_phi, ani_ckb_epsilon, length, width, ani_thd)

    fig = pygmt.Figure()

    # -------------- initial model --------------
    fig.basemap(
        region=region,
        frame=["xa1","ya1",f"+tInitial model (depth {depth:g} km)"],
        projection=projection,
    )
    pygmt.makecpt(cmap=REFERENCE_CMAP, series=REFERENCE_SERIES, background=True, reverse=False)
    x = vel_init[:,0]; y = vel_init[:,1]; value = np.zeros_like(vel_init[:,2])
    grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing,region=region)
    fig.grdimage(grid = grid)
    _draw_borders(fig, region, projection)
    fig.text(text=REFERENCE_LABEL, x=mid_lon, y=mid_lat, font="12p,Helvetica,gray30", justify="CM")
    _plot_quakes(fig, earthquake[0], earthquake[1], ev_idx1, "c0.1c", "red")
    _plot_quakes(fig, earthquake[0], earthquake[1], ev_idx2, "c0.1c", "green")
    _plot_quakes(fig, earthquake[0], earthquake[1], ev_idx3, "c0.1c", "black")
    fig.plot(x=station[0, :], y=station[1, :], style="t0.4c", fill="blue", pen="black")

    fig.shift_origin(xshift=11)
    fig.basemap(region=region_sec, frame=["xa20+lDepth (km)","ya1","Nswe"], projection="X2c/10c")
    x = vel_init_sec[:,3]; y = vel_init_sec[:,1]; value = np.zeros_like(vel_init_sec[:,4])
    grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing_sec_str, region=region_sec)
    fig.grdimage(grid = grid)
    fig.text(text=REFERENCE_LABEL, x=mid_sec_x, y=mid_sec_y, font="12p,Helvetica,gray30", justify="CM")
    _plot_quakes(fig, earthquake[2], earthquake[1], ev_idx1, "c0.1c", "red")
    _plot_quakes(fig, earthquake[2], earthquake[1], ev_idx2, "c0.1c", "green")
    _plot_quakes(fig, earthquake[2], earthquake[1], ev_idx3, "c0.1c", "black")

    fig.shift_origin(xshift=4)

    # -------------- true model --------------
    fig.basemap(
        region=region,
        frame=["xa1","ya1",f"+tTrue model and locations (depth {depth:g} km)"],
        projection=projection,
    )
    pygmt.makecpt(cmap="../../utils/svel13_chen.cpt", series=[-vel_range, vel_range], background=True, reverse=False)
    x = vel_ckb[:,0]; y = vel_ckb[:,1]; value = (vel_ckb[:,2] - vel_init[:,2])/vel_init[:,2] * 100
    grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing,region=region)
    fig.grdimage(grid = grid)
    _draw_borders(fig, region, projection)
    _plot_quakes(fig, earthquake[0], earthquake[1], ev_idx1, "c0.1c", "red")
    _plot_quakes(fig, earthquake[0], earthquake[1], ev_idx2, "c0.1c", "green")
    _plot_quakes(fig, earthquake[0], earthquake[1], ev_idx3, "c0.1c", "black")
    if ani_ckb.size:
        fig.plot(ani_ckb, style='j', fill='yellow1', pen='0.5p,black')

    fig.shift_origin(xshift=11)
    fig.basemap(region=region_sec, frame=["xa20+lDepth (km)","ya1","Nswe"], projection="X2c/10c")
    x = vel_ckb_sec[:,3]; y = vel_ckb_sec[:,1]
    value = np.divide(vel_ckb_sec[:,4] - vel_init_sec[:,4], vel_init_sec[:,4], out=np.zeros_like(vel_init_sec[:,4]), where=vel_init_sec[:,4]!=0) * 100
    grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing_sec_str, region=region_sec)
    fig.grdimage(grid = grid)
    _plot_quakes(fig, earthquake[2], earthquake[1], ev_idx1, "c0.1c", "red")
    _plot_quakes(fig, earthquake[2], earthquake[1], ev_idx2, "c0.1c", "green")
    _plot_quakes(fig, earthquake[2], earthquake[1], ev_idx3, "c0.1c", "black")

    # ------------------- colorbar -------------------
    fig.shift_origin(xshift=-11, yshift=-1.5)
    fig.colorbar(frame = [f"a{VELOCITY_TICK_PCT:g}","x+ldlnVp (%)"], position="+e+w4c/0.3c+h") 

    fig.shift_origin(xshift=6, yshift=-1)
    fig.basemap(region=region, frame=["wesn"], projection="X6c/1.5c")
    ani = [
        [ani_lons[0], ani_lat, 45, 0.02*length, width],
        [ani_lons[1], ani_lat, 45, 0.05*length, width],
        [ani_lons[2], ani_lat, 45, 0.10*length, width],
        ]
    fig.plot(ani, style='j', fill='yellow1', pen='0.5p,black')
    fig.text(text=["0.02", "0.05", "0.10"], x=ani_lons, y=[ani_lat]*3, font="16p,Helvetica", justify="CM")
    fig.shift_origin(xshift= 11, yshift=2.5)

    fig.show()
    fig.savefig("img/model_setting.png", dpi=300)

# %%
# read inversion model
tag = "inv"
data_file = "OUTPUT_FILES/OUTPUT_FILES_%s/final_model.h5"%(tag)
model = ATTModel.read(data_file, par_file)
inv_model = model.to_xarray()
vel_inv_sec = inv_model.interp_sec(start, end, field='vel', val = 1)
x_sec = vel_inv_sec[:,3]; y_sec = vel_inv_sec[:,1]
value_true_sec = np.divide(
    vel_ckb_sec[:,4] - vel_init_sec[:,4],
    vel_init_sec[:,4],
    out=np.zeros_like(vel_init_sec[:,4]),
    where=vel_init_sec[:,4]!=0,
) * 100
value_sec = np.divide(
    vel_inv_sec[:,4] - vel_init_sec[:,4],
    vel_init_sec[:,4],
    out=np.zeros_like(vel_init_sec[:,4]),
    where=vel_init_sec[:,4]!=0,
) * 100

# plot true + inversion models
region = region_plan
frame  = ["xa1","ya1","nSWe"]
projection = "M10c"
spacing = spacing_plan
vel_range = VELOCITY_LIMIT_PCT

for depth in depth_list:
    depth_tag = _depth_tag(depth)
    vel_init = initial_model.interp_dep(depth, field='vel')
    vel_true = ckb_model.interp_dep(depth, field='vel')
    vel_inv = inv_model.interp_dep(depth, field='vel')
    ani_true_phi = ckb_model.interp_dep(depth, field='phi', samp_interval=samp_interval)
    ani_true_epsilon = ckb_model.interp_dep(depth, field='epsilon', samp_interval=samp_interval)
    ani_true = _build_ani_vectors(ani_true_phi, ani_true_epsilon, length, width, ani_thd)
    ani_inv_phi = inv_model.interp_dep(depth, field='phi', samp_interval=samp_interval)
    ani_inv_epsilon = inv_model.interp_dep(depth, field='epsilon', samp_interval=samp_interval)
    ani_inv = _build_ani_vectors(ani_inv_phi, ani_inv_epsilon, length, width, ani_thd)
    fig = pygmt.Figure()

    # -------------- true model --------------
    fig.basemap(
        region=region,
        frame=["xa1","ya1",f"+tTrue model (depth {depth:g} km)"],
        projection=projection,
    )
    pygmt.makecpt(cmap="../../utils/svel13_chen.cpt", series=[-vel_range, vel_range], background=True, reverse=False)
    x = vel_true[:,0]; y = vel_true[:,1]; value = (vel_true[:,2] - vel_init[:,2])/vel_init[:,2] * 100
    grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing,region=region)
    fig.grdimage(grid = grid)
    _draw_borders(fig, region, projection)
    if ani_true.size:
        fig.plot(ani_true, style='j', fill='yellow1', pen='0.5p,black')

    fig.shift_origin(xshift=11)
    fig.basemap(region=region_sec, frame=["xa20+lDepth (km)","ya1","Nswe"], projection="X2c/10c")
    x = vel_ckb_sec[:,3]; y = vel_ckb_sec[:,1]; value = value_true_sec
    grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing_sec_str, region=region_sec)
    fig.grdimage(grid = grid)
    fig.text(text=REFERENCE_LABEL, x=mid_sec_x, y=mid_sec_y, font="12p,Helvetica,gray30", justify="CM")

    fig.shift_origin(xshift=4)

    # -------------- inversion model --------------
    fig.basemap(
        region=region,
        frame=["xa1","ya1",f"+tInversion results (depth {depth:g} km)"],
        projection=projection,
    )
    pygmt.makecpt(cmap="../../utils/svel13_chen.cpt", series=[-vel_range, vel_range], background=True, reverse=False)
    x = vel_inv[:,0]; y = vel_inv[:,1]; value = (vel_inv[:,2] - vel_init[:,2])/vel_init[:,2] * 100
    grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing,region=region)
    fig.grdimage(grid = grid)
    _draw_borders(fig, region, projection)
    if ani_inv.size:
        fig.plot(ani_inv, style='j', fill='yellow1', pen='0.5p,black')

    fig.shift_origin(xshift=11)
    fig.basemap(region=region_sec, frame=["xa20+lDepth (km)","ya1","Nswe"], projection="X2c/10c")
    grid = pygmt.surface(x=x_sec, y=y_sec, z=value_sec, spacing=spacing_sec_str, region=region_sec)
    fig.grdimage(grid = grid)

    # ------------------- colorbar -------------------
    fig.shift_origin(xshift=-11, yshift=-1.5)
    fig.colorbar(frame = [f"a{VELOCITY_TICK_PCT:g}","x+ldlnVp (%)"], position="+e+w4c/0.3c+h") 

    fig.shift_origin(xshift=6, yshift=-1)
    fig.basemap(region=region, frame=["wesn"], projection="X6c/1.5c")
    ani = [
        [ani_lons[0], ani_lat, 45, 0.02*length, width],
        [ani_lons[1], ani_lat, 45, 0.05*length, width],
        [ani_lons[2], ani_lat, 45, 0.10*length, width],
        ]
    fig.plot(ani, style='j', fill='yellow1', pen='0.5p,black')
    fig.text(text=["0.02", "0.05", "0.10"], x=ani_lons, y=[ani_lat]*3, font="16p,Helvetica", justify="CM")
    fig.shift_origin(xshift= 11, yshift=2.5)

    fig.show()
    fig.savefig(f"img/model_true_inv_depth_{depth_tag}.png", dpi=300)
