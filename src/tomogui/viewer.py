"""Browser-facing adapter over the original HDF5 viewer's readers and math."""
from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
import math
from pathlib import Path
import threading

import numpy as np

from .workspace import viewer_modules


def clean_array(values):
    array = np.asarray(values, dtype=float)
    return np.where(np.isfinite(array), array, None).tolist()


class Viewer:
    def __init__(self, workspace):
        self.workspace = workspace
        self.readers, self.model, self.engine = viewer_modules(workspace.root)
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def load(self, path):
        path = self.workspace.path(path)
        key = (str(path), path.stat().st_mtime_ns)
        with self.lock:
            if key not in self.cache:
                self.cache[key] = self.readers.read_volume(path)
                for array in self.cache[key].fields.values():
                    array.flags.writeable = False
            self.cache.move_to_end(key)
            while len(self.cache) > 3:
                self.cache.popitem(last=False)
            return self.cache[key]

    def info(self, path):
        volume = self.load(path)
        return {'path': str(volume.path), 'fields': volume.list_fields(), 'shape': volume.shape,
                'coords': {axis: clean_array(volume.get_coord(axis)) for axis in volume.axis_order},
                'labels': volume.coord_labels, 'type': volume.file_type}

    def evaluate(self, payload):
        paths = {key: payload.get(key) for key in ('A','B','C') if payload.get(key)}
        if 'A' not in paths:
            raise ValueError('Open model A first.')
        volumes = {key: self.load(path) for key,path in paths.items()}
        expression = payload.get('expression') or 'A.' + volumes['A'].list_fields()[0]
        compiled = self.engine.compile_expression(expression, {key: set(v.list_fields()) for key,v in volumes.items()})
        referenced = [volumes[key] for key,fields in compiled.references.items() if fields]
        base = referenced[0]
        for volume in referenced[1:]:
            match = self.model.match_volumes(base, volume)
            if not match.ok:
                raise ValueError(match.reason)
        values = self.engine.evaluate_expression(compiled, {key: v.fields for key,v in volumes.items()})
        mask_expr = payload.get('mask')
        if mask_expr:
            mask_compiled = self.engine.compile_expression(mask_expr, {key: set(v.list_fields()) for key,v in volumes.items()})
            for key,fields in mask_compiled.references.items():
                if fields:
                    match = self.model.match_volumes(base, volumes[key])
                    if not match.ok:
                        raise ValueError(match.reason)
            mask = self.engine.evaluate_expression(mask_compiled, {key: v.fields for key,v in volumes.items()})
            threshold = float(payload.get('threshold', .5))
            values = np.where(mask >= threshold, values, np.nan)
        if not np.isfinite(values).any():
            raise ValueError('No finite values remain in this expression or mask.')
        return base, values

    def slice(self, payload):
        base, values = self.evaluate(payload)
        axis = payload.get('axis', 'z')
        coordinate = base.get_coord(axis)
        requested = float(payload.get('position', coordinate[len(coordinate)//2]))
        if not math.isfinite(requested):
            raise ValueError('Slice position must be finite.')
        index = int(np.argmin(np.abs(coordinate - requested)))
        section = self.model.extract_slice(values, base.coords, axis, index, base.coord_labels)
        stride_u = max(1, math.ceil(len(section.u_coords) / 400))
        stride_v = max(1, math.ceil(len(section.v_coords) / 400))
        limits = [float(np.nanmin(values)), float(np.nanmax(values))]
        return {'values': clean_array(section.values[::stride_u, ::stride_v].T),
                'u': clean_array(section.u_coords[::stride_u]), 'v': clean_array(section.v_coords[::stride_v]),
                'u_label': section.u_label, 'v_label': section.v_label, 'axis': axis,
                'position': section.slice_coord, 'limits': limits}

    def points(self, payload):
        base, values = self.evaluate(payload)
        stride = max(1, math.ceil((values.size / 5000) ** (1/3)))
        sampled = values[::stride, ::stride, ::stride]
        x,y,z = np.meshgrid(*(base.coords[k][::stride] for k in ('x','y','z')), indexing='ij')
        valid = np.isfinite(sampled)
        points = np.column_stack((x[valid],y[valid],z[valid],sampled[valid]))
        points = points[::max(1,math.ceil(len(points)/5000))]
        return {'points': points.tolist(), 'limits': [float(np.nanmin(values)),float(np.nanmax(values))],
                'labels': base.coord_labels, 'stride': stride}

    def png(self, payload):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        section = self.slice(payload)
        figure = Figure(figsize=(8,6), dpi=160)
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(111)
        values = np.array(section['values'], dtype=float)
        extent = [section['u'][0], section['u'][-1], section['v'][0], section['v'][-1]]
        image = axes.imshow(values, origin='lower', extent=extent, aspect='auto', cmap='seismic_r')
        if section['axis'] == 'z' and section['u_label'] == 'longitude':
            axes.set_aspect(1 / max(.01, math.cos(math.radians(sum(extent[2:])/2))))
        if section['v_label'] == 'depth':
            axes.invert_yaxis()
        axes.set(xlabel=section['u_label'], ylabel=section['v_label'], title=payload.get('expression','Model slice'))
        figure.colorbar(image, ax=axes)
        figure.tight_layout()
        stream = BytesIO(); figure.savefig(stream, format='png')
        return stream.getvalue()
