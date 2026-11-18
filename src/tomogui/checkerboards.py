"""Checker counts, physical extents and model-cell sampling restrictions."""
from __future__ import annotations

import math
import numpy as np

PRESETS = (8, 4, 2)


def geometry(grid, counts, *, strict=True):
    if len(counts) != 3:
        raise ValueError('Provide checker counts for longitude, latitude and depth.')
    result = []
    errors = []
    for axis, key, index, count in zip(('longitude', 'latitude', 'depth'),
            ('min_max_lon', 'min_max_lat', 'min_max_dep'), (2, 1, 0), counts):
        if isinstance(count, bool) or not math.isfinite(float(count)) or int(count) != float(count) or count < 2:
            raise ValueError('Checker counts must be integers of at least 2 per axis.')
        lo, hi = map(float, grid['domain'][key])
        nodes = int(grid['n_rtp'][index])
        spacing = (hi - lo) / max(1, nodes - 1)
        # Perturb only whole grid intervals at/under the surface, preserving the
        # existing exclusion of negative-depth air cells.
        first = max(0, math.ceil((max(0, lo) - lo) / spacing - 1e-9)) if axis == 'depth' and spacing > 0 else 0
        intervals = nodes - 1 - first
        length = intervals * spacing / int(count)
        cells = intervals / int(count)
        if spacing <= 0 or cells < 2 - 1e-9:
            errors.append(f'{axis}: {count} checkers have {cells:g} cells each; at least 2 required')
        if length > (hi - lo) / 2 + 1e-9:
            errors.append(f'{axis}: checker exceeds half of the grid span')
        result.append({'axis': axis, 'count': int(count), 'cells_per_checker': cells,
                       'length': length, 'units': 'km' if axis == 'depth' else 'degrees',
                       'first_node': first, 'last_node': nodes - 1, 'spacing': spacing})
    if errors and strict:
        raise ValueError('Invalid checkerboard: ' + '; '.join(errors) + '. Refine the model grid or reduce checker counts.')
    return {'valid': not errors, 'axes': result, 'errors': errors}


def preset_report(grid):
    return [{'counts': [n, n, n], **geometry(grid, [n, n, n], strict=False)} for n in PRESETS]


def build(module, initial, params, grid, config, output):
    counts = [config[key] for key in ('n_pert_x', 'n_pert_y', 'n_pert_z')]
    report = geometry(grid, counts)
    checker = module.Checker(str(initial), para_fname=str(params))
    xi0, eta0 = checker.xi.copy(), checker.eta.copy()
    # Equivalent sinusoidal logic, evaluated on interval endpoints rather than
    # N samples across N-1 intervals. This makes n=2 exactly half-span per lobe.
    factors = []
    for nodes, axis in zip((grid['n_rtp'][2], grid['n_rtp'][1], grid['n_rtp'][0]), report['axes']):
        values = np.zeros(nodes)
        start = axis['first_node']
        values[start:] = np.sin(np.linspace(0, axis['count'] * np.pi, nodes - start))
        values[np.abs(values) < 1e-12] = 0
        factors.append(values)
    perturbation = factors[2][:, None, None] * factors[1][None, :, None] * factors[0][None, None, :]
    checker.perturbation = perturbation
    checker.vel_pert = checker.vel * (1 + perturbation * config['pert_vel'])
    checker.dlnv = perturbation * config['pert_vel']
    checker.epsilon = np.abs(perturbation) * config['pert_ani']
    checker.phi = np.where(perturbation < 0, config['ani_dir'] + 90, config['ani_dir'])
    checker.xi = xi0 + checker.epsilon * np.cos(np.deg2rad(2 * checker.phi))
    checker.eta = eta0 + checker.epsilon * np.sin(np.deg2rad(2 * checker.phi))
    checker.write(str(output))
    return report


def fit_waves(grid, waves, *, allow_fit=False):
    """Validate paper lobe widths; fitting is explicit and recorded in previews."""
    import copy
    fitted = copy.deepcopy(waves)
    spans = [grid['domain'][key][1] - grid['domain'][key][0]
             for key in ('min_max_lon', 'min_max_lat', 'min_max_dep')]
    spacing = [span / (n - 1) for span, n in zip(spans, reversed(grid['n_rtp']))]
    factors = []
    for keys, lower, upper in [(('velocity_period_deg', 'anisotropy_period_deg'),
                                2 * max(spacing[:2]), min(spans[:2]) / 2),
                               (('depth_period_km',), 2 * spacing[2], spans[2] / 2)]:
        widths = [wave[key] for wave in waves.values() for key in keys]
        low_factor, high_factor = lower / min(widths), upper / max(widths)
        if low_factor > high_factor + 1e-9:
            raise ValueError('The model grid cannot support all three Kd wavelengths. Refine its spacing.')
        factor = min(high_factor, max(low_factor, 1.0))
        if not allow_fit and abs(factor - 1) > 1e-9:
            raise ValueError('Kd wavelength violates the two-cell/half-span limits. Enable explicit wavelength fitting or change the grid/domain.')
        for wave in fitted.values():
            for key in keys:
                wave[key] *= factor
        factors.append(factor)
    return {'waves': fitted, 'horizontal_factor': factors[0], 'depth_factor': factors[1],
            'adjusted': any(abs(f - 1) > 1e-9 for f in factors)}
