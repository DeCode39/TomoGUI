"""Observed-data handoffs and filtered CS/CR generation for staged refinement."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import yaml

from .data import read_src_rec

DEFAULTS = {'enabled': False, 'use_cs': True, 'use_cr': True,
            'cs_azimuth_deg': 30.0, 'cr_separation_km': 10.0,
            'cs_shift_km': 3.0, 'cr_top_weight': 1.0,
            'allow_ungraded': False, 'max_pairs': 1000000}


def settings(value=None):
    result = {**DEFAULTS, **(value or {})}
    for key in ('enabled', 'use_cs', 'use_cr', 'allow_ungraded'):
        if not isinstance(result[key], bool):
            raise ValueError(f'{key} must be true or false.')
    for key, maximum in [('cs_azimuth_deg', 180), ('cr_separation_km', 1000), ('cs_shift_km', 1000), ('cr_top_weight', 1000)]:
        number = float(result[key])
        if not math.isfinite(number) or not 0 < number <= maximum:
            raise ValueError(f'{key} must be positive and at most {maximum}.')
        result[key] = number
    if isinstance(result['max_pairs'], bool) or int(result['max_pairs']) != float(result['max_pairs']) or not 1 <= int(result['max_pairs']) <= 10000000:
        raise ValueError('Pair limit must be an integer from 1 to 10000000.')
    result['max_pairs'] = int(result['max_pairs'])
    if result['enabled'] and not (result['use_cs'] or result['use_cr']):
        raise ValueError('Enable CS or CR for differential refinement.')
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def distance_3d(a, b):
    lat1, lon1, dep1 = map(float, a)
    lat2, lon2, dep2 = map(float, b)
    dl = math.radians(lon2 - lon1)
    dp = math.radians(lat2 - lat1)
    h = math.sin(dp / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dl / 2)**2
    horizontal = 2 * 6371.0 * math.asin(math.sqrt(min(1.0, max(0.0, h))))
    return math.hypot(horizontal, dep2 - dep1)


def event_locations(events):
    result = {}
    for event in events:
        if len(event.parts) < 13 or event.parts[12] in result:
            raise ValueError('Differential preparation requires unique, persistent event IDs.')
        result[event.parts[12]] = tuple(map(float, event.parts[7:10]))
    return result


def quality_snapshot(source):
    events = read_src_rec(Path(source), allow_differential=True)
    result = []
    keys = set()
    for event in events:
        for pick in event.picks:
            key = (event.parts[12], pick[2], pick[6])
            if key in keys:
                raise ValueError('Duplicate event/station/phase readings need review before differential preparation.')
            keys.add(key)
            result.append({'event': key[0], 'station': key[1], 'phase': key[2],
                           'weight': float(pick[8]) if len(pick) == 9 else None})
    return result


def as_srcrec(events):
    from pytomoatt.src_rec import SrcRec, setup_rec_points_dd
    sr = SrcRec(fname='tomogui_observed')
    sources, picks = [], []
    for index, event in enumerate(events):
        p = event.parts
        origin = datetime(*map(int, p[1:6])) + timedelta(seconds=float(p[6]))
        sources.append([origin, *map(float, p[7:11]), len(event.picks), p[12], float(p[13]) if len(p) > 13 else 1.0])
        for pick in event.picks:
            picks.append([index, int(pick[1]), pick[2], *map(float, pick[3:6]), pick[6], float(pick[7]), float(pick[8]) if len(pick) > 8 else 1.0])
    sr.src_points = pd.DataFrame(sources, columns=['origin_time','evla','evlo','evdp','mag','num_rec','event_id','weight'])
    sr.src_points.index.name = 'src_index'
    sr.rec_points = pd.DataFrame(picks, columns=['src_index','rec_index','staname','stla','stlo','stel','phase','tt','weight'])
    if sr.rec_points.duplicated(['src_index','staname','phase']).any():
        raise ValueError('Duplicate event/station/phase observations cannot be paired unambiguously.')
    for kind in ('cs','cr'):
        setattr(sr, 'rec_points_' + kind, pd.DataFrame(columns=setup_rec_points_dd(kind)[0]))
    sr.update()
    return sr


def generate_pairs(observed, reference, eligibility_catalog, quality, options, *, include_cs=True):
    """Keep all absolute rows; only eligible subsets contribute differential rows."""
    options = settings(options)
    observed_events = read_src_rec(Path(observed), allow_differential=True)
    original = event_locations(read_src_rec(Path(reference), allow_differential=True))
    baseline = event_locations(read_src_rec(Path(eligibility_catalog), allow_differential=True))
    latest = event_locations(observed_events)
    if not set(latest) <= set(original) or not set(latest) <= set(baseline):
        raise ValueError('Observed catalog contains event IDs absent from the frozen baseline/reference.')
    shifts = {key: distance_3d(original[key], baseline[key]) for key in latest}
    eligible = {key for key, value in shifts.items() if value < options['cs_shift_km']}
    quality = {(row['event'],row['station'],row['phase']): row['weight'] for row in quality}
    sr = as_srcrec(observed_events)
    ids = sr.src_points['event_id'].to_dict()
    cs_count = cr_count = 0
    if include_cs and options['use_cs']:
        selected = sr.copy()
        selected.rec_points = selected.rec_points[selected.rec_points['src_index'].map(ids).isin(eligible)].copy()
        potential = sum(n * (n - 1) // 2 for n in selected.rec_points.groupby(['src_index','phase']).size())
        if potential > options['max_pairs'] * 10:
            raise ValueError('CS candidate set exceeds the safe preview budget. Reduce the catalog or increase the explicit pair limit.')
        if not selected.rec_points.empty:
            selected.calc_distaz()
            before_ids = selected.src_points['event_id'].to_dict()
            directions = {(before_ids[r.src_index], r.staname, r.phase): r.baz
                          for r in selected.rec_points.itertuples()}
            # 0.2.8 lacks same_phase and compares raw azimuths, missing pairs
            # across north. Reuse its CS builder, then apply the circular test
            # ourselves; the same path gives consistent results in 0.3.x.
            selected.generate_double_difference(type='cs', max_azi_gap=361,
                max_dist_gap=1e9, recalc_baz=False)
            local_ids = selected.src_points['event_id'].to_dict()
            keep = []
            for row in selected.rec_points_cs.itertuples():
                phase = row.phase.rsplit(',', 1)[0]
                event_id = local_ids[row.src_index]
                a = directions[(event_id, row.staname1, phase)]
                b = directions[(event_id, row.staname2, phase)]
                keep.append(abs((a - b + 180) % 360 - 180) < options['cs_azimuth_deg'])
            selected.rec_points_cs = selected.rec_points_cs.loc[keep].copy()
            reverse_ids = {name:index for index,name in ids.items()}
            selected.rec_points_cs['src_index'] = selected.rec_points_cs['src_index'].map(local_ids).map(reverse_ids)
            sr.rec_points_cs = selected.rec_points_cs.copy()
        cs_count = len(sr.rec_points_cs)
    unknown = 0
    if options['use_cr']:
        from pytomoatt.src_rec import setup_rec_points_dd
        rows = []
        eligible_rows = []
        for index, pick in sr.rec_points.iterrows():
            q = quality.get((ids[pick.src_index], pick.staname, pick.phase))
            if q is None:
                unknown += 1
                accept = bool(options['allow_ungraded'])
            else:
                accept = math.isclose(q, options['cr_top_weight'], rel_tol=1e-7, abs_tol=1e-9)
            if accept:
                eligible_rows.append(index)
        for (_, phase), group in sr.rec_points.loc[eligible_rows].groupby(['staname','phase'], sort=True):
            records = list(group.itertuples(index=False))
            locations = [latest[ids[r.src_index]] for r in records]
            if len(locations) < 2:
                continue
            coordinates = np.asarray(locations)
            lat, lon = np.radians(coordinates[:,0]), np.radians(coordinates[:,1])
            # Surface chord + depth is a lower bound on surface-distance + depth.
            # The final exact filter below uses the documented 3D metric.
            points = np.column_stack((6371*np.cos(lat)*np.cos(lon),6371*np.cos(lat)*np.sin(lon),6371*np.sin(lat),coordinates[:,2]))
            tree = cKDTree(points)
            potential = (sum(tree.query_ball_point(points, options['cr_separation_km'], return_length=True)) - len(points)) // 2
            if potential > options['max_pairs'] * 10:
                raise ValueError('CR candidate set exceeds the safe preview budget. Reduce the catalog or increase the explicit pair limit.')
            for i,j in sorted(tree.query_pairs(options['cr_separation_km'])):
                a,b = records[i],records[j]
                if a.src_index == b.src_index or distance_3d(locations[i],locations[j]) >= options['cr_separation_km']:
                    continue
                rows.append([a.src_index,a.rec_index,a.staname,a.stla,a.stlo,a.stel,
                             b.src_index,ids[b.src_index],*locations[j],phase+',cr',a.tt-b.tt,(a.weight+b.weight)/2])
                if len(rows) + cs_count > options['max_pairs']:
                    raise ValueError('Differential pair limit exceeded; review selection or raise the explicit limit.')
        sr.rec_points_cr = pd.DataFrame(rows, columns=setup_rec_points_dd('cr')[0])
        cr_count = len(rows)
    if cs_count + cr_count > options['max_pairs']:
        raise ValueError('Differential pair limit exceeded.')
    sr.update()
    errors = []
    if include_cs and options['use_cs'] and not cs_count:
        errors.append('No CS pairs meet the azimuth and baseline event-shift criteria.')
    if options['use_cr'] and not cr_count:
        errors.append('No CR pairs meet both-pick top-quality and 3D separation criteria. Review the saved quality weights; ungraded picks are not assumed top quality.')
    return sr, {'absolute': len(sr.rec_points), 'cs': cs_count, 'cr': cr_count,
                'eligible_cs_events': len(eligible), 'events': len(latest), 'ungraded_picks': unknown,
                'event_shifts_km': shifts, 'ready': not errors, 'errors': errors,
                'settings': options, 'distance_definition': 'hypot(great-circle epicentral km, depth difference km)',
                'cs_shift_reference': 'original baseline input to final baseline observed catalog'}


def observed_output(folder, step):
    params = yaml.safe_load((folder / f'3_input_params/input_params_joint_step{step}.yaml').read_text())
    output = folder / params['output_setting']['output_dir']
    files = list(output.glob('*_obs.dat'))
    if not files:
        raise ValueError(f'Missing completed observed catalog for step {step}: {output}')
    def order(path):
        import re
        return tuple(map(int, re.findall(r'\d+', path.name)))
    return max(files, key=order)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='ascii')


def complete_baseline(folder):
    started = json.loads((folder / 'tomogui_completed.json').read_text()).get('started', 0) if (folder / 'tomogui_completed.json').exists() else 0
    params = yaml.safe_load((folder / '3_input_params/input_params_joint_step2.yaml').read_text())
    model = folder / params['output_setting']['output_dir'] / 'final_model.h5'
    observed = observed_output(folder, 3)
    if not model.is_file():
        raise ValueError('Baseline joint inversion has no final model.')
    for step in (1,2,3):
        if observed_output(folder, step).stat().st_mtime < started - 2:
            raise ValueError('An observed handoff predates this execution.')
    if model.stat().st_mtime < started - 2:
        raise ValueError('The model predates this execution.')
    read_src_rec(observed, allow_differential=True)
    write_json(folder / 'tomogui_completed.json', {'state': 'completed', 'started': started, 'model': str(model.resolve()), 'observed': str(observed.resolve()),
        'model_sha256': sha256(model), 'observed_sha256': sha256(observed)})


def prepare_step(folder, step):
    config = json.loads((folder / 'tomogui_differential.json').read_text())
    for filename, digest in config.get('frozen_hashes', {}).items():
        if sha256(filename) != digest:
            raise ValueError('Frozen baseline inputs changed; prepare a new differential experiment.')
    observed = Path(config['baseline_observed']) if step == 1 else observed_output(folder, step-1)
    sr, report = generate_pairs(observed, config['reference'], config.get('eligibility_catalog', config['baseline_observed']), config['quality'],
                                config['settings'], include_cs=step == 2)
    if not report['ready']:
        raise ValueError('; '.join(report['errors']))
    target = folder / f'1_src_rec_files/step{step}.dat'
    temporary = target.with_suffix('.tmp')
    sr.write(str(temporary))
    temporary.replace(target)
    write_json(folder / f'pair_report_step{step}.json', {**report, 'observed_input': str(observed), 'input_sha256': sha256(observed)})


def install_runner(folder, *, refinement=False):
    """Insert explicit observed-data preparation between completed MPI stages."""
    for filename in ('run_tomogui_local.sh', 'run_tomogui_slurm.sh'):
        path = folder / filename
        lines = []
        for line in path.read_text().splitlines():
            if refinement and '"$TOMOATT_BIN" -i 3_input_params/input_params_joint_step' in line:
                step = int(line.split('joint_step')[1][0])
                lines.append(f'"$PYTHON_BIN" -m tomogui.differential prepare-step --folder . --step {step}')
            if line.startswith('if [[ -f plot_output.py'):
                lines.append('"$PYTHON_BIN" -m tomogui.differential complete-baseline --folder .')
            lines.append(line)
            if line.startswith('export PATH='):
                lines.append('"$PYTHON_BIN" -m tomogui.differential begin --folder .')
        path.write_text('\n'.join(lines) + '\n', encoding='ascii')


def set_data_flags(folder, *, refinement=False):
    """A file can contain all data families; each update uses only its own set."""
    for step in (1,2,3):
        path = folder / f'3_input_params/input_params_joint_step{step}.yaml'
        params = yaml.safe_load(path.read_text())
        for section in ('model_update','relocation'):
            if section not in params:
                continue
            target = params[section]
            target.setdefault('abs_time', {})['use_abs_time'] = True
            target.setdefault('cs_dif_time', {})['use_cs_time'] = bool(refinement and section == 'model_update' and step == 2)
            target.setdefault('cr_dif_time', {})['use_cr_time'] = bool(refinement and section == 'relocation')
        path.write_text(yaml.safe_dump(params, sort_keys=False), encoding='ascii')


def clone_inputs(source, destination):
    import shutil
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('1_src_rec_files','2_models','3_input_params'):
        shutil.copytree(source / name, destination / name)
    for pattern in ('*.py','*.sh'):
        for path in source.glob(pattern):
            shutil.copy2(path, destination / path.name)


def phase_baseline(data, phase):
    folder = Path(data['root']) / f'inversion_{phase}'
    marker = folder / 'tomogui_completed.json'
    if not marker.is_file():
        raise ValueError(f'Complete all three absolute-only stages for {phase} first (missing completion marker).')
    result = json.loads(marker.read_text())
    if result.get('state') != 'completed':
        raise ValueError('The baseline execution did not complete.')
    if result.get('started', 0) < data.get('selected_at', {}).get(phase, 0):
        raise ValueError('The baseline predates the selected Kd coefficient. Complete the baseline again.')
    for key in ('model','observed'):
        if sha256(result[key]) != result[key + '_sha256']:
            raise ValueError('The baseline changed after completion. Complete/record the baseline again.')
    return result


def preview(ws, payload):
    from .runs import manifest
    _, data = manifest(ws)
    if not data.get('validated') or any(p not in data.get('selected_kd', {}) for p in data['phases']):
        raise ValueError('Select Kd and accept baseline checkerboard recovery before differential refinement.')
    options = settings(payload.get('settings', data['settings'].get('differential')))
    if not options['enabled']:
        raise ValueError('Enable differential refinement in the run settings.')
    reports = {}
    for phase in data['phases']:
        baseline = phase_baseline(data, phase)
        reference = Path(data['root']) / f'dataset/reference_{phase}.dat'
        quality = json.loads((Path(data['root']) / f'dataset/quality_{phase}.json').read_text())
        _, reports[phase] = generate_pairs(baseline['observed'], reference, baseline['observed'], quality, options)
        reports[phase].pop('event_shifts_km', None)
    return {'phases': reports, 'ready': all(report['ready'] for report in reports.values()), 'settings': options}


def prepare(ws, payload):
    import shutil
    import uuid
    from .runs import manifest, runner, save
    from .resolution import stage_resolution
    path, data = manifest(ws)
    plan = preview(ws, payload)
    if not plan['ready']:
        raise ValueError('; '.join(error for report in plan['phases'].values() for error in report['errors']))
    root = Path(data['root']) / 'differential' / uuid.uuid4().hex[:12]
    root.mkdir(parents=True)
    differential = {'root': str(root), 'settings': plan['settings'], 'phases': {}, 'validated': False, 'pair_preview': plan}
    for phase in data['phases']:
        baseline = phase_baseline(data, phase)
        folder = root / f'refinement_{phase}'
        clone_inputs(Path(data['root']) / f'inversion_{phase}', folder)
        for source, name in [(baseline['model'],'2_models/baseline_final.h5'),
                             (baseline['observed'],'1_src_rec_files/baseline_observed.dat'),
                             (Path(data['root']) / f'dataset/reference_{phase}.dat','1_src_rec_files/reference.dat')]:
            shutil.copy2(source, folder / name)
        config = {'settings': plan['settings'], 'baseline_observed': str(folder / '1_src_rec_files/baseline_observed.dat'),
                  'reference': str(folder / '1_src_rec_files/reference.dat'),
                  'quality': json.loads((Path(data['root']) / f'dataset/quality_{phase}.json').read_text()),
                  'baseline_model_sha256': baseline['model_sha256']}
        config['frozen_hashes'] = {config[key]: sha256(config[key]) for key in ('reference','baseline_observed')}
        write_json(folder / 'tomogui_differential.json', config)
        for step in (1,2,3):
            param = folder / f'3_input_params/input_params_joint_step{step}.yaml'
            params = yaml.safe_load(param.read_text())
            params['source']['src_rec_file'] = f'1_src_rec_files/step{step}.dat'
            if step != 3:
                params['model']['init_model_path'] = '2_models/baseline_final.h5'
            param.write_text(yaml.safe_dump(params, sort_keys=False), encoding='ascii')
        set_data_flags(folder, refinement=True)
        # The option to omit either family must apply to the solver flags too.
        for param in (folder / '3_input_params').glob('*joint*.yaml'):
            params = yaml.safe_load(param.read_text())
            if not plan['settings']['use_cs'] and 'model_update' in params:
                params['model_update']['cs_dif_time']['use_cs_time'] = False
            if not plan['settings']['use_cr'] and 'relocation' in params:
                params['relocation']['cr_dif_time']['use_cr_time'] = False
            param.write_text(yaml.safe_dump(params, sort_keys=False), encoding='ascii')
        runner(folder, 8, False, {}, dual=True)
        install_runner(folder, refinement=True)
        differential['phases'][phase] = str(folder)
    differential.update(stage_resolution(ws, data, differential))
    differential['input_hashes'] = {str(p): sha256(p) for p in root.rglob('*') if p.is_file() and (
        p.suffix in ('.yaml', '.py', '.sh', '.dat') or p.name == 'tomogui_differential.json' or
        (p.suffix == '.h5' and p.parent.name == '2_models'))}
    data['differential'] = differential
    save(path, data)
    return {'root': str(root), 'differential': differential, 'message': 'Differential refinement and paired resolution cases staged.'}


def execute(ws, payload):
    import subprocess
    from .runs import manifest, save
    from .worker import run
    path, data = manifest(ws)
    dd = data.get('differential')
    if not dd:
        raise ValueError('Prepare differential refinement first.')
    check_settings(dd, payload)
    check_inputs(dd)
    stage = payload.get('stage', 'resolution')
    if stage not in {'resolution','refinement'}:
        raise ValueError('Choose differential resolution or refinement.')
    if stage == 'refinement':
        if not dd.get('validated'):
            raise ValueError('Score and accept the complete differential resolution suite first.')
        for filename, digest in dd['accepted_outputs'].items():
            if sha256(filename) != digest:
                raise ValueError('Resolution outputs changed after acceptance; score and accept again.')
    profile = payload.get('profile', 'slurm')
    if profile not in {'local','slurm'}:
        raise ValueError('Choose local MPI or Slurm.')
    signals = {}
    if stage == 'resolution':
        dd['validated'] = False
        for key in ('scores', 'comparisons', 'scored_outputs', 'accepted_outputs'):
            dd.pop(key, None)
        save(path, data)
        pending = [Path(row['run_dir']) for row in dd['cases']]
    else:
        pending = [Path(folder) for folder in dd['phases'].values()]
    for folder in pending:
        write_json(folder / 'tomogui_completed.json', {'state': 'queued', 'started': time.time()})
    if stage == 'resolution':
        for key, folder in dd['signals'].items():
            folder = Path(folder)
            if profile == 'local':
                run(['bash', folder / 'run_tomogui_local.sh'], folder)
            else:
                job = subprocess.check_output(['sbatch','--parsable',folder / 'run_tomogui_slurm.sh'], cwd=folder, text=True).strip().split(';')[0]
                if not job.isdigit():
                    raise ValueError('Invalid signal job ID returned by Slurm.')
                signals[key] = job
        folders = [(row['run_dir'], row['signal_key']) for row in dd['cases']]
    else:
        folders = [(folder, None) for folder in dd['phases'].values()]
    for text, key in folders:
        folder = Path(text)
        if profile == 'local':
            run(['bash', folder / 'run_tomogui_local.sh'], folder)
        else:
            dependency = ['--dependency=afterok:' + signals[key]] if key in signals else []
            run(['sbatch', *dependency, folder / 'run_tomogui_slurm.sh'], folder)
    return {'message': 'Stage completed.' if profile == 'local' else 'Jobs submitted; inspect actual outputs for completion.'}


def accept(ws, payload):
    from .runs import manifest, save
    path, data = manifest(ws)
    dd = data.get('differential') or {}
    check_settings(dd, payload)
    check_inputs(dd)
    if len(dd.get('scores', [])) != len(dd.get('cases', [])) or not dd.get('scores'):
        raise ValueError('Complete and score every paired resolution case first.')
    for filename, digest in dd.get('scored_outputs', {}).items():
        if sha256(filename) != digest:
            raise ValueError('Resolution outputs changed after scoring; score again before acceptance.')
    if not dd.get('scored_outputs'):
        raise ValueError('Score all completed resolution cases before acceptance.')
    accepted = dict(dd.get('input_hashes', {}))
    accepted.update(dd['scored_outputs'])
    for row in dd['cases']:
        marker = Path(row['run_dir']) / 'tomogui_completed.json'
        if not marker.exists():
            raise ValueError('All three resolution stages must finish before acceptance.')
        result = json.loads(marker.read_text())
        if result.get('state') != 'completed':
            raise ValueError('All three resolution stages must finish before acceptance.')
        for name in ('model','observed'):
            if sha256(result[name]) != result[name + '_sha256']:
                raise ValueError('A completed resolution output was modified.')
            accepted[result[name]] = result[name + '_sha256']
    dd['accepted_outputs'] = accepted
    dd['validated'] = True
    save(path, data)
    return {'message': 'Differential resolution accepted; actual refinement is enabled.'}


def check_inputs(dd):
    for filename, digest in dd.get('input_hashes', {}).items():
        if sha256(filename) != digest:
            raise ValueError('Staged differential inputs changed. Prepare a new experiment before executing or accepting recovery.')


def check_settings(dd, payload):
    if 'settings' in payload and settings(payload['settings']) != dd.get('settings'):
        raise ValueError('Settings changed. Preview and prepare a new differential experiment to apply them.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['prepare-step','complete-baseline','begin'])
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--step', type=int, choices=[1,2,3])
    args = parser.parse_args()
    folder = args.folder.resolve()
    if args.operation == 'begin':
        write_json(folder / 'tomogui_completed.json', {'state': 'running', 'started': time.time()})
    elif args.operation == 'complete-baseline':
        complete_baseline(folder)
    else:
        if args.step is None:
            parser.error('--step is required')
        prepare_step(folder, args.step)


if __name__ == '__main__':
    main()
