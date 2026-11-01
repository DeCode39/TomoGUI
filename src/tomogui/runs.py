"""Stage reproducible runs using the maintained preparation and scoring helpers."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import yaml

from .defaults import current_defaults, grid_estimate, resolution_defaults
from .workspace import load_module


def patch_scalar(path: Path, name: str, value) -> None:
    text = path.read_text(encoding='utf-8')
    text, count = re.subn(rf'(?m)^(\s*{re.escape(name)}:\s*)[^#\n]+', lambda m: m[1] + str(value) + ' ', text)
    if not count:
        raise ValueError(f'{name} was not found in {path.name}.')
    path.write_text(text, encoding='utf-8')


def manifest(ws, path=None):
    state = ws.read_state()
    run = state.get('run')
    if not run:
        raise ValueError('Generate a run first.')
    path = ws.path(path or run['manifest'])
    if path.name != 'tomogui_run.json' or not path.is_relative_to(ws.state_dir / 'runs'):
        raise ValueError('Select a TomoGUI-generated run manifest.')
    return path, json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def runner(folder: Path, tasks: int, synthetic: bool, config: dict, dual: bool = False,
           signal_only: bool = False, inversion_only: bool = False, shared_signal: str = ''):
    """Keep Slurm directives from the copied template; avoid the legacy multi-size loop."""
    import shlex
    template = folder / 'run_this_example_slurm.sh'
    header = '\n'.join(line for line in template.read_text(encoding='utf-8').splitlines() if line.startswith('#SBATCH'))
    options = ' '.join(shlex.quote(str(s)) for s in ['--n-pert', ','.join(str(config[k]) for k in ('n_pert_x','n_pert_y','n_pert_z')),
                          '--pert-vel', config['pert_vel'], '--pert-ani', config['pert_ani'], '--ani-dir', config['ani_dir']])
    prepare = '' if dual or inversion_only else 'python prepare_input_files.py' + (' ' + options if synthetic else '') + '\n'
    files = ['input_params_signal.yaml', 'input_params_inv.yaml'] if synthetic else ['input_params_joint_step1.yaml','input_params_joint_step2.yaml','input_params_joint_step3.yaml']
    if signal_only:
        files = ['input_params_signal.yaml']
    if inversion_only:
        files = ['input_params_inv.yaml']
    for slurm in (False, True):
        launch = 'srun --mpi="${SLURM_MPI_TYPE:-pmix}"' if slurm else '${MPI_LAUNCHER:-mpirun}'
        lines = ['#!/usr/bin/env bash'] + ([header] if slurm else [])
        lines += ['set -eo pipefail', 'cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"' if slurm else 'cd "$(dirname "$0")"']
        lines += ['# Use the generating environment; override TOMOGUI_PYTHON on another machine.',
                  'PYTHON_BIN="${TOMOGUI_PYTHON:-' + sys.executable.replace('"', '\\"') + '}"',
                  'export PATH="$(dirname "$PYTHON_BIN"):$PATH"']
        lines += ['TOMOATT_BIN="${TOMOATT_BIN:-$HOME/TomoATT/build/bin/TOMOATT}"',
                  f'NTASKS="${{SLURM_NTASKS:-{tasks}}}"', prepare]
        if shared_signal:
            signal_path = shlex.quote(shared_signal)
            lines += [f'for attempt in {{1..60}}; do [[ -s {signal_path} ]] && break; sleep 2; done',
                      f'test -s {signal_path}']
        for index, name in enumerate(files):
            lines.append(f'{launch} -n "$NTASKS" "$TOMOATT_BIN" -i 3_input_params/{name}')
            if name == 'input_params_signal.yaml':
                lines += ['python assign_gaussian_noise.py',
                          'test -s OUTPUT_FILES/OUTPUT_FILES_signal/src_rec_file_forward_noisy.dat']
        if not signal_only:
            lines += ['if [[ -f plot_output.py ]]; then python plot_output.py || echo "Plotting failed; inspect logs."; fi']
        destination = folder / ('run_tomogui_slurm.sh' if slurm else 'run_tomogui_local.sh')
        destination.write_text('\n'.join(lines) + '\n', encoding='ascii', newline='\n')


def preview(ws, payload):
    """Resolve the exact settings and layout without creating files or models."""
    state = ws.read_state()
    region = state.get('region')
    if not region:
        raise ValueError('Apply a filtered region first.')
    phases = payload.get('phases', ['P','S'])
    if not phases or any(p not in {'P','S'} for p in phases) or len(set(phases)) != len(phases):
        raise ValueError('Select unique phases from P and S.')
    for phase in phases:
        if phase not in region['summary']['phases']:
            raise ValueError(f'The final src_rec does not contain {phase} picks.')
    defaults = current_defaults(ws.root)
    kds = [float(v) for v in payload.get('kd_candidates', defaults['kd_candidates'])]
    if not kds or len(kds) > 32 or len(set(kds)) != len(kds) or any(not math.isfinite(v) or not 0 <= v <= .95 for v in kds):
        raise ValueError('Choose up to 32 unique Kdensity values between 0 and 0.95.')
    prefix = re.sub(r'[^A-Za-z0-9_-]', '_', payload.get('name', 'TomoGUI'))[:60] or 'TomoGUI'
    workflow = defaults['kd_workflow']
    model_settings = dict(payload.get('model') or (state.get('model') or {}).get('settings') or {})
    model_settings['grid'] = payload.get('grid', model_settings.get('grid', {}))
    if model_settings.get('kind') == 'initial':
        model_settings['kind'] = 'checkerboard'
    if model_settings.get('kind', 'checkerboard') not in {'checkerboard', 'dual_layer'}:
        raise ValueError('Choose a checkerboard or the existing dual-layer validation preset.')
    if model_settings.get('kind', 'checkerboard') == 'checkerboard':
        config = {**defaults['checkerboard'], **model_settings.get('checkerboard', {})}
        for key in ('n_pert_x', 'n_pert_y', 'n_pert_z'):
            value = config[key]
            if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != float(value) or not 1 <= int(value) <= 100:
                raise ValueError('Checkerboard counts must be integers from 1 to 100.')
        for key in ('pert_vel', 'pert_ani', 'ani_dir'):
            value = float(config[key])
            if not math.isfinite(value) or (key != 'ani_dir' and not 0 <= value < 1):
                raise ValueError('Perturbation strengths must be finite fractions in [0, 1).')
    grids = {}
    entries = ['dataset/', 'tomogui_run.json']
    for phase in phases:
        metadata_path = Path(region['metadata_path']).parent / phase / 'region_metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else region['metadata']
        grids[phase] = grid_estimate(ws.root, metadata, model_settings['grid'])
        if math.prod(grids[phase]['n_rtp']) > 20_000_000:
            raise ValueError('The model exceeds 20 million nodes. Increase grid spacing before generation.')
        paper = ws.root / f'RESULTS/JpGU/templates/checkerboard_{phase}_paper_base/prepare_input_files.py'
        if not paper.is_file():
            raise ValueError(f'Missing existing sinusoidal model helper: {paper}')
        for kind in ('checkerboard', 'inversion'):
            entries.extend([f'{kind}_{phase}/', '  1_src_rec_files/src_rec_config.dat', '  2_models/', '  3_input_params/',
                            '  run_tomogui_local.sh', '  run_tomogui_slurm.sh'])
        entries.append(f'resolution_kd_{phase}/')
        for wave_tag in workflow['waves']:
            entries.append(f'  {wave_tag}/signal/  (shared noisy observations)')
            entries.extend(f'  {wave_tag}/kd{str(kd).replace(".", "p")}/  (inversion inputs + runner)' for kd in kds)
    return {'prefix': prefix, 'parent': str(ws.state_dir / 'runs'), 'destination': prefix + '_<timestamp>',
            'phases': phases, 'grids': grids, 'entries': entries, 'kd_candidates': kds, 'kd_workflow': workflow,
            'signal_jobs': len(phases) * len(workflow['waves']), 'sweep_jobs': len(phases) * len(workflow['waves']) * len(kds),
            'keep_pick_weights': payload.get('keep_pick_weights', state.get('keep_pick_weights', defaults['keep_pick_weights'])),
            'heavy_outputs': bool(payload.get('heavy_outputs')), 'model_settings': model_settings,
            'sequence': defaults['sequence']}


def generate(ws, payload):
    from .worker import model_preview
    plan = preview(ws, payload)
    state = ws.read_state()
    region = state['region']
    phases = plan['phases']; kds = plan['kd_candidates']
    defaults = current_defaults(ws.root)
    folder = ws.artifact('runs')
    prep = load_module(ws.root / 'Scripts/prepare_files.py')
    prep.RESULTS_DIR = folder
    prefix = plan['prefix']
    args = [str(ws.root / 'Scripts/prepare_files.py'), '--region-metadata', region['metadata_path'], '--phases', ','.join(phases), '--prefix', prefix]
    if payload.get('keep_pick_weights', state.get('keep_pick_weights', defaults['keep_pick_weights'])):
        args += ['--include-weights']
    if payload.get('heavy_outputs'):
        args += ['--enable-azcov-outputs']
    old_args = sys.argv
    try:
        sys.argv = args
        prep.main()
    finally:
        sys.argv = old_args
    target = next(p for p in folder.iterdir() if p.is_dir())
    record = {'root': str(target), 'phases': phases, 'sequence': defaults['sequence'], 'selected_kd': {},
              'validated': False, 'settings': payload, 'cases': [], 'grids': {}, 'signals': {}}
    model_settings = plan['model_settings']
    record['kd_workflow'] = plan['kd_workflow']
    for phase in phases:
        model_settings['phase'] = phase
        model = model_preview(ws, model_settings)['model']
        from .viewer import Viewer
        import h5py
        coordinates = Viewer(ws).load(model['initial']).coords
        grid = model['grid']; record['grids'][phase] = grid
        config = {**defaults['checkerboard'], **model_settings.get('checkerboard', {})}
        name = '_'.join(str(n) for n in grid['n_rtp'])
        for kind in ('checkerboard', 'inversion'):
            case = target / f'{kind}_{phase}'
            for path in (case / '3_input_params').glob('*.yaml'):
                prep._update_yaml_domain(path, grid['domain']['min_max_dep'], grid['domain']['min_max_lat'], grid['domain']['min_max_lon'], grid['n_rtp'])
                if re.search(r'(?m)^\s*n_inv_dep_lat_lon:', path.read_text(encoding='utf-8')):
                    patch_scalar(path, 'n_inv_dep_lat_lon', grid['n_inv_dep_lat_lon'])
            for script in case.glob('*.py'):
                text = script.read_text(encoding='utf-8')
                text = re.sub(r'(model_(?:init|ckb)_N)\d+_\d+_\d+', lambda m: m[1]+name, text)
                script.write_text(text, encoding='utf-8')
            (case / '2_models').mkdir(exist_ok=True)
            shutil.copy2(model['initial'], case / f'2_models/model_init_N{name}.h5')
            # PyTomoATT versions can omit coordinates from the native HDF5.
            # The existing paper helper expects these physical 1D axes inline.
            with h5py.File(case / f'2_models/model_init_N{name}.h5', 'a') as handle:
                for axis, key in [('x', 'lon'), ('y', 'lat'), ('z', 'dep')]:
                    if key not in handle:
                        handle.create_dataset(key, data=coordinates[axis])
            if kind == 'checkerboard':
                shutil.copy2(model['path'], case / f'2_models/model_ckb_N{name}.h5')
            tasks = min(56, max(1, prep._count_unique_stations(case / '1_src_rec_files/src_rec_config.dat')))
            runner(case, tasks, kind == 'checkerboard', config, model_settings.get('kind') == 'dual_layer' and kind == 'checkerboard')
        template = target / f'checkerboard_{phase}'
        sweep_helper = load_module(ws.root / 'RESULTS/resolution_kd_sweep/prepare_resolution_kd_sweep.py')
        paper_path = ws.root / f'RESULTS/JpGU/templates/checkerboard_{phase}_paper_base/prepare_input_files.py'
        for wave_tag, wave in plan['kd_workflow']['waves'].items():
            signal_folder = target / f'resolution_kd_{phase}' / wave_tag / 'signal'
            shutil.copytree(template, signal_folder)
            # Copy and patch the existing sinusoidal helper, preserving its 3%
            # amplitude, buffer, azimuths, and wavelength conventions.
            paper_copy = signal_folder / 'prepare_input_files.py'
            shutil.copy2(paper_path, paper_copy)
            sweep_helper.patch_prepare_input(paper_copy, wave)
            text = paper_copy.read_text(encoding='utf-8').replace('model_ckb_N55_61_61.h5', f'model_ckb_N{name}.h5')
            paper_copy.write_text(text, encoding='utf-8')
            paper = load_module(paper_copy)
            paper.build_paper_checkerboard(signal_folder / f'2_models/model_init_N{name}.h5', signal_folder / f'2_models/model_ckb_N{name}.h5')
            sweep_helper.patch_noise_seed(signal_folder / 'assign_gaussian_noise.py', int(wave['seed']))
            runner(signal_folder, tasks, True, config, dual=True, signal_only=True)
            signal_key = phase + '_' + wave_tag
            record['signals'][signal_key] = str(signal_folder)
            for kd in kds:
                tag = 'kd' + str(kd).replace('.', 'p')
                case = signal_folder.parent / tag
                shutil.copytree(signal_folder, case)
                patch_scalar(case / '3_input_params/input_params_inv.yaml', 'Kdensity_coe', kd)
                shared_signal = '../signal/OUTPUT_FILES/OUTPUT_FILES_signal/src_rec_file_forward_noisy.dat'
                patch_scalar(case / '3_input_params/input_params_inv.yaml', 'src_rec_file', shared_signal)
                runner(case, tasks, True, config, inversion_only=True, shared_signal=shared_signal)
                output = yaml.safe_load((case / '3_input_params/input_params_inv.yaml').read_text(encoding='utf-8'))['output_setting']['output_dir']
                record['cases'].append({'phase': phase, 'signal_key': signal_key, 'kdensity_coe': kd, 'run_tag': signal_key+'_'+tag,
                                    'run_dir': str(case), 'initial_model': str(case / f'2_models/model_init_N{name}.h5'),
                                    'true_model': str(case / f'2_models/model_ckb_N{name}.h5'),
                                    'recovered_model': str(case / output / 'final_model.h5'),
                                    'wave_tag': wave_tag, 'wave_weight': wave['weight'],
                                    **{key: wave[key] for key in ('velocity_period_deg', 'anisotropy_period_deg', 'depth_period_km')}})
    path = target / 'tomogui_run.json'; save(path, record)
    return {'root': str(target), 'state_updates': {'run': {'root': str(target), 'manifest': str(path)}}}


def execute(ws, payload):
    from .worker import run
    path, data = manifest(ws)
    stage = payload.get('stage', 'resolution_kd')
    if stage not in data['sequence']:
        raise ValueError('Unknown workflow stage.')
    if stage != 'resolution_kd' and not all(p in data['selected_kd'] for p in data['phases']):
        raise ValueError('Score the resolution runs and select a Kd coefficient for each phase first.')
    if stage == 'inversion' and not data['validated']:
        raise ValueError('Inspect and accept the checkerboard recovery before actual inversion.')
    folders = [Path(row['run_dir']) for row in data['cases']] if stage == 'resolution_kd' else [Path(data['root']) / f'{stage}_{p}' for p in data['phases']]
    profile = payload.get('profile', 'slurm')
    if profile not in {'local', 'slurm'}:
        raise ValueError('Use local MPI or Slurm from the environment running TomoGUI.')
    signal_jobs = {}
    if stage == 'resolution_kd':
        for phase, folder_text in data.get('signals', {}).items():
            signal_folder = Path(folder_text)
            if profile == 'local':
                run(['bash', signal_folder / 'run_tomogui_local.sh'], signal_folder)
            else:
                job_id = subprocess.check_output(['sbatch','--parsable',str(signal_folder / 'run_tomogui_slurm.sh')], cwd=signal_folder, text=True).strip().split(';')[0]
                if not job_id.isdigit():
                    raise ValueError('Slurm returned an invalid signal job ID.')
                signal_jobs[phase] = job_id
                print('Submitted shared signal job:', phase, job_id, flush=True)
    for folder in folders:
        script = folder / ('run_tomogui_local.sh' if profile == 'local' else 'run_tomogui_slurm.sh')
        if profile == 'local':
            run(['bash', script], folder)
        else:
            phase = next((row.get('signal_key', row['phase']) for row in data['cases'] if Path(row['run_dir']) == folder), None)
            dependencies = ['--dependency=afterok:'+signal_jobs[phase]] if phase in signal_jobs else []
            run(['sbatch', *dependencies, script], folder)
    return {'message': 'Slurm submissions sent; completion is determined from output files.' if profile == 'slurm' else 'Stage completed.', 'stage': stage}


def score(ws, payload):
    path, data = manifest(ws)
    scorer = load_module(ws.root / 'RESULTS/resolution_kd_sweep/score_resolution_kd_sweep.py')
    scorer.ROOT = Path(data['root'])
    # Preserve the scorer's metric, while using the existing viewer's mapping
    # for native TomoATT files that keep coordinates in a sibling YAML.
    from .viewer import Viewer
    import numpy as np
    reader = Viewer(ws)
    def read_scoring_model(model_path, *, coords_from=None):
        volume = reader.load(model_path)
        if volume.coord_labels.get('x') != 'longitude':
            raise ValueError(f'Physical coordinates could not be resolved for {model_path}. Keep its parameter YAML with the model.')
        arrays = {key: np.transpose(value, (2,1,0)) for key,value in volume.fields.items()}
        arrays.update(dep=volume.coords['z'], lat=volume.coords['y'], lon=volume.coords['x'])
        if coords_from is not None:
            for key in ('dep','lat','lon'):
                if arrays[key].shape != coords_from[key].shape or not np.allclose(arrays[key],coords_from[key]):
                    raise ValueError('Resolution scoring requires matching physical grids.')
        return arrays
    scorer.read_model = read_scoring_model
    args = argparse.Namespace(threshold=.7, energy_floor=.02, artifact_weight=10.0,
                             window_depth_km=10.0, window_lat_deg=.5, window_lon_deg=.5)
    scores = []
    for row in data['cases']:
        scoring_row = {key: str(value) for key,value in row.items()}
        # The legacy scorer labels sinusoidal-paper wavelengths. Its metric also
        # works on the current checkerboard; those labels are inapplicable here.
        for key in ('velocity_period_deg', 'anisotropy_period_deg', 'depth_period_km'):
                scoring_row.setdefault(key, 'nan')
        result = scorer.compute_scores(scoring_row, args)
        if result:
            for key in ('velocity_period_deg', 'anisotropy_period_deg', 'depth_period_km'):
                result.pop(key, None)
            result['phase'] = row['phase']
            scores.append(result)
    if not scores:
        raise ValueError('No completed sweep models found. Run or download the resolution outputs first.')
    if any(not math.isfinite(float(row['score'])) for row in scores):
        raise ValueError('Non-finite resolution scores; inspect true/recovered models and signal support.')
    summary = Path(data['root']) / 'resolution_scores.csv'
    scorer.write_csv(summary, scores)
    data['scores'] = scores
    aggregates = []
    for phase in data['phases']:
        expected_waves = {row['wave_tag'] for row in data['cases'] if row['phase'] == phase}
        phase_scores = [row for row in scores if row['phase'] == phase]
        for row in scorer.aggregate(phase_scores):
            row['phase'] = phase
            row['complete'] = row['n_wave_cases'] == len(expected_waves)
            aggregates.append(row)
    data['aggregate_scores'] = aggregates
    save(path, data)
    # Apply a recommendation only after the entire phase sweep has completed.
    for phase in data['phases']:
        rows = [row for row in aggregates if row['phase'] == phase]
        expected_count = len({row['kdensity_coe'] for row in data['cases'] if row['phase'] == phase})
        if len(rows) == expected_count and all(row['complete'] for row in rows):
            best = max(rows, key=lambda row: (row['weighted_score'], -row['kdensity_coe']))
            select_kd(ws, {'phase': phase, 'value': best['kdensity_coe']})
    return {'scores': scores, 'aggregate_scores': aggregates, 'csv': str(summary)}


def select_kd(ws, payload):
    path, data = manifest(ws)
    if payload.get('accept_validation'):
        for phase in data['phases']:
            params = Path(data['root']) / f'checkerboard_{phase}/3_input_params/input_params_inv.yaml'
            output = yaml.safe_load(params.read_text(encoding='utf-8'))['output_setting']['output_dir']
            if not (params.parents[1] / output / 'final_model.h5').exists():
                raise ValueError(f'Missing checkerboard recovery for {phase}.')
            if (params.parents[1] / output / 'final_model.h5').stat().st_mtime < data.get('selected_at',{}).get(phase,0):
                raise ValueError(f'{phase} recovery predates the selected coefficient. Run checkerboard validation again.')
        data['validated'] = True
    else:
        phase = payload['phase']; kd = float(payload['value'])
        if phase not in data['phases'] or not any(row['phase'] == phase and float(row['kdensity_coe']) == kd for row in data.get('scores', [])):
            raise ValueError('Select a coefficient with a completed, scored resolution run.')
        if data.get('aggregate_scores') and not any(row['phase'] == phase and row['kdensity_coe'] == kd and row['complete'] for row in data['aggregate_scores']):
            raise ValueError('Complete all wavelength tests for this coefficient before selecting it.')
        for folder in (Path(data['root']) / f'checkerboard_{phase}', Path(data['root']) / f'inversion_{phase}'):
            for params in (folder / '3_input_params').glob('*.yaml'):
                if re.search(r'(?m)^\s*Kdensity_coe:', params.read_text(encoding='utf-8')):
                    patch_scalar(params, 'Kdensity_coe', kd)
        if data['selected_kd'].get(phase) != kd:
            data.setdefault('selected_at',{})[phase] = time.time()
        data['selected_kd'][phase] = kd
        data['validated'] = False
    save(path, data)
    return {'selected_kd': data['selected_kd'], 'validated': data['validated']}


def postprocess(ws, payload):
    from .worker import run
    folder = ws.path(payload['path'])
    script = folder / 'run_postprocess.py'
    if not script.exists():
        script = ws.root / 'Scripts/run_postprocess.py'
    run([sys.executable, script, '--run-dir', folder], folder)
    return {'message': 'Postprocess chain finished. Review the log for skipped or failed optional products.'}
