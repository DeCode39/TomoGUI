"""Paired ABS and CS/CR resolution experiments around the baseline final model."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import yaml

from .checkerboards import PRESETS, build
from .differential import clone_inputs, install_runner, set_data_flags, write_json
from .workspace import load_module


def stage_resolution(ws, data, dd):
    from .runs import runner, patch_scalar
    cases, signals = [], {}
    for phase, folder_text in dd['phases'].items():
        refinement = Path(folder_text)
        config = json.loads((refinement / 'tomogui_differential.json').read_text())
        grid = data['grids'][phase]
        helper = load_module(ws.root / f'RESULTS/TEST/checkerboard_{phase}/prepare_input_files.py')
        sweep = load_module(ws.root / 'RESULTS/resolution_kd_sweep/prepare_resolution_kd_sweep.py')
        for count in PRESETS:
            base = Path(dd['root']) / f'resolution_{phase}' / f'ckb{count}{count}{count}'
            signal = base / 'signal'
            clone_inputs(Path(data['root']) / f'checkerboard_{phase}', signal)
            initial = signal / '2_models/baseline_final.h5'
            shutil.copy2(refinement / '2_models/baseline_final.h5', initial)
            true = signal / '2_models/true_model.h5'
            signal_yaml = signal / '3_input_params/input_params_signal.yaml'
            patch_scalar(signal_yaml, 'init_model_path', '2_models/true_model.h5')
            patch_scalar(signal_yaml, 'src_rec_file', config['baseline_observed'])
            model_options = {**helper._base_checkerboard_defaults(),
                             'n_pert_x':count,'n_pert_y':count,'n_pert_z':count}
            build(helper, initial, signal_yaml, grid, model_options, true)
            sweep.patch_noise_seed(signal / 'assign_gaussian_noise.py', 170500 + count)
            runner(signal, 8, True, model_options, dual=True, signal_only=True)
            key = phase + f'_{count}'
            signals[key] = str(signal)
            noisy = signal / 'OUTPUT_FILES/OUTPUT_FILES_signal/src_rec_file_forward_noisy.dat'
            for branch in ('absolute','differential'):
                folder = base / branch
                clone_inputs(refinement, folder)
                # Copy the immutable config explicitly; clone_inputs excludes JSON.
                branch_config = {**config, 'baseline_observed': str(noisy),
                                 'eligibility_catalog': config['baseline_observed']}
                if branch == 'differential':
                    write_json(folder / 'tomogui_differential.json', branch_config)
                for step in (1,2,3):
                    param = folder / f'3_input_params/input_params_joint_step{step}.yaml'
                    params = yaml.safe_load(param.read_text())
                    if branch == 'absolute':
                        if step == 1:
                            params['source']['src_rec_file'] = str(noisy)
                        else:
                            # Restore the maintained observed-file handoff tags.
                            original = yaml.safe_load((Path(data['root']) / f'inversion_{phase}' /
                                f'3_input_params/input_params_joint_step{step}.yaml').read_text())
                            params['source']['src_rec_file'] = original['source']['src_rec_file']
                    param.write_text(yaml.safe_dump(params, sort_keys=False), encoding='ascii')
                if branch == 'absolute':
                    set_data_flags(folder, refinement=False)
                runner(folder, 8, False, {}, dual=True, shared_signal=str(noisy))
                install_runner(folder, refinement=branch == 'differential')
                params = yaml.safe_load((folder / '3_input_params/input_params_joint_step2.yaml').read_text())
                cases.append({'phase': phase, 'branch': branch, 'counts': [count]*3,
                    'kdensity_coe': data['selected_kd'][phase],
                    'run_tag': phase + f'_{count}_' + branch, 'wave_tag': f'ckb{count}', 'wave_weight': 1,
                    'signal_key': key, 'run_dir': str(folder), 'initial_model': str(initial),
                    'true_model': str(true), 'recovered_model': str(folder / params['output_setting']['output_dir'] / 'final_model.h5')})
    return {'cases': cases, 'signals': signals}


def score(ws, payload):
    from .runs import manifest, save, compute_model_scores
    from .differential import sha256, check_inputs, check_settings
    path, data = manifest(ws)
    dd = data.get('differential')
    if not dd:
        raise ValueError('Prepare differential resolution first.')
    check_settings(dd, payload)
    check_inputs(dd)
    fingerprints = {}
    for row in dd['cases']:
        marker = Path(row['run_dir']) / 'tomogui_completed.json'
        if not marker.is_file():
            raise ValueError('All three stages of every paired resolution case must complete before scoring.')
        completed = json.loads(marker.read_text())
        if completed.get('state') != 'completed':
            raise ValueError('All three stages of every paired resolution case must complete before scoring.')
        fingerprints[str(marker)] = sha256(marker)
        for key in ('model','observed'):
            if sha256(completed[key]) != completed[key + '_sha256']:
                raise ValueError('Resolution outputs changed after completion.')
            fingerprints[completed[key]] = completed[key + '_sha256']
    scores = compute_model_scores(ws, dd['cases'], Path(dd['root']), relative_anisotropy=True)
    by_tag = {row['run_tag']: row for row in scores}
    comparisons = []
    for phase in data['phases']:
        for count in PRESETS:
            control = by_tag[f'{phase}_{count}_absolute']
            refined = by_tag[f'{phase}_{count}_differential']
            comparisons.append({'phase': phase, 'count': count, 'absolute_score': control['score'],
                                'differential_score': refined['score'], 'change': refined['score']-control['score']})
    dd['scores'] = scores
    dd['scored_outputs'] = fingerprints
    dd['comparisons'] = comparisons
    dd['validated'] = False
    save(path, data)
    return {'comparisons': comparisons, 'message': 'Paired recovery scored. Review velocity/anisotropy masks and scores before accepting.'}
