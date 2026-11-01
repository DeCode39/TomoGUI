"""Opt-in acceptance run on existing workspace files; never executes the solver."""
from pathlib import Path
import hashlib
import json
import shutil
import uuid

import yaml

from tomogui.data import detect
from tomogui.runs import generate, preview
from tomogui.worker import prepare_data, filter_region
from tomogui.workspace import Workspace


def digest(path):
    with path.open('rb') as handle:
        value = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
        return value.hexdigest()


def main():
    source = Path(__file__).resolve().parents[2]
    destination = source / 'TomoGUI/.test-output' / ('existing_' + uuid.uuid4().hex[:8])
    destination.mkdir()
    cases = [
        ('salute', 'SALUTEv5/simulp_evt.dat', 'SALUTEv5/simulp_sta.dat'),
        ('src_rec', 'SALUTEv5_FULL_reg/region_001/P/src_rec.dat', None),
        ('isc', 'isc_arrivals/DATA/isc_arrivals_2000-01.csv', None),
    ]
    report = []
    for name, relative, stations in cases:
        root = destination / name
        root.mkdir()
        for resource in ('Scripts', 'RESULTS', 'h5_viewer', 'custom_models'):
            (root / resource).symlink_to(source / resource, target_is_directory=True)
        original = source / relative
        before = digest(original)
        shutil.copy2(original, root / original.name)
        if stations:
            shutil.copy2(source / stations, root / Path(stations).name)
        ws = Workspace(root)
        info = detect(root / original.name, root)
        assert not ws.state_dir.exists(), 'Source preview wrote project files'
        imported = prepare_data(ws, {'path': original.name, 'stations': Path(stations).name if stations else ''})
        ws.update_state(**imported['state_updates'])
        filtered = filter_region(ws, {'phases': 'P', 'bounds': {
            'lat_min': 22, 'lat_max': 23.5, 'lon_min': 120, 'lon_max': 121.5,
            'depth_min': 0, 'depth_max': 155}, 'settings': {'require_all_azimuth_bins': False}})
        ws.update_state(**filtered['state_updates'])
        # Explicitly tune the test grid to limit disk use across 72 sweep cases.
        payload = {'name': 'Acceptance_' + name, 'phases': ['P'],
                   'grid': {'receiver_spacing': 12, 'depth_spacing': 5}}
        snapshots = set(ws.state_dir.rglob('*'))
        plan = preview(ws, payload)
        assert snapshots == set(ws.state_dir.rglob('*')), 'Folder preview wrote files'
        result = generate(ws, payload)
        record = json.loads((Path(result['root']) / 'tomogui_run.json').read_text())
        assert len(record['cases']) == plan['sweep_jobs'] == 24
        for row in record['cases']:
            params = yaml.safe_load((Path(row['run_dir']) / '3_input_params/input_params_inv.yaml').read_text())
            assert params['domain']['n_rtp'] == plan['grids']['P']['n_rtp']
            assert Path(row['true_model']).is_file()
        assert digest(original) == before, 'Original source was modified'
        report.append({'file': relative, 'detected': info['kind'],
                       'imported_events': imported['summary']['n_events'],
                       'imported_picks': imported['summary']['n_picks'],
                       'region_events': filtered['region']['summary']['n_events'],
                       'region_picks': filtered['region']['summary']['n_picks'],
                       'grid': plan['grids']['P']['n_rtp'], 'sweep_cases': len(record['cases']),
                       'root': result['root'], 'source_sha256_unchanged': True})
        (destination / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report[-1]), flush=True)
    print('Acceptance report:', destination / 'report.json', flush=True)


if __name__ == '__main__':
    main()
