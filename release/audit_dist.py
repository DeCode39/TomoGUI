"""Verify release archives contain portable resources and no research artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    archives = sorted(args.directory.glob('*.whl')) + sorted(args.directory.glob('*.tar.gz'))
    if len(archives) != 2:
        raise SystemExit('Expected exactly one wheel and one source distribution.')
    required = ['Scripts/prepare_files.py', 'h5_viewer/readers.py',
                'RESULTS/TEST/checkerboard_P/prepare_input_files.py',
                'RESULTS/TEST/inversion_S/3_input_params/input_params_joint_step2.yaml',
                'RESULTS/JpGU/templates/checkerboard_S_paper_base/prepare_input_files.py',
                'RESULTS/resolution_kd_sweep/score_resolution_kd_sweep.py', 'provenance.json', 'LICENSE']
    for archive in archives:
        if archive.suffix == '.whl':
            with zipfile.ZipFile(archive) as handle:
                names = handle.namelist()
        else:
            with tarfile.open(archive) as handle:
                names = handle.getnames()
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts:
                raise SystemExit(f'Unsafe archive entry: {name}')
            if any(part in {'.git', '.venv', '.test-output', '.tomogui', '__pycache__'} for part in path.parts):
                raise SystemExit(f'Local artifact included: {name}')
            if path.suffix.lower() in {'.pyc', '.pyo', '.h5', '.hdf5', '.dat', '.csv'}:
                raise SystemExit(f'Data/cache file included: {name}')
        for relative in required:
            if not any(name.endswith('/tomogui/resources/workflow/' + relative) for name in names):
                raise SystemExit(f'Missing packaged resource: {relative}')
        print(f'Archive audit passed: {archive.name} ({len(names)} entries)')


if __name__ == '__main__':
    main()
