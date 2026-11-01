"""Refresh the explicitly selected, data-free workflow sources for a release."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='Original workflow workspace')
    args = parser.parse_args()
    source = args.source.resolve()
    package = Path(__file__).resolve().parents[1]
    target = package / 'src/tomogui/resources/workflow'
    paths = [f'Scripts/{name}' for name in (
        'prepare_files.py', 'region_scanner_pipeline.py', 'region_scanner_cwb_square.py',
        'convert_salute_to_src_rec.py', 'convert_cwb_to_src_rec.py', 'generate_vpvs_folder.py',
        'plot_qc.py', 'build_azcov_archive.py', 'plot_azimuthal_coverage.py',
        'plot_single_ray_trace.py', 'run_postprocess.py', 'clean_run_results.sh')]
    paths += [f'h5_viewer/{name}.py' for name in ('__init__', 'readers', 'data_model', 'expression_engine', 'utils')]
    paths += ['custom_models/build_custom_model.py', 'custom_models/model_definitions/dual_anisotropy_ns_ew.txt']
    for phase in ('P', 'S'):
        for kind in ('checkerboard', 'inversion'):
            folder = source / f'RESULTS/TEST/{kind}_{phase}'
            paths += [str(p.relative_to(source)).replace('\\', '/') for p in sorted(folder.glob('*.py'))]
            paths += [str(p.relative_to(source)).replace('\\', '/') for p in sorted(folder.glob('*.sh'))]
            paths += [str(p.relative_to(source)).replace('\\', '/') for p in sorted((folder / '3_input_params').glob('*.yaml'))]
        paths.append(f'RESULTS/JpGU/templates/checkerboard_{phase}_paper_base/prepare_input_files.py')
    paths += ['RESULTS/TEST/inversion_VpVs/plot_output.py', 'RESULTS/TEST/inversion_VpVs/3_input_params/input_params_ref.yaml']
    paths += [f'RESULTS/resolution_kd_sweep/{name}.py' for name in (
        'prepare_resolution_kd_sweep', 'score_resolution_kd_sweep', 'resolution_window')]
    manifest = []
    for relative in sorted(set(paths)):
        original = source / relative
        raw = original.read_bytes()
        text = raw.decode('utf-8-sig')
        changes = []
        if relative.endswith('.sh'):
            text = re.sub(r'^#SBATCH\s+(?:-p\s|--partition(?:=|\s)|--account(?:=|\s)).*\n', '', text, flags=re.M)
            text = text.replace('source ~/miniforge3/etc/profile.d/conda.sh\nconda activate tomoatt\n',
                                '# Use the Python/MPI environment from which this job is submitted.\n')
            changes.append('Removed site-specific Slurm account/partition and conda activation')
        # Copy sources as ASCII, preserving escaped Unicode in strings/comments.
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text.encode('ascii', 'backslashreplace').decode('ascii'), encoding='ascii', newline='\n')
        manifest.append({'source': relative, 'source_sha256': hashlib.sha256(raw).hexdigest(), 'changes': changes})
    relative = 'RESULTS/JpGU/build_jpgu_resolution_sweeps.py'
    raw = (source / relative).read_bytes()
    tree = ast.parse(raw.decode('utf-8'))
    values = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'KDENSITY_VALUES' for t in n.targets))
    (target / relative).write_text('"""Kd settings snapshot from the original JpGU builder."""\nKDENSITY_VALUES = '
                                   + repr(values) + '\n', encoding='ascii')
    manifest.append({'source': relative, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                     'changes': ['Only the KDENSITY_VALUES setting is needed and included']})
    for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md'):
        (target / name).write_bytes((package / name).read_bytes())
    (target / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='ascii')
    print(f'Bundled {len(manifest)} selected source/configuration files; no catalogs or model/results data.')


if __name__ == '__main__':
    main()
