"""Launch with `tomogui --workspace /path/to/TEST` or `python -m tomogui`."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import threading
import webbrowser

from .workspace import discover_workspace, initialize_workspace, runtime_info
from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(description='TomoGUI local preparation and HDF5 inspection workspace')
    parser.add_argument('--version', action='version', version='TomoGUI ' + __version__)
    project = parser.add_mutually_exclusive_group()
    project.add_argument('--workspace', type=Path, help='Existing TomoGUI project or legacy workflow workspace')
    project.add_argument('--init', type=Path, metavar='NEW_PROJECT', help='Create a new project with bundled workflow helpers; never overwrite an existing folder')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--check', action='store_true', help='Report workspace and dependency availability, then exit')
    args = parser.parse_args(argv)
    if args.init:
        root = initialize_workspace(args.init)
        print(f'Created project: {root}\nNext: tomogui --workspace "{root}" --check')
        return 0
    root = discover_workspace(args.workspace)
    if args.check:
        from .defaults import current_defaults
        modules = ('pytomoatt','h5py','numpy','yaml','scipy','pandas','matplotlib')
        dependencies = {key: bool(importlib.util.find_spec(key)) for key in modules}
        binary = os.environ.get('TOMOATT_BIN', str(Path.home() / 'TomoATT/build/bin/TOMOATT'))
        solver = shutil.which(binary)
        tools = {key: shutil.which(key) for key in ('bash','mpirun','sbatch','srun')}
        print(json.dumps({'version': __version__, 'workspace': str(root), 'runtime': runtime_info(),
                          'defaults': current_defaults(root), 'dependencies': dependencies,
                          'execution': {'TOMOATT_BIN': binary, 'solver': solver, **tools},
                          'preparation_ready': all(dependencies.values()),
                          'local_inversion_ready': all(dependencies.values()) and bool(solver and tools['mpirun']),
                          'slurm_ready': all(dependencies.values()) and bool(solver and tools['sbatch'] and tools['srun'])}, indent=2))
        return 0 if all(dependencies.values()) else 1
    from .server import create_server
    server = create_server(root, port=args.port)
    url = f'http://127.0.0.1:{server.server_port}'
    runtime = runtime_info()
    print(f'TomoGUI: {url}\nWorkspace: {root}\nRuntime: {runtime["platform"]}\nPython: {runtime["python"]}', flush=True)
    if not args.no_browser:
        threading.Timer(.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
